import copy
from components.episode_buffer import EpisodeBatch
from functools import partial
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.flex_qmix import FlexQMixer, LinearFlexQMixer
from components.action_selectors import parse_avail_actions
import torch as th
import torch.nn.functional as F
import torch.distributions as D
from torch.optim import RMSprop, Adam


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.use_copa = self.args.hier_agent['copa']

        self.params = list(mac.parameters())
        if self.use_copa:
            self.params += list(self.mac.coach.parameters())
            if self.args.hier_agent['copa_vi_loss']:
                self.params += list(self.mac.copa_recog.parameters())

        self.last_target_update_episode = 0
        self.last_alloc_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "flex_qmix":
                assert args.entity_scheme, "FlexQMixer only available with entity scheme"
                self.mixer = FlexQMixer(args)
            elif args.mixer == "lin_flex_qmix":
                assert args.entity_scheme, "FlexQMixer only available with entity scheme"
                self.mixer = LinearFlexQMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps,
                                 weight_decay=args.weight_decay)

        if self.args.hier_agent["task_allocation"] == "aql":
            self.alloc_pi_params = list(mac.alloc_pi_params())
            if self.args.hier_agent["alloc_opt"] == "rmsprop":
                OptClass = partial(RMSprop, alpha=args.optim_alpha)
            elif self.args.hier_agent["alloc_opt"] == "adam":
                OptClass = Adam
            else:
                raise Exception("Optimizer not recognized")
            self.alloc_pi_optimiser = OptClass(
                params=self.alloc_pi_params, lr=args.lr, eps=args.optim_eps,
                weight_decay=args.weight_decay)
            self.alloc_q_params = list(mac.alloc_q_params())
            self.alloc_q_optimiser = OptClass(
                params=self.alloc_q_params, lr=args.lr, eps=args.optim_eps,
                weight_decay=args.alloc_q_weight_decay)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.log_alloc_stats_t = -self.args.learner_log_interval - 1

    def _get_mixer_ins(self, batch):
        if not self.args.entity_scheme:
            return (batch["state"][:, :-1],
                    batch["state"][:, 1:])
        else:
            entities = []
            bs, max_t, ne, ed = batch["entities"].shape
            entities.append(batch["entities"])
            if self.args.entity_last_action:
                last_actions = th.zeros(bs, max_t, ne, self.args.n_actions,
                                        device=batch.device,
                                        dtype=batch["entities"].dtype)
                last_actions[:, 1:, :self.args.n_agents] = batch["actions_onehot"][:, :-1]
                entities.append(last_actions)

            entities = th.cat(entities, dim=3)
            mix_ins = {"entities": entities[:, :-1],
                       "entity_mask": batch["entity_mask"][:, :-1]}
            targ_mix_ins = {"entities": entities[:, 1:],
                            "entity_mask": batch["entity_mask"][:, 1:]}
            if self.args.multi_task:
                # use same subtask assignments for prediction and target
                mix_ins["entity2task_mask"] = batch["entity2task_mask"][:, :-1]
                targ_mix_ins["entity2task_mask"] = batch["entity2task_mask"][:, :-1]
            return mix_ins, targ_mix_ins

    def _make_meta_batch(self, batch: EpisodeBatch):
        reward = batch['reward']
        terminated = batch['terminated'].float()
        reset = batch['reset'].float()
        mask = batch['filled'].float()
        allocs = 1 - batch['entity2task_mask'][:, :, :self.args.n_agents].float()
        mask[:, 1:] = mask[:, 1:] * (1 - reset[:, :-1])
        bs, ts, _ = mask.shape
        t_added = batch['t_added'].reshape(bs, 1, 1).repeat(1, ts, 1)

        timeout = reset - terminated

        decision_points = batch['hier_decision'].float()

        seg_rewards = th.zeros_like(reward)
        cuml_rewards = th.zeros_like(reward[:, 0])
        seg_terminated = th.zeros_like(terminated)
        cuml_terminated = th.zeros_like(terminated[:, 0])
        cuml_timeout = th.zeros_like(timeout[:, 0])

        for t in reversed(range(reward.shape[1])):
            # sum rewards between hierarchical decision points
            cuml_rewards += reward[:, t]
            seg_rewards[:, t] = cuml_rewards
            cuml_rewards *= 1 - decision_points[:, t]

            # track whether env terminated between decision points
            cuml_terminated = cuml_terminated.max(terminated[:, t])
            seg_terminated[:, t] = cuml_terminated
            cuml_terminated *= 1 - decision_points[:, t]

            # mask out decision point if a env timeout happens (since we can't bootstrap from next decision point)
            cuml_timeout = cuml_timeout.max(timeout[:, t])
            mask[:, t] *= (1 - cuml_timeout)
            cuml_timeout *= 1 - decision_points[:, t]

        # scale by action length to keep gradients around same magnitude as low-level controllers
        seg_rewards /= self.args.hier_agent['action_length']

        last_alloc = th.zeros_like(allocs)
        was_reset = th.zeros_like(reset[:, [0]])
        for t in range(1, reward.shape[1]):
            # make sure that last_alloc doesn't copy final assignment from previous episode
            was_reset = (was_reset + reset[:, [t - 1]]).min(th.ones_like(was_reset))
            last_alloc[:, t] = allocs[:, t - 1] * (1 - was_reset)
            was_reset *= (1 - decision_points[:, [t]])

        # mask out last decision point in each trajectory if not terminal state (since we can't bootstrap)
        bs, ts, _ = decision_points.shape
        last_dp_ind = (
            decision_points * th.arange(
                ts, dtype=decision_points.dtype,
                device=decision_points.device).reshape(1, ts, 1)
        ).squeeze().argmax(dim=1)
        mask[th.arange(bs), last_dp_ind] *= seg_terminated[th.arange(bs), last_dp_ind]

        entity2task_mask = batch['entity2task_mask'].clone()

        d_inds = (decision_points == 1).reshape(bs, ts)
        max_bs = self.args.hier_agent['max_bs']
        meta_batch = {
            'reward': seg_rewards[d_inds][:max_bs],
            'terminated': seg_terminated[d_inds][:max_bs],
            'mask': mask[d_inds][:max_bs],
            'entities': batch['entities'][d_inds][:max_bs],
            'obs_mask': batch['obs_mask'][d_inds][:max_bs],
            'entity_mask': batch['entity_mask'][d_inds][:max_bs],
            'entity2task_mask': entity2task_mask[d_inds][:max_bs],
            'task_mask': batch['task_mask'][d_inds][:max_bs],
            'avail_actions': batch['avail_actions'][d_inds][:max_bs],
            'last_alloc': last_alloc[d_inds][:max_bs],
            't_added': t_added[d_inds][:max_bs],
        }
        return meta_batch

    def alloc_train_aql(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        meta_batch = self._make_meta_batch(batch)
        rewards = meta_batch['reward']
        terminated = meta_batch['terminated']
        mask = meta_batch['mask']
        stats = {}

        # Compute Q-values (evaluate task allocation stored in entity2task_mask)
        alloc_q, q_stats = self.mac.evaluate_allocation(meta_batch, calc_stats=True)

        # Compute proposal allocations (test_mode=True to remove stochasticity in critic, pass in target_mac for stability in bootstrap targets)
        new_alloc, pi_stats = self.mac.compute_allocation(meta_batch, calc_stats=True, test_mode=True, target_mac=self.target_mac)

        # Compute target Q-values
        target_alloc_q = pi_stats['targ_best_prop_values']
        target_alloc_q = self.target_mac.alloc_critic.denormalize(target_alloc_q)

        # Compute TD-loss (don't bootstrap from next state if previous state is
        # terminal)
        targets = (rewards[:-1] + self.args.gamma * (1 - terminated[:-1]) * target_alloc_q[1:]).detach()
        if self.args.popart:
            targets = self.mac.alloc_critic.popart_update(
                targets, mask[:-1])

        td_error = (alloc_q[:-1] - targets.detach())
        td_mask = mask[:-1].expand_as(td_error)
        if self.args.hier_agent['decay_old'] > 0:
            cutoff = self.args.hier_agent['decay_old']
            ratio = (cutoff - t_env + meta_batch['t_added'][:-1].float()) / cutoff
            ratio = ratio.max(th.zeros_like(ratio))
            td_mask *= ratio
        masked_td_error = td_error * td_mask
        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error ** 2).sum() / td_mask.sum()
        stats['losses/alloc_q_loss'] = td_loss.cpu().item()

        # backprop Q loss
        q_loss = td_loss
        self.alloc_q_optimiser.zero_grad()
        q_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.alloc_q_params, self.args.grad_norm_clip)
        stats['train_metrics/alloc_q_grad_norm'] = grad_norm
        self.alloc_q_optimiser.step()

        # Log allocation metrics
        stats['alloc_metrics/best_prob'] = pi_stats['best_prob'].mean().cpu().item()
        # Compute what % of agents changed their task allocation (if previous alloc exists)
        active_ag = 1 - meta_batch['entity_mask'][:, :self.args.n_agents].float()
        ag_changed = (meta_batch['last_alloc'].argmax(dim=2) != new_alloc.detach().argmax(dim=2)).float()
        prev_al_exists = (meta_batch['last_alloc'].sum(dim=(1, 2)) >= 1).float()
        perc_changed_per_step = ((ag_changed * active_ag).sum(dim=1) / active_ag.sum(dim=1))
        perc_changed = (perc_changed_per_step * prev_al_exists).sum() / prev_al_exists.sum()
        stats['alloc_metrics/perc_ag_changed'] = perc_changed.cpu().item()
        # Measure abs value of difference between # of agents and # of entities
        # in each subtask (may not be useful for all tasks)
        nonagent2task = 1 - meta_batch['entity2task_mask'][:, self.args.n_agents:].float()
        ag_per_task = new_alloc.detach().sum(dim=1)
        nag_per_task = nonagent2task.sum(dim=1)
        absdiff_per_task = (ag_per_task - nag_per_task).abs()
        abs_diff_mean = absdiff_per_task.sum(dim=1) / (1 - meta_batch['task_mask'].float()).sum(dim=1)
        stats['alloc_metrics/ag_task_concentration'] = abs_diff_mean.mean().cpu().item()

        # Maximize probability of best allocation
        all_prop_log_pi = pi_stats['log_pi']  # log_pi of all sampled proposal actions
        bs = all_prop_log_pi.shape[0]
        best_prop_log_pi = all_prop_log_pi[th.arange(bs), pi_stats['best_prop_inds']]
        amort_step_loss = -best_prop_log_pi
        masked_amort_step_loss = amort_step_loss * mask
        amort_loss = masked_amort_step_loss.sum() / mask.sum()
        stats['losses/alloc_amort_loss'] = amort_loss.cpu().item()


        active_task = 1 - meta_batch['task_mask'].float().unsqueeze(1)
        ag2task = pi_stats['all_allocs'].detach()  # (bs, n_prop, na, nt)
        task_has_agents = (ag2task.sum(dim=2) > 0).float()
        any_task_no_agents = (task_has_agents.sum(dim=2, keepdim=True)
                              != active_task.sum(dim=2, keepdim=True)).float()
        stats['alloc_metrics/any_task_no_agents_pi'] = any_task_no_agents.mean().cpu().item()

        # entropy term
        entropy = pi_stats['entropy']
        entropy_loss = -entropy.mean()
        stats['losses/alloc_entropy'] = -entropy_loss.cpu().item()

        pi_loss = (amort_loss
                   + self.args.hier_agent['entropy_loss'] * entropy_loss)

        # backprop policy loss
        self.alloc_pi_optimiser.zero_grad()
        pi_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.alloc_pi_params, self.args.grad_norm_clip)
        stats['train_metrics/alloc_pi_grad_norm'] = grad_norm
        self.alloc_pi_optimiser.step()

        if (episode_num - self.last_alloc_target_update_episode) / self.args.alloc_target_update_interval >= 1.0:
            self._update_alloc_targets()
            self.last_alloc_target_update_episode = episode_num

        if t_env - self.log_alloc_stats_t >= self.args.learner_log_interval:
            for name, value in stats.items():
                self.logger.log_stat(name, value, t_env)
            self.log_alloc_stats_t = t_env

        return stats, new_alloc

    def _broadcast_decisions_to_batch(self, decisions, decision_pts):
        decision_pts = decision_pts.squeeze(-1)
        bs, ts = decision_pts.shape
        bcast_decisions = {k: th.zeros_like(v[[0]]).unsqueeze(0).repeat(bs * rep, ts, *(1 for _ in range(len(v.shape) - 1))) for k, (v, rep) in decisions.items()}
        for decname in bcast_decisions:
            value, rep = decisions[decname]
            bcast_decisions[decname][decision_pts.repeat(rep, 1)] = value
        for t in range(1, ts):
            for decname in bcast_decisions:
                rep = decisions[decname][1]
                prev_value = bcast_decisions[decname][:, t - 1]
                bcast_decisions[decname][:, t] = ((decision_pts[:, t].repeat(rep).reshape(bs * rep, 1, 1).to(prev_value.dtype) * bcast_decisions[decname][:, t])
                                                  + ((1 - decision_pts[:, t].repeat(rep).reshape(bs * rep, 1, 1)).to(prev_value.dtype) * prev_value))
        return bcast_decisions

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        # episode over (not including timeout) - determines when to bootstrap
        terminated = batch["terminated"][:, :-1].float()
        # env reset (either terminated or timed out) - determines what timesteps
        # to learn from - we can't learn from final ts bc there is no
        # transition
        reset = batch["reset"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - reset[:, :-1])
        org_mask = mask.clone()
        avail_actions = batch["avail_actions"]
        if self.args.agent['subtask_cond'] is not None:
            # Learning separate controllers for each task
            rewards = batch['task_rewards'][:, :-1]
            terminated = batch['tasks_terminated'][:, :-1].float()
            mask = mask.repeat(1, 1, self.args.n_tasks)
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            task_has_agents = (1 - batch['entity2task_mask'][:, :-1, :self.args.n_agents]).sum(2) > 0
            mask *= task_has_agents.float()

        # # Calculate estimated Q-Values
        # mac_out = []
        self.mac.init_hidden(batch.batch_size)
        # enable things like dropout on mac and mixer, but not target_mac and target_mixer
        self.mac.train()
        self.target_mac.eval()
        if self.mixer is not None:
            self.mixer.train()
            self.target_mixer.eval()

        coach_h = None
        targ_coach_h = None
        coach_z = None
        targ_coach_z = None

        imagine_inps = None
        if self.args.agent['imagine']:
            imagine_inps, imagine_groups = self.mac.agent.make_imagined_inputs(batch)
        if self.use_copa:
            coach_h = self.mac.coach.encode(batch, imagine_inps=imagine_inps)
            targ_coach_h = self.target_mac.coach.encode(batch)
            decision_points = batch['hier_decision'].squeeze(-1)
            bs_rep = 1
            if self.args.agent['imagine']:
                bs_rep = 3
            coach_h_t0 = coach_h[decision_points.repeat(bs_rep, 1)]
            targ_coach_h_t0 = targ_coach_h[decision_points]
            coach_z_t0, coach_mu_t0, coach_logvar_t0 = self.mac.coach.strategy(coach_h_t0)
            coach_mu_t0 = coach_mu_t0.chunk(bs_rep, dim=0)[0]
            coach_logvar_t0 = coach_logvar_t0.chunk(bs_rep, dim=0)[0]
            targ_coach_z_t0, _, _ = self.target_mac.coach.strategy(targ_coach_h_t0)

            bcast_ins = {
                'coach_z_t0': (coach_z_t0, bs_rep),
                'coach_mu_t0': (coach_mu_t0, 1),
                'coach_logvar_t0': (coach_logvar_t0, 1),
                'targ_coach_z_t0': (targ_coach_z_t0, 1),
            }
            bcast_decisions = self._broadcast_decisions_to_batch(bcast_ins, batch['hier_decision'])
            coach_z = bcast_decisions['coach_z_t0']
            coach_mu = bcast_decisions['coach_mu_t0']
            coach_logvar = bcast_decisions['coach_logvar_t0']
            targ_coach_z = bcast_decisions['targ_coach_z_t0']


        batch_mult = 1
        if self.args.agent['imagine']:
            batch_mult += 2

        all_mac_out, mac_info = self.mac.forward(
            batch, t=None,
            coach_z=coach_z,
            imagine_inps=imagine_inps)
        rep_actions = actions.repeat(batch_mult, 1, 1, 1)
        all_chosen_action_qvals = th.gather(all_mac_out[:, :-1], dim=3, index=rep_actions).squeeze(3)  # Remove the last dim

        mac_out_tup = all_mac_out.chunk(batch_mult, dim=0)
        caq_tup = all_chosen_action_qvals.chunk(batch_mult, dim=0)

        mac_out = mac_out_tup[0]
        chosen_action_qvals = caq_tup[0]
        if self.args.agent['imagine']:
            caq_imagine = th.cat(caq_tup[1:], dim=2)

        self.target_mac.init_hidden(batch.batch_size)

        target_mac_out, _ = self.target_mac.forward(batch, coach_z=targ_coach_z, t=None, target=True)
        if self.args.agent['subtask_cond'] is not None:
            allocs = (1 - batch['entity2task_mask'][:, :, :self.args.n_agents])
            avail_actions_targ = parse_avail_actions(avail_actions[:, 1:], allocs[:, :-1], self.args)
        else:
            avail_actions_targ = avail_actions[:, 1:]
        target_mac_out = target_mac_out[:, 1:]

        # Mask out unavailable actions
        target_mac_out[avail_actions_targ == 0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()[:, 1:]
            mac_out_detach[avail_actions_targ == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            mix_ins, targ_mix_ins = self._get_mixer_ins(batch)

            chosen_action_qvals = self.mixer(chosen_action_qvals, mix_ins)
            gamma = self.args.gamma

            target_max_qvals = self.target_mixer(target_max_qvals, targ_mix_ins)
            target_max_qvals = self.target_mixer.denormalize(target_max_qvals)
            # Calculate 1-step Q-Learning targets
            targets = (rewards + gamma * (1 - terminated) * target_max_qvals).detach()
            if self.args.popart:
                targets = self.mixer.popart_update(
                    targets, mask)

            if self.args.agent['imagine']:
                # don't need last timestep
                imagine_groups = [gr[:, :-1] for gr in imagine_groups]
                caq_imagine = self.mixer(caq_imagine, mix_ins,
                                         imagine_groups=imagine_groups)
        else:
            targets = (rewards + self.args.gamma * (1 - terminated) * target_max_qvals).detach()

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        if self.args.agent['imagine']:
            im_prop = self.args.lmbda
            im_td_error = (caq_imagine - targets.detach())
            im_masked_td_error = im_td_error * mask
            im_loss = (im_masked_td_error ** 2).sum() / mask.sum()
            loss = (1 - im_prop) * loss + im_prop * im_loss

        if self.use_copa and self.args.hier_agent['copa_vi_loss']:
            # VI loss
            q_mu, q_logvar = self.mac.copa_recog(batch)
            q_t = D.normal.Normal(q_mu, (0.5 * q_logvar).exp())
            coach_z = coach_z.chunk(bs_rep, dim=0)[0]  # if combining with REFIL, only train full info Z
            log_prob = q_t.log_prob(coach_z).clamp_(-1000, 0).sum(-1)
            # entropy loss
            p_ = D.normal.Normal(coach_mu, (0.5 * coach_logvar).exp())
            entropy = p_.entropy().clamp_(0, 10).sum(-1)

            # mask inactive agents
            agent_mask = 1 - batch['entity_mask'][:, :, :self.args.n_agents].float()
            log_prob = (log_prob * agent_mask).sum(-1) / (agent_mask.sum(-1) + 1e-8)
            entropy = (entropy * agent_mask).sum(-1) / (agent_mask.sum(-1) + 1e-8)

            vi_loss = (-log_prob[:, :-1] * org_mask.squeeze(-1)).sum() / org_mask.sum()
            entropy_loss = (-entropy[:, :-1] * org_mask.squeeze(-1)).sum() / org_mask.sum()
            
            loss += vi_loss * self.args.vi_lambda + entropy_loss * self.args.vi_lambda / 10

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("losses/q_loss", loss.item(), t_env)
            if self.args.agent['imagine']:
                self.logger.log_stat("losses/im_loss", im_loss.item(), t_env)
            if self.use_copa and self.args.hier_agent['copa_vi_loss']:
                self.logger.log_stat("losses/copa_vi_loss", vi_loss.item(), t_env)
                self.logger.log_stat("losses/copa_entropy_loss", entropy_loss.item(), t_env)
            self.logger.log_stat("train_metrics/q_grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("train_metrics/td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("train_metrics/q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("train_metrics/target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def _update_alloc_targets(self):
        self.target_mac.load_alloc_state(self.mac)
        self.logger.console_logger.info("Updated allocation target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}opt.th".format(path))

    def load_models(self, path, pi_only=False, evaluate=False):
        self.mac.load_models(path, pi_only=pi_only)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path, pi_only=pi_only)
        if not evaluate and not pi_only:
            if self.mixer is not None:
                self.mixer.load_state_dict(th.load("{}mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.optimiser.load_state_dict(th.load("{}opt.th".format(path), map_location=lambda storage, loc: storage))

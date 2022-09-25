from .basic_controller import BasicMAC
import torch as th
from modules.agents import ALLOC_CRITIC_REGISTRY, ALLOC_POLICY_REGISTRY
from modules.agents.copa import Coach, RecognitionModel
from components.epsilon_schedules import DecayThenFlatSchedule
from utils.allocation import random_allocs


# This multi-agent controller shares parameters between agents and takes
# entities + observation masks as input
class EntityMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(EntityMAC, self).__init__(scheme, groups, args)
        self.task_allocations = None
        e_start_val, e_finish_val, e_decay_time_frac = args.hier_agent['alloc_eps'].split('-')
        e_decay_time = int(float(e_decay_time_frac) * self.args.t_max)
        self.alloc_eps_schedule = DecayThenFlatSchedule(float(e_start_val), float(e_finish_val), e_decay_time, decay="linear")
        self.alloc_eps = float(e_start_val)

        self.prop_alloc_eps = 0
        self.prop_alloc_eps_schedule = None
        if args.hier_agent['prop_alloc_eps'] != '':
            ce_start_val, ce_finish_val, ce_decay_time_frac = args.hier_agent['prop_alloc_eps'].split('-')
            ce_decay_time = int(float(ce_decay_time_frac) * self.args.t_max)
            self.prop_alloc_eps_schedule = DecayThenFlatSchedule(float(ce_start_val), float(ce_finish_val), ce_decay_time, decay="linear")
            self.prop_alloc_eps = float(ce_start_val)

    def alloc_pi_params(self):
        return self.alloc_policy.parameters()

    def alloc_q_params(self):
        return self.alloc_critic.parameters()

    def compute_allocation(self, meta_batch, t_env=None, t_ep=None, acting=False,
                           test_mode=False, calc_stats=False, target_mac=None,
                           **kwargs):
        all_allocs = self.alloc_policy(meta_batch, calc_stats=calc_stats, n_proposals=self.args.hier_agent['n_proposals'], test_mode=test_mode, **kwargs)
        if calc_stats:
            all_allocs, stats = all_allocs
            stats['all_allocs'] = all_allocs
        # evaluate allocations and take best
        evaluations = self.evaluate_allocation(meta_batch, override_alloc=all_allocs, test_mode=test_mode)
        evaluations = evaluations.squeeze(2)
        best_prop_inds = evaluations.argmax(dim=1)
        allocs = all_allocs[th.arange(all_allocs.shape[0]), best_prop_inds]
        if calc_stats:
            stats['best_prop_inds'] = best_prop_inds  # used to select which action to maximize log_prob of
            if target_mac is not None:
                stats['targ_best_prop_values'] = target_mac.evaluate_allocation(
                    meta_batch, override_alloc=allocs, test_mode=test_mode)

        if acting and not test_mode:
            # epsilon greedy
            rand_allocs = random_allocs(meta_batch['task_mask'], meta_batch['entity_mask'], self.n_agents)
            assert t_env is not None, "Must provide t_env for epsilon greedy exploration schedule"
            assert t_ep is not None, "Must provide t_ep for epsilon greedy exploration schedule"
            self.alloc_eps = self.alloc_eps_schedule.eval(t_env)
            if self.prop_alloc_eps_schedule is not None:
                self.prop_alloc_eps = self.prop_alloc_eps_schedule.eval(t_env)
            # We do pure random sampling independently per agent, while we
            # copy entire proposed allocations (since they're sampled
            # autoregressively and agents' assignments depend on other
            # agents). Pure random will overwrite proposals, as it is
            # applied second.
            prop_draw = th.rand_like(rand_allocs[:, 0, 0])
            prop_eps_mask = (prop_draw <= self.prop_alloc_eps)
            allocs[prop_eps_mask] = all_allocs[:, 0][prop_eps_mask]  # take randomly sampled proposal action

            eps_draw = th.rand_like(rand_allocs[:, :, 0])
            eps_mask = (eps_draw <= self.alloc_eps)
            allocs[eps_mask] = rand_allocs[eps_mask]
        if calc_stats:
            return allocs, stats
        return allocs


    def evaluate_allocation(self, meta_batch, **kwargs):
        return self.alloc_critic(meta_batch, **kwargs)

    def _make_meta_batch(self, ep_batch, t_ep):
        # Add quantities necessary for meta-controller (only used for acting as
        # this doesn't compute rewards/term, etc.)
        decision_pts = ep_batch['hier_decision'][:, t_ep].flatten()
        d_inds = (decision_pts == 1)
        meta_batch = {
            'entities': ep_batch['entities'][d_inds, t_ep],
            'obs_mask': ep_batch['obs_mask'][d_inds, t_ep],
            'entity_mask': ep_batch['entity_mask'][d_inds, t_ep],
            'entity2task_mask': ep_batch['entity2task_mask'][d_inds, t_ep],
            'task_mask': ep_batch['task_mask'][d_inds, t_ep],
            'avail_actions': ep_batch['avail_actions'][d_inds, t_ep],
        }
        if self.learned_alloc:
            meta_batch['last_alloc'] = self.task_allocations[d_inds]
        return meta_batch

    def _build_agents(self, input_shapes):
        agent_input_shape, hier_input_shape = input_shapes
        super()._build_agents(agent_input_shape)
        if self.learned_alloc:
            self.alloc_critic = ALLOC_CRITIC_REGISTRY[self.args.hier_agent['alloc_critic']](hier_input_shape, self.args)
            self.alloc_policy = ALLOC_POLICY_REGISTRY[self.args.hier_agent['alloc_policy']](hier_input_shape, self.args)
        if self.use_copa:
            self.coach = Coach(self.args)
            if self.args.hier_agent['copa_vi_loss']:
                self.copa_recog = RecognitionModel(self.args)


    def _build_inputs(self, batch, t, target=False, imagine_inps=None):
        # Assumes homogenous agents with entity + observation mask inputs.
        bs = batch.batch_size
        entity_parts = []
        entity_parts.append(batch["entities"][:, t])  # bs, ts, n_entities, vshape
        if self.args.entity_last_action:
            ent_acs = th.zeros(bs, t.stop - t.start, self.args.n_entities,
                               self.args.n_actions, device=batch.device,
                               dtype=batch["entities"].dtype)
            if t.start == 0:
                ent_acs[:, 1:, :self.args.n_agents] = (
                    batch["actions_onehot"][:, slice(0, t.stop - 1)])
            else:
                ent_acs[:, :, :self.args.n_agents] = (
                    batch["actions_onehot"][:, slice(t.start - 1, t.stop - 1)])
            entity_parts.append(ent_acs)
        entities = th.cat(entity_parts, dim=3)
        if imagine_inps is not None:
            imagine_inps['entities'] = entities.repeat(2, 1, 1, 1)
            imagine_inps['reset'] = imagine_inps['reset'].float()
        rets = {'entities': entities,
                'obs_mask': batch["obs_mask"][:, t],
                'entity_mask': batch["entity_mask"][:, t],
                'reset': batch["reset"][:, t].float()}
        if self.args.multi_task:
            rets['entity2task_mask'] = batch['entity2task_mask'][:, t]
            rets['task_mask'] = batch['task_mask'][:, t]
            if target:
                # if we're computing a bootstrapping target, use the task
                # assignments from the previous step
                task_t = slice(max(t.start - 1, 0), t.stop - 1)
                rets['entity2task_mask'] = batch['entity2task_mask'][:, task_t]
                if rets['entity2task_mask'].shape[1] < (t.stop - t.start):
                    rets['entity2task_mask'] = th.cat([rets['entity2task_mask'][:, [0]],
                                                       rets['entity2task_mask']],
                                                      dim=1)
        return rets, imagine_inps

    def _get_input_shape(self, scheme):
        agent_input_shape = scheme["entities"]["vshape"]
        if self.args.entity_last_action:
            agent_input_shape += scheme["actions_onehot"]["vshape"][0]
        hier_input_shape = scheme["entities"]["vshape"]
        return agent_input_shape, hier_input_shape

    def init_hidden(self, batch_size):
        super().init_hidden(batch_size)
        if self.use_alloc:
            self.task_allocations = th.zeros(
                batch_size, self.n_agents, self.args.n_tasks, device=self.agent._base.fc1.weight.device)
        if self.use_copa:
            self.coach_z = th.zeros(
                batch_size, self.n_agents, self.args.rnn_hidden_dim, device=self.agent._base.fc1.weight.device
            )

    def load_state(self, other_mac):
        super().load_state(other_mac)
        if self.use_copa:
            self.coach.load_state_dict(other_mac.coach.state_dict())
            if self.args.hier_agent['copa_vi_loss']:
                self.copa_recog.load_state_dict(other_mac.copa_recog.state_dict())

    def load_alloc_state(self, other_mac):
        if self.learned_alloc:
            self.alloc_policy.load_state_dict(other_mac.alloc_policy.state_dict())
            self.alloc_critic.load_state_dict(other_mac.alloc_critic.state_dict())

    def cuda(self):
        super().cuda()
        if self.learned_alloc:
            self.alloc_policy.cuda()
            self.alloc_critic.cuda()
        if self.use_copa:
            self.coach.cuda()
            if self.args.hier_agent['copa_vi_loss']:
                self.copa_recog.cuda()

    def eval(self):
        super().eval()
        if self.learned_alloc:
            self.alloc_policy.eval()
            self.alloc_critic.eval()
        if self.use_copa:
            self.coach.eval()
            if self.args.hier_agent['copa_vi_loss']:
                self.copa_recog.eval()

    def train(self):
        super().train()
        if self.learned_alloc:
            self.alloc_policy.train()
            self.alloc_critic.train()
        if self.use_copa:
            self.coach.train()
            if self.args.hier_agent['copa_vi_loss']:
                self.copa_recog.train()

    def save_models(self, path):
        super().save_models(path)
        if self.learned_alloc:
            th.save(self.alloc_policy.state_dict(), "{}alloc_pi.th".format(path))
            th.save(self.alloc_critic.state_dict(), "{}alloc_q.th".format(path))
        if self.use_copa:
            th.save(self.coach.state_dict(), "{}copa_coach.th".format(path))
            if self.args.hier_agent['copa_vi_loss']:
                th.save(self.copa_recog.state_dict(), "{}copa_recog.th".format(path))

    def load_models(self, path, pi_only=False):
        super().load_models(path)
        if pi_only:
            return
        if self.learned_alloc:
            self.alloc_policy.load_state_dict(th.load("{}alloc_pi.th".format(path), map_location=lambda storage, loc: storage))
            self.alloc_critic.load_state_dict(th.load("{}alloc_q.th".format(path), map_location=lambda storage, loc: storage))
        if self.use_copa:
            self.coach.load_state_dict(th.load("{}copa_coach.th".format(path), map_location=lambda storage, loc: storage))
            if self.args.hier_agent['copa_vi_loss']:
                self.copa_recog.load_state_dict(th.load("{}copa_recog.th".format(path), map_location=lambda storage, loc: storage))

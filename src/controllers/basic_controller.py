from wandb import agent
from modules.agents import Agent
from components.action_selectors import REGISTRY as action_REGISTRY
from utils.allocation import random_allocs
import torch as th


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.learned_alloc = args.hier_agent['task_allocation'] == 'aql'
        self.heuristic_alloc = args.hier_agent['task_allocation'] == 'heuristic'
        self.random_alloc = args.hier_agent['task_allocation'] in ['random', 'random_fixed']
        self.use_alloc = args.hier_agent['task_allocation'] is not None
        self.use_copa = args.hier_agent['copa']
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        active_allocs = None
        # Add task allocations if applicable
        if self.use_alloc or self.use_copa:
            decision_pts = ep_batch['hier_decision'][:, t_ep].flatten()
            if decision_pts.sum() >= 1:
                if self.use_alloc:
                    # only update task allocation if at any decision points
                    if self.learned_alloc:
                        meta_batch = self._make_meta_batch(ep_batch, t_ep)
                        new_allocs = self.compute_allocation(meta_batch, t_ep=t_ep, t_env=t_env, acting=True, test_mode=test_mode)
                    elif self.heuristic_alloc:
                        # heuristic is computed at every step in the env and stored in entity2task_mask
                        new_allocs = 1 - ep_batch['entity2task_mask'][:, t_ep, :self.n_agents][decision_pts == 1]
                    elif self.random_alloc:
                        new_allocs = random_allocs(ep_batch['task_mask'][:, t_ep][decision_pts == 1],
                                                ep_batch['entity_mask'][:, t_ep][decision_pts == 1],
                                                self.n_agents)
                    # update task allocation at decision points
                    self.task_allocations[decision_pts == 1] = new_allocs.to(self.task_allocations.dtype)
                if self.use_copa:
                    if not self.learned_alloc:
                        meta_batch = self._make_meta_batch(ep_batch, t_ep)
                    z_team, z_team_mu, _ = self.coach(meta_batch)
                    if test_mode:
                        z_team = z_team_mu
                    self.coach_z[decision_pts == 1] = z_team
            if self.use_alloc:
                # Add decided allocations to entity2task_mask
                ep_batch['entity2task_mask'][:, t_ep, :self.n_agents] = (1 - self.task_allocations).detach().to(th.uint8)
                active_allocs = self.task_allocations.detach()[bs]

        agent_outputs, _ = self.forward(ep_batch, t_ep, test_mode=test_mode, acting=True)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, allocs=active_allocs, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, coach_z=None, acting=False, test_mode=False, target=False, imagine_inps=None):
        if t is None:
            t = slice(0, ep_batch["avail_actions"].shape[1])
            int_t = False
        elif type(t) is int:
            t = slice(t, t + 1)
            int_t = True

        agent_inputs, imagine_inps = self._build_inputs(ep_batch, t, target=target, imagine_inps=imagine_inps)
        agent_inputs['hidden_state'] = self.hidden_states
        if self.use_copa:
            if acting:
                agent_inputs['coach_z'] = self.coach_z.unsqueeze(1)
            else:
                assert coach_z is not None, "Must provide coach_z for training with COPA"
                agent_inputs['coach_z'] = coach_z

        agent_outs, self.hidden_states, info = self.agent(agent_inputs, imagine_inps=imagine_inps)

        if int_t:
            agent_outs = agent_outs.squeeze(1)
        return agent_outs, info

    def init_hidden(self, batch_size):
        hidden = self.agent.init_hidden()
        if hidden is not None:
            self.hidden_states = hidden.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def eval(self):
        self.agent.eval()

    def train(self):
        self.agent.train()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = Agent(input_shape, self.args, **self.args.agent)

    def _build_inputs(self, batch, t, target=False, imagine_inps=None):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs, ts, na, os = batch["obs"].shape
        obs_parts = []
        obs_parts.append(batch["obs"][:, t])  # btav
        if self.args.obs_last_action:
            if t.start == 0:
                acs = th.zeros_like(batch["actions_onehot"][:, t])
                acs[:, 1:] = batch["actions_onehot"][:, slice(0, t.stop - 1)]
            else:
                acs = batch["actions_onehot"][:, slice(t.start - 1, t.stop - 1)]
            obs_parts.append(acs)
        if self.args.obs_agent_id:
            obs_parts.append(th.eye(self.n_agents, device=batch.device).view(1, 1, self.n_agents, self.n_agents).expand(bs, t.stop - t.start, -1, -1))
        return {'obs': th.cat(obs_parts, dim=3),
                'reset': batch["reset"][:, t].float()}

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

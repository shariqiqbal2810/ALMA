import numpy as np
import torch as th
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn.functional import softmax
from wandb import agent
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}

class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, allocs=None, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions

REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        # Was there so I used it
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time, decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, allocs=None, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        avail_actions = parse_avail_actions(avail_actions, allocs, self.args)

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:,:,0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical((avail_actions >= 1).float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


class CopyAllocActionSelector():
    def __init__(self, args) -> None:
        self.args = args

    def select_action(self, agent_inputs, avail_actions, t_env, allocs=None, test_mode=False):
        return agent_inputs.max(dim=2)[1]

REGISTRY['copy_alloc'] = CopyAllocActionSelector


def parse_avail_actions(avail_actions, allocs, args):
    if allocs is None or not args.mask_subtask_actions:
        return avail_actions
    ag_task_ids = allocs.argmax(dim=-1, keepdim=True).to(th.int)
    # actions available no matter the subtask
    all_avail = avail_actions == 1
    # actions available to specific subtasks
    subtask_avail = (avail_actions - 2) == ag_task_ids
    # when actions apply to agents, we check if that agent is allocated to the same task and mask out if not
    action_ag_id = avail_actions - 999  # bs, ts, na, nac
    action_ag_id[action_ag_id < 0] = 0
    same_task = (ag_task_ids == ag_task_ids.transpose(-1, -2))
    same_task = th.cat([th.zeros_like(same_task[..., [0]]), same_task], dim=-1)
    ag_ac_avail = same_task.gather(-1, action_ag_id.long())
    parsed_avail_actions = (all_avail + subtask_avail + ag_ac_avail).to(th.int)
    return parsed_avail_actions
    
    

REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector

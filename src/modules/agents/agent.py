import torch as th
import torch.nn as nn
from .bases import EntityBase, StandardBase
from .heads import RecurrentHead, FeedforwardHead
from .allocation_common import TaskEmbedder


class Agent(nn.Module):
    def __init__(self, input_shape, args,
                 recurrent=False, entity_scheme=True,
                 subtask_cond=None,
                 **kwargs):
        super().__init__()
        self.args = args
        if subtask_cond == 'full_obs':
            self.task_embed = TaskEmbedder(args.attn_embed_dim, args)
        if entity_scheme:
            self._base = EntityBase(input_shape, args)
        else:
            self._base = StandardBase(input_shape, args)
        out_dim = args.n_actions
        self.use_copa = self.args.hier_agent['copa']
        head_in_dim = args.rnn_hidden_dim
        if recurrent:
            self._head = RecurrentHead(args, in_dim=head_in_dim, out_dim=out_dim)
        else:
            self._head = FeedforwardHead(args, in_dim=head_in_dim, out_dim=out_dim)
        self.subtask_cond = subtask_cond

    def init_hidden(self):
        # make hidden states on same device as model
        return self._base.init_hidden()

    def _compute_network(self, inputs):
        x = self._base(inputs)
        if self.use_copa:
            x += inputs['coach_z']
        q, h = self._head(x, inputs)
        if 'entity_mask' in inputs:
            entity_mask = inputs['entity_mask']
            agent_mask = entity_mask[:, :, :self.args.n_agents]
            # zero out output for inactive agents
            q = q.masked_fill(agent_mask.unsqueeze(3), 0)
        return q, h

    def _logical_not(self, inp):
        return 1 - inp

    def _logical_or(self, inp1, inp2):
        out = inp1 + inp2
        out[out > 1] = 1
        return out

    def _groupmask2attnmask(self, group_mask):
        """
        Creates mask that only allows entities in the same group to attend to
        each other. Assumes each entity gets either binary or vector of binary
        group indicators
        """
        if len(group_mask.shape) == 3:
            bs, ts, ne = group_mask.shape
            ng = 1
        elif len(group_mask.shape) == 4:
            bs, ts, ne, ng = group_mask.shape
        else:
            raise Exception("Unrecognized group_mask shape")
        in1 = (1 - group_mask.to(th.float)).reshape(bs * ts, ne, ng)
        in2 = in1.transpose(1, 2)
        attn_mask = 1 - th.bmm(in1, in2)
        return attn_mask.reshape(bs, ts, ne, ne).to(th.uint8)

    def make_imagined_inputs(self, inputs):
        entities = inputs['entities']
        obs_mask = inputs['obs_mask']
        entity_mask = inputs['entity_mask']
        bs, ts, ne, ed = entities.shape

        # create random split of entities (once per episode)
        groupA_probs = th.rand(bs, 1, 1, device=entities.device).repeat(1, 1, ne)

        groupA = th.bernoulli(groupA_probs).to(th.uint8)
        groupB = self._logical_not(groupA)
        # mask out entities not present in env
        groupA = self._logical_or(groupA, entity_mask[:, [0]])
        groupB = self._logical_or(groupB, entity_mask[:, [0]])

        # convert entity mask to attention mask
        groupAattnmask = self._groupmask2attnmask(groupA)
        groupBattnmask = self._groupmask2attnmask(groupB)
        # create attention mask for interactions between groups
        interactattnmask = self._logical_or(self._logical_not(groupAattnmask),
                                            self._logical_not(groupBattnmask))
        # get within group attention mask
        withinattnmask = self._logical_not(interactattnmask)

        activeattnmask = self._groupmask2attnmask(entity_mask[:, [0]])
        # get masks to use for mixer (no obs_mask but mask out unused entities)
        Wattnmask_noobs = self._logical_or(withinattnmask, activeattnmask).repeat(1, ts, 1, 1)
        Iattnmask_noobs = self._logical_or(interactattnmask, activeattnmask).repeat(1, ts, 1, 1)
        # mask out agents that aren't observable (also expands time dim due to shape of obs_mask)
        withinattnmask = self._logical_or(withinattnmask, obs_mask)
        interactattnmask = self._logical_or(interactattnmask, obs_mask)

        new_inputs = {}
        if 'entity2task_mask' in inputs:
            new_inputs['entity2task_mask'] = inputs['entity2task_mask'].repeat(2, 1, 1, 1)
        new_inputs['entities'] = entities.repeat(2, 1, 1, 1)
        new_inputs['obs_mask'] = th.cat([withinattnmask, interactattnmask], dim=0)
        new_inputs['imagine_mask'] = th.cat([Wattnmask_noobs, Iattnmask_noobs], dim=0)
        new_inputs['entity_mask'] = entity_mask.repeat(2, 1, 1)
        new_inputs['reset'] = inputs['reset'].repeat(2, 1, 1)
        return new_inputs, (Wattnmask_noobs, Iattnmask_noobs)

    def _mask_by_task(self, inputs):
        # at each timestep, this is a matrix where entry [i, j] = 0 if entity i
        # belongs to task j, otherwise it's 1. Shape: (bs, ts, num_entities,
        # num_tasks)
        entity2task_mask = inputs['entity2task_mask']
        # we want an observability matrix (n_entites, n_entities) that = 0 if i,
        # j belong to the same task
        task_obs_mask = self._groupmask2attnmask(
            entity2task_mask)
        inputs['obs_mask'] = self._logical_or(inputs['obs_mask'],
                                              task_obs_mask)
        return inputs

    def _observe_tasks(self, inputs):
        entity2task = 1 - inputs['entity2task_mask'].float()
        inputs['task_embeds'] = self.task_embed(entity2task)
        return inputs

    def forward(self, inputs, imagine_inps=None):
        info = {}
        input_list = [inputs]
        if self.subtask_cond == 'mask':
            input_list = [self._mask_by_task(inp) for inp in input_list]
        if self.subtask_cond == 'full_obs':
            input_list = [self._observe_tasks(inp) for inp in input_list]
        if self.subtask_cond is not None and self.subtask_cond not in ['mask', 'full_obs']:
            raise Exception("Subtask conditioning not recognized")
        if imagine_inps is not None:
            input_list.append(imagine_inps)

        # concatenate all inputs along batch_dimension
        in_keys = set.intersection(*[set(d) for d in input_list])
        coach_z = None
        if 'coach_z' in inputs:
            # only in inputs so won't show up in intersection
            coach_z = inputs['coach_z']
        inputs = {k: th.cat([inp_dict[k] for inp_dict in input_list], dim=0)
                  for k in in_keys}
        if coach_z is not None:
            inputs['coach_z'] = coach_z
        
        q, h = self._compute_network(inputs)
        return q, h, info

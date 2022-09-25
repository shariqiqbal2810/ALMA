import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import EntityAttentionLayer
from functools import partial
from utils.rl_utils import ExponentialMeanStd


class AttentionHyperNet(nn.Module):
    """
    mode='matrix' gets you a <n_agents x mixing_embed_dim> sized matrix
    mode='vector' gets you a <mixing_embed_dim> sized vector by averaging over agents
    mode='scalar' gets you a scalar by averaging over agents and embed dim
    ...per set of entities
    """
    def __init__(self, args, extra_dims=0, mode='matrix', use_copa=False):
        super(AttentionHyperNet, self).__init__()
        self.args = args
        self.mode = mode
        self.use_copa = use_copa
        self.extra_dims = extra_dims
        self.entity_dim = args.entity_shape
        if self.args.entity_last_action:
            self.entity_dim += args.n_actions
        if extra_dims > 0:
            self.entity_dim += extra_dims

        out_dim = args.mixing_embed_dim

        hypernet_embed = args.hypernet_embed
        self.fc1 = nn.Linear(self.entity_dim, hypernet_embed)
        self.attn = EntityAttentionLayer(hypernet_embed,
                                            hypernet_embed,
                                            hypernet_embed, args)

        self.fc2 = nn.Linear(hypernet_embed, out_dim)
        

    def _duplicate_per_task(self, weights, entity2task_mask):
        if entity2task_mask is None:
            return weights
        # we duplicate generated weights for each task and mask out
        # non-included agents
        agent2task_mask = entity2task_mask[:, :self.args.n_agents]
        weights = weights.unsqueeze(1).repeat(1, self.args.n_tasks, 1, 1)
        # weights.shape = (bs, nt, na, hd)
        # agent2task_mask.shape = (bs, na, nt)
        weights = weights.masked_fill(
            agent2task_mask.transpose(1, 2).unsqueeze(-1), 0)
        # new weights.shape = (bs, nt, na, hd)
        bs, nt, na, hd = weights.shape
        return weights.reshape(bs * nt, na, hd)

    def forward(self, entities, entity_mask,
                attn_mask=None, entity2task_mask=None,
                nonneg=th.abs):
        agent_mask = entity_mask[:, :self.args.n_agents]
        x1 = F.relu(self.fc1(entities))
        if attn_mask is None:
            # create attn_mask from entity mask
            attn_mask = 1 - th.bmm((1 - agent_mask.to(th.float)).unsqueeze(2),
                                (1 - entity_mask.to(th.float)).unsqueeze(1))
        x2 = self.attn(x1, pre_mask=attn_mask.to(th.uint8),
                    post_mask=agent_mask)
        x3 = self.fc2(x2)
        x3 = x3.masked_fill(agent_mask.unsqueeze(2), 0)
        if self.mode == 'matrix':
            x3 = nonneg(x3)
            x3 = self._duplicate_per_task(x3, entity2task_mask)
        elif self.mode == 'vector':
            x3 = self._duplicate_per_task(x3, entity2task_mask)
            x3 = x3.mean(dim=-2)
            x3 = nonneg(x3)
        elif self.mode == 'alt_vector':
            return nonneg(x3.mean(dim=-1))
        elif self.mode == 'scalar':
            x3 = self._duplicate_per_task(x3, entity2task_mask)
            x3 = x3.mean(dim=(-2, -1))
            x3 = nonneg(x3)
        else:
            raise Exception("Mode not recognized")
        return x3


class FlexQMixer(nn.Module):
    def __init__(self, args):
        super(FlexQMixer, self).__init__()
        self.args = args

        self.n_agents = args.n_agents

        self.embed_dim = args.mixing_embed_dim

        self.use_copa = self.args.hier_agent['copa']

        extra_dims = 0
        if args.mixer_subtask_cond == 'full_obs':
            extra_dims += args.n_tasks

        self.hyper_w_1 = AttentionHyperNet(args, extra_dims=extra_dims, mode='matrix', use_copa=self.use_copa)
        self.hyper_w_final = AttentionHyperNet(args, extra_dims=extra_dims, mode='vector', use_copa=self.use_copa)
        self.hyper_b_1 = AttentionHyperNet(args, extra_dims=extra_dims, mode='vector', use_copa=self.use_copa)
        # V(s) instead of a bias for the last layers
        self.V = AttentionHyperNet(args, extra_dims=extra_dims, mode='scalar', use_copa=self.use_copa)

        self.non_lin = F.elu
        if getattr(self.args, "mixer_non_lin", "elu") == "tanh":
            self.non_lin = F.tanh

        self.use_popart = self.args.popart
        if self.use_popart:
            self.norm_agg_inds = None
            n_out = 1
            if self.args.mixer_subtask_cond is not None:
                # normalize all subtask outputs equivalently since weights that
                # produce them are shared
                self.norm_agg_inds = (list(range(self.args.n_tasks)),)
                n_out = self.args.n_tasks
            self.targ_rms = ExponentialMeanStd(alpha=0.01, agg_inds=self.norm_agg_inds)

            self.popart_weight = nn.parameter.Parameter(
                th.ones(1, 1, n_out))
            self.popart_bias = nn.parameter.Parameter(
                th.zeros(1, 1, n_out))

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

    def popart_update(self, targets, mask):
        assert self.use_popart
        if self.targ_rms.mean is not None:
            old_mean = self.targ_rms.mean.clone()
            old_var = self.targ_rms.var.clone()
            self.targ_rms.update(targets, mask)
        else:
            self.targ_rms.update(targets, mask)
            old_mean = self.targ_rms.mean.clone()
            old_var = self.targ_rms.var.clone()
        sd_ratio = old_var.sqrt() / self.targ_rms.var.sqrt()
        self.popart_weight.data.mul_(sd_ratio)
        self.popart_bias.data.mul_(sd_ratio).add_((old_mean - self.targ_rms.mean) / self.targ_rms.var.sqrt())
        return (targets - self.targ_rms.mean) / self.targ_rms.var.sqrt()

    def denormalize(self, q):
        if self.targ_rms.mean is None or not self.use_popart:
            return q
        return q * self.targ_rms.var.sqrt() + self.targ_rms.mean

    def load_state_dict(self, state_dict):
        if self.use_popart:
            targ_rms_state_dict, state_dict = state_dict
            self.targ_rms.load_state_dict(targ_rms_state_dict)
        super().load_state_dict(state_dict)

    def state_dict(self):
        if self.use_popart:
            return self.targ_rms.state_dict(), super().state_dict()
        return super().state_dict()

    def forward(self, agent_qs, inputs, imagine_groups=None):
        entities = inputs['entities']
        entity_mask = inputs['entity_mask']
        if self.args.mixer_subtask_cond == 'full_obs':
            entity_tasks = 1 - inputs['entity2task_mask'].float()
            entities = th.cat([entities, entity_tasks],
                              dim=3)
        bs, max_t, ne, ed = entities.shape

        entities = entities.reshape(bs * max_t, ne, ed)
        entity_mask = entity_mask.reshape(bs * max_t, ne)
        entity2task_mask = None
        n_out = 1
        if self.args.mixer_subtask_cond is not None:
            # We want as many mixing network outputs as we have tasks, so we
            # make a copy of the network for each task and zero out the
            # weights that correspond to agents not belonging to the task.
            # This is done by providing a different entity mask for each
            # task.
            entity2task_mask = inputs['entity2task_mask'].reshape(
                bs * max_t, ne, self.args.n_tasks)
            n_out = self.args.n_tasks

        if self.args.softmax_mixing_weights:
            nonneg = partial(F.softmax, dim=-1)
        else:
            nonneg = th.abs

        attn_mask = None
        if self.args.mixer_subtask_cond == 'mask':
            attn_mask = self._groupmask2attnmask(inputs['entity2task_mask'])
            attn_mask = attn_mask.reshape(bs * max_t, ne, ne)
        if imagine_groups is not None:
            agent_qs = agent_qs.view(bs * max_t, 1, self.n_agents * 2)
            Wmask, Imask = imagine_groups
            Wmask = Wmask.reshape(bs * max_t, ne, ne)
            Imask = Imask.reshape(bs * max_t, ne, ne)
            if attn_mask is not None:
                Wmask = self._logical_or(Wmask, attn_mask)
                Imask = self._logical_or(Imask, attn_mask)
            w1_W = self.hyper_w_1(entities, entity_mask,
                                  attn_mask=Wmask,
                                  entity2task_mask=entity2task_mask,
                                  nonneg=nonneg)
            w1_I = self.hyper_w_1(entities, entity_mask,
                                  attn_mask=Imask,
                                  entity2task_mask=entity2task_mask,
                                  nonneg=nonneg)
            w1 = th.cat([w1_W, w1_I], dim=1)
        else:
            agent_qs = agent_qs.view(bs * max_t, 1, self.n_agents)
            # First layer
            w1 = self.hyper_w_1(entities, entity_mask, attn_mask=attn_mask,
                                entity2task_mask=entity2task_mask,
                                nonneg=nonneg)
        b1 = self.hyper_b_1(entities, entity_mask, attn_mask=attn_mask,
                            entity2task_mask=entity2task_mask)
        w1 = w1.view(bs * max_t * n_out, agent_qs.shape[-1], self.embed_dim)
        b1 = b1.view(bs * max_t * n_out, 1, self.embed_dim)

        # Second layer
        w_final = self.hyper_w_final(entities, entity_mask, attn_mask=attn_mask,
                                     entity2task_mask=entity2task_mask,
                                     nonneg=nonneg)
        w_final = w_final.view(bs * max_t * n_out, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(entities, entity_mask, attn_mask=attn_mask,
                   entity2task_mask=entity2task_mask).view(
                       bs * max_t * n_out, 1, 1)


        # Mixing network forward
        if entity2task_mask is not None:
            # separate forward pass per task
            agent_qs = agent_qs.repeat_interleave(n_out, dim=0)
        hidden = self.non_lin(th.bmm(agent_qs, w1) + b1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, max_t, n_out)
        if self.use_popart:
            q_tot = self.popart_weight * q_tot + self.popart_bias
        return q_tot


class LinearFlexQMixer(nn.Module):
    def __init__(self, args):
        super(LinearFlexQMixer, self).__init__()
        self.args = args

        self.n_agents = args.n_agents

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = AttentionHyperNet(args, mode='alt_vector')
        self.V = AttentionHyperNet(args, mode='scalar')

    def forward(self, agent_qs, inputs, imagine_groups=None, ret_ingroup_prop=False):
        entities, entity_mask = inputs
        bs, max_t, ne, ed = entities.shape

        entities = entities.reshape(bs * max_t, ne, ed)
        entity_mask = entity_mask.reshape(bs * max_t, ne)
        if imagine_groups is not None:
            agent_qs = agent_qs.view(-1, self.n_agents * 2)
            Wmask, Imask = imagine_groups
            w1_W = self.hyper_w_1(entities, entity_mask,
                                  attn_mask=Wmask.reshape(bs * max_t,
                                                          self.n_agents, ne))
            w1_I = self.hyper_w_1(entities, entity_mask,
                                  attn_mask=Imask.reshape(bs * max_t,
                                                          self.n_agents, ne))
            w1 = th.cat([w1_W, w1_I], dim=1)
        else:
            agent_qs = agent_qs.view(-1, self.n_agents)
            # First layer
            w1 = self.hyper_w_1(entities, entity_mask)
        w1 = w1.view(bs * max_t, -1)
        if self.args.softmax_mixing_weights:
            w1 = F.softmax(w1, dim=1)
        else:
            w1 = th.abs(w1)
        v = self.V(entities, entity_mask)

        q_cont = agent_qs * w1
        q_tot = q_cont.sum(dim=1) + v
        # Reshape and return
        q_tot = q_tot.view(bs, -1, 1)
        if ret_ingroup_prop:
            ingroup_w = w1.clone()
            ingroup_w[:, self.n_agents:] = 0  # zero-out out of group weights
            ingroup_prop = (ingroup_w.sum(dim=1)).mean()
            return q_tot, ingroup_prop
        return q_tot

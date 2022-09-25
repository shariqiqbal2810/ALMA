import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ..layers import EntityAttentionLayer
from .allocation_common import groupmask2attnmask, TaskEmbedder, CountEmbedder
from utils.rl_utils import ExponentialMeanStd


class StandardAllocCritic(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args

        self.args = args
        self.in_fc_ent = nn.Linear(input_shape, args.alloc_embed_dim)
        self.in_fc_alloc = TaskEmbedder(args.alloc_embed_dim, args)
        self.attn = EntityAttentionLayer(args.alloc_embed_dim,
                                         args.alloc_embed_dim,
                                         args.alloc_embed_dim, args,
                                         n_heads=args.alloc_n_heads,
                                         use_layernorm=False)
        self.count_embed = CountEmbedder(args.alloc_embed_dim, args)
        self.out_dim = 1
        self.out_fc = nn.Linear(args.alloc_embed_dim, self.out_dim)

        self.use_popart = self.args.popart
        if self.use_popart:
            self.targ_rms = ExponentialMeanStd(alpha=0.01)
            self.popart_weight = nn.parameter.Parameter(
                th.ones(1, self.out_dim))
            self.popart_bias = nn.parameter.Parameter(
                th.zeros(1, self.out_dim))

    def load_state_dict(self, state_dict):
        if self.use_popart:
            targ_rms_state_dict, state_dict = state_dict
            self.targ_rms.load_state_dict(targ_rms_state_dict)
        super().load_state_dict(state_dict)

    def state_dict(self):
        if self.use_popart:
            return self.targ_rms.state_dict(), super().state_dict()
        return super().state_dict()

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

    def forward(self, batch, override_alloc=None, test_mode=False, calc_stats=False):
        entities = batch['entities']
        bs = entities.shape[0]
        entity_mask = batch['entity_mask']
        attn_mask = groupmask2attnmask(entity_mask)
        entity2task = 1 - batch['entity2task_mask'].float()
        multi_eval = False
        repeat_fn = lambda x: x
        if override_alloc is not None:
            if len(override_alloc.shape) == 4:
                multi_eval = True
                bs, np, na, nt = override_alloc.shape
                override_alloc = override_alloc.reshape(bs * np, na, nt)
                repeat_fn = lambda x: x.repeat_interleave(np, dim=0)
            entity2task = repeat_fn(entity2task)
            entity2task[:, :self.args.n_agents] = override_alloc
        x1_ent = self.in_fc_ent(entities)
        x1_alloc = self.in_fc_alloc(entity2task)
        x1_count = self.count_embed(entity2task)
        x1 = F.relu(repeat_fn(x1_ent) + x1_alloc + x1_count)
        x2 = F.relu(self.attn(x1, pre_mask=repeat_fn(attn_mask),
                              post_mask=repeat_fn(entity_mask[:, :self.args.n_agents])))
        out_shape = (bs, self.out_dim)
        if multi_eval:
            out_shape = (bs, np, self.out_dim)
        out = self.out_fc(x2.mean(dim=1)).reshape(*out_shape)
        if self.use_popart:
            if multi_eval:
                out = out * self.popart_weight.unsqueeze(1) + self.popart_bias.unsqueeze(1)
            else:
                out = out * self.popart_weight + self.popart_bias
        if calc_stats:
            return out, {}
        return out

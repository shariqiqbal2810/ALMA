import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical


class EntityAttentionLayer(nn.Module):
    def __init__(self, in_dim, embed_dim, out_dim, args, self_attention=True,
                 indep_out_heads=False, scalar_valued=False,
                 n_heads=None, use_layernorm=False):
        super(EntityAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.n_heads = args.attn_n_heads
        if n_heads is not None:
            self.n_heads = n_heads
        self.n_agents = args.n_agents
        self.args = args
        self.self_attention = self_attention
        self.indep_out_heads = indep_out_heads
        self.scalar_valued = scalar_valued
        self.layernorm = None
        if use_layernorm:
            self.layernorm = nn.LayerNorm(self.in_dim)

        self.value_dim = self.embed_dim
        if self.scalar_valued:
            self.n_heads = 1
            self.value_dim = 1
            self.value_head_dim = 1
            self.out_dim = 1

        assert self.embed_dim % self.n_heads == 0, "Embed dim must be divisible by n_heads"
        self.head_dim = self.embed_dim // self.n_heads
        self.register_buffer('scale_factor',
                             th.scalar_tensor(self.head_dim).sqrt())
        self.value_head_dim = self.head_dim
        if self.scalar_valued:
            self.value_head_dim = 1


        if self.self_attention:
            self.in_trans = nn.Linear(self.in_dim, self.embed_dim * 2 + self.value_dim, bias=False)
        else:
            self.in_trans_q = nn.Linear(self.in_dim, self.embed_dim, bias=False)
            self.in_trans_kv = nn.Linear(self.in_dim, self.embed_dim + self.value_dim, bias=False)

        if self.indep_out_heads:
            self.out_trans = nn.Linear(self.value_dim // self.n_heads, self.out_dim)
        else:
            self.out_trans = nn.Linear(self.value_dim, self.out_dim)


    def forward(self, entities, pre_mask=None, diff_mask=None, post_mask=None, test_mode=False):
        """
        entities: Entity representations
            shape: batch size, # of entities, embedding dimension
        pre_mask: Which agent-entity pairs are not available (observability and/or padding).
                  Mask out before attention.
            shape: batch_size, # of queries, # of entities
        diff_mask: Differentiable mask. Same shape as pre-mask but inverted (i.e. 0 means mask out).
                   Different mask for each attention head (duplicated in batch dimension)
            shape: batch_size * # of heads, # of queries, # of entities
        post_mask: Which agents/entities are not available. Zero out their outputs to
                   prevent gradients from flowing back. Shape of 2nd dim determines
                   whether to compute queries for all entities or just agents.
            shape: batch size, # of agents (or entities)

        Return shape: batch size, # of agents, embedding dimension
        """
        if self.self_attention:
            if self.layernorm is not None:
                entities = self.layernorm(entities)
            entities_t = entities.transpose(0, 1)
            n_queries = post_mask.shape[1]
            pre_mask = pre_mask[:, :n_queries]
            if diff_mask is not None:
                diff_mask = diff_mask[:, :n_queries]
            ne, bs, _ = entities_t.shape
            if self.scalar_valued:
                combined_out = self.in_trans(entities_t)
                query, key = combined_out[..., :-1].chunk(2, dim=2)
                value = combined_out[..., -1:]
            else:
                query, key, value = self.in_trans(entities_t).chunk(3, dim=2)
            query = query[:n_queries]
        else:
            entities_q, entities_kv = entities
            if self.layernorm is not None:
                entities_q = self.layernorm(entities_q)
                entities_kv = self.layernorm(entities_kv)
            entities_q_t = entities_q.transpose(0, 1)
            entities_kv_t = entities_kv.transpose(0, 1)
            n_queries = entities_q.shape[1]
            bs, ne, _ = entities_kv.shape
            query = self.in_trans_q(entities_q_t)
            if self.scalar_valued:
                combined_out = self.in_trans_kv(entities_kv_t)
                key = combined_out[..., :-1]
                value = combined_out[..., -1:]
            else:
                key, value = self.in_trans_kv(entities_kv_t).chunk(2, dim=2)

        query_spl = query.reshape(n_queries, bs * self.n_heads, self.head_dim).transpose(0, 1)
        key_spl = key.reshape(ne, bs * self.n_heads, self.head_dim).permute(1, 2, 0)
        value_spl = value.reshape(ne, bs * self.n_heads, self.value_head_dim).transpose(0, 1)

        attn_logits = th.bmm(query_spl, key_spl) / self.scale_factor

        if pre_mask is not None:
            pre_mask_rep = pre_mask.repeat_interleave(self.n_heads, dim=0)
            attn_logits = attn_logits.masked_fill(pre_mask_rep[:, :, :ne], -float('Inf'))

        # if diff_mask is not None:
        #     # https://arxiv.org/pdf/2102.12871.pdf
        #     attn_logits = attn_logits - (1 - diff_mask) * 1000

        attn_weights = F.softmax(attn_logits, dim=2)

        # some weights might be NaN (if agent is inactive and all entities were masked)
        attn_weights = attn_weights.masked_fill(attn_weights != attn_weights, 0)

        if diff_mask is not None:
            diff_mask_rep = diff_mask.repeat_interleave(self.n_heads, dim=0)
            # do the actual masking
            attn_weights = attn_weights * (diff_mask_rep)
            # re-normalize
            attn_weights = attn_weights / (attn_weights.sum(dim=2, keepdim=True) + 1e-8)

        attn_outs = th.bmm(attn_weights, value_spl)
        attn_outs = attn_outs.transpose(
            0, 1).reshape(n_queries, bs, self.value_dim)
        attn_outs = attn_outs.transpose(0, 1)
        if self.indep_out_heads:
            attn_outs = attn_outs.reshape(bs, n_queries, self.n_heads, self.value_head_dim)
        attn_outs = self.out_trans(attn_outs)
        if post_mask is not None:
            n_broadcast_dims = len(attn_outs.shape) - 2
            attn_outs = attn_outs.masked_fill(post_mask.reshape(bs, n_queries, *[1 for _ in range(n_broadcast_dims)]), 0)
        return attn_outs

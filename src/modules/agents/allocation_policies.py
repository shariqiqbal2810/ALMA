import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import RelaxedOneHotCategorical
from ..layers import EntityAttentionLayer
from .allocation_common import groupmask2attnmask, TaskEmbedder, COUNT_NORM_FACTOR

class AutoregressiveAllocPolicy(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args

        self.args = args
        self.pi_ag_attn = self.args.hier_agent['pi_ag_attn']
        self.pi_pointer_net = self.args.hier_agent['pi_pointer_net']
        self.subtask_mask = self.args.hier_agent['subtask_mask']
        self.sel_task_upd = self.args.hier_agent['sel_task_upd']
        self.pi_autoreg = self.args.hier_agent['pi_autoreg']

        embd_upd_in_shape = args.attn_embed_dim * 2
        if not self.pi_pointer_net:
            input_shape += args.n_tasks
            embd_upd_in_shape += args.n_tasks
        if not self.sel_task_upd:
            embd_upd_in_shape += args.attn_embed_dim
        self.fc1 = nn.Linear(input_shape, args.attn_embed_dim)
        self.task_embed = TaskEmbedder(args.attn_embed_dim, args)
        self.attn = EntityAttentionLayer(args.attn_embed_dim,
                                         args.attn_embed_dim,
                                         args.attn_embed_dim, args)
        if self.pi_autoreg:
            self.embed_update = nn.Linear(embd_upd_in_shape, args.attn_embed_dim)
            if self.pi_pointer_net:
                self.count_embed = nn.Linear(2, args.attn_embed_dim)
        elif self.pi_pointer_net:
            # can only embed nonagent entity counts since agent allocs are decided all at once
            self.count_embed = nn.Linear(1, args.attn_embed_dim)
        if not self.pi_pointer_net:
            self.out_fc = nn.Linear(args.attn_embed_dim * 2, args.n_tasks)
        self.register_buffer('sample_temp',
                             th.scalar_tensor(1))  # TODO: anneal or try other values?
        self.register_buffer('scale_factor',
                             th.scalar_tensor(args.attn_embed_dim).sqrt())

    @property
    def device(self):
        return self.fc1.weight.device

    def _autoreg_forward(self, task_embeds, task_nonag_counts, agent_embeds, task_mask, entity_mask, avail_actions,
                         calc_stats=False, test_mode=False, repeat_fn=lambda x: x):
        nt = self.args.n_tasks
        bs, na, _ = agent_embeds.shape
        stats = {}
        task_embeds = repeat_fn(task_embeds)
        task_mask = repeat_fn(task_mask)
        if task_nonag_counts is not None:
            task_nonag_counts = repeat_fn(task_nonag_counts)
        prop_bs = task_mask.shape[0]
        allocs = repeat_fn(th.zeros((bs, na, nt), device=agent_embeds.device))
        all_log_pi = th.zeros_like(allocs)

        task_ag_counts = th.zeros_like(allocs[:, 0])

        agent_mask = entity_mask[:, :na]

        prev_alloc_mask = th.zeros_like(task_mask)

        for ai in range(self.args.n_agents):
            # compute pointer-net logits (scale as in dot-product attention)
            curr_agent_embed = agent_embeds[:, [ai]]
            curr_agent_embed = repeat_fn(curr_agent_embed)

            if self.pi_pointer_net:
                count_embeds = self.count_embed(th.stack([task_nonag_counts, task_ag_counts], dim=-1))
                logits = th.bmm(curr_agent_embed, (task_embeds + count_embeds).transpose(1, 2)).squeeze(1) / self.scale_factor
            else:
                # curr_agent_embed.shape = (bs, 1, hd), task_embeds.shape = (bs, hd)
                logit_ins = th.cat([curr_agent_embed.squeeze(1), task_embeds], dim=1)
                logits = self.out_fc(F.relu(logit_ins))

            # mask inactive tasks s.t. softmax is 0
            curr_mask = task_mask.clone()
            masked_logits = logits.masked_fill(curr_mask, th.finfo(logits.dtype).min)
            # mask for inactive agents
            curr_agent_mask = repeat_fn(1 - agent_mask[:, [ai]].float())
            dist = RelaxedOneHotCategorical(self.sample_temp, logits=masked_logits)
            # sample action
            soft_ac = dist.rsample()
            if calc_stats:
                # NOTE: we can use dist.logits as log prob as pytorch
                # Categorical distribution normalizes logits such that
                # th.exp(dist.logits) is the probability
                all_log_pi[:, ai] = dist.logits
                ag_log_pi = dist.logits.gather(1, soft_ac.argmax(dim=1, keepdim=True))
                stats['log_pi'] = stats.get('log_pi', 0) + ag_log_pi * curr_agent_mask
                stats['best_prob'] = stats.get('best_prob', 0) + dist.probs.max(dim=1, keepdim=True)[0] * curr_agent_mask
                entropy = -(dist.logits * dist.probs).sum(dim=1, keepdim=True)
                stats['entropy'] = stats.get('entropy', 0) + entropy * curr_agent_mask
            # make one-hot sample that acts like a continuous sample in the backward pass
            onehot_ac = F.one_hot(soft_ac.argmax(dim=1), num_classes=nt).float()
            hard_ac = onehot_ac - soft_ac.detach() + soft_ac
            hard_ac = hard_ac * curr_agent_mask
            prev_alloc_mask += hard_ac.detach().to(th.uint8)
            task_ag_counts += hard_ac.detach() * COUNT_NORM_FACTOR
            allocs[:, ai] = hard_ac
            # update embedding of selected task to incorporate new agent (only if agent is active)
            if self.pi_pointer_net:
                if self.sel_task_upd:
                    # only update selected task embeddings
                    embed_upd_in = th.cat([task_embeds, curr_agent_embed.repeat(1, nt, 1)], dim=2)
                    task_embeds = task_embeds + self.embed_update(F.relu(embed_upd_in)) * hard_ac.detach().unsqueeze(2) * curr_agent_mask.unsqueeze(2)
                else:
                    # update all task embeddings (use copy of selected task to condition on the previous agents' allocations)
                    sel_task_embed = (task_embeds * hard_ac.detach().unsqueeze(2)).sum(dim=1, keepdim=True)
                    embed_upd_in = th.cat([task_embeds, curr_agent_embed.repeat(1, nt, 1), sel_task_embed.repeat(1, nt, 1)], dim=2)
                    task_embeds = task_embeds + self.embed_update(F.relu(embed_upd_in)) * curr_agent_mask.unsqueeze(2)
            else:
                embed_upd_in = th.cat([task_embeds, curr_agent_embed.squeeze(1), hard_ac.detach()], dim=1)
                task_embeds = task_embeds + self.embed_update(F.relu(embed_upd_in)) * curr_agent_mask
        if calc_stats:
            stats['all_log_pi'] = all_log_pi
        return allocs, stats

    def _standard_forward(self, task_embeds, task_nonag_counts, agent_embeds, task_mask, entity_mask,
                         calc_stats=False, test_mode=False, repeat_fn=lambda x: x):
        nt = self.args.n_tasks
        bs, na, _ = agent_embeds.shape
        stats = {}
        task_embeds = repeat_fn(task_embeds) + repeat_fn(self.count_embed(task_nonag_counts.unsqueeze(-1)))
        task_mask = repeat_fn(task_mask)
        agent_embeds = repeat_fn(agent_embeds)
        allocs = repeat_fn(th.zeros((bs, na, nt), device=agent_embeds.device))

        if self.pi_pointer_net:
            logits = th.bmm(agent_embeds, task_embeds.transpose(1, 2)) / self.scale_factor
        else:
            raise NotImplementedError
            # curr_agent_embed.shape = (bs, na, hd), task_embeds.shape(bs, nt, hd)
            logit_ins = th.cat([agent_embeds.unsqueeze(2).repeat(1, 1, nt, 1),
                                task_embeds.unsqueeze(1).repeat(1, na, 1, 1)], dim=3)
            logits = self.out_fc(F.relu(logit_ins)).squeeze(3)

        # mask inactive tasks s.t. softmax is 0
        masked_logits = logits.masked_fill(task_mask.unsqueeze(1), th.finfo(logits.dtype).min)
        # mask for inactive agents
        agent_mask = repeat_fn(1 - entity_mask[:, :na].float())
        dist = RelaxedOneHotCategorical(self.sample_temp, logits=masked_logits)
        # sample action
        soft_ac = dist.rsample()
        if calc_stats:
            # NOTE: we can use dist.logits as log prob as pytorch
            # Categorical distribution normalizes logits such that
            # th.exp(dist.logits) is the probability
            stats['all_log_pi'] = dist.logits
            ag_log_pi = dist.logits.gather(2, soft_ac.argmax(dim=2, keepdim=True)).squeeze(2)
            stats['log_pi'] = (ag_log_pi * agent_mask).sum(dim=1, keepdim=True)
            stats['best_prob'] = (dist.probs.max(dim=2)[0] * agent_mask).sum(dim=1, keepdim=True)
            entropy = -(dist.logits * dist.probs).sum(dim=2)
            stats['entropy'] = (entropy * agent_mask).sum(dim=1, keepdim=True)
        # make one-hot sample that acts like a continuous sample in the backward pass
        onehot_ac = F.one_hot(soft_ac.argmax(dim=2), num_classes=nt).float()
        allocs = onehot_ac - soft_ac.detach() + soft_ac
        allocs = allocs * agent_mask.unsqueeze(2)
        return allocs, stats

    def forward(self, batch, calc_stats=False, test_mode=False, n_proposals=-1):
        # copy entity2task mask and zero out assignments
        entities = batch['entities']
        entity_mask = batch['entity_mask']
        entity2task_mask = batch['entity2task_mask']
        avail_actions = batch['avail_actions']

        nag = self.args.n_agents
        entity2task = 1 - entity2task_mask.float()
        last_alloc = batch['last_alloc']
        entity2task[:, :nag] = last_alloc

        # observe which task agents were assigned to in previous step + which
        # task non-agent entities belong to
        if not self.pi_pointer_net:
            entities = th.cat([entities, entity2task], dim=-1)
        x1 = self.fc1(entities)
        if self.pi_pointer_net:
            x1 += self.task_embed(entity2task)

        # compute attention for non-agent entities and get embedding for each task
        nonagent_x1 = x1[:, nag:]
        if self.pi_pointer_net and self.subtask_mask:
            nonagent_attn_mask = groupmask2attnmask(
                entity2task_mask[:, nag:])
        else:
            nonagent_attn_mask = groupmask2attnmask(
                entity_mask[:, nag:])
        nonagent_mask = entity_mask[:, nag:]
        nonagent_x2 = self.attn(F.relu(nonagent_x1), pre_mask=nonagent_attn_mask,
                                post_mask=nonagent_mask)
        if self.pi_pointer_net:
            nonagent_entity2task = entity2task[:, nag:]  # (bs, n_nonagent, nt)
            # sum up embeddings of non-agent entities belonging to each task
            task_x2 = th.bmm(nonagent_entity2task.transpose(1, 2), nonagent_x2)
            # count nonagent entities present in each task
            task_nonag_cnt = nonagent_entity2task.sum(dim=1) * COUNT_NORM_FACTOR
        else:
            task_x2 = nonagent_x2.mean(dim=1)
            task_nonag_cnt = None

        # get agent embeddings
        agent_x1 = x1[:, :nag]
        if self.pi_ag_attn:
            ag_mask = entity_mask[:, :nag]
            active_mask = groupmask2attnmask(ag_mask)
            inverse_causal_mask = th.diag(
                th.ones(nag, device=ag_mask.device)
            ).cumsum(dim=1).transpose(0,1).to(th.uint8)
            ag_attn_mask = (active_mask + inverse_causal_mask).min(th.ones_like(active_mask))
            agent_embeds = agent_x1 + self.attn(
                F.relu(agent_x1), pre_mask=ag_attn_mask, post_mask=ag_mask)
        else:
            agent_embeds = agent_x1

        repeat = 1
        if n_proposals > 0:
            repeat = n_proposals
        repeat_fn = lambda x: x.repeat_interleave(repeat, dim=0)

        if self.pi_autoreg:
            allocs, stats = self._autoreg_forward(
                task_x2, task_nonag_cnt, agent_embeds, batch['task_mask'], entity_mask, avail_actions,
                calc_stats=calc_stats, test_mode=test_mode, repeat_fn=repeat_fn)
        else:
            allocs, stats = self._standard_forward(
                task_x2, task_nonag_cnt, agent_embeds, batch['task_mask'], entity_mask,
                calc_stats=calc_stats, test_mode=test_mode, repeat_fn=repeat_fn)

        if n_proposals > 1:
            allocs = allocs.reshape(-1, n_proposals, nag, self.args.n_tasks)
        if calc_stats:
            stats['best_prob'] = stats['best_prob'] / repeat_fn(1 - entity_mask[:, :nag].float()).sum(dim=1, keepdim=True)
            for k, v in stats.items():
                stats[k] = v.reshape(-1, n_proposals, *v.shape[1:])
            return allocs, stats
        return allocs

import torch as th
import torch.nn as nn

COUNT_NORM_FACTOR = 0.2

def groupmask2attnmask(group_mask, last_ind_observe=None):
    """
    Creates mask that only allows entities in the same group to attend to
    each other. Assumes each entity gets either binary or vector of binary
    group indicators
    """
    if len(group_mask.shape) == 2:
        assert last_ind_observe is None, "No group index provided"
        bs, ne = group_mask.shape
        ng = 1
    elif len(group_mask.shape) == 3:
        bs, ne, ng = group_mask.shape
    else:
        raise Exception("Unrecognized group_mask shape")
    in1 = (1 - group_mask.to(th.float)).reshape(bs, ne, ng)
    in2 = in1.transpose(1, 2)
    if last_ind_observe is not None:
        in2 = in2.clone()  # so in-place operation doesn't modify in1 as well
        in2[:, -1] = (1 - last_ind_observe).float()
    attn_mask = 1 - th.bmm(in1, in2)
    return attn_mask.reshape(bs, ne, ne).to(th.uint8)


class TaskEmbedder(nn.Module):
    """
    Learn a set of sub-task embeddings. Embeddings are selected randomly per
    forward pass but are always consistent across the batch and entities. In
    other words two agents assigned to task 1 will always get the same embedding
    as each other but it may be a different embedding each time. We learn more
    embeddings than we need in case we want to generalize to settings with more
    subtasks than we are training on.
    """
    def __init__(self, embed_dim, args) -> None:
        super().__init__()
        self.args = args
        self.n_tasks = args.n_tasks
        self.n_extra_tasks = args.n_extra_tasks
        in_dim = self.n_tasks + self.n_extra_tasks
        self.fc = nn.Linear(in_dim, embed_dim, bias=False)

    def forward(self, task_one_hot):
        extra_dims = th.zeros_like(task_one_hot[..., [0]]).repeat_interleave(self.n_extra_tasks, -1)
        task_one_hot = th.cat([task_one_hot, extra_dims], dim=-1)
        shuff_task_one_hot = task_one_hot[..., th.randperm(self.n_tasks + self.n_extra_tasks)]
        return self.fc(shuff_task_one_hot)


class CountEmbedder(nn.Module):
    """
    Create embedding that encodes the quantity of agents/non-agent entities
    belonging to each subtask and returns a vector for each entity based on the
    subtask they belong to
    """
    def __init__(self, embed_dim, args) -> None:
        super().__init__()
        self.args = args
        self.count_embed = nn.Linear(2, embed_dim)

    def forward(self, entity2task):
        x1_pertask_count = self.count_embed(th.stack(
            [entity2task[:, :self.args.n_agents].sum(dim=1),
            entity2task[:, self.args.n_agents:].sum(dim=1)], dim=-1)) * COUNT_NORM_FACTOR
        x1_count = th.bmm(entity2task, x1_pertask_count)
        return x1_count


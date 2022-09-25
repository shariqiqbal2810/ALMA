from torch.distributions import OneHotCategorical

def random_allocs(task_mask, entity_mask, n_agents):
    active_tasks = (1 - task_mask).float()
    task_probs = active_tasks / active_tasks.sum(dim=1, keepdim=True)
    task_probs = task_probs.unsqueeze(1).repeat(1, n_agents, 1)
    rand_allocs = OneHotCategorical(probs=task_probs).sample()
    # mask out inactive agents
    rand_allocs *= (1 - entity_mask[:, :n_agents].float()).unsqueeze(2)
    return rand_allocs
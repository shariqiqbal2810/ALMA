from os import stat
import torch as th


def build_td_lambda_targets__old(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    bs = rewards.size(0)
    max_t = rewards.size(1)
    targets = rewards.new(target_qs.size()).zero_()[:,:-1] # Produce 1 less target than the inputted Q-Values
    running_target = rewards.new(bs, n_agents).zero_()
    terminated = terminated.float()
    for t in reversed(range(max_t)):
        if t == max_t - 1:
            running_target = mask[:, t] * (rewards[:, t] + gamma * (1 - terminated[:, t]) * target_qs[:, t])
        else:
            running_target = mask[:, t] * (
                    terminated[:, t] * rewards[:, t]  # Just the reward if the env terminates
                 + (1 - terminated[:, t]) * (rewards[:, t] + gamma * (td_lambda * running_target + (1 - td_lambda) * target_qs[:, t]))
            )
        targets[:, t, :] = running_target
    return targets


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]

class RunningMeanStd(object):
    # https://github.com/openai/random-network-distillation/blob/master/mpi_util.py
    def __init__(self, epsilon=1e-4):
        self.mean = 0
        self.var = 1
        self.count = epsilon

    def update(self, x):
        batch_dims = tuple(range(len(x.shape) - 1))
        batch_mean, batch_std = th.mean(x, dim=batch_dims, keepdim=True), th.std(x, dim=batch_dims, keepdim=True)
        batch_count = th.tensor(x.shape)[:-1].prod()
        batch_var = th.pow(batch_std, 2)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + th.pow(delta, 2) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class ExponentialMeanStd(object):
    # https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
    def __init__(self, alpha=0.001, agg_inds=None):
        self._mean = None
        self._var = None
        self.alpha = alpha
        self.agg_inds = agg_inds
        if self.agg_inds is not None:
            self.total_dims = sum(len(inds) for inds in self.agg_inds)

    def update(self, x, mask=None):
        if mask is None:
            mask = th.ones_like(x)
        batch_dims = tuple(range(len(x.shape) - 1))
        if self._mean is None:
            self._mean = masked_mean(x, mask, agg_inds=self.agg_inds, dim=batch_dims, keepdim=True)
        deltas = x - self.mean
        self._mean += self.alpha * masked_mean(deltas, mask, agg_inds=self.agg_inds, dim=batch_dims, keepdim=True)
        if self._var is None:
            self._var = masked_mean(deltas**2, mask, agg_inds=self.agg_inds, dim=batch_dims, keepdim=True)
        self._var = (1 - self.alpha) * (self._var + self.alpha * masked_mean(deltas**2, mask, agg_inds=self.agg_inds, dim=batch_dims, keepdim=True))

    @property
    def mean(self):
        if self.agg_inds is None:
            return self._mean
        if self._mean is None:
            return None
        exp_mean = th.zeros_like(self._mean[..., [0]]).repeat_interleave(self.total_dims, dim=-1)
        for i, inds in enumerate(self.agg_inds):
            exp_mean[..., inds] = self._mean[..., [i]]
        return exp_mean

    @property
    def var(self):
        if self.agg_inds is None:
            return self._var
        if self._var is None:
            return None
        exp_var = th.zeros_like(self._var[..., [0]]).repeat_interleave(self.total_dims, dim=-1)
        for i, inds in enumerate(self.agg_inds):
            exp_var[..., inds] = self._var[..., [i]]
        return exp_var

    def load_state_dict(self, state_dict):
        self._mean = state_dict['mean']
        self._var = state_dict['var']

    def state_dict(self):
        if self._mean is None:
            return {'mean': None, 'var': None}
        return {'mean': self._mean.clone(),
                'var': self._var.clone()}


def masked_mean(x, mask, agg_inds=None, **kwargs):
    masked_sum = (x * mask).sum(**kwargs)
    total_items = mask.sum(**kwargs)
    if agg_inds is not None:
        masked_sum = th.cat([masked_sum[..., inds].sum(dim=-1, keepdim=True) for inds in agg_inds], dim=-1)
        total_items = th.cat([total_items[..., inds].sum(dim=-1, keepdim=True) for inds in agg_inds], dim=-1)
    return masked_sum / (total_items + 1e-8)

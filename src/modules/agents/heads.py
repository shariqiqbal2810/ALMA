import torch as th
import torch.nn as nn


class RecurrentHead(nn.Module):
    def __init__(self, args, out_dim=None):
        super().__init__()
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        if out_dim is None:
            out_dim = args.n_actions
        self.fc_out = nn.Linear(args.rnn_hidden_dim, out_dim)
        self.args = args

    def forward(self, x, inputs):
        bs, ts, na, ed = x.shape
        raise NotImplementedError("Need to duplicate hidden state for REFIL")

        h = inputs['hidden_state'].reshape(-1, self.args.rnn_hidden_dim)
        hs = []
        for t in range(ts):
            curr_x = x[:, t].reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(curr_x, h)
            hs.append(h.reshape(bs, self.args.n_agents, self.args.rnn_hidden_dim))
            # re-init hidden if env is reset
            h = h * (1 - inputs['reset'][:, t]).view(bs, 1).repeat_interleave(self.args.n_agents, 0)
        hs = th.stack(hs, dim=1)  # Concat over time

        q = self.fc_out(hs)
        q = q.reshape(bs, ts, self.args.n_agents, -1)
        return q, hs


class FeedforwardHead(nn.Module):
    def __init__(self, args, in_dim=None, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = args.n_actions
        if in_dim is None:
            in_dim = args.rnn_hidden_dim
        self.fc_out = nn.Linear(in_dim, out_dim)

    def forward(self, x, inputs):
        return self.fc_out(x), x

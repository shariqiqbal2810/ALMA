import torch.nn as nn
import torch.nn.functional as F
from ..layers import EntityAttentionLayer


class EntityBase(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.attn_embed_dim)
        self.attn = EntityAttentionLayer(args.attn_embed_dim,
                                            args.attn_embed_dim,
                                            args.attn_embed_dim, args)
        self.fc2 = nn.Linear(args.attn_embed_dim, args.rnn_hidden_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs):
        entities = inputs['entities']
        obs_mask = inputs['obs_mask']
        entity_mask = inputs['entity_mask']
        bs, ts, ne, ed = entities.shape
        entities = entities.reshape(bs * ts, ne, ed)
        obs_mask = obs_mask.reshape(bs * ts, ne, ne)
        entity_mask = entity_mask.reshape(bs * ts, ne)
        agent_mask = entity_mask[:, :self.args.n_agents]
        x1 = self.fc1(entities)
        if 'task_embeds' in inputs:
            x1 += inputs['task_embeds'].reshape(bs * ts, ne, -1)
        attn_outs = self.attn(F.relu(x1), pre_mask=obs_mask,
                              post_mask=agent_mask)
        return F.relu(self.fc2(F.relu(attn_outs))).reshape(
            bs, ts, self.args.n_agents, self.args.rnn_hidden_dim)


class StandardBase(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs):
        return F.relu(self.fc1(inputs['obs']))

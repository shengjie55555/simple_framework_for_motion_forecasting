import torch
import torch.nn as nn
from einops import repeat, rearrange
from model.decoder import SimpleDecoder as Decoder
from model.layers import Linear, LinearRes
from utils.data_utils import gpu


class MHL(nn.Module):
    def __init__(self, cfg, device):
        super(MHL, self).__init__()
        a_dim, out_dim, a_num_layer, num_mode, pred_len = \
            cfg["n_agent"], cfg["n_feature"], \
            cfg["n_layer_agent"], cfg["num_mode"], cfg["pred_len"]
        self.device = device
        self.agent_encoder = AgentEncoder(a_dim, out_dim, a_num_layer)
        self.interaction_encoder = InteractionEncoder(out_dim)
        self.decoder = Decoder(out_dim, num_mode, pred_len, is_cat=True)

    def forward(self, data):
        agents, agent_ids, agent_ctrs = agent_gather(gpu(data["trajs_obs"], self.device),
                                                     gpu(data["pad_obs"], self.device))
        agents = self.agent_encoder(agents)
        i_agents = self.interaction_encoder(agents, agent_ids)

        agents = torch.cat([agents, i_agents], dim=-1)
        reg, cls = self.decoder(agents)

        out = dict()
        out['cls'], out['reg'] = [], []
        for i in range(len(agent_ids)):
            ids = agent_ids[i]
            ctrs = agent_ctrs[i].view(-1, 1, 1, 2)
            reg[ids] = reg[ids] + ctrs
            out['cls'].append(cls[ids])
            out['reg'].append(reg[ids])

        return out


def agent_gather(trajs_obs, pad_obs):
    batch_size = len(trajs_obs)
    num_agents = [len(x) for x in trajs_obs]

    agents = []
    for i in range(batch_size):
        feats = torch.zeros_like(trajs_obs[i][:, :, :2])
        feats[:, 1:, :] = trajs_obs[i][:, 1:, :] - trajs_obs[i][:, :-1, :]
        agents.append(torch.cat([trajs_obs[i][:, :, :2], feats, pad_obs[i].unsqueeze(2)], dim=-1))

    agents = torch.cat(agents, 0)  # n * obs_len * 5 (x, y, dx, dy, mask)
    agents[:, :, :-1] *= agents[:, :, -1:]
    agents[:, 1:, :-1] *= agents[:, :-1, -1:]

    agent_ids = []
    count = 0
    for i in range(batch_size):
        ids = torch.arange(count, count + num_agents[i]).to(agents.device)
        agent_ids.append(ids)
        count += num_agents[i]

    agent_ctrs = [agents[ids, -1, :2] for ids in agent_ids]

    return agents, agent_ids, agent_ctrs


class AgentEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers):
        super(AgentEncoder, self).__init__()
        assert out_dim % (2 ** num_layers) == 0, "feature dim: {}, num_layer: {}".format(out_dim, num_layers)

        self.backbone = nn.ModuleList()
        for i in range(num_layers):
            hidden_dim = out_dim // (2 ** (num_layers - i - 1))
            self.backbone.append(LinearRes(in_dim, hidden_dim, hidden_dim))
            in_dim = hidden_dim

        self.h0 = nn.Parameter(torch.zeros(1, 1, out_dim))  # num_layer_in_LSTM * n * out_dim
        self.c0 = nn.Parameter(torch.zeros(1, 1, out_dim))  # num_layer_in_LSTM * n * out_dim
        self.lstm = nn.LSTM(input_size=out_dim, hidden_size=out_dim)

    def forward(self, x):
        for layer in self.backbone:
            x = layer(x)
        n = x.shape[0]
        h0 = repeat(self.h0, "h n d -> h (n1 n) d", n1=n)
        c0 = repeat(self.c0, "h n d-> h (n1 n) d", n1=n)
        output, (hn, cn) = self.lstm(x.transpose(0, 1), (h0, c0))
        return hn.squeeze(0)


class InteractionEncoder(nn.Module):
    def __init__(self, out_dim):
        super(InteractionEncoder, self).__init__()
        self.n_head = 6
        self.scale = out_dim ** -0.5

        self.to_q = Linear(out_dim, self.n_head * out_dim, act=False)
        self.to_k = Linear(out_dim, self.n_head * out_dim, act=False)
        self.to_v = Linear(out_dim, self.n_head * out_dim)
        self.to_out = nn.Sequential(
            Linear(self.n_head * out_dim, out_dim),
            nn.Linear(out_dim, out_dim, bias=False)
        )

        self.linear_1 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.linear_2 = nn.Linear(out_dim, out_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, agents, agent_ids):
        batch_size = len(agent_ids)
        hi, wi = [], []
        for i in range(batch_size):
            ids = agent_ids[i]
            hi.append(repeat(ids, "d -> (d n)", n=len(ids)))
            wi.append(repeat(ids, "d -> (n d)", n=len(ids)))
        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)

        nodes = agents
        res = nodes
        q = self.to_q(nodes[hi])
        k = self.to_k(nodes[wi])
        v = self.to_v(nodes[wi])

        query, key, value = map(lambda t: rearrange(t, "n (h d) -> n h d", h=self.n_head).unsqueeze(-2), [q, k, v])

        att = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        att = (att - att.max()).exp()
        att_sum = torch.zeros(nodes.shape[0], self.n_head, 1, 1).to(nodes.device)
        att_sum.index_add_(0, hi, att)
        att = att / att_sum[hi]

        out = torch.matmul(att, value).squeeze(-2)
        out = torch.zeros(nodes.shape[0], self.n_head, nodes.shape[1]).to(nodes.device).index_add_(0, hi, out)
        out = rearrange(out, "n h d -> n (h d)")
        out = self.to_out(out)

        nodes = self.linear_1(nodes)
        nodes = self.relu(self.norm(nodes + out))

        nodes = self.linear_2(nodes)
        nodes = self.relu(nodes + res)

        return nodes

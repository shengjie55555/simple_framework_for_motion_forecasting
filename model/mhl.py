import torch
import torch.nn as nn
from einops import repeat, rearrange
from model.decoder import SimpleDecoder as Decoder
from model.atds import Linear
from utils.data_utils import gpu


class MHL(nn.Module):
    def __init__(self, cfg, device):
        super(MHL, self).__init__()
        n_agt, n_out, n_agt_layer, num_mode, pred_len = \
            cfg["n_agent"], cfg["n_feature"], \
            cfg["n_agent_layer"], cfg["num_mode"], cfg["pred_len"]
        self.device = device
        self.agent_encoder = AgentEncoder(n_agt, n_out, n_agt_layer)
        self.interaction_encoder = InteractionEncoder(n_out)
        self.decoder = Decoder(n_out, num_mode, pred_len)

    def forward(self, data):
        agents, agent_ids, agent_ctrs = agent_gather(gpu(data["feats"], self.device),
                                                     gpu(data["locs"], self.device))
        agents = self.agent_encoder(agents)

        agents = self.interaction_encoder(agents, agent_ids)

        out = self.decoder(agents, agent_ids, agent_ctrs)
        rot, orig = gpu(data["rot"], self.device), gpu(data["orig"], self.device)

        # transform prediction to world coordinates
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(1, 1, 1, -1)
        return out


def agent_gather(agents, locs):
    batch_size = len(agents)
    num_agents = [len(x) for x in agents]

    agents = torch.cat(agents, dim=0)
    locs = torch.cat(locs, dim=0)
    agents = torch.cat([locs[..., :2], agents], dim=-1)

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
            self.backbone.append(LinearRes(in_dim, hidden_dim))
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


class LinearRes(nn.Module):
    def __init__(self, n_in, n_out):
        super(LinearRes, self).__init__()
        self.linear1 = nn.Linear(n_in, n_out, bias=False)
        self.linear2 = nn.Linear(n_out, n_out, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = nn.LayerNorm([20, n_out])
        self.norm2 = nn.LayerNorm([20, n_out])

        if n_in != n_out:
            self.transform = nn.Sequential(
                nn.Linear(n_in, n_out, bias=False),
                nn.LayerNorm([20, n_out]))
        else:
            self.transform = None

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)

        if self.transform is not None:
            out += self.transform(x)
        else:
            out += x

        out = self.relu(out)
        return out


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

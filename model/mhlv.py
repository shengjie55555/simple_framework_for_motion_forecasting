import torch
import torch.nn as nn
from einops import repeat, rearrange
from model.decoder import SimpleDecoder as Decoder
from model.atds import Linear
from utils.data_utils import gpu


class MHLV(nn.Module):
    def __init__(self, cfg, device):
        super(MHLV, self).__init__()
        n_agt, n_lane, n_out, n_agent_layer, n_lane_layer, num_mode, pred_len = \
            cfg["n_agent"], cfg["n_lane"], cfg["n_feature"], \
            cfg["n_agent_layer"], cfg["n_lane_layer"], cfg["num_mode"], cfg["pred_len"]
        self.device = device
        self.agent_encoder = AgentEncoder(n_agt, n_out, n_agent_layer)
        self.lane_encoder = LaneEncoder(n_lane, n_out, n_lane_layer)
        self.interaction_encoder = InteractionEncoder(n_out)
        self.decoder = Decoder(n_out, num_mode, pred_len)

    def forward(self, data):
        agents, agent_ids, agent_ctrs = agent_gather(gpu(data["trajs_obs"], self.device),
                                                     gpu(data["pad_obs"], self.device))
        lanes, lane_ids, lane_ctrs, lane_seg = graph_gather(gpu(data["graph"], self.device))

        agents = self.agent_encoder(agents)
        lanes = self.lane_encoder(lanes, lane_seg)

        agents = self.interaction_encoder(agents, agent_ids, lanes, lane_ids)

        out = self.decoder(agents, agent_ids, agent_ctrs)

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


def graph_gather(graphs):
    batch_size = len(graphs)
    lane_ids = []
    count = 0
    for i in range(batch_size):
        ids = torch.arange(count, count + graphs[i]["num_nodes"]).to(graphs[i]["feats"].device)
        lane_ids.append(ids)
        count += graphs[i]["num_nodes"]

    lanes = torch.cat([x["feats"] for x in graphs], dim=0)
    lane_ctrs = torch.cat([x["ctrs"] for x in graphs], dim=0)

    lane_seg = []
    for key in ["turn", "control", "intersect"]:
        lane_seg.append(torch.cat([x[key] for x in graphs], dim=0).type(torch.float32))
    lane_seg = torch.cat([lane_seg[0], lane_seg[1].unsqueeze(-1), lane_seg[2].unsqueeze(-1)], dim=-1)

    return lanes, lane_ids, lane_ctrs, lane_seg


class LinearRes(nn.Module):
    def __init__(self, n_in, n_out, norm="ln", n_l=10):
        super(LinearRes, self).__init__()
        self.linear1 = nn.Linear(n_in, n_out, bias=False)
        self.linear2 = nn.Linear(n_out, n_out, bias=False)
        self.relu = nn.ReLU(inplace=True)

        assert norm in ["ln", "gn"], "{} is not in [ln, gn]".format(norm)
        if norm == "ln":
            self.norm1 = nn.LayerNorm([n_l, n_out])
            self.norm2 = nn.LayerNorm([n_l, n_out])
        else:
            self.norm1 = nn.GroupNorm(1, n_out)
            self.norm2 = nn.GroupNorm(1, n_out)

        if n_in != n_out:
            self.transform = nn.Sequential(
                nn.Linear(n_in, n_out, bias=False),
                nn.LayerNorm([n_l, n_out]) if norm == "ln" else nn.GroupNorm(1, n_out))
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


class AgentEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers):
        super(AgentEncoder, self).__init__()
        assert out_dim % (2 ** num_layers) == 0, "feature dim: {}, num_layer: {}".format(out_dim, num_layers)

        self.backbone = nn.ModuleList()
        for i in range(num_layers):
            hidden_dim = out_dim // (2 ** (num_layers - i - 1))
            self.backbone.append(LinearRes(in_dim, hidden_dim, "ln", 20))
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


class LaneEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers):
        super(LaneEncoder, self).__init__()
        assert out_dim % (2 ** num_layers) == 0, "feature dim: {}, num_layer: {}".format(out_dim, num_layers)

        self.seg = LinearRes(in_dim * 2, out_dim, "gn")

        self.backbone = nn.ModuleList()
        for i in range(num_layers):
            hidden_dim = out_dim // (2 ** (num_layers - i))
            self.backbone.append(LinearRes(in_dim, hidden_dim, "ln", 10))
            in_dim = hidden_dim + hidden_dim

        self.max_pool = nn.MaxPool1d(kernel_size=10, stride=1)

        self.fuse = LinearRes(out_dim, out_dim, "gn")

    def forward(self, lanes, lane_seg):
        for layer in self.backbone:
            lanes = layer(lanes)
            global_feats = repeat(self.max_pool(lanes.transpose(1, 2)).squeeze(-1), "n d -> n l d", l=10)
            lanes = torch.cat([lanes, global_feats], dim=-1)
        lanes = self.max_pool(lanes.transpose(1, 2)).squeeze(-1)

        lanes = self.fuse(lanes + self.seg(lane_seg))

        return lanes


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

    def forward(self, agents, agent_ids, lanes, lane_ids):
        batch_size = len(agent_ids)
        lane_ids = [ids + agents.shape[0] for ids in lane_ids]
        hi, wi = [], []
        for i in range(batch_size):
            ids = torch.cat([agent_ids[i], lane_ids[i]], dim=0)
            hi.append(repeat(ids, "d -> (d n)", n=len(ids)))
            wi.append(repeat(ids, "d -> (n d)", n=len(ids)))
        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)

        nodes = torch.cat([agents, lanes], dim=0)
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

        agent_ids = torch.cat(agent_ids, dim=0)

        return nodes[agent_ids]

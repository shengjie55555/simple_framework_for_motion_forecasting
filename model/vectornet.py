import torch
import torch.nn as nn
from einops import repeat, rearrange
from model.layers import Linear, LinearRes
from utils.data_utils import gpu


class VectorNet(nn.Module):
    def __init__(self, cfg, device):
        super(VectorNet, self).__init__()
        a_dim, m_dim, out_dim, a_num_layer, m_num_layer, num_mode, pred_len = \
            cfg["n_agent"], cfg["n_lane"], cfg["n_feature"], \
            cfg["n_layer_agent"], cfg["n_layer_lane"], cfg["num_mode"], cfg["pred_len"]
        self.device = device
        self.agent_encoder = AgentEncoder(a_dim, out_dim, a_num_layer)
        self.lane_encoder = LaneEncoder(m_dim, out_dim, m_num_layer)
        self.interaction_encoder = InteractionEncoder(out_dim)
        self.decoder = Decoder(out_dim, num_mode, pred_len)

    def forward(self, data):
        agents, agent_ids, agent_ctrs = agent_gather(gpu(data["trajs_obs"], self.device),
                                                     gpu(data["pad_obs"], self.device))
        lanes, lane_ids, lane_ctrs, lane_seg = graph_gather(gpu(data["graph"], self.device))

        agents = self.agent_encoder(agents)
        lanes = self.lane_encoder(lanes, lane_seg)

        i_agents, i_lanes = self.interaction_encoder(agents, agent_ids, lanes, lane_ids)

        reg, cls = self.decoder(agents, i_agents)

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


class LaneEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers):
        super(LaneEncoder, self).__init__()
        assert out_dim % (2 ** num_layers) == 0, "feature dim: {}, num_layer: {}".format(out_dim, num_layers)

        self.seg = LinearRes(in_dim * 2, out_dim, out_dim)

        self.backbone = nn.ModuleList()
        for i in range(num_layers):
            hidden_dim = out_dim // (2 ** (num_layers - i))
            self.backbone.append(LinearRes(in_dim, hidden_dim, hidden_dim))
            in_dim = hidden_dim + hidden_dim

        self.max_pool = nn.MaxPool1d(kernel_size=10, stride=1)

        self.fuse = LinearRes(out_dim, out_dim, out_dim)

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
        lane_ids = torch.cat(lane_ids, dim=0)

        return nodes[agent_ids], nodes[lane_ids]


class Decoder(nn.Module):
    def __init__(self, out_dim, num_mode, pred_len):
        super(Decoder, self).__init__()
        self.out_dim = out_dim
        self.num_mode = num_mode
        self.pred_len = pred_len

        self.traj_decoder = nn.ModuleList(nn.Sequential(
            LinearRes(2 * out_dim, out_dim, out_dim),
            nn.Linear(out_dim, self.pred_len * 2, bias=False)
        ) for _ in range(num_mode))

        self.score_decoder = nn.ModuleDict({
            "goal": Linear(2, out_dim),
            "score": nn.Sequential(
                LinearRes(3 * out_dim, out_dim, out_dim),
                nn.Linear(out_dim, 1, bias=False),
                nn.Softmax(dim=-2)
            ),
        })

    def forward(self, agents, i_agents):
        agents = torch.cat([agents, i_agents], dim=-1)
        preds = []
        for i in range(self.num_mode):
            preds.append(self.traj_decoder[i](agents))
        reg = torch.cat([pred.unsqueeze(-2) for pred in preds]).view(agents.shape[0], self.num_mode, self.pred_len, -1)

        goals = reg[:, :, -1].detach()
        goals = self.score_decoder["goal"](goals)
        agents = torch.cat([repeat(agents, "n d -> n h d", h=self.num_mode), goals], dim=-1)
        cls = self.score_decoder["score"](agents).squeeze(-1)

        return reg, cls


class Loss(nn.Module):
    def __init__(self, cfg, device):
        super(Loss, self).__init__()
        self.cfg = cfg
        self.device = device
        self.pred_loss = PredLoss(cfg)

    def forward(self, out, data):
        loss_out = self.pred_loss(out, gpu(data["trajs_fut"], self.device), gpu(data["pad_fut"], self.device))
        loss_out["loss"] = loss_out["cls_loss"] / (loss_out["num_cls"] + 1e-10) + \
                           loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        return loss_out


class PredLoss(nn.Module):
    def __init__(self, cfg):
        super(PredLoss, self).__init__()
        self.cfg = cfg
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out, gt_preds, has_preds):
        cls, reg = out["cls"], out["reg"]
        cls = torch.cat([x for x in cls], 0)
        reg = torch.cat([x for x in reg], 0)
        gt_preds = torch.cat([x for x in gt_preds], 0)
        has_preds = torch.cat([x for x in has_preds], 0) == 1

        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0
        loss_out["reg_loss"] = zero.clone()
        loss_out["num_reg"] = 0

        num_mode, pred_len = self.cfg["num_mode"], self.cfg["pred_len"]

        last = has_preds.float() + 0.1 * torch.arange(pred_len).float().to(has_preds.device) / float(pred_len)
        max_last, last_ids = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_ids = last_ids[mask]

        row_ids = torch.arange(len(last_ids)).long().to(last_ids.device)
        dist = []
        for j in range(num_mode):
            dist.append(
                torch.sqrt(
                    (
                            (reg[row_ids, j, last_ids] - gt_preds[row_ids, last_ids])
                            ** 2
                    ).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_ids = dist.min(1)
        row_ids = torch.arange(len(min_ids)).long().to(min_ids.device)

        mgn = cls[row_ids, min_ids].unsqueeze(1) - cls
        mask0 = (min_dist < self.cfg["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.cfg["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.cfg["mgn"]
        coef = self.cfg["cls_coef"]
        loss_out["cls_loss"] += coef * (
                self.cfg["mgn"] * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()

        reg = reg[row_ids, min_ids]
        coef = self.cfg["reg_coef"]
        loss_out["reg_loss"] += coef * self.reg_loss(
            reg[has_preds], gt_preds[has_preds]
        )
        loss_out["num_reg"] += has_preds.sum().item()

        return loss_out

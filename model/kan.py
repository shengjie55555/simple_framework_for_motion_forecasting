import torch
import torch.nn as nn
import torch.nn.functional as F
from fractions import gcd
from einops import rearrange
from utils.data_utils import gpu, to_long
from collections import defaultdict


class KAN(nn.Module):
    def __init__(self, config, device):
        super(KAN, self).__init__()
        self.config = config
        self.device = device

        self.agent_encoder = AgentEncoder(config)
        self.map_encoder = MapEncoder(config)

        self.a2m = A2M(config)
        self.m2m = M2M(config)
        self.m2a = M2A(config)
        self.a2a = A2A(config)

        self.pyramid_decoder = PyramidDecoder(config)

    def forward(self, data):
        # construct agent feature
        agents, agent_idcs, agent_locs = agent_gather(gpu(data["feats"], self.device), gpu(data["locs"], self.device))
        agent_ctrs = gpu(data["ctrs"], self.device)
        agents = self.agent_encoder(agents, agent_locs)

        # construct map features
        graph = graph_gather(to_long(gpu(data["graph"], self.device)))
        nodes, node_idcs, node_ctrs = self.map_encoder(graph)

        # interactions
        nodes = self.a2m(nodes, graph, agents, agent_idcs, agent_ctrs)
        nodes = self.m2m(nodes, graph)
        agents = self.m2a(agents, agent_idcs, agent_ctrs, nodes, node_idcs, node_ctrs)
        agents = self.a2a(agents, agent_idcs, agent_ctrs)

        # prediction
        out = self.pyramid_decoder(agents, agent_idcs, agent_ctrs, nodes, node_idcs, node_ctrs)
        rot, orig = gpu(data["rot"], self.device), gpu(data["orig"], self.device)

        # transform prediction to world coordinates
        for i in range(len(out["reg"])):
            if "scaling_ratio" in data.keys():
                out["reg"][i] /= data["scaling_ratio"][i]
                out["key_points"][i] /= data["scaling_ratio"][i]
            if "flip" in data.keys() and data["flip"][i] < self.config["flip"]:
                out["reg"][i] = torch.cat((out["reg"][i][..., 1:], out["reg"][i][..., :1]), dim=-1)
                out['key_points'][i] = torch.cat((out["key_points"][i][..., 1:], out["key_points"][i][..., :1]), dim=-1)
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(1, 1, 1, -1)
            out['key_point'][i] = torch.matmul(out['key_point'][i], rot[i]) + orig[i]
            out['key_points'][i] = torch.matmul(out['key_points'][i], rot[i]) + orig[i].view(1, 1, 1, -1)
        return out


def agent_gather(agents, locs):
    batch_size = len(agents)
    num_agents = [len(x) for x in agents]

    agents = [x.transpose(1, 2) for x in agents]
    agents = torch.cat(agents, 0)

    locs = [x.transpose(1, 2) for x in locs]
    agent_locs = torch.cat(locs, 0)

    agent_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_agents[i]).to(agents.device)
        agent_idcs.append(idcs)
        count += num_agents[i]
    return agents, agent_idcs, agent_locs


def graph_gather(graphs):
    batch_size = len(graphs)
    node_idcs = []
    count = 0
    counts = []
    for i in range(batch_size):
        counts.append(count)
        idcs = torch.arange(count, count + graphs[i]["num_nodes"]).to(
            graphs[i]["feats"].device
        )
        node_idcs.append(idcs)
        count = count + graphs[i]["num_nodes"]

    graph = dict()
    graph["idcs"] = node_idcs
    graph["ctrs"] = [x["ctrs"] for x in graphs]

    for key in ["feats", "turn", "control", "intersect"]:
        graph[key] = torch.cat([x[key] for x in graphs], 0)

    for k1 in ["pre", "suc"]:
        graph[k1] = []
        for i in range(len(graphs[0]["pre"])):
            graph[k1].append(dict())
            for k2 in ["u", "v"]:
                graph[k1][i][k2] = torch.cat(
                    [graphs[j][k1][i][k2] + counts[j] for j in range(batch_size)], 0
                )

    for k1 in ["left", "right"]:
        graph[k1] = dict()
        for k2 in ["u", "v"]:
            temp = [graphs[i][k1][k2] + counts[i] for i in range(batch_size)]
            temp = [
                x if x.dim() > 0 else graph["pre"][0]["u"].new().resize_(0)
                for x in temp
            ]
            graph[k1][k2] = torch.cat(temp)
    return graph


class AgentEncoder(nn.Module):
    def __init__(self, config):
        super(AgentEncoder, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_in = 3
        n_out = [32, 64, 128]
        blocks = [Res1d, Res1d, Res1d]
        num_blocks = [2, 2, 2]

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        n = config["n_agent"]
        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], n, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(n, n, norm=norm, ng=ng)

        self.subgraph = Attention(n)
        self.input = nn.Sequential(
            nn.Linear(30, n),
            nn.ReLU(inplace=True),
            Linear(n, n, ng=ng, act=False),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, agents, agent_locs):
        out = agents

        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        out = self.output(out).transpose(1, 2)
        out = self.subgraph(out)
        agent_locs = rearrange(agent_locs[:, :, 10:], 'b c l -> b (c l)')
        out += self.input(agent_locs)
        out = self.relu(out)
        return out


class MapEncoder(nn.Module):
    def __init__(self, config):
        super(MapEncoder, self).__init__()
        self.config = config
        n_map = config["n_map"]
        norm = "GN"
        ng = 1

        self.input = nn.Sequential(
            nn.Linear(2, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, norm=norm, ng=ng, act=False),
        )
        self.seg = nn.Sequential(
            nn.Linear(2, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, norm=norm, ng=ng, act=False),
        )

        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["num_scales"]):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(gcd(ng, n_map), n_map))
                elif key in ["ctr2"]:
                    fuse[key].append(Linear(n_map, n_map, norm=norm, ng=ng, act=False))
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, graph):
        if (
            len(graph["feats"]) == 0
            or len(graph["pre"][-1]["u"]) == 0
            or len(graph["suc"][-1]["u"]) == 0
        ):
            temp = graph["feats"]
            return (
                temp.new().resize_(0),
                [temp.new().long().resize_(0) for x in graph["node_idcs"]],
                temp.new().resize_(0),
            )

        ctrs = torch.cat(graph["ctrs"], 0)
        feat = self.input(ctrs)
        feat += self.seg(graph["feats"])
        feat = self.relu(feat)

        """fuse map"""
        res = feat
        for i in range(len(self.fuse["ctr"])):
            temp = self.fuse["ctr"][i](feat)
            for key in self.fuse:
                if key.startswith("pre") or key.startswith("suc"):
                    k1 = key[:3]
                    k2 = int(key[3:])
                    temp.index_add_(
                        0,
                        graph[k1][k2]["u"],
                        self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                    )

            if len(graph["left"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["left"]["u"],
                    self.fuse["left"][i](feat[graph["left"]["v"]]),
                )
            if len(graph["right"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["right"]["u"],
                    self.fuse["right"][i](feat[graph["right"]["v"]]),
                )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            feat = self.relu(feat)
            res = feat
        return feat, graph["idcs"], graph["ctrs"]


class A2M(nn.Module):
    def __init__(self, config):
        super(A2M, self).__init__()
        self.config = config
        n_map = config["n_map"]
        norm = "GN"
        ng = 1

        """fuse meta, static, dyn"""
        self.meta = Linear(n_map + 4, n_map, norm=norm, ng=ng)
        att = []
        for i in range(2):
            att.append(Att(n_map, config["n_agent"]))
        self.att = nn.ModuleList(att)

    def forward(self, feat, graph, agents, agent_idcs, agent_ctrs):
        """meta, static and dyn fuse using attention"""
        meta = torch.cat(
            (
                graph["turn"],
                graph["control"].unsqueeze(1),
                graph["intersect"].unsqueeze(1),
            ),
            1,
        )
        feat = self.meta(torch.cat((feat, meta), 1))
        for i in range(len(self.att)):
            feat = self.att[i](
                feat,
                graph["idcs"],
                graph["ctrs"],
                agents,
                agent_idcs,
                agent_ctrs,
                self.config["agent2map_dist"],
            )
        return feat


class M2M(nn.Module):
    def __init__(self, config):
        super(M2M, self).__init__()
        self.config = config
        n_map = config["n_map"]
        norm = "GN"
        ng = 1

        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["num_scales"]):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(gcd(ng, n_map), n_map))
                elif key in ["ctr2"]:
                    fuse[key].append(Linear(n_map, n_map, norm=norm, ng=ng, act=False))
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat, graph):
        """fuse map"""
        res = feat
        for i in range(len(self.fuse["ctr"])):
            temp = self.fuse["ctr"][i](feat)
            for key in self.fuse:
                if key.startswith("pre") or key.startswith("suc"):
                    k1 = key[:3]
                    k2 = int(key[3:])
                    temp.index_add_(
                        0,
                        graph[k1][k2]["u"],
                        self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                    )

            if len(graph["left"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["left"]["u"],
                    self.fuse["left"][i](feat[graph["left"]["v"]]),
                )
            if len(graph["right"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["right"]["u"],
                    self.fuse["right"][i](feat[graph["right"]["v"]]),
                )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            feat = self.relu(feat)
            res = feat
        return feat


class M2A(nn.Module):
    def __init__(self, config):
        super(M2A, self).__init__()
        self.config = config

        n_agent = config["n_agent"]
        n_map = config["n_map"]

        att = []
        for i in range(2):
            att.append(Att(n_agent, n_map))
        self.att = nn.ModuleList(att)

    def forward(self, agents, agent_idcs, agent_ctrs, nodes,
                node_idcs, node_ctrs):
        for i in range(len(self.att)):
            agents = self.att[i](
                agents,
                agent_idcs,
                agent_ctrs,
                nodes,
                node_idcs,
                node_ctrs,
                self.config["map2agent_dist"],
            )
        return agents


class A2A(nn.Module):
    def __init__(self, config):
        super(A2A, self).__init__()
        self.config = config

        n_agent = config["n_agent"]

        att = []
        for i in range(2):
            att.append(Att(n_agent, n_agent))
        self.att = nn.ModuleList(att)

    def forward(self, agents, agent_idcs, agent_ctrs):
        for i in range(len(self.att)):
            agents = self.att[i](
                agents,
                agent_idcs,
                agent_ctrs,
                agents,
                agent_idcs,
                agent_ctrs,
                self.config["agent2agent_dist"],
            )
        return agents


class Att(nn.Module):
    def __init__(self, n_agt, n_ctx):
        super(Att, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_ctx),
            nn.ReLU(inplace=True),
            Linear(n_ctx, n_ctx, norm=norm, ng=ng),
        )

        self.query = Linear(n_agt, n_ctx, norm=norm, ng=ng)

        self.ctx = nn.Sequential(
            Linear(3 * n_ctx, n_agt, norm=norm, ng=ng),
            nn.Linear(n_agt, n_agt, bias=False),
        )

        self.agt = nn.Linear(n_agt, n_agt, bias=False)
        self.norm = nn.GroupNorm(gcd(ng, n_agt), n_agt)
        self.linear = Linear(n_agt, n_agt, norm=norm, ng=ng, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, agts, agt_idcs, agt_ctrs, ctx, ctx_idcs, ctx_ctrs, dist_th):
        res = agts
        if len(ctx) == 0:
            agts = self.agt(agts)
            agts = self.relu(agts)
            agts = self.linear(agts)
            agts += res
            agts = self.relu(agts)
            return agts

        batch_size = len(agt_idcs)
        hi, wi = [], []
        hi_count, wi_count = 0, 0
        for i in range(batch_size):
            dist = agt_ctrs[i].view(-1, 1, 2) - ctx_ctrs[i].view(1, -1, 2)
            dist = torch.sqrt((dist ** 2).sum(2))
            mask = dist <= dist_th

            idcs = torch.nonzero(mask, as_tuple=False)
            if len(idcs) == 0:
                continue

            hi.append(idcs[:, 0] + hi_count)
            wi.append(idcs[:, 1] + wi_count)
            hi_count += len(agt_idcs[i])
            wi_count += len(ctx_idcs[i])
        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)

        agt_ctrs = torch.cat(agt_ctrs, 0)
        ctx_ctrs = torch.cat(ctx_ctrs, 0)
        dist = agt_ctrs[hi] - ctx_ctrs[wi]
        dist = self.dist(dist)

        query = self.query(agts[hi])

        ctx = ctx[wi]
        ctx = torch.cat((dist, query, ctx), 1)
        ctx = self.ctx(ctx)

        agts = self.agt(agts)
        agts.index_add_(0, hi, ctx)
        agts = self.norm(agts)
        agts = self.relu(agts)

        agts = self.linear(agts)
        agts += res
        agts = self.relu(agts)
        return agts


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn_base = nn.Parameter(torch.arange(-3, 3, 6 / 20).unsqueeze(0).unsqueeze(0))
        self.attn = nn.Linear(dim, 1)
        self.attend = nn.Softmax(dim=-1)

        self.to_out = nn.Linear(dim, dim)
        self.norm = nn.GroupNorm(1, dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        attn = self.attn(x).transpose(1, 2)
        attn = attn + self.attn_base
        attn = self.attend(attn)
        out = torch.matmul(attn, x)
        out = self.to_out(out)
        return self.norm(out[:, -1, :])


class AttDest(nn.Module):
    def __init__(self, n_agt):
        super(AttDest, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_agt),
            nn.ReLU(inplace=True),
            Linear(n_agt, n_agt, norm=norm, ng=ng),
        )

        self.agt = Linear(2 * n_agt, n_agt, norm=norm, ng=ng)

    def forward(self, agts, agt_ctrs, dest_ctrs):
        n_agt = agts.size(1)
        num_mods = dest_ctrs.size(1)

        dist = (agt_ctrs.unsqueeze(1) - dest_ctrs).view(-1, 2)
        dist = self.dist(dist)
        agts = agts.unsqueeze(1).repeat(1, num_mods, 1).view(-1, n_agt)

        agts = torch.cat((dist, agts), 1)
        agts = self.agt(agts)
        return agts


class SingleKeyPointDecoder(nn.Module):
    def __init__(self, config):
        super(SingleKeyPointDecoder, self).__init__()
        self.config = config

        ng = 1
        norm = "GN"
        n_agent = config["n_agent"]

        self.pred = nn.Sequential(
            LinearRes(n_agent, n_agent, norm=norm, ng=ng),
            nn.Linear(n_agent, 2)
        )

        self.m2a = M2A(config)

    def forward(self, agents, agent_idcs, agent_ctrs, nodes, node_idcs, node_ctrs):
        key_point = self.pred(agents)
        key_point_ctrs = []
        for k in range(len(agent_idcs)):
            idcs = agent_idcs[k]
            ctrs = agent_ctrs[k]
            key_point[idcs] = key_point[idcs] + ctrs
            key_point_ctrs.append(key_point[idcs])
        agents = self.m2a(agents, agent_idcs, key_point_ctrs, nodes, node_idcs, node_ctrs)
        return agents, key_point


class MultipleKeyPointsDecoder(nn.Module):
    def __init__(self, config):
        super(MultipleKeyPointsDecoder, self).__init__()
        self.config = config
        self.n_key_points = config["n_key_points"]
        ng = 1
        norm = "GN"
        n_agent = config["n_agent"]

        pred = []
        for i in range(self.n_key_points):
            pred.append(
                nn.Sequential(
                    LinearRes(n_agent, n_agent, norm=norm, ng=ng),
                    nn.Linear(n_agent, 2))
            )
        self.pred = nn.ModuleList(pred)

        self.att_dest = AttDest(n_agent)
        self.cls = nn.Sequential(
            LinearRes(n_agent, n_agent, norm=norm, ng=ng), nn.Linear(n_agent, 1)
        )

        self.m2a = M2A(config)
        self.a2a = A2A(config)

    def forward(self, agents, agent_idcs, agent_ctrs, nodes, node_idcs, node_ctrs):
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](agents))
        key_points = torch.cat([x.unsqueeze(1) for x in preds], 1)
        key_points = key_points.view(key_points.size()[0], key_points.size()[1], -1, 2)

        for i in range(len(agent_idcs)):
            idcs = agent_idcs[i]
            ctrs = agent_ctrs[i].view(-1, 1, 1, 2)
            key_points[idcs] = key_points[idcs] + ctrs

        dest_ctrs = key_points[:, :, -1].detach()
        feats = self.att_dest(agents, torch.cat(agent_ctrs, 0), dest_ctrs)
        cls1 = self.cls(feats).view(-1, self.n_key_points)
        cls = torch.softmax(cls1, dim=-1)

        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        key_points = key_points[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)

        key_point_ctrs = []
        for i in range(len(agent_idcs)):
            idcs = agent_idcs[i]
            key_point_ctrs.append(key_points[idcs, 0, 0])

        agents = self.m2a(agents, agent_idcs, key_point_ctrs, nodes, node_idcs, node_ctrs)
        agents = self.a2a(agents, agent_idcs, key_point_ctrs)
        return agents, key_points, cls


class TrajectoryDecoder(nn.Module):
    def __init__(self, config):
        super(TrajectoryDecoder, self).__init__()
        self.config = config

        ng = 1
        norm = "GN"
        n_agent = config["n_agent"]

        pred = []
        for i in range(config["num_mods"]):
            pred.append(nn.Sequential(
                LinearRes(n_agent, n_agent, norm=norm, ng=ng),
                nn.Linear(n_agent, config["num_preds"] * 2)
            ))
        self.pred = nn.ModuleList(pred)

        self.att_dest = AttDest(n_agent)
        self.cls = nn.Sequential(
            LinearRes(n_agent, n_agent, norm=norm, ng=ng), nn.Linear(n_agent, 1)
        )

    def forward(self, agents, agent_idcs, agent_ctrs):
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](agents))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size()[0], reg.size()[1], -1, 2)

        for i in range(len(agent_idcs)):
            idcs = agent_idcs[i]
            ctrs = agent_ctrs[i].view(-1, 1, 1, 2)
            reg[idcs] = reg[idcs] + ctrs

        dest_ctrs = reg[:, :, -1].detach()
        feats = self.att_dest(agents, torch.cat(agent_ctrs, 0), dest_ctrs)
        cls1 = self.cls(feats).view(-1, self.config["num_mods"])
        cls = torch.softmax(cls1, dim=-1)

        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)
        return reg, cls


class PyramidDecoder(nn.Module):
    def __init__(self, config):
        super(PyramidDecoder, self).__init__()
        self.config = config

        self.single_key_point_decoder = SingleKeyPointDecoder(config)
        self.multiple_key_points_decoder = MultipleKeyPointsDecoder(config)
        self.trajectory_decoder = TrajectoryDecoder(config)

    def forward(self, agents, agent_idcs, agent_ctrs, nodes, node_idcs, node_ctrs):
        agents, key_point = self.single_key_point_decoder(agents, agent_idcs, agent_ctrs, nodes, node_idcs, node_ctrs)
        agents, key_points, key_point_cls = self.multiple_key_points_decoder(
            agents, agent_idcs, agent_ctrs, nodes, node_idcs, node_ctrs)
        reg, cls = self.trajectory_decoder(agents, agent_idcs, agent_ctrs)

        out = defaultdict(list)
        for i in range(len(agent_idcs)):
            idcs = agent_idcs[i]
            out['cls'].append(cls[idcs])
            out['reg'].append(reg[idcs])
            out['key_point'].append(key_point[idcs])
            out['key_points'].append(key_points[idcs])
            out['key_point_cls'].append(key_point_cls[idcs])
        return out


class Conv1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Conv1d, self).__init__()
        assert (norm in ['GN', 'BN', 'SyncBN'])

        self.conv = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=(int(kernel_size) - 1) // 2, stride=stride,
                              bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Linear(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32, act=True):
        super(Linear, self).__init__()
        assert (norm in ['GN', 'BN', 'SyncBN'])

        self.linear = nn.Linear(n_in, n_out, bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Res1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Res1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace = True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == 'GN':
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.bn1 = nn.BatchNorm1d(n_out)
            self.bn2 = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        if stride != 1 or n_out != n_in:
            if norm == 'GN':
                self.downsample = nn.Sequential(
                        nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                        nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.downsample = nn.Sequential(
                        nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


class LinearRes(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32):
        super(LinearRes, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.linear1 = nn.Linear(n_in, n_out, bias=False)
        self.linear2 = nn.Linear(n_out, n_out, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if norm == 'GN':
            self.norm1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.norm2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm1 = nn.BatchNorm1d(n_out)
            self.norm2 = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        if n_in != n_out:
            if norm == 'GN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
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

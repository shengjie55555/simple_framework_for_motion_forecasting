import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import Conv1d, Conv1dRes
from model.decoder import SimpleDecoder as Decoder
from utils.data_utils import gpu


class ATDS(nn.Module):
    def __init__(self, cfg, device):
        super(ATDS, self).__init__()
        a_dim, out_dim, num_mode, pred_len = cfg["n_agent"], cfg["n_feature"], cfg["num_mode"], cfg["pred_len"]
        self.device = device
        self.agent_encoder = AgentEncoder(a_dim, out_dim)
        self.decoder = Decoder(out_dim, num_mode, pred_len)

    def forward(self, data):
        agents, agent_ids, agent_ctrs = agent_gather(gpu(data["trajs_obs"], self.device),
                                                     gpu(data["pad_obs"], self.device))
        agents = self.agent_encoder(agents)
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

    agents = [x.transpose(1, 2) for x in agents]
    agents = torch.cat(agents, 0)  # n  * 5 (x, y, dx, dy, mask) * obs_len
    agents[:, :-1] *= agents[:, -1:]
    agents[:, :-1, 1:] *= agents[:, -1:, :-1]

    agent_ids = []
    count = 0
    for i in range(batch_size):
        ids = torch.arange(count, count + num_agents[i]).to(agents.device)
        agent_ids.append(ids)
        count += num_agents[i]

    agent_ctrs = [agents[ids, :2, -1] for ids in agent_ids]

    return agents, agent_ids, agent_ctrs


class AgentEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AgentEncoder, self).__init__()
        out_dims = [out_dim // 2 ** 2, out_dim // 2, out_dim]
        blocks = [Conv1dRes, Conv1dRes, Conv1dRes]
        num_blocks = [2, 2, 2]

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](in_dim, out_dims[i]))
            else:
                group.append(blocks[i](in_dim, out_dims[i], stride=2))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](out_dims[i], out_dims[i]))
            groups.append(nn.Sequential(*group))
            in_dim = out_dims[i]
        self.groups = nn.ModuleList(groups)

        lateral = []
        for i in range(len(out_dims)):
            lateral.append(Conv1d(out_dims[i], out_dim, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Conv1dRes(out_dim, out_dim)

    def forward(self, x):
        out = x

        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        out = self.output(out)[:, :, -1]
        return out

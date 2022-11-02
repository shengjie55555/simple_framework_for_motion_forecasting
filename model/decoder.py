import torch
import torch.nn as nn
from model.layers import Linear, LinearRes
from einops import repeat


class SimpleDecoder(nn.Module):
    def __init__(self, out_dim, num_mode, pred_len, is_cat=False):
        super(SimpleDecoder, self).__init__()
        self.out_dim = out_dim
        self.num_mode = num_mode
        self.pred_len = pred_len

        if is_cat:
            self.in_dim = 2 * self.out_dim
        else:
            self.in_dim = self.out_dim

        self.traj_decoder = nn.ModuleList(nn.Sequential(
            LinearRes(self.in_dim, out_dim, out_dim),
            nn.Linear(out_dim, self.pred_len * 2)
        ) for _ in range(num_mode))

        self.score_decoder = nn.ModuleDict({
            "goal": nn.Sequential(
                nn.Linear(2, out_dim),
                nn.ReLU(inplace=True),
                Linear(out_dim, out_dim)
            ),
            "score": nn.Sequential(
                Linear(self.in_dim + self.out_dim, out_dim),
                nn.Linear(out_dim, 1),
                nn.Softmax(dim=-2)
            ),
        })

    def forward(self, agents):
        preds = []
        for i in range(self.num_mode):
            preds.append(self.traj_decoder[i](agents))
        reg = torch.cat([pred.unsqueeze(-2) for pred in preds]).view(agents.shape[0], self.num_mode, self.pred_len, -1)

        goals = reg[:, :, -1].detach()
        goals = self.score_decoder["goal"](goals)
        agents = torch.cat([repeat(agents, "n d -> n h d", h=self.num_mode), goals], dim=-1)
        cls = self.score_decoder["score"](agents).squeeze(-1)

        return reg, cls


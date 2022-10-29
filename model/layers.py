import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, act=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = act
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.act:
            x = self.relu(self.norm(self.linear(x)))
        else:
            x = self.norm(self.linear(x))
        return x


class LinearRes(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(LinearRes, self).__init__()
        self.linear_1 = Linear(in_dim, hidden_dim)
        self.linear_2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

        self.shortcut = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        ) if in_dim != out_dim else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.linear_2(self.linear_1(x)) + self.shortcut(x))

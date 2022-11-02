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


class Conv1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, act=True):
        super(Conv1d, self).__init__()
        padding = (int(kernel_size) - 1) // 2
        self.conv = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=padding, stride=stride,
                              bias=False)
        self.norm = nn.GroupNorm(1, n_out)  # nn.LayerNorm([n_out, seq_len])
        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        if self.act:
            return self.relu(self.norm(self.conv(x)))
        else:
            return self.norm(self.conv(x))


class Conv1dRes(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, act=True):
        super(Conv1dRes, self).__init__()
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = Conv1d(n_in, n_out, kernel_size, stride, act=True)
        self.conv2 = nn.Sequential(
            nn.Conv1d(n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(1, n_out))

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(1, n_out))
        else:
            self.shortcut = nn.Identity()

        self.act = act
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv2(self.conv1(x)) + self.shortcut(x)
        if self.act:
            return self.relu(out)
        else:
            return out


import torch
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, enable=True, log_dir=""):
        self.enable = enable
        self.log_dir = log_dir

        if self.enable:
            self.writer = SummaryWriter(self.log_dir)
        print("logger enable_flags: {}, log_dir: {}".format(self.enable, self.log_dir))

    def add_scalar(self, title, value, it):
        if self.enable:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            self.writer.add_scalar(title, value, it)

    def add_dict(self, data, it, prefix=""):
        if self.enable:
            for key, val in data.items():
                title = prefix + key
                self.add_scalar(title, val, it)

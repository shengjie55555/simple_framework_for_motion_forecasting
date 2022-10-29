import os.path

import torch
import random
import numpy as np
from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate


def worker_init_fn(pid):
    np_seed = int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


def save_ckpt(net, opt, epoch, save_dir):
    state_dict = net.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "state_dict": state_dict,
            "opt_state": opt.state_dict()
        },
        os.path.join(save_dir, str(epoch) + ".pth")
    )


def load_prev_weights(net, path, cfg, opt=None, rank=0, is_ddp=False):
    ckpt = torch.load(path, map_location="cuda:0")
    if opt is not None:
        opt.load_state_dict(ckpt["opt_state"])
    pre_state = ckpt["state_dict"]
    state_dict = net.state_dict()
    loaded_modules = []
    for k, v in pre_state.items():
        if not is_ddp:
            k = k.replace('module.', '', 1)
            module = k.split('.')[0]
        else:
            module = k.split('.')[1]
        if module in cfg["ignored_modules"]:
            continue
        elif k in state_dict.keys() and state_dict[k].shape == v.shape:
            state_dict[k] = v
            loaded_modules.append(k)
    net.load_state_dict(state_dict)
    if rank == 0:
        print(f'loaded parameters {len(loaded_modules)}/{len(state_dict)}')


class AverageLoss(object):
    def __init__(self):
        self.loss_out = {}

    def reset(self):
        self.loss_out = {}

    def update(self, loss_out):
        # initialization
        if len(self.loss_out.keys()) == 0:
            for key in loss_out.keys():
                if key != "loss":
                    self.loss_out[key] = 0

        for key in loss_out.keys():
            if key == "loss":
                continue
            if isinstance(loss_out[key], torch.Tensor):
                self.loss_out[key] += loss_out[key].detach().cpu().item()
            else:
                self.loss_out[key] += loss_out[key]

    def get(self):
        cls = self.loss_out["cls_loss"] / (self.loss_out["num_cls"] + 1e-10)
        reg = self.loss_out["reg_loss"] / (self.loss_out["num_reg"] + 1e-10)
        loss = cls + reg
        loss_out = {
            "cls_loss": cls,
            "reg_loss": reg,
            "loss": loss
        }
        return loss_out


class AverageMetrics(object):
    def __init__(self, cfg):
        self.reg = {}
        self.cls = {}
        self.gts = {}
        self.city = {}
        self.cfg = cfg

    def reset(self):
        self.reg, self.cls, self.city, self.gts = {}, {}, {}, {}

    def update(self, post_out, data):
        reg = [x[0:1].detach().cpu().numpy() for x in post_out["reg"]]
        cls = [x[0:1].detach().cpu().numpy() for x in post_out["cls"]]
        for j, seq_id in enumerate(data["seq_id"]):
            self.reg[seq_id] = reg[j].squeeze()
            self.cls[seq_id] = [x for x in cls[j].squeeze()]
            self.gts[seq_id] = data["trajs_fut"][j][0].detach().cpu().numpy()
            self.city[seq_id] = data["city_name"][j]

    def get(self):
        res_1 = get_displacement_errors_and_miss_rate(
            self.reg, self.gts, 1, self.cfg["pred_len"], 2, self.cls)
        res_k = get_displacement_errors_and_miss_rate(
            self.reg, self.gts, self.cfg["num_mode"], self.cfg["pred_len"], 2, self.cls)

        metrics_out = {
            'minade_1': res_1['minADE'],
            'minfde_1': res_1['minFDE'],
            'mr_1': res_1['MR'],
            'brier_fde_1': res_1['brier-minFDE'],
            'minade_k': res_k['minADE'],
            'minfde_k': res_k['minFDE'],
            'mr_k': res_k['MR'],
            'brier_fde_k': res_k['brier-minFDE']
        }

        return metrics_out

import os.path
import importlib

import torch
import random
import numpy as np
from datetime import datetime


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


def update_cfg(args, cfg, is_eval=False):
    if is_eval:
        model_name = args.model_name
        cfg["val_batch_size"] = args.val_batch_size
        cfg["save_dir"] = os.path.join("results", model_name, "weights/")
        cfg["cfg"] = os.path.join("results", model_name, "cfg.txt")
        cfg["images"] = os.path.join("results", model_name, "images/")
        cfg["competition_files"] = os.path.join("results", model_name, "competition/")
        cfg["processed_val"] = args.val_dir
        return cfg
    model_name = args.model.lower() + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["epoch"] = args.train_epochs
    cfg["train_batch_size"] = args.train_batch_size
    cfg["val_batch_size"] = args.val_batch_size
    cfg["save_dir"] = os.path.join("results", model_name, "weights/")
    cfg["cfg"] = os.path.join("results", model_name, "cfg.txt")
    cfg["images"] = os.path.join("results", model_name, "images/")
    cfg["competition_files"] = os.path.join("results", model_name, "competition/")
    cfg["train_log"] = os.path.join("results", model_name, "train_log.csv")
    cfg["val_log"] = os.path.join("results", model_name, "val_log.csv")
    cfg["log_dir"] = "logs/" + model_name
    cfg["processed_train"] = args.train_dir
    cfg["processed_val"] = args.val_dir
    cfg["num_val"] = args.num_val
    cfg["num_display"] = args.num_display
    cfg["train_workers"] = args.workers
    return cfg


class Loader(object):
    def __init__(self, model_name):
        self.module_dict = {
            "mhl": "mhl",
            "mhlv": "mhlv",
            "mhll": "mhll",
            "mhld": "mhld",
            "atds": "atds",
        }
        self.class_dict = {
            "mhl": "MHL",
            "mhlv": "MHLV",
            "mhll": "MHLL",
            "mhld": "MHLD",
            "atds": "ATDS",
        }
        assert model_name in ["mhl", "MHL", "mhlv", "MHLV", "mhll", "MHLL", "mhld", "MHLD", "atds", "ATDS"], \
            '{} is not in ["mhl", "MHL", "mhlv", "MHLV", "mhll", "MHLL", "mhld", "MHLD", "atds", "ATDS"]'.format(model_name)
        self.model_name = model_name.lower()

    def load(self):
        config = self.load_config()
        dataset_cls = self.load_dataset()
        model_cls = self.load_model()
        loss_cls = self.load_loss()
        al_cls = self.load_average_loss_logger()
        am_cls = self.load_average_metrics_logger()
        vis_cls = self.load_vis()
        return config, dataset_cls, model_cls, loss_cls, al_cls, am_cls, vis_cls

    @staticmethod
    def load_attr(package_name, module_name, attr_name):
        module = importlib.import_module(".{}".format(module_name), package=package_name)
        assert hasattr(module, attr_name), "attribution {} is not in {}".format(attr_name, module_name)
        attr = getattr(module, attr_name)
        return attr

    def load_model(self):
        package_name = "model"
        module_name = self.model_name
        attr_name = self.class_dict[self.model_name]
        return self.load_attr(package_name, module_name, attr_name)

    def load_config(self):
        package_name = "config"
        module_name = "cfg_{}".format(self.module_dict[self.model_name])
        attr_name = "config"
        return self.load_attr(package_name, module_name, attr_name)

    def load_dataset(self):
        package_name = "utils"
        module_name = "dataset"
        attr_name = "{}Dataset".format(self.class_dict[self.model_name])
        return self.load_attr(package_name, module_name, attr_name)

    def load_loss(self):
        package_name = "model"
        module_name = "loss"
        attr_name = "{}Loss".format(self.class_dict[self.model_name])
        return self.load_attr(package_name, module_name, attr_name)

    def load_average_loss_logger(self):
        package_name = "utils"
        module_name = "log_utils"
        attr_name = "{}AverageLoss".format(self.class_dict[self.model_name])
        return self.load_attr(package_name, module_name, attr_name)

    def load_average_metrics_logger(self):
        package_name = "utils"
        module_name = "log_utils"
        attr_name = "{}AverageMetrics".format(self.class_dict[self.model_name])
        return self.load_attr(package_name, module_name, attr_name)

    def load_vis(self):
        package_name = "visualize"
        module_name = "vis_{}".format(self.module_dict[self.model_name])
        attr_name = "Vis"
        return self.load_attr(package_name, module_name, attr_name)

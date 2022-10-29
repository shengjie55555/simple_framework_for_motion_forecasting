import os
import torch
import pandas as pd
import numpy as np


def gpu(data, device):
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x, device=device) for x in data]
    elif isinstance(data, dict):
        data = {key: gpu(_data, device=device) for key, _data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().to(device, non_blocking=True)
    return data


def to_long(data):
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    elif isinstance(data, dict):
        data = {k: to_long(v) for k, v in data.items()}
    elif torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data


def from_numpy(data):
    if isinstance(data, dict):
        data = {k: from_numpy(v) for k, v in data.items()}
    elif isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    elif isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    return data


def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch


def create_dirs(cfg):
    # create dirs
    for x in [cfg["save_dir"], cfg["images"], cfg["competition_files"]]:
        if not os.path.exists(x):
            os.makedirs(x)
            print("mkdir " + x)
    # cfg
    f = open(cfg["cfg"], 'a')
    f.write("hyper parameters".center(50, "-") + "\n")
    for k, v in cfg.items():
        f.write(k + str(":") + str(v) + '\n')
    f.write("".center(50, "-") + "\n")
    f.close()
    # train & val log
    df = pd.DataFrame(columns=cfg["log_columns"])
    if not os.path.exists(cfg["train_log"]):
        df.to_csv(cfg["train_log"], index=False)
    if not os.path.exists(cfg["val_log"]):
        df.to_csv(cfg["val_log"], index=False)


def save_log(epoch, post_loss, post_metrics, cfg, mode="train"):
    value = [epoch]
    for k, v in post_loss.items():
        value.append(v)
    for k, v in post_metrics.items():
        value.append(v)
    df = pd.DataFrame([value])
    if "{}_log".format(mode) in cfg.keys():
        df.to_csv(cfg["{}_log".format(mode)], mode="a", header=False, index=False)


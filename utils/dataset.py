import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ProcessedDataset(Dataset):
    def __init__(self, path, mode):
        super(ProcessedDataset, self).__init__()
        self.path = path
        self.mode = mode
        self.obs_len = 20
        self.train = False

        if self.mode == "train":
            self.train = True

        file_list = os.listdir(path)
        file_list.sort(key=lambda x: int(x.split(".")[0]))
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        file_name = self.file_list[item]
        df = pd.read_pickle(os.path.join(self.path, file_name))

        df_dict = {}
        for key in list(df.keys()):
            df_dict[key] = df[key].values[0]

        seq_id = df_dict["SEQ_ID"]
        city_name = df_dict["CITY_NAME"]
        orig = df_dict["ORIG"]
        rot = df_dict["ROT"]
        ts = df_dict["TIMESTAMP"]

        trajs = df_dict["TRAJS"]
        pad_flags = 1 - df_dict["PAD_FLAGS"]  # 0 for missing
        graph = df_dict["GRAPH"]

        data = {
            "seq_id": seq_id,
            "city_name": city_name,
            "orig": orig,
            "rot": rot,
            "ts": np.diff(ts, prepend=ts[0])[:self.obs_len],
            "trajs_obs": trajs[:, :self.obs_len],
            "pad_obs": pad_flags[:, :self.obs_len],
            "trajs_fut": trajs[:, self.obs_len:, :2],
            "pad_fut": pad_flags[:, self.obs_len:],
            "graph": graph
        }

        if self.train is True:
            data = DataAug.simple_aug(data)

        return data


class DataAug(object):
    @classmethod
    def simple_aug(cls, data):
        return data

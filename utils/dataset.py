import os
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ATDS2Dataset(Dataset):
    def __init__(self, path, mode):
        super(ATDS2Dataset, self).__init__()
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
        pad_flags = df_dict["PAD_FLAGS"]  # 0 for missing
        graph = df_dict["GRAPH"]

        # feats, locs, ctrs
        locs = np.concatenate([trajs[:, :self.obs_len],
                               np.expand_dims(pad_flags[:, :self.obs_len], axis=-1)], axis=-1)
        feats = np.zeros_like(trajs[:, :self.obs_len, :2])
        feats[:, 1:, :] = locs[:, 1:, :2] - locs[:, :-1, :2]
        feats = np.concatenate([feats, np.expand_dims(pad_flags[:, :self.obs_len], axis=-1)], axis=-1)
        ctrs = locs[:, -1, :2]

        data = {
            "argo_id": seq_id,
            "city": city_name,
            "orig": orig,
            "gt_preds": trajs[:, self.obs_len:],
            "has_preds": pad_flags[:, self.obs_len:].astype(np.bool_),
            "rot": rot,
            "feats": feats,
            "ctrs": ctrs,
            "graph": graph,
            "locs": locs
        }
        return data


class MHLVDataset(Dataset):
    def __init__(self, path, mode):
        super(MHLVDataset, self).__init__()
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
        pad_flags = df_dict["PAD_FLAGS"]  # 0 for missing
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


class ATDSDataset(Dataset):
    def __init__(self, path, mode):
        super(ATDSDataset, self).__init__()
        self.path = path
        self.mode = mode
        self.obs_len = 20
        self.train = False

        if self.mode == "train":
            self.train = True

        file_list = os.listdir(path)
        file_list.sort(key=lambda x: int(x[:-7]))
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        file = self.file_list[item]
        with open(self.path + file, "rb") as f:
            data = pickle.load(f)
            locs = data["feats"].copy()
            loc = data["ctrs"].copy()
            feat = data["feats"][..., :2].copy()
            for i in reversed(range(feat.shape[1])):
                locs[:, i, :2] = loc
                pre = feat[:, i]
                loc -= pre
            locs[..., 0] *= locs[..., -1]
            locs[..., 1] *= locs[..., -1]
            data["locs"] = locs.copy()
            if self.train is True:
                data = DataAug.simple_aug(data)

            return data


class MHLDataset(ATDSDataset):
    def __init__(self, path, mode):
        super(MHLDataset, self).__init__(path, mode)


class MHLLDataset(ATDSDataset):
    def __init__(self, path, mode):
        super(MHLLDataset, self).__init__(path, mode)


class MHLDDataset(ATDSDataset):
    def __init__(self, path, mode):
        super(MHLDDataset, self).__init__(path, mode)


class DataAug(object):
    @classmethod
    def simple_aug(cls, data):
        return data

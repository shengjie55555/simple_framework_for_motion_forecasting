import torch
from torch.utils.tensorboard import SummaryWriter
from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate


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


class MHLVAverageLoss(object):
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


class MHLVAverageMetrics(object):
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


class ATDSAverageLoss(object):
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
        key_points = self.loss_out["key_points_loss"] / (self.loss_out["num_key_points"] + 1e-10)
        loss = cls + reg + key_points
        loss_out = {
            "cls_loss": cls,
            "reg_loss": reg,
            "key_points": key_points,
            "loss": loss
        }
        return loss_out


class ATDSAverageMetrics(object):
    def __init__(self, cfg):
        self.reg = {}
        self.cls = {}
        self.gts = {}
        self.city = {}
        self.cfg = cfg
        self.seq_id = 0

    def reset(self):
        self.reg, self.cls, self.city, self.gts = {}, {}, {}, {}
        self.seq_id = 0

    def update(self, post_out, data):
        reg = [x[0:1].detach().cpu().numpy() for x in post_out["reg"]]
        cls = [x[0:1].detach().cpu().numpy() for x in post_out["cls"]]
        for j in range(len(reg)):
            self.reg[self.seq_id + j] = reg[j].squeeze()
            self.cls[self.seq_id + j] = [x for x in cls[j].squeeze()]
            self.gts[self.seq_id + j] = data["gt_preds"][j][0].detach().cpu().numpy()
            self.city[self.seq_id + j] = data["city"][j]
        self.seq_id += len(reg)

    def get(self):
        res_1 = get_displacement_errors_and_miss_rate(
            self.reg, self.gts, 1, self.cfg["num_preds"], 2, self.cls)
        res_k = get_displacement_errors_and_miss_rate(
            self.reg, self.gts, self.cfg["num_mods"], self.cfg["num_preds"], 2, self.cls)

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


class KANAverageLoss(ATDSAverageLoss):
    def __init__(self):
        super(KANAverageLoss, self).__init__()

    def get(self):
        cls = self.loss_out["cls_loss"] / (self.loss_out["num_cls"] + 1e-10)
        reg = self.loss_out["reg_loss"] / (self.loss_out["num_reg"] + 1e-10)
        key_point = self.loss_out["key_point_loss"] / (self.loss_out["num_key_point"] + 1e-10)
        key_points = self.loss_out["key_points_loss"] / (self.loss_out["num_key_points"] + 1e-10)
        key_point_cls = self.loss_out["key_point_cls_loss"] / (self.loss_out["num_key_point_cls"] + 1e-10)
        reg_end = self.loss_out["reg_end_loss"] / (self.loss_out["num_reg_end"] + 1e-10)
        loss = cls + reg + key_point + key_points + key_point_cls + reg_end
        loss_out = {
            "cls_loss": cls,
            "reg_loss": reg,
            "key_point": key_point,
            "key_points": key_points,
            "key_point_cls": key_point_cls,
            "reg_end": reg_end,
            "loss": loss
        }
        return loss_out


class KANAverageMetrics(ATDSAverageMetrics):
    def __init__(self, cfg):
        super(KANAverageMetrics, self).__init__(cfg)


class MHLAverageLoss(MHLVAverageLoss):
    def __init__(self):
        super(MHLAverageLoss, self).__init__()


class MHLAverageMetrics(ATDSAverageMetrics):
    def __init__(self, cfg):
        super(MHLAverageMetrics, self).__init__(cfg)

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


class MHLLAverageLoss(MHLVAverageLoss):
    def __init__(self):
        super(MHLLAverageLoss, self).__init__()


class MHLLAverageMetrics(MHLAverageMetrics):
    def __init__(self, cfg):
        super(MHLLAverageMetrics, self).__init__(cfg)


class MHLDAverageLoss(MHLVAverageLoss):
    def __init__(self):
        super(MHLDAverageLoss, self).__init__()


class MHLDAverageMetrics(MHLAverageMetrics):
    def __init__(self, cfg):
        super(MHLDAverageMetrics, self).__init__(cfg)

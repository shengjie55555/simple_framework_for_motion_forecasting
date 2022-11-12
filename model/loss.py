import torch
import torch.nn as nn
from utils.data_utils import gpu


class VectorNetLoss(nn.Module):
    def __init__(self, cfg, device):
        super(VectorNetLoss, self).__init__()
        self.cfg = cfg
        self.device = device
        self.pred_loss = VectorNetPredLoss(cfg)

    def forward(self, out, data):
        loss_out = self.pred_loss(out, gpu(data["trajs_fut"], self.device), gpu(data["pad_fut"], self.device))
        loss_out["loss"] = loss_out["cls_loss"] / (loss_out["num_cls"] + 1e-10) + \
                           loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        return loss_out


class MHLLoss(nn.Module):
    def __init__(self, cfg, device):
        super(MHLLoss, self).__init__()
        self.cfg = cfg
        self.device = device
        self.pred_loss = VectorNetPredLoss(cfg)

    def forward(self, out, data):
        loss_out = self.pred_loss(out, gpu(data["gt_preds"], self.device), gpu(data["has_preds"], self.device))
        loss_out["loss"] = loss_out["cls_loss"] / (loss_out["num_cls"] + 1e-10) + \
                           loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        return loss_out


class LaneGCNLoss(nn.Module):
    def __init__(self, cfg, device):
        super(LaneGCNLoss, self).__init__()
        self.cfg = cfg
        self.device = device
        self.pred_loss = VectorNetPredLoss(cfg)

    def forward(self, out, data):
        loss_out = self.pred_loss(out, gpu(data["gt_preds"], self.device), gpu(data["has_preds"], self.device))
        loss_out["loss"] = loss_out["cls_loss"] / (loss_out["num_cls"] + 1e-10) + \
                           loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        return loss_out


class DSLoss(nn.Module):
    def __init__(self, cfg, device):
        super(DSLoss, self).__init__()
        self.cfg = cfg
        self.device = device
        self.pred_loss = VectorNetPredLoss(cfg)

    def forward(self, out, data):
        loss_out = self.pred_loss(out, gpu(data["gt_preds"], self.device), gpu(data["has_preds"], self.device))
        loss_out["loss"] = loss_out["cls_loss"] / (loss_out["num_cls"] + 1e-10) + \
                           loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        return loss_out


class ATDSLoss(nn.Module):
    def __init__(self, config, device):
        super(ATDSLoss, self).__init__()
        self.config = config
        self.device = device
        self.pred_loss = ATDSPredLoss(config)

    def forward(self, out, data):
        loss_out = self.pred_loss(out, gpu(data["gt_preds"], self.device), gpu(data["has_preds"], self.device))
        loss_out["loss"] = loss_out["cls_loss"] / (loss_out["num_cls"] + 1e-10) + \
                           loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10) + \
                           loss_out["key_points_loss"] / (loss_out["num_key_points"] + 1e-10)
        return loss_out


class VectorNetPredLoss(nn.Module):
    def __init__(self, cfg):
        super(VectorNetPredLoss, self).__init__()
        self.cfg = cfg
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out, gt_preds, has_preds):
        cls, reg = out["cls"], out["reg"]
        cls = torch.cat([x for x in cls], 0)
        reg = torch.cat([x for x in reg], 0)
        gt_preds = torch.cat([x for x in gt_preds], 0)
        has_preds = torch.cat([x for x in has_preds], 0) == 1

        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0
        loss_out["reg_loss"] = zero.clone()
        loss_out["num_reg"] = 0

        num_mode, pred_len = self.cfg["num_mode"], self.cfg["pred_len"]

        last = has_preds.float() + 0.1 * torch.arange(pred_len).float().to(has_preds.device) / float(pred_len)
        max_last, last_ids = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_ids = last_ids[mask]

        row_ids = torch.arange(len(last_ids)).long().to(last_ids.device)
        dist = []
        for j in range(num_mode):
            dist.append(
                torch.sqrt(
                    (
                            (reg[row_ids, j, last_ids] - gt_preds[row_ids, last_ids])
                            ** 2
                    ).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_ids = dist.min(1)
        row_ids = torch.arange(len(min_ids)).long().to(min_ids.device)

        mgn = cls[row_ids, min_ids].unsqueeze(1) - cls
        mask0 = (min_dist < self.cfg["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.cfg["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.cfg["mgn"]
        coef = self.cfg["cls_coef"]
        loss_out["cls_loss"] += coef * (
                self.cfg["mgn"] * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()

        reg = reg[row_ids, min_ids]
        coef = self.cfg["reg_coef"]
        loss_out["reg_loss"] += coef * self.reg_loss(
            reg[has_preds], gt_preds[has_preds]
        )
        loss_out["num_reg"] += has_preds.sum().item()

        return loss_out


class ATDSPredLoss(nn.Module):
    def __init__(self, config):
        super(ATDSPredLoss, self).__init__()
        self.config = config
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out, gt_preds, has_preds):
        cls, reg, key_points = out["cls"], out["reg"], out["key_points"]
        cls = torch.cat([x for x in cls], 0)
        reg = torch.cat([x for x in reg], 0)
        key_points = torch.cat([x for x in key_points], 0)
        gt_preds = torch.cat([x for x in gt_preds], 0)
        has_preds = torch.cat([x for x in has_preds], 0)

        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0
        loss_out["reg_loss"] = zero.clone()
        loss_out["num_reg"] = 0
        loss_out["key_points_loss"] = zero.clone()
        loss_out["num_key_points"] = 0

        num_mods, num_preds = self.config["num_mods"], self.config["num_preds"]

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(has_preds.device) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        key_points = key_points[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        dist = []
        for j in range(num_mods):
            dist.append(
                torch.sqrt(
                    (
                            (reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                            ** 2
                    ).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        coef = self.config["cls_coef"]
        loss_out["cls_loss"] += coef * (
                self.config["mgn"] * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()

        reg = reg[row_idcs, min_idcs]
        coef = self.config["reg_coef"]
        loss_out["reg_loss"] += coef * self.reg_loss(
            reg[has_preds], gt_preds[has_preds]
        )
        loss_out["num_reg"] += has_preds.sum().item()

        key_points_idx = self.config["key_points"]
        key_points = key_points[row_idcs, min_idcs]
        has_preds = has_preds[:, [-1, key_points_idx[0], key_points_idx[1], -1]]
        gt_preds = gt_preds[:, [-1, key_points_idx[0], key_points_idx[1], -1]]
        coef = self.config["key_points_coef"]
        loss_out["key_points_loss"] += coef * self.reg_loss(
            key_points[has_preds], gt_preds[has_preds]
        )
        loss_out["num_key_points"] = has_preds.sum().item()

        return loss_out

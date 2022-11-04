import torch
import torch.nn as nn
from model.atds import Linear, LinearRes, AttDest


class SimpleDecoder(nn.Module):
    def __init__(self, n_agt, num_mode, pred_len):
        super(SimpleDecoder, self).__init__()
        ng = 1
        self.n_agt = n_agt
        self.num_mode = num_mode
        self.pred_len = pred_len

        pred = []
        for i in range(num_mode):
            pred.append(
                nn.Sequential(
                    LinearRes(n_agt, n_agt, ng=ng),
                    nn.Linear(n_agt, 2 * pred_len),
                )
            )
        self.pred = nn.ModuleList(pred)

        self.att_dest = AttDest(n_agt)
        self.cls = nn.Sequential(
            LinearRes(n_agt, n_agt, ng=ng), nn.Linear(n_agt, 1)
        )

    def forward(self, agents, agent_ids, agent_ctrs):
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](agents))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size()[0], reg.size()[1], -1, 2)

        for i in range(len(agent_ids)):
            ids = agent_ids[i]
            ctrs = agent_ctrs[i].view(-1, 1, 1, 2)
            reg[ids] = reg[ids] + ctrs

        dest_ctrs = reg[:, :, -1].detach()
        feats = self.att_dest(agents, torch.cat(agent_ctrs, 0), dest_ctrs)
        cls1 = self.cls(feats).view(-1, self.num_mode)
        cls = torch.softmax(cls1, dim=-1)

        cls, sort_ids = cls.sort(1, descending=True)
        row_ids = torch.arange(len(sort_ids)).long().to(sort_ids.device)
        row_ids = row_ids.view(-1, 1).repeat(1, sort_ids.size(1)).view(-1)
        sort_ids = sort_ids.view(-1)
        reg = reg[row_ids, sort_ids].view(cls.size(0), cls.size(1), -1, 2)

        out = dict()
        out["cls"], out["reg"] = [], []
        for i in range(len(agent_ids)):
            ids = agent_ids[i]
            out["cls"].append(cls[ids])
            out["reg"].append(reg[ids])
        return out

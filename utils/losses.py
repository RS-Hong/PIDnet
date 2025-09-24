import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255, loss_weight=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    def forward(self, logit, label):
        loss = F.cross_entropy(logit, label.long(), ignore_index=self.ignore_index, reduction='mean')
        return self.loss_weight * loss


class OhemCrossEntropy(nn.Module):
    def __init__(self, thres=0.9, min_kept=131072, ignore_index=255, loss_weight=1.0):
        super().__init__()
        self.thres = thres
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.ce = CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logit, label):
        n, c, h, w = logit.size()
        label = label.long()
        valid_mask = label != self.ignore_index
        num_valid = valid_mask.sum()
        if num_valid == 0:
            return logit.sum() * 0.0

        prob = F.softmax(logit, dim=1)
        prob = torch.gather(prob, 1, label.unsqueeze(1)).squeeze(1)
        mask = (prob <= self.thres) & valid_mask

        if mask.sum() < self.min_kept:
            prob_sorted, idx = torch.sort(prob[valid_mask], descending=False)
            thres_prob = prob_sorted[min(self.min_kept - 1, len(prob_sorted) - 1)] if len(
                prob_sorted) > 0 else self.thres
            mask = (prob <= thres_prob) & valid_mask

        loss = self.ce(logit, label)
        loss = loss * mask.float()
        return self.loss_weight * loss.sum() / max(mask.sum(), 1.0)


class BoundaryLoss(nn.Module):
    def __init__(self, loss_weight=20.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, logit, gtmaps):
        logit = torch.sigmoid(logit)
        pos_pixels = (gtmaps == 1).float()
        neg_pixels = (gtmaps == 0).float()
        pos_loss = -pos_pixels * torch.log(logit + 1e-6)
        neg_loss = -neg_pixels * torch.log(1.0 - logit + 1e-6) * 0.1  # downweight neg
        return self.loss_weight * (pos_loss + neg_loss).mean()


class PIDNetLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(PIDNetLoss, self).__init__()
        self.ce = CrossEntropyLoss(ignore_index=ignore_index, loss_weight=0.4)
        self.ohem_ce = OhemCrossEntropy(thres=0.9, min_kept=131072, ignore_index=ignore_index, loss_weight=1.0)
        self.boundary = BoundaryLoss(loss_weight=20.0)
        self.ohem_edge = OhemCrossEntropy(thres=0.9, min_kept=131072, ignore_index=ignore_index, loss_weight=1.0)

    def forward(self, outputs, targets, edge_targets):
        main_out, aux_out, edge_out = outputs
        loss = self.ce(aux_out, targets)  # aux 0.4
        loss += self.ohem_ce(main_out, targets)  # main 1.0
        loss += self.boundary(edge_out, edge_targets)  # bound 20.0
        loss += self.ohem_edge(edge_out, targets)  # edge as seg 1.0
        return loss
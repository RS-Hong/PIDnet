import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255, loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    def forward(self, logit, label):
        loss = F.cross_entropy(logit, label.long(), ignore_index=self.ignore_index, reduction='mean')
        return self.loss_weight * loss

class OhemCrossEntropy(nn.Module):
    def __init__(self, thres=0.9, min_kept=131072, ignore_index=255, loss_weight=1.0):
        super(OhemCrossEntropy, self).__init__()
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
            thres_prob = prob_sorted[min(self.min_kept - 1, len(prob_sorted) - 1)] if len(prob_sorted) > 0 else self.thres
            mask = (prob <= thres_prob) & valid_mask

        loss = self.ce(logit, label)
        loss = loss * mask.float()
        return self.loss_weight * loss.sum() / max(mask.sum(), 1.0)

class BoundaryLoss(nn.Module):
    def __init__(self, loss_weight=2.0):  # 进一步降低权重以提高稳定性
        super(BoundaryLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, logit, gtmaps):
        logit = torch.sigmoid(logit)
        gtmaps = gtmaps.unsqueeze(1)
        pos_pixels = (gtmaps == 1).float()
        neg_pixels = (gtmaps == 0).float()
        pos_loss = -pos_pixels * torch.log(logit + 1e-6)
        neg_loss = -neg_pixels * torch.log(1.0 - logit + 1e-6) * 0.1
        return self.loss_weight * (pos_loss + neg_loss).mean()

class PIDNetLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(PIDNetLoss, self).__init__()
        self.ce = CrossEntropyLoss(ignore_index=ignore_index, loss_weight=0.4)
        self.ohem_ce = OhemCrossEntropy(thres=0.9, min_kept=131072, ignore_index=ignore_index, loss_weight=1.0)
        self.boundary = BoundaryLoss(loss_weight=2.0)

    def forward(self, outputs, targets, edge_targets):
        main_out, aux_out, edge_out = outputs
        loss = self.ce(aux_out, targets)
        loss += self.ohem_ce(main_out, targets)
        loss += self.boundary(edge_out, edge_targets)
        return loss

def weights_init(net, init_type='kaiming_normal'):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight.data, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1, no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            return lr
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    def poly_lr(current_iter, total_iters):
        return lr * (1 - current_iter / total_iters) ** 0.9

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    elif lr_decay_type == "step":
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)
    elif lr_decay_type == "poly":
        func = poly_lr
    else:
        raise ValueError(f"Unsupported lr_decay_type: {lr_decay_type}")

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, iter, total_iters):
    lr = lr_scheduler_func(iter, total_iters) if lr_scheduler_func.__name__ == 'poly_lr' else lr_scheduler_func(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
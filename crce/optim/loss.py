#!/usr/bin/env python3

from crce import const
from torch import nn
import torch


class ContextRegularizedCrossEntropy(nn.Module):
    """Context-Regularized Cross-Entropy (https://arxiv.org/abs/2611.08157/)."""

    def __init__(self, get_contrastive_cams_fn, binary_cls=False, device=const.DEVICE):
        super().__init__()

        if not binary_cls: self.ce = nn.CrossEntropyLoss()
        self.get_contrastive_cams = get_contrastive_cams_fn
        self.binary_cls = binary_cls
        self.device = device

    def forward(self, y_pred, y, cls_bias=None):
        # if region annotations are missing, default to cross-entropy
        y[0][y[0].sum(1).sum(1) == 0] = 1

        if self.binary_cls:
            label = y[1].ravel()
            hrc = y_pred[1][:, 0]
            fg_mask = y[0]

            crce = torch.log(1 + torch.exp((-1)**label * (fg_mask * hrc).sum(dim=[1, 2]) + ((1 - fg_mask) * hrc.abs()).sum(dim=[1, 2]) + (-1)**label * cls_bias)).mean()
        else:
            cc = self.get_contrastive_cams(y[1], y_pred[1]).to(self.device)

            fg_mask = y[0].unsqueeze(1).repeat(1, const.N_CLASSES, 1, 1).to(torch.int).to(self.device)
            ablation = (-cc * fg_mask + cc.abs() * (1 - fg_mask)).sum(dim=[2, 3])

            if cls_bias is not None: ablation = ablation + (cls_bias[None,] - cls_bias[y[1].argmax(-1)].unsqueeze(-1))
            crce = self.ce(ablation, y[1])

        self.prev = crce.item()
        return crce


class CFALoss(nn.Module):
    """Causal Feature Alignment (https://openaccess.thecvf.com/content/WACV2024/papers/Venkataramani_Causal_Feature_Alignment_Learning_To_Ignore_Spurious_Background_Features_WACV_2024_paper.pdf)."""

    def __init__(self, binary_cls=False):
        super().__init__()

        if binary_cls: self.ce = nn.BCEWithLogitsLoss()
        else: self.ce = nn.CrossEntropyLoss()

    def forward(self, erm_feature_map, ft_feature_map, y_pred, y):
        ce_loss = self.ce(y_pred, y)
        cosine_loss = (1 - torch.cosine_similarity(erm_feature_map, ft_feature_map, dim=1)).mean()

        self.prev_ce = ce_loss.item()
        self.prev_cosine = cosine_loss.item()
        return ce_loss + cosine_loss


class EnergyLoss(nn.Module):
    """Energy Loss (https://arxiv.org/abs/2303.11932)."""

    def __init__(self, binary_cls=False):
        super().__init__()

        if binary_cls: self.ce = nn.BCEWithLogitsLoss()
        else: self.ce = nn.CrossEntropyLoss()

    def forward(self, y_pred, y):
        y[0][y[0].sum(1).sum(1) == 0] = 1

        ce_loss = self.ce(y_pred[0], y[1])

        grad_cam = y_pred[1].relu().clamp(min=1e-8)
        fg_mask = y[0].unsqueeze(1)
        energy_loss = 1 - ((grad_cam * fg_mask).sum(dim=[-1, -2]) / grad_cam.sum(dim=[-1, -2])).mean()

        self.prev_epg = energy_loss.item()
        return ce_loss + energy_loss


class GALSLoss(nn.Module):
    """Guiding Attention with Language Specification (https://arxiv.org/abs/2202.08926)."""

    def __init__(self, binary_cls=False):
        super().__init__()

        if binary_cls: self.ce = nn.BCEWithLogitsLoss()
        else: self.ce = nn.CrossEntropyLoss()

        self.l1 = nn.L1Loss()

    def forward(self, y_pred, y):
        ce_loss = self.ce(y_pred[0], y[1])

        fg_mask = y[0]
        grad_cam = y_pred[1][y[1].to(torch.bool)].relu()

        return ce_loss + self.l1(grad_cam, fg_mask)


class ABNLoss(nn.Module):
    """Attention Branch Networks (https://arxiv.org/abs/1812.10025)."""

    def __init__(self, binary_cls=False):
        super().__init__()

        if binary_cls: self.ce = nn.BCEWithLogitsLoss()
        else: self.ce = nn.CrossEntropyLoss()

        self.l1 = nn.L1Loss()

    def forward(self, y_pred, y, ab_logits):
        ce_loss = self.ce(y_pred[0], y[1]) + self.ce(ab_logits, y[1])

        fg_mask = y[0]
        abn_attention = y_pred[1][:, 0]

        return ce_loss + self.l1(abn_attention, fg_mask)

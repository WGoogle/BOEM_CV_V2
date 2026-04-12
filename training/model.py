"""
Builds a UNet ResNet34 segmentation model via segmentation_models_pytorch
and provides a combined BCE + Dice loss for binary segmentation of sparse
targets like polymetallic nodules.
"""
from __future__ import annotations
import logging
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)


def build_model(model_cfg, **_kw):
    model = smp.create_model(
        arch=model_cfg["architecture"],
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        in_channels=model_cfg.get("in_channels", 3),
        classes=model_cfg.get("classes", 1),
        activation=None,  # raw logits — sigmoid applied in loss / inference
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(
        "  Model: %s + %s  (weights=%s, in=%d, classes=%d, params=%.1fM)",
        model_cfg["architecture"],
        model_cfg["encoder_name"],
        model_cfg.get("encoder_weights", "imagenet"),
        model_cfg.get("in_channels", 3),
        model_cfg.get("classes", 1),
        n_params,
    )
    return model


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        p = probs.view(probs.size(0), -1)
        t = targets.view(targets.size(0), -1)

        intersection = (p * t).sum(dim=1)
        denom = p.sum(dim=1) + t.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss (Abraham & Khan, 2019).

    Tversky index generalises Dice by weighting FP and FN independently.
    Setting alpha < beta penalises false negatives more heavily — exactly
    what sparse small-object segmentation needs (missing a nodule is worse
    than a spurious detection).

    The focal modulation (1 - TI)^gamma further down-weights easy, well-
    segmented samples so the gradient is dominated by hard examples.

    Defaults:
        alpha = 0.3  (FP weight — lenient on false positives)
        beta  = 0.7  (FN weight — harsh on missed nodules)
        gamma = 4/3  (focal exponent — mild focus on hard samples)
    """

    def __init__(self, alpha=0.3, beta=0.7, gamma=4 / 3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        p = probs.view(probs.size(0), -1)
        t = targets.view(targets.size(0), -1)

        tp = (p * t).sum(dim=1)
        fp = (p * (1.0 - t)).sum(dim=1)
        fn = ((1.0 - p) * t).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        focal_tversky = (1.0 - tversky).pow(self.gamma)
        return focal_tversky.mean()


class CombinedLoss(nn.Module):
    # BCE + Dice (legacy baseline)

    def __init__(self, bce_weight=0.5, dice_weight=0.5, **_kw):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return (
            self.bce_weight * self.bce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
        )


class FocalTverskyCombinedLoss(nn.Module):
    """BCE + Focal Tversky — drop-in replacement for CombinedLoss.

    BCE stabilises early training (per-pixel cross-entropy provides dense,
    well-behaved gradients even when predictions are random).  Focal
    Tversky drives the model toward recall on sparse small objects once
    the predictions become meaningful.
    """

    def __init__(
        self,
        bce_weight=0.3,
        ftl_weight=0.7,
        alpha=0.3,
        beta=0.7,
        gamma=4 / 3,
        **_kw,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.ftl = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.bce_weight = bce_weight
        self.ftl_weight = ftl_weight

    def forward(self, logits, targets):
        return (
            self.bce_weight * self.bce(logits, targets)
            + self.ftl_weight * self.ftl(logits, targets)
        )

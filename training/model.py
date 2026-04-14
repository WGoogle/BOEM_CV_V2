"""
Builds a UNet ResNet34 segmentation model via segmentation_models_pytorch
and provides a combined BCE + Dice loss for binary segmentation
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
        activation=None, 
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

class CombinedLoss(nn.Module):
    # BCE + Dice
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

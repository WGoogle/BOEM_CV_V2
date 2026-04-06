"""
training.model — Model Factory and Loss Functions
===================================================
Builds a U-Net with ResNet34 encoder via segmentation_models_pytorch
and provides a combined BCE + Dice loss optimised for binary
segmentation of sparse targets (polymetallic nodules).

Design notes:
  - Combined loss balances pixel-level accuracy (BCE) with region
    overlap (Dice).  This is critical because nodules occupy a small
    fraction of each patch — pure BCE would converge to predicting
    all-background.
  - DiceLoss uses smooth=1 (Laplace smoothing) so gradients remain
    well-defined when a patch has zero foreground.
  - The model outputs raw logits; sigmoid is applied inside the loss
    and must be applied explicitly at inference time.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_model(model_cfg: dict) -> nn.Module:
    """Instantiate a segmentation model from config.MODEL dict.

    Parameters
    ----------
    model_cfg : dict
        Must contain: architecture, encoder_name, encoder_weights,
        in_channels, classes.

    Returns
    -------
    nn.Module
        SMP model with sigmoid activation stripped (returns logits).
    """
    model = smp.create_model(
        arch=model_cfg["architecture"],
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=model_cfg["encoder_weights"],
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["classes"],
        activation=None,  # raw logits — sigmoid applied in loss / inference
    )
    return model


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Operates on logits (applies sigmoid internally).
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Flatten spatial dims
        probs_flat  = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Weighted sum of BCE + Dice loss.

    Both components operate on raw logits.

    Parameters
    ----------
    bce_weight : float
        Weight for binary cross-entropy component.
    dice_weight : float
        Weight for Dice loss component.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.bce_weight  * self.bce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
        )

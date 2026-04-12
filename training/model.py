"""
Builds a segmentation model via segmentation_models_pytorch and provides a combined BCE + Dice loss optimised 
for binary segmentation of sparse targets like polymetallic nodules.

Supported architectures (via config.MODEL presets in config.py):
  - "Segformer" + "mit_b0".."mit_b5" — hierarchical transformer (default, SegFormer-B2). Xie et al., NeurIPS 2021.
  - "Unet" + "tu-convnext_tiny" — CNN failsafe, Liu et al., CVPR 2022.
"""
from __future__ import annotations
import logging
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)
_STRIDE32_ARCHS = {"segformer"}

def _assert_patch_divisibility(arch, patch_size):
    if arch.lower() in _STRIDE32_ARCHS and patch_size % 32 != 0:
        raise ValueError(
            f"{arch} requires input spatial dims divisible by 32, "
            f"got patch_size={patch_size}. Adjust config.PATCHING['patch_size']."
        )

def _create_smp(model_cfg):
    return smp.create_model(
        arch=model_cfg["architecture"],
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        in_channels=model_cfg.get("in_channels", 3),
        classes=model_cfg.get("classes", 1),
        activation=None,  # raw logits — sigmoid applied in loss / inference
    )

def build_model(
    model_cfg,
    patch_size = None,
    failsafe_cfg = None,
):
    if failsafe_cfg is None:
        try:
            import config  # type: ignore
            failsafe_cfg = getattr(config, "FAILSAFE_MODEL", None)
        except Exception:  # noqa: BLE001
            failsafe_cfg = None

    arch = model_cfg["architecture"]
    if patch_size is not None:
        _assert_patch_divisibility(arch, patch_size)

    # Try the primary preset first.
    try:
        model = _create_smp(model_cfg)
        active_cfg = model_cfg
        used_failsafe = False
    except Exception as exc: 
        if failsafe_cfg is None:
            raise
        logger.warning(
            "  Failed to build primary model %s + %s (%s). "
            "Falling back to failsafe preset %s + %s.",
            arch, model_cfg.get("encoder_name"), exc,
            failsafe_cfg["architecture"], failsafe_cfg["encoder_name"],
        )
        if patch_size is not None:
            _assert_patch_divisibility(failsafe_cfg["architecture"], patch_size)
        model = _create_smp(failsafe_cfg)
        active_cfg = failsafe_cfg
        used_failsafe = True

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    tag = "FAILSAFE" if used_failsafe else "primary"
    logger.info(
        "  Model [%s]: %s + %s  (weights=%s, in=%d, classes=%d, params=%.1fM)",
        tag,
        active_cfg["architecture"],
        active_cfg["encoder_name"],
        active_cfg.get("encoder_weights", "imagenet"),
        active_cfg.get("in_channels", 3),
        active_cfg.get("classes", 1),
        n_params,
    )
    return model

class DiceLoss(nn.Module):
    # Soft Dice loss for binary segmentation.
    def __init__(self, smooth = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat   = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        denom        = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()

class FocalTverskyLoss(nn.Module):
    # Focal Tversky loss — precision-biased region loss.
    # The Tversky index generalises the Dice coefficient by allowing asymmetric weighting of FP and FN
    def __init__(
        self,
        alpha = 0.7,
        beta = 0.3,
        gamma = 4.0 / 3.0,
        smooth = 1e-6,
    ):
        super().__init__()
        if alpha <= 0 or beta <= 0:
            raise ValueError(f"alpha and beta must be positive; got {alpha=}, {beta=}")
        self.alpha = alpha
        self.beta  = beta
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
        loss = torch.pow(torch.clamp(1.0 - tversky, min=0.0), self.gamma)
        return loss.mean()

class CombinedLoss(nn.Module):
    # BCE + Focal-Tversky — the active training loss.

    def __init__(
        self,
        bce_weight = 0.3,
        tversky_weight = 0.7,
        alpha = 0.7,
        beta = 0.3,
        gamma = 4.0 / 3.0,
        bce_pos_weight = None,
        label_smoothing = 0.0,
        dice_weight = None,
    ):
        super().__init__()
        if dice_weight is not None:
            tversky_weight = dice_weight

        if bce_pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(float(bce_pos_weight))
            )
        else:
            self.bce = nn.BCEWithLogitsLoss()

        self.tversky        = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.bce_weight     = bce_weight
        self.tversky_weight = tversky_weight

        # Label smoothing — symmetric two-sided, shrinks hard targets toward 0.5 so the network stops chasing 
        # obviously-wrong pixels to logit infinity. Cheapest possible robustness against proxy-label noise 
        # (Reed et al. 2015 bootstrapping; Müller et al. NeurIPS 2019 "When does label smoothing help?").
        if not 0.0 <= label_smoothing < 0.5:
            raise ValueError(
                f"label_smoothing must be in [0, 0.5), got {label_smoothing}"
            )
        self.label_smoothing = float(label_smoothing)

    def _smooth(self, targets):
        eps = self.label_smoothing
        if eps == 0.0:
            return targets
        return targets * (1.0 - 2.0 * eps) + eps

    def forward(self, logits, targets):
        t = self._smooth(targets)
        return (
            self.bce_weight     * self.bce(logits, t)
            + self.tversky_weight * self.tversky(logits, t)
        )
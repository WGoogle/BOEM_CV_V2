"""
training.model — Model Factory and Loss Functions
===================================================
Builds a segmentation model via segmentation_models_pytorch and provides
a combined BCE + Dice loss optimised for binary segmentation of sparse
targets (polymetallic nodules).

Supported architectures (via config.MODEL presets in config.py):
  - "Segformer" + "mit_b0".."mit_b5" — hierarchical transformer (default,
    SegFormer-B2). Xie et al., NeurIPS 2021.
  - "Unet" + "tu-convnext_tiny" — CNN failsafe, Liu et al., CVPR 2022.
  - Any other SMP (arch, encoder) pair — passed through unchanged.

If the primary model (``config.MODEL``) fails to initialise — e.g. blocked
pretrained-weight download, a transformer op unsupported on the target
device — build_model() falls back to ``config.FAILSAFE_MODEL`` (the
ConvNeXt-Tiny U-Net) so training can still start. The fallback is logged
so the user is never silently running the wrong architecture.

Design notes:
  - Combined loss balances pixel-level accuracy (BCE) with region
    overlap (Dice). This is critical because nodules occupy a small
    fraction of each patch — pure BCE would converge to predicting
    all-background.
  - DiceLoss uses smooth=1 (Laplace smoothing) so gradients remain
    well-defined when a patch has zero foreground.
  - The model outputs raw logits; sigmoid is applied inside the loss
    and must be applied explicitly at inference time.
  - SegFormer's MiT encoder downsamples the input 32× across four stages,
    so the input spatial dims must be divisible by 32. The pipeline's
    256 px patches satisfy this; build_model asserts it defensively.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)

# Architectures that require spatial dims divisible by 32 (MiT / transformer
# encoders with 4 stride-2 stages).
_STRIDE32_ARCHS = {"segformer"}


def _assert_patch_divisibility(arch: str, patch_size: int) -> None:
    """Fail fast if the configured patch size is incompatible with the encoder."""
    if arch.lower() in _STRIDE32_ARCHS and patch_size % 32 != 0:
        raise ValueError(
            f"{arch} requires input spatial dims divisible by 32, "
            f"got patch_size={patch_size}. Adjust config.PATCHING['patch_size']."
        )


def _create_smp(model_cfg: dict) -> nn.Module:
    """Thin wrapper around smp.create_model that reads our preset schema."""
    return smp.create_model(
        arch=model_cfg["architecture"],
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        in_channels=model_cfg.get("in_channels", 3),
        classes=model_cfg.get("classes", 1),
        activation=None,  # raw logits — sigmoid applied in loss / inference
    )


def build_model(
    model_cfg: dict,
    patch_size: int | None = None,
    failsafe_cfg: dict | None = None,
) -> nn.Module:
    """Instantiate a segmentation model from a config.MODEL dict.

    Parameters
    ----------
    model_cfg : dict
        Primary preset. Must contain: architecture, encoder_name,
        encoder_weights, in_channels, classes.
    patch_size : int | None
        If provided, the patch size is validated against the architecture's
        downsampling constraints (SegFormer/MiT requires divisibility by 32).
        Validation is applied to whichever preset is ultimately used
        (primary or failsafe).
    failsafe_cfg : dict | None
        CNN fallback preset used if the primary preset fails to build. If
        *None*, ``config.FAILSAFE_MODEL`` is used when importable; otherwise
        the primary exception is re-raised.

    Returns
    -------
    nn.Module
        SMP model with sigmoid activation stripped (returns raw logits).
    """
    # Resolve the default failsafe lazily to avoid a hard import cycle with
    # config.py — training.model is imported from 2_train.py which also
    # imports config.
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
    except Exception as exc:  # noqa: BLE001  — SMP/timm/HF raise many types
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
    """Soft Dice loss for binary segmentation.

    Symmetric (equal weight on FP and FN). Kept for legacy / ablation
    studies; the active loss is FocalTverskyLoss below, which is
    precision-biased to suppress sediment/divot false positives.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat   = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        denom        = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss — precision-biased region loss.

    The Tversky index generalises the Dice coefficient by allowing
    asymmetric weighting of FP and FN::

        TP = Σ  p·t
        FP = Σ  p·(1 − t)          ← predicted positive, target negative
        FN = Σ (1 − p)·t           ← predicted negative, target positive

                           TP  +  ε
        T(α, β) = ──────────────────────────────
                   TP + α·FP + β·FN  +  ε

                 L = (1 − T)^γ

    Parameters
    ----------
    alpha : float
        Weight on *false positives*. Larger α → stronger FP penalty →
        higher precision. Default **0.7**, chosen because sediment grains,
        gray divots, and 3-D-relief shading captured as nodules are a
        worse failure mode than missing a faint nodule (the density and
        coverage metrics downstream are sensitive to FP contamination).
    beta : float
        Weight on *false negatives*. Larger β → stronger FN penalty →
        higher recall. Default **0.3**.
    gamma : float
        Focal exponent on (1 − T). γ > 1 focuses learning on batches with
        low current Tversky index (hard cases); γ = 1 reduces to the
        plain Tversky loss. Default **4/3** (Abraham & Khan 2019).
    smooth : float
        Numerical stabiliser. Kept small (1e-6) so the gradient on empty
        patches remains meaningful — a large smooth term biases the
        network toward predicting all-background.

    Notes
    -----
    α + β need not equal 1, but fixing the sum to 1 makes ablations
    easier to interpret (see Salehi et al. 2017, "Tversky loss function
    for image segmentation using 3D fully convolutional deep networks").

    α = β = 0.5 recovers plain Dice loss exactly.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 4.0 / 3.0,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        if alpha <= 0 or beta <= 0:
            raise ValueError(f"alpha and beta must be positive; got {alpha=}, {beta=}")
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p = probs.view(probs.size(0), -1)
        t = targets.view(targets.size(0), -1)

        tp = (p * t).sum(dim=1)
        fp = (p * (1.0 - t)).sum(dim=1)
        fn = ((1.0 - p) * t).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        # clamp to avoid NaNs from numerical drift above 1 before pow().
        loss = torch.pow(torch.clamp(1.0 - tversky, min=0.0), self.gamma)
        return loss.mean()


class CombinedLoss(nn.Module):
    """BCE + Focal-Tversky — the active training loss.

    Optimised for the nodule-segmentation regime where a small number of
    dark blobs must be separated from structurally similar false
    positives (gray sediment divots, 3-D-relief shading, grain texture).

    Design:
      - **Focal-Tversky (α=0.7, β=0.3, γ=4/3)** is the dominant term. It
        directly penalises FP twice as hard as FN, which is what the
        downstream density metric actually cares about: capturing
        sediment as nodules is worse than missing a faint grain nodule.
      - **BCE** contributes a small pixel-level gradient, kept at lower
        weight so it doesn't drown out the region-level precision bias.
      - Optional **pos_weight** on BCE (``bce_pos_weight < 1``) further
        suppresses FP by reducing the gradient that rewards positive
        predictions. Default is ``None`` (symmetric BCE) — tune only if
        Focal-Tversky alone isn't precise enough.
      - ``smooth=1e-6`` (in FocalTversky) so empty patches still produce
        real gradients — the old ``smooth=1.0`` silently encouraged the
        model to predict nothing on sparse patches.

    Parameters
    ----------
    bce_weight : float
        Weight of the BCE term in the convex combination (default 0.3).
    tversky_weight : float
        Weight of the Focal-Tversky term (default 0.7).
    alpha, beta, gamma : float
        Passed through to FocalTverskyLoss. Defaults are the
        precision-favouring values (0.7 / 0.3 / 4⁄3).
    bce_pos_weight : float | None
        If set, used as ``pos_weight`` for BCEWithLogitsLoss. Values
        below 1 bias toward precision (fewer FP); values above 1 bias
        toward recall. Default ``None`` (=1.0, symmetric).
    """

    def __init__(
        self,
        bce_weight: float = 0.3,
        tversky_weight: float = 0.7,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 4.0 / 3.0,
        bce_pos_weight: float | None = None,
        # Legacy kwarg — accepted for backwards compat with old configs
        # that still pass ``dice_weight``. Maps onto tversky_weight.
        dice_weight: float | None = None,
    ) -> None:
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

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.bce_weight     * self.bce(logits, targets)
            + self.tversky_weight * self.tversky(logits, targets)
        )

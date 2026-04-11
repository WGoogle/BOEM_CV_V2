"""
training.dataset — Nodule Segmentation Dataset
================================================
Loads image-mask patch pairs produced by Step 1 and applies
train-time augmentations via Albumentations.

Input modes
-----------
Deep-sea AUV imagery is effectively single-channel: feeding three near-
identical RGB channels into a three-channel ImageNet-pretrained encoder
wastes two thirds of the first-conv capacity. Three input
representations are supported via the ``input_mode`` argument
(propagated from ``config.MODEL["input_mode"]``):

  - ``"rgb"``        — legacy. Preprocessed BGR → RGB → ImageNet normalise.
  - ``"grayscale"``  — single-channel L replicated 3×. Cheapest upgrade;
                       drops the redundant color channels but keeps the
                       pretrained first-conv weights meaningful.
  - ``"engineered"`` — three hand-crafted channels the proxy-label
                       pipeline already knows to be discriminative::

                         Ch 0 = L           (LAB lightness — raw signal)
                         Ch 1 = Sobel mag   (boundary / edge strength)
                         Ch 2 = LCR         (darkness vs local background,
                                             matching PROXY_LABEL.lcr_bg_sigma)

                       The first conv adapts to these new statistics in
                       ~5 epochs; historically worth 1–3 Dice points on
                       small-object tasks because the network no longer
                       has to re-derive what the auto-tuner already knows.

Design notes
------------
  - Images are stored as BGR (OpenCV default).
  - Masks are binary (0/255); normalised to {0, 1} float tensors.
  - Augmentations are applied jointly to image+mask so geometric
    transforms stay aligned.
  - Validation/test splits receive only deterministic normalisation
    (no random augmentations) for consistent evaluation.
  - Normalisation uses ImageNet stats in every mode. The first-conv
    weights adapt during training regardless of the incoming channel
    semantics; keeping one normaliser simplifies the code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── ImageNet normalisation (matches every pretrained encoder we use) ─────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ── Input-mode literals ─────────────────────────────────────────────────
InputMode = Literal["rgb", "grayscale", "engineered"]
_VALID_MODES: tuple[str, ...] = ("rgb", "grayscale", "engineered")

# Default σ for the local-contrast-ratio channel. Mirrors
# config.PROXY_LABEL["lcr_bg_sigma"]; re-hardcoded here to keep the
# dataset module importable without a hard dependency on config.py.
_LCR_BG_SIGMA = 30.0


def _validate_input_mode(mode: str) -> None:
    if mode not in _VALID_MODES:
        raise ValueError(
            f"input_mode must be one of {_VALID_MODES}, got {mode!r}"
        )


def compute_engineered_channels(
    bgr: np.ndarray,
    lcr_sigma: float = _LCR_BG_SIGMA,
) -> np.ndarray:
    """Build the three engineered channels from a preprocessed BGR patch.

    Returns a ``(H, W, 3)`` uint8 array with channels::

        [L, Sobel magnitude, Local Contrast Ratio]

    All three are produced as uint8 so Albumentations intensity
    augmentations (which assume 0–255 inputs) keep working unchanged.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"expected BGR (H,W,3) uint8, got shape {bgr.shape}")

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]                                     # uint8

    # Sobel gradient magnitude — edge / boundary signal. Real nodules
    # have strong circular boundaries; sediment grain has diffuse edges.
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    sobel = cv2.magnitude(gx, gy)                        # float32
    sobel_u8 = np.clip(sobel, 0.0, 255.0).astype(np.uint8)

    # Local contrast ratio — matches the proxy-label pipeline's LCR.
    # Positive where the pixel is darker than the local Gaussian
    # background, i.e. exactly the signal the auto-tuner gates on.
    L_f = L.astype(np.float32)
    bg = cv2.GaussianBlur(L_f, (0, 0), sigmaX=lcr_sigma, sigmaY=lcr_sigma)
    lcr = (bg - L_f) / (bg + 1e-6)
    lcr_u8 = np.clip(lcr * 255.0, 0.0, 255.0).astype(np.uint8)

    return np.stack([L, sobel_u8, lcr_u8], axis=-1)


def _prepare_image(bgr: np.ndarray, input_mode: str) -> np.ndarray:
    """Apply the mode-specific channel transform to a BGR patch.

    Returns a ``(H, W, 3)`` uint8 array ready to feed into the
    Albumentations pipeline.
    """
    _validate_input_mode(input_mode)
    if input_mode == "rgb":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if input_mode == "grayscale":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return np.stack([gray, gray, gray], axis=-1)
    # "engineered"
    return compute_engineered_channels(bgr)


def get_train_augmentations(patch_size: int = 256, input_mode: str = "rgb") -> A.Compose:
    """Augmentation pipeline for training split.

    Tuned for deep-sea AUV mosaics (low-contrast, near-grayscale, rigid
    geological targets, already heavily preprocessed by Step 1).

    What's in:
      - Flips + 90° rotations     — seafloor has no canonical orientation.
      - ``Affine`` jitter         — small shift/scale/rotate models AUV
                                    positioning drift. Replaces the old
                                    ``ShiftScaleRotate`` (deprecated in
                                    Albumentations 2.x).
      - ``GridDistortion``        — models AUV lens / altitude warping
                                    more realistically than ElasticTransform
                                    for rigid blob targets.
      - ``ElasticTransform``      — kept at low probability (p=0.1) as a
                                    mild shape regulariser; nodules are
                                    rigid so this is conservative.
      - ``RandomBrightnessContrast`` — models lighting variation between
                                    mosaic regions.
      - ``CLAHE``                 — randomises the per-patch contrast
                                    enhancement the preprocessing already
                                    does, so the model sees the real
                                    distribution of CLAHE strengths
                                    rather than a narrow snapshot.
      - ``GaussianBlur``          — models camera defocus / altitude jitter.
      - ``ISONoise``              — Poisson-Gaussian-ish sensor noise model;
                                    ``color_shift`` kept tiny because the
                                    input is near-grayscale.
      - ``CoarseDropout``         — occlusion robustness; forces the
                                    model to handle missing regions
                                    without hallucinating nodules.

    What's deliberately OUT:
      - ``HueSaturationValue`` — injects color artifacts that do not
        exist in deep-sea grayscale-ish imagery. Was actively hurting
        generalisation in the old pipeline.
      - MixUp / Copy-Paste — handled in a later pass; Copy-Paste in
        particular requires cross-patch sampling and is worth its own
        wiring rather than being squeezed in here.

    Mode awareness: a couple of the intensity augmentations assume their
    input is a genuine RGB image:

      - ``CLAHE``     — meaningful on each channel independently in
                        "rgb" and "grayscale" modes (all channels are
                        lightness-like) but weird on the Sobel /
                        LCR channels of "engineered" mode.
      - ``ISONoise``  — models sensor noise with a color shift that
                        only makes sense on true RGB.

    In "engineered" (and "grayscale") mode these are swapped for a
    plain ``GaussNoise`` and CLAHE is dropped. The geometric and
    dropout transforms are identical across modes.
    """
    _validate_input_mode(input_mode)
    is_rgb_like = input_mode == "rgb"

    return A.Compose([
        # ── Geometric (applied to image + mask jointly) ──────────────
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.90, 1.10),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.4,
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.2,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.2,
        ),
        A.ElasticTransform(
            alpha=20,
            sigma=5,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.1,  # rigid targets → keep this low
        ),

        # ── Intensity (image only) ───────────────────────────────────
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=0.4,
        ),
        *(
            [A.CLAHE(
                clip_limit=(1.0, 2.0),   # matches preprocessing clip range
                tile_grid_size=(8, 8),   # matches preprocessing tile grid
                p=0.3,
            )] if is_rgb_like else []
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        *(
            [A.ISONoise(
                color_shift=(0.005, 0.015),  # tiny — imagery is near-grayscale
                intensity=(0.10, 0.30),
                p=0.25,
            )]
            if is_rgb_like
            else [A.GaussNoise(std_range=(0.01, 0.05), p=0.25)]
        ),

        # ── Occlusion (applied to image only; mask stays intact so the
        #     model has to learn to tolerate the hole, not ignore the
        #     pixels under it) ───────────────────────────────────────
        A.CoarseDropout(
            num_holes_range=(2, 8),
            hole_height_range=(8, 16),
            hole_width_range=(8, 16),
            fill=0,
            p=0.25,
        ),

        # ── Final normalisation ──────────────────────────────────────
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_augmentations(input_mode: str = "rgb") -> A.Compose:
    """Deterministic normalisation only — no random transforms.

    ``input_mode`` is accepted for symmetry with
    :func:`get_train_augmentations` and future mode-specific
    normalisation stats; currently all modes share ImageNet norm.
    """
    _validate_input_mode(input_mode)
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class NoduleSegmentationDataset(Dataset):
    """PyTorch dataset for paired image + binary mask patches.

    Parameters
    ----------
    records : list[dict]
        Patch records from patch_manifest.json.  Each record must contain
        ``image_path`` and ``mask_path`` keys.
    transform : A.Compose | None
        Albumentations pipeline.  If *None*, a deterministic validation
        transform is used (``get_val_augmentations(input_mode)``).
    corrected_masks_dir : Path | str | None
        Directory containing manually corrected masks.  If a corrected
        mask exists for a patch, it is used instead of the proxy label.
    input_mode : {"rgb", "grayscale", "engineered"}
        Channel-representation mode. See the module docstring. The
        incoming BGR patch is converted via :func:`_prepare_image`
        **before** the Albumentations pipeline runs, so all intensity
        augmentations see the engineered channels directly. Callers are
        responsible for passing a ``transform`` that was constructed
        with the matching ``input_mode`` (the runners
        ``2_train.py`` / ``5_audit_labels.py`` do this automatically).
    """

    def __init__(
        self,
        records: list[dict],
        transform: A.Compose | None = None,
        corrected_masks_dir: str | None = None,
        input_mode: str = "rgb",
    ) -> None:
        _validate_input_mode(input_mode)
        self.records = records
        self.input_mode = input_mode
        self.transform = transform or get_val_augmentations(input_mode=input_mode)
        self.corrected_masks_dir = Path(corrected_masks_dir) if corrected_masks_dir else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]

        # Load BGR image
        bgr = cv2.imread(rec["image_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Image not found: {rec['image_path']}")

        # Mode-specific channel transform (RGB / grayscale-3× / engineered)
        image = _prepare_image(bgr, self.input_mode)

        # Load mask: prefer corrected mask over proxy label
        mask_path = rec["mask_path"]
        if self.corrected_masks_dir:
            patch_id = rec.get("patch_id", "")
            corrected = self.corrected_masks_dir / f"{patch_id}.png"
            if corrected.exists():
                mask_path = str(corrected)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = (mask > 127).astype(np.float32)

        # Apply augmentation (jointly to image + mask)
        augmented = self.transform(image=image, mask=mask)
        image_t = augmented["image"]                       # (3, H, W) float32
        mask_t  = augmented["mask"].unsqueeze(0).float()   # (1, H, W) float32

        return image_t, mask_t

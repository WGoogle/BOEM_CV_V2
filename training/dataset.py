"""
training.dataset — Nodule Segmentation Dataset
================================================
Loads image-mask patch pairs produced by Step 1 and applies
train-time augmentations via Albumentations.

Design notes:
  - Images are stored as BGR (OpenCV default); converted to RGB here.
  - Masks are binary (0/255); normalised to {0, 1} float tensors.
  - Augmentations are applied jointly to image+mask so geometric
    transforms stay aligned.
  - Validation/test splits receive only deterministic normalisation
    (no random augmentations) for consistent evaluation.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── ImageNet normalisation (matches ResNet34 pretrained encoder) ─────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_augmentations(patch_size: int = 256) -> A.Compose:
    """Augmentation pipeline for training split.

    Combines intensity + geometric transforms that are physically
    plausible for deep-sea AUV imagery:
      - Flips / 90-degree rotations  (seafloor has no canonical orientation)
      - Slight affine perturbations   (simulates AUV positioning jitter)
      - Brightness / contrast shifts  (simulates lighting variation)
      - Gaussian noise / blur         (simulates sensor noise / defocus)
      - Elastic distortion            (mild, for shape generalisation)
    """
    return A.Compose([
        # ── Geometric ────────────────────────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.10, rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101, p=0.4,
        ),
        A.ElasticTransform(
            alpha=30, sigma=120 * 0.05,
            border_mode=cv2.BORDER_REFLECT_101, p=0.2,
        ),

        # ── Intensity (image-only via additional_targets default) ────
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=0.4,
        ),
        A.HueSaturationValue(
            hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=15, p=0.3,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.03), p=0.2),

        # ── Final normalisation ──────────────────────────────────────
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_augmentations() -> A.Compose:
    """Deterministic normalisation only — no random transforms."""
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
        Albumentations pipeline.  If *None*, raw tensors are returned
        with only ImageNet normalisation applied.
    """

    def __init__(
        self,
        records: list[dict],
        transform: A.Compose | None = None,
    ) -> None:
        self.records = records
        self.transform = transform or get_val_augmentations()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]

        # Load image (BGR) → RGB
        image = cv2.imread(rec["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {rec['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale) → binary {0, 1}
        mask = cv2.imread(rec["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {rec['mask_path']}")
        mask = (mask > 127).astype(np.float32)

        # Apply augmentation (jointly to image + mask)
        augmented = self.transform(image=image, mask=mask)
        image_t = augmented["image"]                       # (3, H, W) float32
        mask_t  = augmented["mask"].unsqueeze(0).float()   # (1, H, W) float32

        return image_t, mask_t

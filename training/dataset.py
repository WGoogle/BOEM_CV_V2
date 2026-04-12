"""
Loads image-mask patch pairs produced by Step 1.
Applies train-time augmentations via Albumentations.
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalisation
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Input-mode literals 
InputMode = Literal["rgb", "grayscale", "engineered"]
_VALID_MODES: tuple[str, ...] = ("rgb", "grayscale", "engineered")

# For local contrast ratio
_LCR_BG_SIGMA = 30.0

def _validate_input_mode(mode):
    if mode not in _VALID_MODES:
        raise ValueError(
            f"input_mode must be one of {_VALID_MODES}, got {mode!r}"
        )

def compute_engineered_channels(
    bgr,
    lcr_sigma = _LCR_BG_SIGMA,
):
    # Returns [L, Sobel magnitude, Local Contrast Ratio]

    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"expected BGR (H,W,3) uint8, got shape {bgr.shape}")

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]                                   

    # Sobel gradient magnitude — edge / boundary signal.
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    sobel = cv2.magnitude(gx, gy)                    
    sobel_u8 = np.clip(sobel, 0.0, 255.0).astype(np.uint8)

    # Local contrast ratio — matches the proxy-label pipeline's LCR.
    L_f = L.astype(np.float32)
    bg = cv2.GaussianBlur(L_f, (0, 0), sigmaX=lcr_sigma, sigmaY=lcr_sigma)
    lcr = (bg - L_f) / (bg + 1e-6)
    lcr_u8 = np.clip(lcr * 255.0, 0.0, 255.0).astype(np.uint8)

    return np.stack([L, sobel_u8, lcr_u8], axis=-1)

def _prepare_image(bgr, input_mode):
    # Apply the mode-specific channel transform to a BGR patch

    _validate_input_mode(input_mode)
    if input_mode == "rgb":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if input_mode == "grayscale":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return np.stack([gray, gray, gray], axis=-1)
    return compute_engineered_channels(bgr)

def get_train_augmentations(patch_size = 256, input_mode = "rgb"):
    """
    Augmentation pipeline for training split.
    Tuned for deep-sea AUV mosaics (low-contrast, near-grayscale, rigid
    geological targets, already heavily preprocessed by Step 1).

    Contains Flips + 90 degree rotations, Affine jitter, GridDistortion, ElasticTransform,
    RandomBrightnessContrast, CLAHE (RGB/grayscale only), GaussianBlur, ISONoise (RGB only), and CoarseDropout
    """
    _validate_input_mode(input_mode)
    is_rgb_like = input_mode == "rgb"

    return A.Compose([
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

        # Intensity (image only)
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=0.4,
        ),
        *(
            [A.CLAHE(
                clip_limit=(1.0, 2.0),  
                tile_grid_size=(8, 8),   
                p=0.3,
            )] if is_rgb_like else []
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        *(
            [A.ISONoise(
                color_shift=(0.005, 0.015),  
                intensity=(0.10, 0.30),
                p=0.25,
            )]
            if is_rgb_like
            else [A.GaussNoise(std_range=(0.01, 0.05), p=0.25)]
        ),

        # Occlusion
        A.CoarseDropout(
            num_holes_range=(2, 8),
            hole_height_range=(8, 16),
            hole_width_range=(8, 16),
            fill=0,
            p=0.25,
        ),

        # Final normalisation 
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_val_augmentations(input_mode = "rgb"):
    # Deterministic normalisation only — no random transforms.
    _validate_input_mode(input_mode)
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

class CopyPasteAugmentation:
    """
    Simple Copy-Paste augmentation (Ghiasi et al. CVPR 2021).
    Crop real nodules from high-coverage "source" patches and paste them into whichever patch the dataset is 
    currently returning, updating both image and mask.
   """
    def __init__(
        self,
        source_records,
        corrected_masks_dir = None,
        p = 0.5,
        max_objects = 3,
        min_source_coverage = 5.0,
        max_object_area_frac = 0.25,
    ):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if max_objects < 1:
            raise ValueError(f"max_objects must be ≥ 1, got {max_objects}")

        self.p = p
        self.max_objects = max_objects
        self.max_object_area_frac = max_object_area_frac
        self.corrected_masks_dir = (
            Path(corrected_masks_dir) if corrected_masks_dir else None
        )
        self.sources: list[dict] = [
            r for r in source_records
            if r.get("label_stats", {}).get("coverage_pct", 0.0) >= min_source_coverage
        ]

    def __bool__(self):
        return bool(self.sources)

    def _resolve_mask_path(self, rec):
        # Prefer a corrected mask over the proxy label when available
        if self.corrected_masks_dir is not None:
            patch_id = rec.get("patch_id", "")
            corrected = self.corrected_masks_dir / f"{patch_id}.png"
            if corrected.exists():
                return str(corrected)
        return rec["mask_path"]

    def __call__(
        self,
        bgr,
        mask,
    ):
        if not self.sources or np.random.random() > self.p:
            return bgr, mask

        src = self.sources[int(np.random.randint(len(self.sources)))]
        src_img = cv2.imread(src["image_path"], cv2.IMREAD_COLOR)
        if src_img is None:
            return bgr, mask

        src_mask_path = self._resolve_mask_path(src)
        src_mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
        if src_mask is None:
            return bgr, mask
        src_mask_bin = (src_mask > 127).astype(np.uint8)

        n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(
            src_mask_bin, connectivity=8,
        )
        if n_cc <= 1:
            return bgr, mask

        H, W = mask.shape[:2]
        max_obj_area = int(self.max_object_area_frac * H * W)

        cc_indices = np.arange(1, n_cc)
        np.random.shuffle(cc_indices)

        out_bgr  = bgr.copy()
        out_mask = mask.copy()

        n_pasted = 0
        for ci in cc_indices:
            if n_pasted >= self.max_objects:
                break
            x, y, w, h, area = stats[ci]
            if area < 4 or w >= W or h >= H or area > max_obj_area:
                continue

            x2, y2 = x + w, y + h
            if x2 > src_img.shape[1] or y2 > src_img.shape[0]:
                continue

            src_crop     = src_img[y:y2, x:x2]
            src_crop_cc  = (labels[y:y2, x:x2] == ci).astype(np.uint8)

            px = int(np.random.randint(0, W - w + 1))
            py = int(np.random.randint(0, H - h + 1))

            # Update image
            mask3 = src_crop_cc[:, :, None]  # (h, w, 1)
            roi_img = out_bgr[py:py + h, px:px + w]
            roi_img[:] = roi_img * (1 - mask3) + src_crop * mask3

            # Update mask
            roi_mask = out_mask[py:py + h, px:px + w]
            np.maximum(roi_mask, src_crop_cc.astype(np.float32), out=roi_mask)
            n_pasted += 1
        return out_bgr, out_mask

class NoduleSegmentationDataset(Dataset):
    # PyTorch dataset for paired image + binary mask patches.
    def __init__(
        self,
        records,
        transform = None,
        corrected_masks_dir = None,
        input_mode = "rgb",
        copy_paste = None,
        mirror_pad = 0,
    ):
        _validate_input_mode(input_mode)
        if mirror_pad < 0:
            raise ValueError(f"mirror_pad must be ≥ 0, got {mirror_pad}")
        self.records = records
        self.input_mode = input_mode
        self.transform = transform or get_val_augmentations(input_mode=input_mode)
        self.corrected_masks_dir = Path(corrected_masks_dir) if corrected_masks_dir else None

        self.copy_paste = copy_paste
        self.mirror_pad = mirror_pad

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        bgr = cv2.imread(rec["image_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Image not found: {rec['image_path']}")

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

        if self.copy_paste is not None:
            bgr, mask = self.copy_paste(bgr, mask)

        image = _prepare_image(bgr, self.input_mode)
        augmented = self.transform(image=image, mask=mask)
        image_t = augmented["image"]                       
        mask_t  = augmented["mask"].unsqueeze(0).float()  

        if self.mirror_pad > 0:
            p = self.mirror_pad
            image_t = F.pad(
                image_t.unsqueeze(0), (p, p, p, p), mode="reflect",
            ).squeeze(0)

        return image_t, mask_t
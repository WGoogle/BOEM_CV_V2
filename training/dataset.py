"""
Loads image-mask patch pairs produced by Step 1.
Applies train-time augmentations via Albumentations.
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalisation (for RGB mode and as a fallback for engineered mode) -- we will use engineered mode
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

import json as _json
import logging as _logging
_norm_logger = _logging.getLogger(__name__)

def compute_channel_stats(records, input_mode):
    # Compute per-channel (mean, std) over all patches for a given input mode.

    n = 0
    ch_sum = np.zeros(3, dtype=np.float64)
    ch_sq  = np.zeros(3, dtype=np.float64)

    for rec in records:
        bgr = cv2.imread(rec["image_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        img = _prepare_image(bgr, input_mode).astype(np.float64) / 255.0
        pixels = img.shape[0] * img.shape[1]
        for c in range(3):
            ch = img[:, :, c]
            ch_sum[c] += ch.sum()
            ch_sq[c]  += (ch ** 2).sum()
        n += pixels

    mean = ch_sum / n
    std  = np.sqrt(ch_sq / n - mean ** 2)
    return tuple(np.round(mean, 4).tolist()), tuple(np.round(std, 4).tolist())


def get_normalization_stats(input_mode, records=None, cache_dir=None):
    # Return (mean, std) tuples appropriate for the given input mode.

    if input_mode != "engineered":
        return IMAGENET_MEAN, IMAGENET_STD
    if cache_dir is None:
        try:
            import config as _cfg
            cache_dir = _cfg.CHECKPOINTS_DIR
        except Exception:
            cache_dir = None
    cache_file = (
        Path(cache_dir) / "engineered_norm_stats.json" if cache_dir is not None else None
    )

    # Try to load from cache
    if cache_file is not None and cache_file.exists():
        with open(cache_file) as f:
            cached = _json.load(f)
        cached_n = cached.get("num_patches", -1)
        current_n = len(records) if records is not None else cached_n
        if cached_n == current_n:
            _norm_logger.debug("  Engineered norm stats loaded from cache (%d patches)", cached_n)
            return tuple(cached["mean"]), tuple(cached["std"])
        _norm_logger.info(
            "  Patch count changed (%d → %d) — recomputing engineered norm stats",
            cached_n, current_n,
        )

    # Compute from scratch (requires records)
    if records is None:
        # Inference path — no records available, cache missing or stale
        if cache_file is not None and cache_file.exists():
            with open(cache_file) as f:
                cached = _json.load(f)
            _norm_logger.warning(
                "  No records supplied; using stale engineered norm cache (%d patches)",
                cached.get("num_patches", -1),
            )
            return tuple(cached["mean"]), tuple(cached["std"])
        _norm_logger.warning(
            "  No engineered norm cache found and no records supplied — "
            "falling back to ImageNet stats (results will be suboptimal)"
        )
        return IMAGENET_MEAN, IMAGENET_STD

    _norm_logger.info("  Computing engineered channel stats over %d patches ...", len(records))
    mean, std = compute_channel_stats(records, "engineered")
    _norm_logger.info("  Engineered mean=%s  std=%s", mean, std)

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"mean": list(mean), "std": list(std),
                   "num_patches": len(records)}
        with open(cache_file, "w") as f:
            _json.dump(payload, f, indent=2)
        _norm_logger.info("  Cached → %s", cache_file)

    return mean, std

_VALID_MODES: tuple[str, ...] = ("rgb", "engineered")

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
    return compute_engineered_channels(bgr)

def get_train_augmentations(patch_size = 256, input_mode = "rgb", norm_stats = None):
    """
    Augmentation pipeline for training split.
    Contains Flips + 90 degree rotations, Affine jitter, ElasticTransform,
    RandomBrightnessContrast, GaussianBlur, and GaussNoise.
    """
    _validate_input_mode(input_mode)
    if norm_stats is not None:
        norm_mean, norm_std = norm_stats
    else:
        norm_mean, norm_std = get_normalization_stats(input_mode)

    return A.Compose([
        # Geometric
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

        # Intensity
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=0.4,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.03), p=0.2),

        # Final normalisation
        A.Normalize(mean=norm_mean, std=norm_std),
        ToTensorV2(),
    ])

def get_val_augmentations(input_mode = "rgb", norm_stats = None):
    # Deterministic normalisation only
    _validate_input_mode(input_mode)
    if norm_stats is not None:
        norm_mean, norm_std = norm_stats
    else:
        norm_mean, norm_std = get_normalization_stats(input_mode)
    return A.Compose([
        A.Normalize(mean=norm_mean, std=norm_std),
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

            # Update image — feather alpha to avoid teaching the model to detect paste seams
            alpha = cv2.GaussianBlur(
                src_crop_cc.astype(np.float32), ksize=(0, 0), sigmaX=1.5,
            )[:, :, None]
            roi_img = out_bgr[py:py + h, px:px + w]
            blended = roi_img.astype(np.float32) * (1 - alpha) + src_crop.astype(np.float32) * alpha
            roi_img[:] = np.clip(blended, 0, 255).astype(roi_img.dtype)

            # Update mask — keep hard edges for crisp supervision
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
    ):
        _validate_input_mode(input_mode)
        self.records = records
        self.input_mode = input_mode
        self.transform = transform or get_val_augmentations(input_mode=input_mode)
        self.corrected_masks_dir = Path(corrected_masks_dir) if corrected_masks_dir else None

        self.copy_paste = copy_paste

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

        return image_t, mask_t
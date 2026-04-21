"""
preprocess.py — Deploy-side Step-1 filter chain on a raw mosaic.

Runs the same patch-wise filter chain used to train the model:
  gray-world WB -> MSR -> CLAHE -> bilateral -> sediment fade -> unsharp

Returns the reassembled preprocessed mosaic as a BGR uint8 array (in memory).
The model was trained on the output of this chain, so feeding raw imagery
directly to the predictor will give degraded results.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

from preprocessing.patcher import MosaicPatcher
from preprocessing.auto_tuner import PatchAutoTuner
from preprocessing.filters import FilterPipeline
from preprocessing.geo_resolution import extract_meters_per_pixel

# Frozen copy of the relevant sections of the training-time config.
# Must stay in sync with the project root config.py values used to train
# the shipped checkpoint.
PATCHING = {
    "patch_size":   256,
    "overlap":      32,
    "min_std":      3.0,
    "min_mean":     10.0,
    "max_black_fraction": 0.25,
    "max_noise":    4.5,
}

AUTO_TUNER = {
    "msr_blend_range":      (0.3, 1.0),
    "clahe_clip_range":     (1.0, 2.0),
    "clahe_blend_range":    (0.3, 1.0),
    "clahe_tile_grid":      (8, 8),
    "unsharp_strength_range": (0.1, 0.5),
    "bilateral_d":          7,
    "bilateral_sigma_color_range": (30, 60),
    "bilateral_sigma_space_range": (30, 60),
    "block_size_range":     (11, 51),
    "c_offset_range":       (2, 15),
    "morph_open_range":     (1, 3),
    "morph_close_range":    (5, 11),
    "contour_area_min_range": (2, 8),
    "max_contour_area":       5000,
    "eccentricity_range":  (0.80, 0.90),
    "solidity_range":      (0.45, 0.65),
    "circularity_range":   (0.25, 0.40),
}

PREPROCESSING = {
    "filter_chain": [
        "gray_world_white_balance",
        "multi_scale_retinex",
        "clahe_lab",
        "bilateral_denoise",
        "sediment_fade",
        "unsharp_mask",
    ],
    "msr_sigmas":               [5, 20, 80],
    "msr_gain":                 1.0,
    "sediment_fade_blur_sigma":  15.0,
    "sediment_fade_strength":    0.6,
    "unsharp_sigma":    2.0,
    "unsharp_strength": 0.5,
}

FALLBACK_METERS_PER_PIXEL = 0.005


def preprocess_mosaic(raw_path, *, progress=True):
    """
    Load a raw mosaic, run the Step-1 filter chain patch-wise, and reassemble.

    Returns
    -------
    preprocessed_bgr : np.ndarray (H, W, 3) uint8
    raw_bgr          : np.ndarray (H, W, 3) uint8   original, for overlays
    meters_per_pixel : float | None                 from GeoTIFF tags or fallback
    """
    raw_path = Path(raw_path)

    patcher  = MosaicPatcher(**PATCHING)
    tuner    = PatchAutoTuner(AUTO_TUNER)
    pipeline = FilterPipeline(PREPROCESSING)

    mosaic = patcher.load_mosaic(raw_path)
    H, W = mosaic.shape[:2]

    mpp = extract_meters_per_pixel(raw_path, fallback=FALLBACK_METERS_PER_PIXEL)
    if progress and mpp is not None:
        print(
            f"  Spatial resolution: {mpp:.6f} m/px ({mpp * 1000:.2f} mm/px)  "
            f"— image covers {W * mpp:.1f} m x {H * mpp:.1f} m"
        )

    patches, infos = patcher.extract_patches(mosaic)
    n_valid = len(patches)
    if n_valid == 0:
        raise RuntimeError(
            f"No valid patches in {raw_path.name} — mosaic rejected by quality gate."
        )

    preprocessed_patches = []
    for i, patch_bgr in enumerate(patches, 1):
        params = tuner.analyse(patch_bgr)
        pre, _ = pipeline.run(patch_bgr, params)
        preprocessed_patches.append(pre)
        if progress and (i % 50 == 0 or i == n_valid):
            print(f"    preprocessed {i}/{n_valid} patches")

    # Reassemble preprocessed patches; rejected tiles keep their raw pixels so
    # borders don't become voids. Matches the behaviour of 1_preprocess_and_label.py.
    accum = np.zeros((H, W, 3), dtype=np.float64)
    count = np.zeros((H, W),    dtype=np.float64)
    pi = 0
    for info in infos:
        if not info.is_valid:
            continue
        patch = preprocessed_patches[pi]
        pi += 1
        y, x = info.y, info.x
        ph = min(info.height, H - y)
        pw = min(info.width,  W - x)
        accum[y:y + ph, x:x + pw] += patch[:ph, :pw].astype(np.float64)
        count[y:y + ph, x:x + pw] += 1.0

    covered = count > 0
    full = mosaic.copy()
    full[covered] = (
        accum[covered] / count[covered, None]
    ).clip(0, 255).astype(np.uint8)

    if progress:
        print(f"  Preprocessing coverage: {100.0 * covered.mean():.1f}%")

    return full, mosaic, mpp

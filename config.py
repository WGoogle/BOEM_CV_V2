"""
Configuration — Polymetallic Nodule Segmentation Pipeline
==========================================================
All paths, hyper-parameters, and feature flags live here.
Downstream modules import this file and read the dicts they need;
no module hard-codes a magic number.

Architectural note (CoralNet-inspired):
  Every processing stage reads its own config section.  Adding a new
  parameter means touching only this file + the function that uses it.
"""

from pathlib import Path

# ----------- PATHS -----------

PROJECT_ROOT    = Path(__file__).parent
DATA_DIR        = PROJECT_ROOT / "data"
RAW_MOSAICS_DIR = DATA_DIR / "raw_mosaics"

OUTPUT_DIR          = PROJECT_ROOT / "outputs"
PREPROCESSED_DIR    = OUTPUT_DIR / "preprocessed"
PROXY_LABELS_DIR    = OUTPUT_DIR / "proxy_labels"
PATCHES_DIR         = OUTPUT_DIR / "patches"
CHECKPOINTS_DIR     = OUTPUT_DIR / "checkpoints"
RESULTS_DIR         = OUTPUT_DIR / "results"
STEP_BY_STEP_DIR    = OUTPUT_DIR / "step_by_step_logs"
LOGS_DIR            = OUTPUT_DIR / "logs"
MANUAL_LABELS_DIR   = DATA_DIR / "manual_labels"

# Create all directories at import time
for _d in [
    DATA_DIR, RAW_MOSAICS_DIR, OUTPUT_DIR, PREPROCESSED_DIR,
    PROXY_LABELS_DIR, PATCHES_DIR, CHECKPOINTS_DIR, RESULTS_DIR,
    STEP_BY_STEP_DIR, LOGS_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)



# ----------- PATCHING: splitting massive mosaics into manageable tiles -----------

PATCHING = {
    # Patch geometry  (pixels)
    "patch_size":   256,            # smaller patches for better local adaptation
                                    # and higher nodule detection sensitivity
    "overlap":      32,             # overlap in pixels between adjacent patches
                                    # to avoid cutting nodules at boundaries

    # Quality gate — reject featureless / black-border patches
    # NOTE: Deep-sea imagery is inherently low-contrast (std ≈ 4-8).
    # Set min_std conservatively low to avoid rejecting valid seafloor.
    "min_std":      3.0,            # minimum grayscale std-dev
    "min_mean":     10.0,           # minimum grayscale mean (rejects black)
    "max_black_fraction": 0.25,     # reject if >25 % of pixels are near-zero
}



# -----------AUTO-TUNER: per-patch adaptive parameter calculation -----------


AUTO_TUNER = {
    # ── CLAHE ────────────────────────────────────────────────────────────────
    # Clip limit is scaled by local contrast: low-contrast patches get a
    # stronger boost, high-contrast patches are left mostly alone.
    # All three enhancement stages (MSR, CLAHE, unsharp) are scaled by the
    # patch's contrast ratio so low-contrast (sediment-heavy) patches stay
    # close to the original and only high-contrast (nodule-rich) patches
    # receive full enhancement.
    "msr_blend_range":      (0.3, 1.0),     # MSR blend: 30% for flat sediment, 100% for nodule-rich
    "clahe_clip_range":     (1.0, 2.0),     # CLAHE clip limit range
    "clahe_blend_range":    (0.3, 1.0),     # CLAHE blend: 30% for flat sediment, 100% for nodule-rich
    "clahe_tile_grid":      (8, 8),         # tile grid for CLAHE
    "unsharp_strength_range": (0.1, 0.5),   # unsharp: 0.1 for flat sediment, 0.5 for nodule-rich

    # ── Bilateral filter ─────────────────────────────────────────────────────
    "bilateral_d":          7,
    "bilateral_sigma_color_range": (30, 60),  # scaled by noise estimate — capped at 60 to avoid
    "bilateral_sigma_space_range": (30, 60),  # blending across nodule–sediment boundaries

    # ── Adaptive thresholding (for proxy-label generation) ───────────────────
    "block_size_range":     (11, 51),        # adaptive threshold block size
    "c_offset_range":       (2, 15),         # constant subtracted from mean

    # ── Morphology ───────────────────────────────────────────────────────────
    "morph_open_range":     (1, 3),          # opening kernel — must be tiny for grain nodules
    "morph_close_range":    (5, 11),         # closing kernel — solidify scattered detections
                                            # into coherent nodule blobs (was 3-7, too small)


    # ── Contour shape filters (adaptive — noise-driven) ──────────────────────
    # All four shape parameters scale with noise_estimate so that noisy patches
    # (which produce pixelated, irregular contours) get relaxed criteria while
    # clean patches get tighter ones.
    #
    # At 5mm/px, a 10mm grain nodule ≈ 2px diameter ≈ 3px² area.
    # Shape filters are skipped entirely for contours < 20px² (size-aware gate
    # in filters.py), so the area floor is the primary grain-nodule guard.
    "contour_area_min_range": (2, 8),    # [low-noise min, high-noise min]
    "max_contour_area":       5000,      # fixed — physical upper bound

    # Eccentricity: high noise → allow more elongated shapes (relax upward)
    "eccentricity_range":  (0.80, 0.90),  # [low-noise max, high-noise max]

    # Solidity & circularity: high noise → lower threshold (relax downward)
    # Tuple is [high-noise floor, low-noise floor] — mapping is inverted in analyse()
    "solidity_range":      (0.45, 0.65),
    "circularity_range":   (0.25, 0.40),
}


# ----------- PREPROCESSING: filter chain applied to every patch -----------

PREPROCESSING = {
    # Ordered list of filter steps.  The runner executes them in this order;
    # each name maps to a function in preprocessing/filters.py.
    # Toggle any step off by removing it from the list.
    "filter_chain": [
        "gray_world_white_balance",
        "multi_scale_retinex",         # illumination/shading removal (safe for divots)
        "clahe_lab",                   # gentle local contrast enhancement
        "bilateral_denoise",
        "sediment_fade",
        "unsharp_mask",
    ],

    # ── Multi-Scale Retinex ───────────────────────────────────────────────────
    # Removes illumination gradients and shading from 3D surface relief,
    # so divots/bumps are not falsely darkened.  CLAHE follows to gently
    # recover local contrast for nodule detection.
    "msr_sigmas":               [5, 20, 80],
    "msr_gain":                 1.0,

    # ── Sediment fade ────────────────────────────────────────────────────────
    "sediment_fade_blur_sigma":  15.0,
    "sediment_fade_strength":    0.6,

    # ── Unsharp mask (edge-selective) ────────────────────────────────────────
    "unsharp_sigma":    2.0,
    "unsharp_strength": 0.5,
}


# ----------- PROXY LABEL GENERATION -----------

PROXY_LABEL = {
    # ── Multi-feature blob detection ─────────────────────────────────────
    # Primary signal: multi-scale black top-hat (compact dark blob detector).
    # Top-hat only fires on features SMALLER than the structuring element,
    # so diffuse gray sediment patches get zero response.
    # Gated by local contrast ratio so detections must actually be darker
    # than their surroundings.

    # Top-hat radii covering grain (1-3px) through large/cluster (18-40px)
    # nodules at 5mm/px resolution.  SE diameter = 2r+1, so the SE must be
    # larger than the nodule for the closing to fully fill it.
    #   r=1-2:  grain nodules (5-15mm = 1-3px)
    #   r=4-8:  medium nodules (25-50mm = 5-10px)
    #   r=12:   large nodules (40-90mm = 8-18px)
    #   r=16-20: very large nodules / clusters (90-200mm = 18-40px)
    "tophat_radii":             [1, 2, 4, 8, 12, 16, 20],

    # Texture gate: suppresses top-hat response in grainy sediment where
    # individual grains fire the morphological filter.
    "texture_sigma":            2.0,
    "texture_threshold":        18.0,

    # ── Local contrast ratio gate ────────────────────────────────────────
    # σ for background estimation in LCR.  30px = 150mm at 5mm/px —
    # large enough to not track individual nodules but small enough to
    # adapt to local illumination.
    "lcr_bg_sigma":             30.0,

    # ── Score thresholding ───────────────────────────────────────────────
    # Combined score = raw_tophat × raw_lcr (absolute magnitude).
    # Real nodules score 9-42; noise/artifacts score < 3.
    # Absolute threshold is the primary gate — patches with no real
    # nodules never exceed it.  Lower for higher recall (more grain
    # nodules), raise for higher precision (fewer false positives).
    "score_threshold":          4.0,
    # Percentile adapts to nodule density:
    #   sparse patches → high percentile (strict, abs_threshold dominates)
    #   dense patches  → low percentile (relaxed, more nodules survive)
    "score_percentile_range":   (70, 90),
    # Dense-patch two-pass thresholding:
    # When ≥ dense_frac_trigger of valid pixels exceed score_threshold,
    # the effective abs threshold is lowered toward dense_score_threshold_min
    # AND the percentile gate is faded out so it no longer dominates.
    # This lets dense-nodule patches keep faint-but-real detections while
    # sparse patches remain unaffected.
    "dense_score_threshold_min": 3.0,   # lowest abs threshold for very dense patches
    "dense_frac_trigger":        0.01,  # fraction of pixels above score_threshold
                                        # that triggers the dense-patch path

    # ── Contour filters ──────────────────────────────────────────────────
    # Per-contour contrast check: interior must be this fraction darker
    # than local background.
    "min_local_contrast":       0.02,
    # Score-gate: contours with mean score ≥ threshold × this multiplier
    # are high-confidence and skip strict shape filters (solidity,
    # eccentricity, circularity). Prevents strong detections from being
    # killed by irregular blob shape at low pixel counts.
    "shape_bypass_score_mult":  2.0,
}


# ----------- STEP-BY-STEP LOGGING -----------

LOGGING = {
    "save_intermediate_steps":  True,       # master toggle
    "max_step_log_patches":     5,          # random subset of patches to log
    "composite_grid_cols":      4,          # columns in the summary grid
}


# ----------- MODEL / TRAINING / INFERENCE  (For Later) -----------


MODEL = {
    "architecture":     "Unet",
    "encoder_name":     "resnet34",
    "encoder_weights":  "imagenet",
    "in_channels":      3,
    "classes":          1,
}

TRAINING = {
    "batch_size":       16,
    "num_epochs":       100,
    "learning_rate":    1e-4,
    "weight_decay":     1e-5,
    "train_split":      0.8,
    "val_split":        0.1,
    "test_split":       0.1,
    "random_seed":      42,
}

INFERENCE = {
    "probability_threshold":    0.5,
    "blend_mode":               "average",
}

METRICS = {
    # Fallback value — used only when the mosaic file lacks GeoTIFF metadata.
    # The pipeline auto-extracts meters_per_pixel from GeoTIFF tags (33550 or
    # 34264) at load time.  This default (5 mm/px) matches the BOEM D1 Node3
    # survey data.
    "meters_per_pixel":         0.005,
    "connectivity":             8,
    "min_nodule_size":          20,
}

VISUALIZATION = {
    "save_preprocessed":            True,
    "save_proxy_labels":            True,
    "save_probability_maps":        True,
    "save_segmentation_overlays":   True,
    "overlay_alpha":                0.4,
}
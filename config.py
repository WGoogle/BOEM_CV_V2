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
    "patch_size":   1024,           # square patches — big enough to contain
                                    # multiple nodules with local context
    "overlap":      128,            # overlap in pixels between adjacent patches
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
    "clahe_clip_range":     (1.0, 1.5),     # conservative — MSR handles illumination, CLAHE just adds gentle local contrast
    "clahe_tile_grid":      (8, 8),         # tile grid for CLAHE

    # ── Bilateral filter ─────────────────────────────────────────────────────
    "bilateral_d":          7,
    "bilateral_sigma_color_range": (30, 75),  # scaled by noise estimate
    "bilateral_sigma_space_range": (30, 75),

    # ── Adaptive thresholding (for proxy-label generation) ───────────────────
    "block_size_range":     (11, 51),        # adaptive threshold block size
    "c_offset_range":       (2, 15),         # constant subtracted from mean

    # ── Morphology ───────────────────────────────────────────────────────────
    "morph_open_range":     (3, 7),          # opening kernel diameter
    "morph_close_range":    (5, 13),         # closing kernel diameter

    # ── Nodule-boost (bottom-hat) ────────────────────────────────────────────
    "nodule_boost_factor":  2.0,
    "morph_radius":         20,              # SE radius for bottom-hat
    "texture_sigma":        2.0,
    "texture_threshold":    18.0,
    "max_darkening":        70,

    # ── Top-hat proxy labelling ──────────────────────────────────────────────
    "tophat_radii":         [12, 20, 30],
    "tophat_percentile":    96,
    "tophat_threshold_floor": 15.0,

    # ── Contour shape filters ────────────────────────────────────────────────
    "min_contour_area":     50,
    "max_contour_area":     3000,
    "max_eccentricity":     0.80,
    "min_solidity":         0.60,
    "min_circularity":      0.45,
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
    "apply_gaussian_blur":      True,
    "gaussian_kernel_size":     15,
    "gaussian_sigma":           5.0,

    "adaptive_abs_intensity":   True,
    "adaptive_abs_percentile":  8,
    "absolute_intensity_max":   85,
}


# ----------- STEP-BY-STEP LOGGING -----------

LOGGING = {
    "save_intermediate_steps":  True,       # master toggle
    "log_every_n_patches":      1,          # log every Nth patch (1 = all)
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
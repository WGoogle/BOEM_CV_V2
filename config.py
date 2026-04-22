"""
Configuration — Polymetallic Nodule Segmentation Pipeline

All paths, hyper-parameters, and feature flags live here.
Downstream modules import this file and read the dicts they need;
no module hard-codes a magic number.

Architectural note (CoralNet-inspired):
  Every processing stage reads its own config section.  Adding a new
  parameter means touching only this file + the function that uses it.
"""
from pathlib import Path

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
CORRECTED_MASKS_DIR = OUTPUT_DIR / "corrected_masks"
ANNOTATION_INBOX    = OUTPUT_DIR / "annotation_inbox"

# Create all directories at import time
for _d in [
    DATA_DIR, RAW_MOSAICS_DIR, OUTPUT_DIR, PREPROCESSED_DIR,
    PROXY_LABELS_DIR, PATCHES_DIR, CHECKPOINTS_DIR, RESULTS_DIR,
    STEP_BY_STEP_DIR, LOGS_DIR, ANNOTATION_INBOX,
]:
    _d.mkdir(parents=True, exist_ok=True)


# ----------- PATCHING: splitting massive mosaics into manageable tiles -----------

PATCHING = {
    # Patch geometry  (pixels)
    "patch_size":   256,           
    "overlap":      32,          

    # Quality gate — reject featureless / black-border patches
    "min_std":      3.0,            # minimum grayscale std-dev
    "min_mean":     10.0,           # minimum grayscale mean (rejects black)
    "max_black_fraction": 0.5,     # reject if >50 % of pixels are near-zero
    "max_noise":    4.5,            # reject extremely grainy patches
}

AUTO_TUNER = {
    "msr_blend_range":      (0.3, 1.0),     # MSR blend: 30% for flat sediment, 100% for nodule-rich
    "clahe_clip_range":     (1.0, 2.0),     # CLAHE clip limit range
    "clahe_blend_range":    (0.3, 1.0),     # CLAHE blend: 30% for flat sediment, 100% for nodule-rich
    "clahe_tile_grid":      (8, 8),         # tile grid for CLAHE
    "unsharp_strength_range": (0.1, 0.5),   # unsharp: 0.1 for flat sediment, 0.5 for nodule-rich

    "bilateral_d":          7,
    "bilateral_sigma_color_range": (30, 60),  # scaled by noise estimate — capped at 60 to avoid
    "bilateral_sigma_space_range": (30, 60),  # blending across nodule–sediment boundaries

    "block_size_range":     (11, 51),        # adaptive threshold block size
    "c_offset_range":       (2, 15),         # constant subtracted from mean

    "morph_open_range":     (1, 3),          # opening kernel — must be tiny for grain nodules
    "morph_close_range":    (5, 11),         # closing kernel — solidify scattered detections
                                            # into coherent nodule blobs (was 3-7, too small)

    "contour_area_min_range": (2, 8),    # [low-noise min, high-noise min]
    "max_contour_area":       5000,      # fixed — physical upper bound
    "eccentricity_range":  (0.80, 0.90),  # [low-noise max, high-noise max]
    "solidity_range":      (0.45, 0.65),
    "circularity_range":   (0.25, 0.40),
}

PREPROCESSING = {
    "filter_chain": [
        "gray_world_white_balance",
        "multi_scale_retinex",         # illumination/shading removal (safe for divots)
        "clahe_lab",                   # gentle local contrast enhancement
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

PROXY_LABEL = {
    "tophat_radii":             [1, 2, 4, 8, 12, 16, 20],

    "texture_sigma":            2.0,
    "texture_threshold":        18.0,

    "lcr_bg_sigma":             30.0,

    "score_threshold":          4.0,

    "score_percentile_range":   (70, 90),
    "dense_score_threshold_min": 3.0,   # lowest abs threshold for very dense patches
    "dense_frac_trigger":        0.01,  # fraction of pixels above score_threshold
                                        # that triggers the dense-patch path

    "min_local_contrast":       0.02,
    "shape_bypass_score_mult":  2.0,
}

LOGGING = {
    "save_intermediate_steps":  True,       # master toggle
    "max_step_log_patches":     5,          # random subset of patches to log
    "composite_grid_cols":      4,          # columns in the summary grid
}

# input_mode: "rgb" → preprocessed BGR→RGB, "engineered" -> [L, Sobel, LCR]
MODEL = {
    "architecture":     "Unet",
    "encoder_name":     "resnet34",
    "encoder_weights":  "imagenet",
    "in_channels":      3,
    "classes":          1,
    "input_mode":       "engineered",
}

TRAINING = {
    "batch_size":       16,
    "num_epochs":       150,
    "learning_rate":    1e-4,
    "weight_decay":     1e-5,
    "train_split":      0.8,
    "val_split":        0.1,
    "test_split":       0.1,
    "random_seed":      42,

    # Early stopping — halt training when val Dice stops improving
    "early_stopping_patience": 15,

    # ReduceLROnPlateau scheduler — cuts LR when val Dice plateaus
    "scheduler_patience":  7,       # epochs without improvement before LR drop
    "scheduler_factor":    0.5,     # multiply LR by this factor on plateau

    # Loss
    "bce_weight":       0.5,
    "dice_weight":      0.5,

    "encoder_lr_multiplier": 1.0,     # uniform LR for baseline — match working Unet run

    # DataLoader workers (set 0 for debugging, 4+ for GPU training)
    "num_workers":      4,

    # Augmentation toggle (disable for ablation studies)
    "augmentation":     True,

    # Copy-Paste augmentation (Ghiasi et al. CVPR 2021)
    "copy_paste":                 True,
    "copy_paste_p":               0.5,   # probability per training example
    "copy_paste_max_objects":     3,     # up to N nodules pasted per call
    "copy_paste_min_coverage":    2.0,   # only mine from patches with coverage >= this %
}

INFERENCE = {
    "threshold_override":       None,
    "probability_threshold":    0.5,
}

METRICS = {
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
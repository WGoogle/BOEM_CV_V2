# Project Memory File

**Read this file first before doing any work.**

---

## 1. PROJECT OVERVIEW

**Project**: Polymetallic Nodule Segmentation Pipeline  
**Owner**: Brian Hwang (GitHub: WGoogle/BOEM_CV)  
**Affiliation**: BOEM (Bureau of Ocean Energy Management) research project  
**Date started with Claude**: March 24, 2026

### What it does
Ingests massive `.tif` seafloor mosaic strips captured by AUVs (Autonomous
Underwater Vehicles), segments polymetallic nodules from the sediment
background, and calculates nodule density (nodules/m²) and coverage (%).

### The core challenge
- Mosaics are enormous continuous strips (tens of thousands of pixels long)
- Lighting, sediment type, depth, and camera altitude vary along the AUV track
- Nodules are small, dark, roughly circular blobs on a gray sediment background
- Sediment texture (grain noise) is easily confused with nodules
- A single set of preprocessing parameters cannot work for the entire mosaic

---

## 2. WHAT THE IMAGERY LOOKS LIKE

Studied from two reference images the user provided:

1. **Raw seafloor mosaic**: Very low contrast (grayscale std ≈ 5), gray
   sediment with scattered dark specks (nodules ~3-10cm diameter in real
   world, appear as dark irregular blobs ~20-3000 pixels in area).  Also has
   bright white specks (likely marine snow or sensor artifacts).  Sediment has
   visible grain texture at fine scale.

2. **CVAT ground truth**: Same mosaic with polygon annotations in CVAT
   (app.cvat.ai).  Labels are "POLYMETALLIC NODULE" with semi-auto
   annotation.  Nodules appear as dark, compact, roughly circular shapes
   outlined with white polygons.

**Key visual characteristics**:
- Nodules are **darker** than surrounding sediment
- Nodules have **smooth surfaces** (low local texture) vs. grainy sediment
- Nodules are **compact/circular** (eccentricity < 0.8, circularity > 0.45)
- Background sediment has **high fine-scale texture** (useful for rejection)
- Illumination is **uneven** (AUV light cone creates falloff at edges)

---

## 3. PREDECESSOR CODEBASE (BOEM_CV)

The original repo at `github.com/WGoogle/BOEM_CV` had:
- A monolithic `1_preprocess_and_label.py` (472 lines) that mixed runner
  logic with CV code
- `preprocessing/preprocessing/preprocess.py` — MosaicPreprocessor class
- `preprocessing/preprocessing/proxy_labels.py` — ProxyLabelGenerator class
- `config.py` with static parameters (no auto-tuning)
- CoralNet-inspired manifest/idempotency pattern (JSON state tracking)
- Double-nested module paths: `preprocessing.preprocessing.preprocess`

### Techniques from BOEM_CV that we ported and kept:
1. Gray-world white balance
2. CLAHE in LAB color space (L channel only)
3. Bilateral filtering (edge-preserving denoise)
4. **Nodule boost** via morphological bottom-hat transform + texture gate
5. **Sediment fade** — blends bright regions toward smooth background
6. Unsharp mask on L channel
7. Multi-scale black top-hat for proxy label generation
8. Percentile-based thresholding with hard floor
9. Absolute intensity gate (only darkest pixels can be nodules)
10. Contour shape filtering (area, eccentricity, solidity, circularity)

### What we changed / improved (V2 overhaul):
- **Eliminated double-nested module paths** (`preprocessing.preprocessing.x`
  → `preprocessing.x`)
- **Added intelligent patching** — mosaic is split into overlapping 1024px
  patches before any processing
- **Added per-patch auto-tuning** — histogram analysis dynamically computes
  CLAHE clip limit, bilateral sigmas, threshold parameters per patch
- **Auto-tuner masks black borders** — all diagnostic signals (contrast,
  noise, skew, uniformity) are computed on valid pixels only (gray > 10),
  preventing mosaic border regions from contaminating parameter estimates
- **Added illumination normalization** — divides L channel by large-scale
  Gaussian blur (σ=51px) to flatten AUV light-cone gradients before CLAHE
- **Fixed nodule boost formula** — old version squared the bottom-hat
  (bh²/255), producing <1 level darkening.  New version soft-thresholds at
  P60 then applies linear boost, achieving -6 to -9 levels on nodule pixels
- **Fixed sediment fade** — old static L-threshold (80) brightened nodules
  by +12-14 levels.  New adaptive threshold (patch median L) with wider
  ramp (40 levels) and minimal blur ensures dark pixels are untouched (+2)
- **Edge-selective unsharp mask** — Sobel-gated sharpening only enhances
  real boundaries (nodule edges), not sediment grain texture
- **Added step-by-step intermediate logging** — every filter step saves a
  numbered image + composite grid for visual debugging
- **Strict separation**: runner script contains zero CV logic; all feature
  code lives in `preprocessing/` module
- **Quality gate** on patches rejects black borders and featureless tiles

---

## 4. CURRENT REPOSITORY STRUCTURE

```
nodule_segmentation/
├── context.md             ← THIS FILE (read first)
├── config.py                     ← All parameters, paths, feature flags
├── 1_preprocess_and_label.py     ← Step 1 runner (thin — no CV logic)
├── 2_train.py                    ← Step 2 placeholder (U-Net training)
├── 3_inference.py                ← Step 3 placeholder (sliding window)
├── preprocessing/
│   ├── __init__.py               ← Public API exports
│   ├── patcher.py                ← MosaicPatcher: split/reassemble large TIFs
│   ├── auto_tuner.py             ← PatchAutoTuner: per-patch parameter calc
│   ├── filters.py                ← FilterPipeline + all CV filters + proxy labels
│   └── geo_resolution.py         ← extract_meters_per_pixel from GeoTIFF metadata
├── data/
│   └── raw_mosaics/              ← Input .TIF/.PNG mosaics go here
└── outputs/
    ├── preprocessed/             ← Per-mosaic subdirs with preprocessed patches
    ├── proxy_labels/             ← Full-mosaic proxy masks + overlays
    ├── patches/                  ← Per-mosaic subdirs: images/ + masks/ + manifest
    ├── step_by_step_logs/        ← Per-patch intermediate image sequences
    ├── checkpoints/              ← Model checkpoints (Step 2)
    ├── results/                  ← Inference outputs (Step 3)
    └── logs/                     ← Pipeline log files
```

---

## 5. HOW THE PIPELINE WORKS (Step 1)

When `python 1_preprocess_and_label.py` runs:

```
For each mosaic in data/raw_mosaics/:
  1. LOAD mosaic via OpenCV (fallback: tifffile for >2GB TIFs)
  1b. EXTRACT meters_per_pixel from GeoTIFF metadata (tags 34264 or 33550)
      - Handles geographic CRS (degree→metre conversion at local latitude)
      - Handles projected CRS (already in metres)
      - Falls back to config.METRICS["meters_per_pixel"] for non-GeoTIFF files
  2. PATCH into overlapping 1024px tiles (stride = 1024 - 128 = 896px)
     - Quality gate rejects: black borders, featureless tiles (std < 3.0)
  3. For each valid patch:
     a. AUTO-TUNE: Analyze L-channel histogram (black borders masked) → compute:
        - CLAHE clip limit (inverse of contrast IQR)
        - Bilateral sigmas (proportional to noise MAD)
        - Threshold block size + C-offset (from brightness skewness)
        - Morph kernel sizes (from illumination uniformity)
     b. FILTER CHAIN (configurable order in config.py):
        00_original → 01_gray_world → 02_illumination_normalize →
        03_clahe_lab → 04_bilateral → 05_nodule_boost →
        06_sediment_fade → 07_unsharp_mask
     c. PROXY LABEL generation:
        01_grayscale → 02_gaussian_blur → 03_tophat_response →
        04_thresholded → 05_intensity_gated → 06_morph_cleaned →
        07_proxy_mask → 08_overlay
     d. SAVE: preprocessed patch, proxy mask, step-by-step images
  4. REASSEMBLE full-mosaic proxy mask (average overlapping regions)
  5. SAVE overlay visualization + patch manifest JSON
  6. UPDATE pipeline manifest (CoralNet-style audit trail)
```

---

## 6. KEY DESIGN DECISIONS & RATIONALE

### Why patch-level processing?
Different regions of the mosaic have different lighting, sediment, and depth.
A CLAHE clip of 1.5 might be perfect for a well-lit center patch but terrible
for a dark edge patch.  Auto-tuning per patch solves this.

### Why the quality gate threshold is low (min_std=3.0)?
Deep-sea imagery is inherently low-contrast.  The test image had grayscale
std ≈ 5.0 across the whole image, and individual patches had std ≈ 4.8-5.0.
A threshold of 5.0 rejected all valid patches.  3.0 is safe for real seafloor
data while still rejecting truly featureless tiles.

### Why bottom-hat instead of Gaussian background subtraction?
The original BOEM_CV code found that Gaussian background subtraction creates
halo artifacts around nodules.  Bottom-hat (morphological close − original)
responds only to dark compact blobs smaller than the structuring element,
with no halo.

### Why texture gate?
Sediment grain texture can trigger the top-hat response at fine scales.
Computing local standard deviation at σ=2px and ramping the response to zero
where texture is high effectively suppresses sediment false positives while
preserving smooth-surfaced nodules.  Threshold raised from 12→18 after
analysis showed the original value over-suppressed the nodule boost signal;
actual sediment texture scores are mean 2.1, P90 2.7 — the gate only needs
to catch outliers, not blanket-suppress.

### Why mask black borders in auto-tuning?
AUV mosaic patches have 15-25% black border pixels from the strip geometry.
Including these zeros in diagnostic signals caused: contrast IQR inflated
(saw 13-20 instead of real 6-12), skewness wildly negative (-1.5 to -5.4
instead of -0.35), illumination uniformity inflated (34-44 instead of 3-8).
This made CLAHE clip max out at 4.0 on nearly every patch regardless of
actual content.  Masking pixels below gray=10 gives accurate per-patch
diagnostics.

### Why illumination normalization before CLAHE?
AUV light cones create 7-11% illumination gradients across individual
patches.  On imagery where total nodule-sediment contrast is only 10-15
levels, this gradient dominates the signal.  Dividing L by a large-scale
Gaussian blur (σ=51px) removes the gradient while preserving local contrast.
Placed before CLAHE so the equaliser works on uniform illumination.  Black
border pixels are filled with the valid-pixel mean before blurring to prevent
edge artefacts, then restored to 0 afterward.

### Why soft-threshold the nodule boost?
The raw bottom-hat response has mean ~8-9 even on background sediment (from
grain texture and small dark features).  The old formula squared this
(bh²/255), which killed the signal — 0.6 levels mean darkening, invisible.
The new formula subtracts the 60th-percentile floor, so background (bh < floor)
gets zero darkening while actual nodules (bh=15-40) get -6 to -9 levels.
This targeted approach darkens only real compact blobs.

### Why adaptive sediment fade threshold?
The old hardcoded L-threshold (80) was below the median brightness of most
patches after CLAHE, causing the fade to brighten nodule pixels by +12-14
levels — actively destroying the contrast CLAHE created.  Using the per-patch
median L as threshold guarantees only the brighter half (sediment) can be
faded.  A wider ramp (40 vs 20 levels) and reduced mask blur (σ=2 vs 5)
prevent bleed into dark pixels at nodule-sediment boundaries.

### Why edge-selective unsharp mask?
Naive unsharp masking amplifies both nodule edges and sediment grain equally.
The edge-selective version computes a Sobel gradient magnitude and gates the
sharpening: only pixels with above-median edge strength receive the full
boost.  This expands dynamic range at real nodule boundaries (DR 71→81,
90→105) without amplifying fine-scale texture that could confuse a U-Net.

### Why auto-extract meters_per_pixel from GeoTIFF?
A hardcoded `meters_per_pixel` is fragile — different surveys, cameras, or
altitudes produce different ground-sample distances.  GeoTIFF files from AUV
mosaic software (e.g. Hypack, QGIS) embed the spatial resolution in tags
33550 (ModelPixelScaleTag) or 34264 (ModelTransformationTag).  Reading it
directly from the file eliminates a manual configuration step and ensures
density calculations (nodules/m²) are always correct.  For geographic CRS
files (EPSG:4326, units in degrees), the code converts to metres using the
local latitude from the same metadata.  The config value serves as a
fallback for plain PNG/TIFF files that lack geo metadata.

### Why pre-CLAHE L channel for nodule boost?
CLAHE creates bright halos around dark nodule edges.  The bottom-hat
transform reads these halos as part of the "bright background", reducing
the response.  Using the L channel captured before CLAHE avoids this.

### Why 232 nodules in the test image?
The test image (raw seafloor screenshot, 1040×2000px) contains many small
dark specks.  232 is the count after all filtering (area 50-3000px,
circularity > 0.45, solidity > 0.6, eccentricity < 0.8).  This number will
change dramatically with real full-resolution TIF mosaics.  The contour
shape filters in config.py are the main tuning knobs for false positive rate.

---

## 7. MODULE API REFERENCE

### `preprocessing.patcher.MosaicPatcher`
```python
patcher = MosaicPatcher(patch_size=1024, overlap=128, min_std=3.0, ...)
mosaic = patcher.load_mosaic(Path("image.tif"))
patches, infos = patcher.extract_patches(mosaic)  # List[ndarray], List[PatchInfo]
full_map = patcher.reassemble(outputs, infos, (H, W))  # for inference
```

### `preprocessing.auto_tuner.PatchAutoTuner`
```python
tuner = PatchAutoTuner(config.AUTO_TUNER)
params = tuner.analyse(patch_bgr)  # returns TunedParams dataclass
# params.clahe_clip_limit, params.bilateral_sigma_color, etc.
```

### `preprocessing.geo_resolution.extract_meters_per_pixel`
```python
from preprocessing import extract_meters_per_pixel

mpp = extract_meters_per_pixel(Path("image.tif"), fallback=0.005)
# Returns float (e.g. 0.005015) from GeoTIFF, or fallback for PNG/non-geo TIF
# Logged automatically: "GeoTIFF resolution = 0.005015 m/px (5.02 mm/px) at lat -15.657°"
```

### `preprocessing.filters.FilterPipeline`
```python
pipeline = FilterPipeline(config.PREPROCESSING)
preprocessed, steps = pipeline.run(patch_bgr, params)
# steps = [("00_original", img), ("01_gray_world_white_balance", img),
#          ("02_illumination_normalize", img), ("03_clahe_lab", img), ...]
FilterPipeline.save_step_images(steps, output_dir, prefix="patch_0001")
```

### `preprocessing.filters.generate_proxy_label`
```python
mask, steps, stats = generate_proxy_label(preprocessed, params, config.PROXY_LABEL)
# mask: (H,W) uint8 binary, steps: intermediate images, stats: dict
```

---

## 8. CONFIG.PY SECTIONS

| Section | Used by | Key parameters |
|---------|---------|----------------|
| `PATCHING` | `patcher.py` | `patch_size`, `overlap`, `min_std`, `max_black_fraction` |
| `AUTO_TUNER` | `auto_tuner.py` | `clahe_clip_range`, `bilateral_sigma_*_range`, `block_size_range`, `tophat_radii`, contour filters |
| `PREPROCESSING` | `filters.py` | `filter_chain` (ordered list), `illum_norm_sigma`, `sediment_fade_*`, `unsharp_*` |
| `PROXY_LABEL` | `filters.py` | `gaussian_*`, `adaptive_abs_*` |
| `LOGGING` | `1_preprocess_and_label.py` | `save_intermediate_steps`, `log_every_n_patches` |
| `MODEL` | `2_train.py` (future) | U-Net architecture params |
| `TRAINING` | `2_train.py` (future) | batch_size, lr, epochs, splits |
| `INFERENCE` | `3_inference.py` (future) | threshold, blend_mode |
| `METRICS` | `3_inference.py` (future) | `meters_per_pixel` (auto-extracted from GeoTIFF; config value used as fallback for non-geo files) |

---

## 9. WHAT'S NOT BUILT YET

### Step 2: Training (`2_train.py`)
- U-Net with ResNet34 encoder (pretrained ImageNet)
- Hybrid BCE + Dice loss
- Augmentation: brightness/contrast, blur, rotation, flips, elastic
- Early stopping + LR scheduling
- Train/val/test split from patch manifest
- Manual label override: if `data/manual_labels/{patch_id}.png` exists,
  it replaces the proxy label (CoralNet pattern: explicit > inferred)

### Step 3: Inference (`3_inference.py`)
- Sliding window over full mosaic using `MosaicPatcher.reassemble()`
- Probability map blending (average overlap regions)
- Binary mask at configurable threshold
- Metrics: nodule count, density (nodules/m²), coverage (%), size distribution
- Per-mosaic JSON + aggregated dataset summary
- Overlay visualizations

### Other future work
- Integration with CVAT for annotation import/export
- Per-patch lat/lon from GeoTIFF transformation matrix (spatial metadata
  already extracted for resolution; extending to per-patch coordinates)
- Multi-GPU batch processing
- Confidence calibration on proxy labels vs. manual labels

---

## 10. KNOWN ISSUES & TUNING NOTES

1. **232 detections on test image seems high** — the test image was a
   low-res screenshot, not a real full-res mosaic.  With real data, adjust:
   - `min_contour_area` (raise to 100+ to reject tiny noise)
   - `tophat_percentile` (raise to 97-98 for stricter thresholding)
   - `min_circularity` (raise to 0.50+ to reject elongated artifacts)

2. **Patch size tradeoff**: 1024px gives good local context but may be too
   large for very non-uniform lighting.  Try 512px for problem areas.

3. **`meters_per_pixel`** is now auto-extracted from GeoTIFF metadata at
   load time (via `preprocessing/geo_resolution.py`).  The config default
   of 0.005 (5 mm/px) is used only as a fallback for non-GeoTIFF files.
   Verified against `CameraMosaic_D1_Node3_L8.tif`: extracted 0.005015 m/px,
   consistent with the config default.  The extracted value and coverage
   dimensions are logged and saved in the pipeline manifest.

4. **tifffile fallback**: For TIFs > 2GB that OpenCV can't load, install
   `pip install tifffile`.  The patcher auto-detects and falls back.

5. **The filter chain is configurable**: Remove steps from
   `config.PREPROCESSING["filter_chain"]` to disable them.  Order matters.

---

## 11. EXTERNAL REFERENCES

- **Original repo**: https://github.com/WGoogle/BOEM_CV
- **Architecture inspiration**: https://github.com/coralnet/coralnet
  (CoralNet's idempotent manifest pattern, modular separation)
- **CoMoNoD algorithm** (traditional nodule delineation):
  https://www.nature.com/articles/s41598-017-13335-x
- **BOEM Symposium poster** by Brian Hwang, Kailash Ramesh, Thang Nguyen (2026)
- **Annotation tool**: CVAT (app.cvat.ai), task ID 2051496

---

*Last updated: March 26, 2026 — preprocessing overhaul: illumination normalization,
nodule boost fix, adaptive sediment fade, edge-selective unsharp, auto-tuner
black border masking, GeoTIFF resolution extraction*
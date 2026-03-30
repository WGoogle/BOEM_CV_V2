# Project Memory File

**Read this file first before doing any work.**

---

## 1. PROJECT OVERVIEW

**Project**: Polymetallic Nodule Segmentation Pipeline  
**Owner**: Brian Hwang (GitHub: WGoogle/BOEM_CV)  
**Affiliation**: BOEM (Bureau of Ocean Energy Management) research project  

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
   *(kept in code but removed from default filter chain — see V2 overhaul)*
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
- **Replaced illumination_normalize + nodule_boost with Multi-Scale Retinex
  (MSR)** — the old approach (divide-by-blur illumination normalization →
  aggressive CLAHE → bottom-hat nodule boost) falsely darkened gray sediment
  divots and bumps into dark blobs resembling nodules.  MSR separates
  reflectance from illumination in log domain (`log(R) = log(I) - log(blur(I))`)
  at multiple Gaussian scales, intrinsically removing both large-scale AUV
  light-cone gradients and medium-scale shading from 3D surface relief
  (divots/bumps) while preserving true reflectance differences (dark nodule
  material vs. bright sediment).  Followed by gentle CLAHE (clip max 1.5)
  to recover local contrast for downstream detection.
- **Nodule boost removed from filter chain** — bottom-hat transform cannot
  distinguish reflectance darkness (real nodules) from shading darkness
  (divot shadows).  MSR handles the illumination separation that nodule_boost
  was attempting.  The function remains in code for optional re-enablement.
- **CLAHE clip range reduced from (1.0, 4.0) to (1.0, 1.5)** — MSR handles
  illumination normalization; CLAHE now only needs to provide gentle local
  contrast enhancement, not compensate for lighting gradients.
- **Fixed sediment fade** — old static L-threshold (80) brightened nodules
  by +12-14 levels.  New adaptive threshold (patch median L) with wider
  ramp (40 levels) and minimal blur ensures dark pixels are untouched (+2)
- **Edge-selective unsharp mask** — Sobel-gated sharpening only enhances
  real boundaries (nodule edges), not sediment grain texture
- **Added step-by-step intermediate logging** — every filter step saves a
  numbered image + composite grid for visual debugging
- **Multi-feature composite proxy labelling** (V2.1) — replaced single-channel
  top-hat proxy labelling with multi-feature composite scoring:
  1. Top-hat (primary signal): multi-scale black top-hat with texture gate
  2. Local Contrast Ratio (LCR gate): darkness relative to local background,
     adapts to illumination variations preprocessing didn't fully remove
  3. Difference of Gaussians (DoG, supplementary): scale-normalized blob
     detection that catches nodules at scales the top-hat misses
  4. Smoothness (soft gate): fine vs. coarse local variance ratio;
     suppresses sediment grain texture responses
  Combined via multiplicative gating: tophat * (1 + lcr_gain*LCR) *
  smoothness^power + dog_blend * DoG (same gates).  This ensures a pixel
  must be BOTH morphologically blob-like AND locally dark AND smooth.
- **Grain-size nodule support** — top-hat radii extended to [1,2,4,8,12]
  (from [12,20,30]) to detect 5-15mm grain nodules that are only 1-3px
  at 5mm/px resolution.  Pre-blur reduced to σ=1.5 (from σ=5), contour
  min_area lowered to 3px² (from 50), morph kernels shrunk to 1-3px opening.
- **Size-aware contour filtering** — grain-size contours (<20px²) skip
  unreliable shape checks (eccentricity, circularity, solidity) since
  pixelated blobs can't produce meaningful shape metrics.  Only area +
  local contrast are checked.  Larger contours get full shape filtering.
- **Watershed separation** for touching nodules — distance transform →
  local maxima (nodule centers) → marker-controlled watershed to split
  clusters into individual detections.
- **Enhanced contour filtering** — two new checks beyond shape:
  1. Local contrast: contour interior must be darker than local background
  2. Boundary gradient: mean Sobel magnitude along contour boundary must
     exceed threshold (real edges produce strong gradients; noise doesn't)
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
        00_original → 01_gray_world_white_balance →
        02_multi_scale_retinex → 03_clahe_lab → 04_bilateral_denoise →
        05_sediment_fade → 06_unsharp_mask
     c. PROXY LABEL generation (multi-feature composite scoring):
        01_grayscale → 02_gaussian_blur →
        03_tophat (multi-scale black top-hat + texture gate) →
        04_local_contrast (LCR: darkness vs. local background) →
        05_dog_blobs (Difference of Gaussians blob detection) →
        06_smoothness (fine vs. coarse local variance) →
        07_composite_score (multiplicative gating: tophat * LCR_boost * smooth_gate + DoG) →
        08_thresholded (percentile-based) →
        09_intensity_gated (adaptive dark-pixel gate) →
        10_morph_cleaned → 11_watershed_split (separate touching nodules) →
        12_proxy_mask (size-aware contour filtering) → 13_overlay
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

### Why Multi-Scale Retinex (MSR) instead of illumination_normalize + nodule_boost?
The original V2 pipeline used: illumination_normalize (divide by blur) →
CLAHE (aggressive, clip up to 4.0) → nodule_boost (bottom-hat + texture gate).
This chain falsely darkened gray sediment divots and 3D bumps into dark blobs
that looked like nodules in the final output.  The root cause:

- **illumination_normalize** (divide-by-blur) only removes large-scale
  gradients, not medium-scale shading from 3D surface relief
- **CLAHE at high clip** amplifies all local contrast equally — real nodule
  darkness AND shading from divots/bumps
- **Bottom-hat (nodule_boost)** responds to any dark compact feature smaller
  than its structuring element — cannot distinguish reflectance darkness
  (nodules) from shading darkness (divot shadows)
- **Texture gate** fails for divots because their shadows are smooth
  (low texture), just like real nodules

MSR solves this at the root: in log domain, `log(R) = log(I) - log(blur(I))`
intrinsically separates reflectance (material property) from illumination
(lighting + 3D shading).  Averaging across scales [5, 20, 80] removes both
fine-scale shading and large-scale gradients.  After MSR, CLAHE only needs
gentle local contrast enhancement (clip max 1.5), and nodule_boost is
unnecessary because MSR already preserves true reflectance differences.

**MSR implementation details**:
- Operates on L channel in LAB colour space
- Black borders (L < 8) filled with valid-pixel mean before blurring
- Output normalized to [0, 255] using P1-P99 percentile stretch
- Stats-matching: output mean/std are matched to original valid-pixel
  statistics so downstream filters (sediment fade, proxy labels) remain
  calibrated
- Optional gain parameter (default 1.0) for contrast scaling

### Why bottom-hat is kept in code but removed from the chain?
The bottom-hat transform + texture gate (`nodule_boost`) remains in
`filters.py` and `_FILTER_REGISTRY` for optional re-enablement via
`config.PREPROCESSING["filter_chain"]`.  It is not in the default chain
because MSR handles illumination/reflectance separation more cleanly.
The bottom-hat approach may still be useful for specific scenarios with
extremely low-contrast nodules where extra darkening is needed.

### Why texture gate in proxy label generation?
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

### Why conservative CLAHE (clip max 1.5) after MSR?
MSR removes illumination/shading but its stats-matching step compresses
the output to the original brightness distribution, which is inherently
low-contrast.  Gentle CLAHE (clip 1.0–1.5) recovers local contrast for
nodule detection without re-introducing the shading amplification that
aggressive CLAHE (clip up to 4.0) caused in the old pipeline.

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
*(Relevant only when nodule_boost is re-enabled in the filter chain.)*
CLAHE creates bright halos around dark nodule edges.  The bottom-hat
transform reads these halos as part of the "bright background", reducing
the response.  Using the L channel captured before CLAHE avoids this.

### Why multi-feature composite scoring instead of single top-hat?
The original single-channel top-hat approach has blind spots:
1. **Fixed scale** — top-hat at radii [12,20,30] misses grain nodules
   (1-3px at 5mm/px resolution) and can't adapt to multi-scale input.
2. **No local context** — the absolute intensity gate uses a global
   percentile, missing nodules in locally bright regions.
3. **No touching-nodule separation** — clusters merge into one contour
   that gets rejected by shape filters.
4. **Additive scoring doesn't work** — features like smoothness score
   high everywhere (P50=0.70), drowning out selective features when
   combined additively.

Multiplicative gating solves this: tophat * LCR_boost * smooth_gate
ensures a detection must score high on ALL features simultaneously.
The DoG supplementary channel catches blobs at unexpected scales.

### Why size-aware contour filtering?
At 5mm/px, polymetallic nodules span a huge pixel-size range:
- **Grain (5-15mm)**: 1-3px diameter, 3-7px² area
- **Medium (25-50mm)**: 5-10px diameter, 20-78px² area
- **Large (40-90mm)**: 8-18px diameter, 50-254px² area

A 3px² contour has too few pixels for meaningful eccentricity,
circularity, or solidity measurements — these metrics are dominated
by pixelation artifacts at this scale.  The pipeline uses a 20px²
threshold: below it, only area + local contrast are checked;
above it, full shape filtering (eccentricity, circularity, solidity,
boundary gradient) is applied.

### Why multiplicative gating over additive weighting?
Early testing showed that additive weighted combination
(score = w1*f1 + w2*f2 + ...) fails because non-selective features
(smoothness P50=0.70, normalized top-hat P50=0.25) elevate the
composite baseline everywhere.  The 94th percentile of positive scores
then cuts at ~0.53, fragmenting detections into tiny disconnected pixels.

Multiplicative gating (tophat * LCR_boost * smooth_gate) naturally
produces near-zero scores where ANY feature is low, concentrating
high scores at true nodule locations.  The proven percentile threshold
from V1 (96th percentile of positive values with hard floor) then
works correctly.

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
#          ("02_multi_scale_retinex", img), ("03_clahe_lab", img),
#          ("04_bilateral_denoise", img), ("05_sediment_fade", img),
#          ("06_unsharp_mask", img)]
FilterPipeline.save_step_images(steps, output_dir, prefix="patch_0001")
```

### `preprocessing.filters.generate_proxy_label`
```python
mask, steps, stats = generate_proxy_label(preprocessed, params, config.PROXY_LABEL)
# mask: (H,W) uint8 binary [0, 255]
# steps: 13 intermediate images (grayscale → blur → 4 features → composite →
#         threshold → intensity gate → morph → watershed → mask → overlay)
# stats: dict with candidates, nodules, coverage, rejections, feature params
```

---

## 8. CONFIG.PY SECTIONS

| Section | Used by | Key parameters |
|---------|---------|----------------|
| `PATCHING` | `patcher.py` | `patch_size`, `overlap`, `min_std`, `max_black_fraction` |
| `AUTO_TUNER` | `auto_tuner.py` | `clahe_clip_range`, `bilateral_sigma_*_range`, `block_size_range`, `tophat_radii`, contour filters |
| `PREPROCESSING` | `filters.py` | `filter_chain` (ordered list), `msr_sigmas`, `msr_gain`, `sediment_fade_*`, `unsharp_*` |
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

1. **Nodule size ranges vs. resolution** — at 5mm/px (BOEM D1 survey):
   - Grain (5-15mm) = 1-3px diameter, 3-7px² area — at resolution limit
   - Medium (25-50mm) = 5-10px diameter, 20-78px² area — reliably detectable
   - Large (40-90mm) = 8-18px diameter, 50-254px² area — easily detected
   Single-pixel (5mm) grain nodules cannot be reliably distinguished from
   sensor noise at this resolution.  `min_contour_area=3` catches 10mm+.

2. **205 detections on D1 Node3 L8** — multi-feature composite scoring
   with grain-size support.  Old pipeline detected 64 on same mosaic using
   single-channel top-hat with min_area=50 (blind to grain nodules).

3. **Patch size tradeoff**: 1024px gives good local context but may be too
   large for very non-uniform lighting.  Try 512px for problem areas.

4. **`meters_per_pixel`** is now auto-extracted from GeoTIFF metadata at
   load time (via `preprocessing/geo_resolution.py`).  The config default
   of 0.005 (5 mm/px) is used only as a fallback for non-GeoTIFF files.
   Verified against `CameraMosaic_D1_Node3_L8.tif`: extracted 0.005015 m/px.

5. **tifffile fallback**: For TIFs > 2GB that OpenCV can't load, install
   `pip install tifffile`.  The patcher auto-detects and falls back.

6. **The filter chain is configurable**: Remove steps from
   `config.PREPROCESSING["filter_chain"]` to disable them.  Order matters.

7. **Divot/bump false darkening** — was the primary issue in the old pipeline.
   Solved by replacing with MSR + gentle CLAHE.  If divot false positives
   reappear, check that CLAHE clip max hasn't been raised above ~1.5.

8. **MSR grain amplification** — at very fine Gaussian scales (σ < 10), MSR's
   log-domain division amplifies sediment grain texture.  The current sigmas
   [5, 20, 80] include σ=5 which can produce some texture amplification, but
   this is controlled by the stats-matching step and gentle CLAHE.

9. **Tuning the proxy label sensitivity** — to catch more/fewer nodules:
   - `tophat_percentile`: lower = more detections (try 92-96)
   - `tophat_threshold_floor`: lower = catch weaker responses (try 5-15)
   - `min_contour_area`: lower = catch smaller grain nodules (min 1)
   - `adaptive_abs_percentile`: higher = allow lighter pixels (try 15-30)
   - `lcr_gain`: higher = more LCR boost for locally dark pixels (try 2-5)
   - `composite_smooth_sigma`: lower = sharper detections, higher = merge nearby

10. **Additive scoring pitfall** — early composite scoring used weighted
    addition (w1*f1 + w2*f2 + ...) which failed because smoothness (P50=0.70)
    and normalized top-hat (P50=0.25) elevate the baseline everywhere.  The
    fix was multiplicative gating: tophat * LCR_boost * smooth_gate.

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

*Last updated: March 27, 2026 — multi-feature composite proxy labelling with
grain-size nodule support; multiplicative gating (top-hat * LCR * smoothness + DoG);
watershed separation; size-aware contour filtering; top-hat radii [1,2,4,8,12];
min_contour_area=3; 205 detections on D1 Node3 L8 (up from 64 with old pipeline)*
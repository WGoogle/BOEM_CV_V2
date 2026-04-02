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
4. **Sediment fade** — blends bright regions toward smooth background
5. Unsharp mask on L channel
6. Multi-scale black top-hat for proxy label generation
7. Percentile-based thresholding with hard floor
8. Absolute intensity gate (only darkest pixels can be nodules)
9. Contour shape filtering (area, eccentricity, solidity, circularity)

### What we changed / improved (V2 overhaul):
- **Eliminated double-nested module paths** (`preprocessing.preprocessing.x`
  → `preprocessing.x`)
- **Added intelligent patching** — mosaic is split into overlapping 256px
  patches (32px overlap) before any processing
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
  material vs. bright sediment).  Followed by gentle CLAHE (clip max 2.0)
  to recover local contrast for downstream detection.
- **Nodule boost deleted entirely** — bottom-hat transform cannot
  distinguish reflectance darkness (real nodules) from shading darkness
  (divot shadows).  MSR handles the illumination separation that nodule_boost
  was attempting.  The function, config entries, and TunedParams fields have
  been fully removed.
- **CLAHE clip range reduced from (1.0, 4.0) to (1.0, 2.0)** — MSR handles
  illumination normalization; CLAHE now only needs to provide gentle local
  contrast enhancement, not compensate for lighting gradients.  Ceiling
  raised from 1.5→2.0 so the lowest-contrast patches get a meaningful lift.
- **Fixed sediment fade** — old static L-threshold (80) brightened nodules
  by +12-14 levels.  New adaptive threshold (patch median L) with wider
  ramp (40 levels) and minimal blur ensures dark pixels are untouched (+2)
- **Edge-selective unsharp mask** — Sobel-gated sharpening only enhances
  real boundaries (nodule edges), not sediment grain texture
- **Added step-by-step intermediate logging** — every filter step saves a
  numbered image + composite grid for visual debugging
- **Multi-feature proxy labelling** (V2.1→V2.2) — replaced the four-feature
  composite scoring (top-hat × LCR × smoothness + DoG) with a simpler,
  more robust two-feature approach:
  1. Top-hat (primary signal): multi-scale black top-hat with texture gate
  2. Local Contrast Ratio (LCR gate): darkness relative to local background
  Combined via **raw multiplication** (`tophat × LCR`) on absolute magnitudes
  (no per-patch normalization).  Real nodules score 9–42; noise scores <3.
  Thresholded by an **absolute score floor** (default 5.0) — patches with no
  real nodules never exceed it — plus a secondary percentile gate (85th) for
  dense-nodule patches.  Removed DoG and smoothness features (unnecessary
  complexity once the absolute-magnitude scoring was adopted).
  Removed watershed separation step (touching nodules handled adequately by
  contour filtering alone).
- **Top-hat radii extended** to [1, 2, 4, 8, 12, 16, 20] (from [1,2,4,8,12])
  to detect very large nodules/clusters (90–200mm = 18–40px at 5mm/px).
  Grain-size support retained at r=1-2.
- **Size-aware contour filtering** — grain-size contours (<20px²) skip
  unreliable shape checks (eccentricity, circularity, solidity) since
  pixelated blobs can't produce meaningful shape metrics.  Only area +
  local contrast are checked.  Larger contours get full shape filtering.
- **Noise-adaptive contour shape filters** — all four shape parameters
  (min_area, max_eccentricity, min_solidity, min_circularity) now scale
  with the auto-tuner's noise estimate.  Noisy patches get relaxed criteria
  (pixelated contours look irregular); clean patches get tighter filters.
  Configured via range tuples in config.py (e.g. `eccentricity_range`,
  `solidity_range`).
- **Contrast-adaptive enhancement blending** — MSR, CLAHE, and unsharp
  mask all share a contrast factor (`t = contrast_IQR / 40`, clamped 0–1).
  Low-contrast sediment-heavy patches receive reduced enhancement (blend
  as low as 30%) to avoid amplifying grain noise, while high-contrast
  nodule-rich patches receive full enhancement.  Controlled by
  `msr_blend_range`, `clahe_blend_range`, and `unsharp_strength_range`.
- **Bilateral filter tuning via grid search** — instead of linearly
  scaling sigma by noise estimate, the auto-tuner runs a grid search over
  sigma_color × sigma_space combinations [50–100] and picks the combo that
  minimises median local variance.  Bilateral d is also adaptive, derived
  from local variance percentiles.
- **Density-adaptive score percentile** — `score_percentile_range` (70–90)
  replaces the fixed percentile gate.  Sparse patches (few pixels above
  abs threshold) use the high end (strict); dense patches (≥5% of pixels
  above abs threshold) use the low end (relaxed) so more nodules survive.
- **Score-gated contour filtering** — contours whose mean combined score
  exceeds `shape_bypass_score_mult × threshold` (default 2×) are treated
  as high-confidence and skip strict shape filters (solidity, eccentricity,
  circularity).  Prevents strong detections from being killed by irregular
  blob shape at low pixel counts.
- **Bilateral filter sigma capped at 60** (from 75) to avoid blending
  across nodule–sediment boundaries.
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
├── test_patch_sizes.py           ← Patch size comparison tool (256/512/1024)
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
  2. PATCH into overlapping 256px tiles (stride = 256 - 32 = 224px)
     - Quality gate rejects: black borders, featureless tiles (std < 3.0)
  3. For each valid patch:
     a. AUTO-TUNE: Analyze L-channel histogram (black borders masked) → compute:
        - Contrast factor t = IQR/40 → drives MSR blend, CLAHE clip+blend, unsharp strength
        - Bilateral d + sigmas (grid search over sigma combinations, min variance)
        - Threshold block size + C-offset (from brightness skewness)
        - Morph kernel sizes (from illumination uniformity)
        - Contour shape filter thresholds (from noise estimate)
     b. FILTER CHAIN (configurable order in config.py):
        00_original → 01_gray_world_white_balance →
        02_multi_scale_retinex → 03_clahe_lab → 04_bilateral_denoise →
        05_sediment_fade → 06_unsharp_mask
     c. PROXY LABEL generation (top-hat × LCR scoring):
        01_grayscale →
        02_tophat_response (multi-scale black top-hat + texture gate) →
        03_local_contrast (LCR: darkness vs. local background) →
        04_combined_score (tophat × LCR, absolute magnitude) →
        05_thresholded (absolute floor + percentile gate) →
        06_morph_cleaned →
        07_proxy_mask (noise-adaptive contour filtering) → 08_overlay
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
for a dark edge patch.  Auto-tuning per 256px patch solves this.

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
gentle local contrast enhancement (clip max 2.0), and the old nodule_boost
was unnecessary because MSR already preserves true reflectance differences.

**MSR implementation details**:
- Operates on L channel in LAB colour space
- Black borders (L < 8) filled with valid-pixel mean before blurring
- Output normalized to [0, 255] using P1-P99 percentile stretch
- Stats-matching: output mean/std are matched to original valid-pixel
  statistics so downstream filters (sediment fade, proxy labels) remain
  calibrated
- Optional gain parameter (default 1.0) for contrast scaling
- **Adaptive blend**: MSR output is blended with the original L channel
  using `msr_blend` (0.3–1.0, scaled by contrast).  Low-contrast
  sediment patches get 30% retinex to avoid amplifying grain noise.

### Why texture gate in proxy label generation?
Sediment grain texture can trigger the top-hat response at fine scales.
Computing local standard deviation at σ=2px and ramping the response to zero
where texture is high effectively suppresses sediment false positives while
preserving smooth-surfaced nodules.  Threshold raised from 12→18 after
analysis showed the original value over-suppressed the top-hat signal;
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

### Why conservative CLAHE (clip max 2.0) after MSR?
MSR removes illumination/shading but its stats-matching step compresses
the output to the original brightness distribution, which is inherently
low-contrast.  Gentle CLAHE (clip 1.0–2.0) recovers local contrast for
nodule detection without re-introducing the shading amplification that
aggressive CLAHE (clip up to 4.0) caused in the old pipeline.  The ceiling
was raised from 1.5→2.0 so the lowest-contrast patches get a meaningful lift.
Additionally, CLAHE output is blended with original L using `clahe_blend`
(0.3–1.0, scaled by contrast) — low-contrast patches get only 30% CLAHE
to prevent grain amplification.

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

### Why top-hat × LCR instead of four-feature composite scoring?
The earlier V2.1 composite (tophat × LCR × smoothness + DoG) was
unnecessarily complex.  Two features are sufficient:
1. **Top-hat** detects compact dark blobs (the morphological signal)
2. **LCR** verifies they are actually darker than surroundings (the
   contrast signal)

Raw multiplication (`tophat × LCR`) on absolute magnitudes (no per-patch
normalization) produces a score with natural separation: real nodules
score 9–42, noise/artifacts score <3.  An **absolute score floor**
(default 5.0) replaces per-patch percentile as the primary gate — patches
with no real nodules never exceed it — eliminating the problem of
percentile thresholds finding "detections" in empty patches.  The
secondary percentile gate (85th) only activates in dense-nodule patches.

DoG and smoothness features were removed: DoG added marginal recall at
the cost of false positives; smoothness was redundant once the absolute
threshold gate was in place.

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

### Why absolute score threshold instead of percentile-only?
Per-patch percentile thresholds (e.g. 96th percentile of positive values)
always find "detections" — even in empty patches — because the percentile
adapts to whatever signal is present.  Using absolute-magnitude scoring
(raw tophat × raw LCR, no normalization) means patches with no real
nodules produce scores well below the absolute floor (5.0), yielding
zero detections correctly.  The percentile gate (85th) acts as a
secondary filter only in patches that have many pixels above the floor.

### Why noise-adaptive contour shape filters?
Contour shape metrics (eccentricity, circularity, solidity) are noisy
for pixelated contours in high-noise patches.  Fixed thresholds either
reject valid detections in noisy regions or accept false positives in
clean regions.  Scaling all four shape parameters with the auto-tuner's
noise estimate (0–1 normalized) resolves this: noisy patches get relaxed
criteria while clean patches get tighter filters.  The area floor also
scales up with noise to reject the extra tiny noise blobs that noisy
patches produce.

### Why score-gated contour filtering?
Contours with very high combined scores (≥ 2× the threshold by default)
are almost certainly real nodules.  Applying strict shape filters
(solidity, eccentricity, circularity) to these can reject them due to
irregular blob shape at low pixel counts — a 10px² blob can't produce
meaningful circularity.  The score gate skips shape checks for
high-confidence detections while still applying them to borderline ones.

### Why density-adaptive percentile thresholding?
A fixed percentile gate (e.g. 85th) works well for average-density
patches but is too strict for dense-nodule patches (rejects many real
nodules) and too lenient for sparse patches (still finds noise).  The
density-adaptive range (70–90) adjusts automatically: patches with many
pixels above the absolute threshold get a relaxed percentile (70th) so
more nodules survive, while sparse patches get a strict percentile (90th)
where the absolute floor already dominates.

### Why grid-search bilateral tuning?
Linear noise-proportional scaling of bilateral sigmas was too coarse —
sigma_color and sigma_space interact non-linearly with actual image
content.  The auto-tuner now tests a grid of (sigma_color, sigma_space)
combinations [50–100 in steps of 10] and selects the pair that minimises
median local variance.  Bilateral d is also adaptive, derived from the
90th percentile of local variance to ensure proper kernel diameter.

### Why 256px patches instead of 1024px?
Smaller patches improve local adaptation: each 256px tile has more
uniform lighting and sediment characteristics than a 1024px tile.
Auto-tuner diagnostics (contrast, noise, skew) are more accurate over
smaller homogeneous regions.  The trade-off is more boundary effects
(nodules split across edges), mitigated by overlap (32px = 12.5%).
The `test_patch_sizes.py` comparison tool allows empirical evaluation
of 256/512/1024 on any mosaic.

---

## 7. MODULE API REFERENCE

### `preprocessing.patcher.MosaicPatcher`
```python
patcher = MosaicPatcher(patch_size=256, overlap=32, min_std=3.0, ...)
mosaic = patcher.load_mosaic(Path("image.tif"))
patches, infos = patcher.extract_patches(mosaic)  # List[ndarray], List[PatchInfo]
full_map = patcher.reassemble(outputs, infos, (H, W))  # for inference
```

### `preprocessing.auto_tuner.PatchAutoTuner`
```python
tuner = PatchAutoTuner(config.AUTO_TUNER)
params = tuner.analyse(patch_bgr)  # returns TunedParams dataclass
# params.msr_blend, params.clahe_clip_limit, params.clahe_blend,
# params.bilateral_d, params.bilateral_sigma_color, params.unsharp_strength, etc.
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
# steps: 8 intermediate images (grayscale → tophat → LCR → combined_score →
#         threshold → morph → mask → overlay)
# stats: dict with candidates, nodules, coverage, rejections, feature params
```

---

## 8. CONFIG.PY SECTIONS

| Section | Used by | Key parameters |
|---------|---------|----------------|
| `PATCHING` | `patcher.py` | `patch_size` (256), `overlap` (32), `min_std`, `min_mean`, `max_black_fraction` |
| `AUTO_TUNER` | `auto_tuner.py` | `msr_blend_range`, `clahe_clip_range`, `clahe_blend_range`, `clahe_tile_grid`, `unsharp_strength_range`, `bilateral_d`, `bilateral_sigma_*_range`, `block_size_range`, `morph_*_range`, noise-adaptive contour filter ranges (`contour_area_min_range`, `eccentricity_range`, `solidity_range`, `circularity_range`) |
| `PREPROCESSING` | `filters.py` | `filter_chain` (ordered list), `msr_sigmas`, `msr_gain`, `sediment_fade_*`, `unsharp_*` |
| `PROXY_LABEL` | `filters.py` | `tophat_radii`, `texture_*`, `lcr_bg_sigma`, `score_threshold`, `score_percentile_range`, `min_local_contrast`, `shape_bypass_score_mult` |
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

3. **Patch size tradeoff**: 256px is the current default for better local
   adaptation.  Use `test_patch_sizes.py` to compare 256/512/1024 on any
   mosaic.  Larger patches give more context for background estimation
   (bg_sigma=30 covers ~180px, which is 70% of a 256px patch).

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
   reappear, check that CLAHE clip max hasn't been raised above ~2.0.

8. **MSR grain amplification** — at very fine Gaussian scales (σ < 10), MSR's
   log-domain division amplifies sediment grain texture.  The current sigmas
   [5, 20, 80] include σ=5 which can produce some texture amplification, but
   this is controlled by the stats-matching step and gentle CLAHE.

9. **Tuning the proxy label sensitivity** — to catch more/fewer nodules:
   - `score_threshold`: lower = more detections, higher = fewer (try 3-8)
   - `score_percentile_range`: widen = more detections (try (60,85) to (80,95))
   - `contour_area_min_range`: lower lo = catch smaller grain nodules
   - `tophat_radii`: add larger radii for bigger nodules/clusters
   - `lcr_bg_sigma`: smaller = more local adaptation, larger = smoother background
   - `eccentricity_range` / `solidity_range` / `circularity_range`: widen
     ranges to relax shape filters in noisy patches

10. **Bilateral sigma capped at 60** (from 75) — higher sigma was blending
    across nodule–sediment boundaries, softening the edges that proxy labelling
    relies on for contour detection.

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

*Last updated: April 2, 2026 — deleted nodule_boost entirely (function, config,
TunedParams fields); patch size reduced to 256px (overlap 32px); added contrast-adaptive
MSR/CLAHE/unsharp blending; grid-search bilateral tuning; density-adaptive score
percentile range (70-90); score-gated contour filtering (shape_bypass_score_mult);
added test_patch_sizes.py comparison tool*
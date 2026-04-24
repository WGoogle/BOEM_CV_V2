# BOEM CV — Polymetallic Nodule Segmentation

A computer-vision pipeline for detecting and segmenting polymetallic nodules in seafloor mosaics. Given a raw GeoTIFF of the seafloor, the trained model outputs a per-pixel nodule mask plus real-world metrics (coverage percentage, nodule count, density per m²).

If you have never seen this project before, read this whole README once, then decide which role you are (see [Dream Scenario](#dream-scenario)). Most people only need the `deploy/` folder and the annotation tool — not the full training pipeline.

---

## What this repo contains

There are two things living in this repo:

1. **The full research pipeline** (scripts `1_` through `5_` at the root) — preprocesses raw mosaics, generates proxy labels, trains a UNet, runs inference, and audits bad labels. This is what the lead maintainer uses to keep improving the model.
2. **The deploy package** ([deploy/](deploy/)) — a self-contained, dependency-light tool that takes a trained model and a raw mosaic and returns metrics. This is what most end users actually run.

Plus the annotation tool ([4_annotate.py](4_annotate.py) + [annotation/](annotation/)), which is how humans correct model mistakes so the next training round is better.

---

## Dream Scenario — who does what

In an ideal setup, the team works like an army with one general:

- **One lead ("the general").** Holds the full repo, has all the raw mosaics, maintains the `outputs/corrected_masks/` directory, runs training, runs audits, knows the ins and outs of every script. This is the only person who needs to understand `1_preprocess_and_label.py` through `5_audit_labels.py`. They distribute annotation batches, import corrections, retrain, and ship updated model weights.
- **Everyone else (collaborators / end users).** Only needs two things from this repo:
  1. The **annotation tool** ([4_annotate.py](4_annotate.py)) to correct patches the lead sends them as a `.zip`.
  2. The **deploy folder** ([deploy/](deploy/)) to run the trained model on their own mosaics and get metrics.

  Collaborators do not need to train, preprocess, or audit anything. They do not need the raw mosaics. They only need Python and the instructions in the two `.txt` guides.

This division matters because the lead is the single source of truth for corrected masks and model weights. If multiple people tried to maintain corrected masks in parallel, they would overlap and overwrite each other's work. The lead prevents that by splitting annotation assignments **by mosaic** (see [ANNOTATION_GUIDE.txt](ANNOTATION_GUIDE.txt)).

---

## Read these first — the two guide files

Two plain-text guides at the repo root are the most important documents in this project for anyone who is not the lead. Read the one that matches your task before running any command:

- **[ANNOTATION_GUIDE.txt](ANNOTATION_GUIDE.txt)** — full walkthrough of the annotation tool for both leads and collaborators. Covers every editor keybind, the lead's export/import workflow, the collaborator's zip-based workflow, and a worked example with three people splitting three mosaics. If you were sent a `.zip` to annotate, start here.
- **[AUDIT_PIPELINE.txt](AUDIT_PIPELINE.txt)** — how to run the label audit, what it writes, and how to feed the audit queue back into the annotation tool. Only the lead runs this.

These guides are the canonical reference. The sections below summarize them, but the `.txt` files are authoritative.

---

## Setup (lead or anyone running the full pipeline)

```bash
git clone https://github.com/WGoogle/BOEM_CV_V2.git
cd BOEM_CV_V2
pip install -r requirements.txt
```

Drop raw mosaic GeoTIFFs into `data/raw_mosaics/`. All paths, hyperparameters, and feature flags live in [config.py](config.py).

If you are a collaborator who only needs to annotate, the same install works — you just never run scripts 1, 2, 3, or 5.

If you only need to run the trained model on a mosaic, skip this section and go straight to [Deploy](#deploy--standalone-prediction-package). The deploy package has its own lighter requirements.

---

## Pipeline (lead only)

The numbered scripts run in order. Each writes to `outputs/` and the next one reads from there.

### 1. Preprocess and label — [1_preprocess_and_label.py](1_preprocess_and_label.py)

```bash
python 1_preprocess_and_label.py          # normal run
python 1_preprocess_and_label.py --force  # reprocess everything
```

For each mosaic in `data/raw_mosaics/`:
- Tiles into 256 px patches with 32 px overlap.
- Rejects featureless or black-border patches (`min_std`, `max_black_fraction` in `PATCHING`).
- Runs the preprocessing filter chain (gray-world white balance, multi-scale Retinex, CLAHE, bilateral denoise, sediment fade, unsharp mask). Per-patch parameters are auto-tuned from local contrast and noise.
- Generates proxy labels via multi-scale black top-hat + local-contrast-ratio gating + shape filters.

Outputs:
- `outputs/patches/<mosaic>/` — preprocessed image + proxy mask + `patch_manifest.json`
- `outputs/preprocessed/`, `outputs/proxy_labels/` — visual artifacts
- `outputs/step_by_step_logs/` — debug views of the filter chain on a sample of patches

### 2. Train — [2_train.py](2_train.py)

```bash
python 2_train.py                 # uses config.TRAINING defaults
python 2_train.py --epochs 150    # override epochs
python 2_train.py --batch-size 8  # override batch size
```

- Loads every `patch_manifest.json` under `outputs/patches/`.
- Stratified train/val/test split by nodule density (seeded for reproducibility).
- **Prefers corrected masks from `outputs/corrected_masks/` over proxy labels when available.** This is how human corrections flow back into the model.
- UNet / `resnet34` encoder + ImageNet weights by default ([config.MODEL](config.py)).
- Loss: weighted BCE + Dice.
- Mixed precision on CUDA; ReduceLROnPlateau scheduler; early stopping on val Dice.
- Sweeps probability thresholds on val after training and re-saves the best checkpoint with the threshold embedded.

Outputs:
- `outputs/checkpoints/checkpoint_best.pt` / `checkpoint_last.pt`
- `outputs/checkpoints/training_history.json`, `split_info.json`
- `outputs/pipeline_manifest.json` — training section updated

### 3. Inference — [3_inference.py](3_inference.py)

```bash
python 3_inference.py
```

Loads `checkpoint_best.pt`, runs predictions on the test split, and writes probability maps, binary masks, and overlay visualizations to `outputs/results/inference/`. Use this to eyeball model quality against proxy labels before deploying.

### 4. Annotate — [4_annotate.py](4_annotate.py)

Correct proxy labels (or labels flagged by the audit in Step 5). See the [Annotation tool](#annotation-tool) section below and **[ANNOTATION_GUIDE.txt](ANNOTATION_GUIDE.txt) for the full guide** — it has the complete keybind list and a three-person worked example.

Saved corrections land in `outputs/corrected_masks/` and are picked up automatically on the next retrain.

### 5. Audit labels — [5_audit_labels.py](5_audit_labels.py)

```bash
python 5_audit_labels.py                      # top 150 worst-DICE patches, skips already-corrected
python 5_audit_labels.py --top-k 100
python 5_audit_labels.py --include-corrected  # re-audit everything, including human-corrected patches
```

Ranks patches by disagreement between proxy labels and model predictions (DICE-based). The worst-K are exported to `outputs/annotation_inbox/` for correction. True-negative patches (both ground truth and prediction empty) are filtered out.

See **[AUDIT_PIPELINE.txt](AUDIT_PIPELINE.txt) for the full guide.**

Outputs in `outputs/annotation_inbox/`:
- `audit_queue.csv` / `audit_queue.json` — ranked patch list
- `visualizations/<patch_id>.png` — side-by-side proxy label vs. model prediction
- `audit_manifest.json` — one entry per audit run for tracking progress

### Loop

After Step 5, run Step 4 on the queued patches, then rerun Step 2. Repeat until the worst-DICE patches stop being obvious model failures.

---

## Deploy — standalone prediction package

The `deploy/` folder is a self-contained inference package for running the trained model on raw mosaics **without the rest of the pipeline**. This is what collaborators and end users run.

See **[deploy/HOW_TO_USE.txt](deploy/HOW_TO_USE.txt)** for the canonical walkthrough.

### Quick start

```bash
# From the repo root:
python deploy/predict.py

# Or from inside deploy/:
python predict.py

# Override the detection threshold (default: whatever DICE-optimized value is baked into the checkpoint):
python deploy/predict.py --threshold 0.3
```

### Step 0 — get the model weights

`checkpoint_best.pt` (~280 MB) is too large for GitHub and is **not** in the repo. If you did not train the model yourself, get the file from the lead and place it at:

```
deploy/checkpoints/checkpoint_best.pt
```

The small `engineered_norm_stats.json` beside it is included in the repo — you do not need to download that.

### Step 1 — install deploy dependencies (one-time)

The deploy package has its own lighter `requirements.txt`:

```bash
pip install -r deploy/requirements.txt
```

A GPU is optional; the predictor auto-selects CUDA then MPS (Apple Silicon) then CPU.

### Step 2 — drop mosaics into `deploy/input/`

Accepted: `.tif` / `.tiff` only. PNG/JPG/BMP screenshots are rejected because the model is out-of-distribution on display-space captures. Large GeoTIFFs (>2 GB) load via `tifffile` automatically.

Pixel size is read from GeoTIFF tags so density output is in real units (nodules/m²). If the metadata is missing it falls back to 5 mm/px (BOEM D1 Node3 default).

### Step 3 — run the predictor

```bash
python deploy/predict.py
```

### Outputs (written to `deploy/predictions/`)

| File | Contents |
|---|---|
| `<mosaic>_metrics.json` | Per-mosaic metrics (machine-readable) |
| `<mosaic>_metrics.txt` | Same metrics, human-readable |
| `summary.json` | Batch-level summary across all processed mosaics |

Each `*_metrics.json` looks like:

```json
{
  "mosaic":                   "CameraMosaic_Node7_CAM_L3.tif",
  "threshold":                0.50,
  "coverage_pct":             12.43,
  "nodule_count":             8142,
  "nodule_pixels":            1284501,
  "seafloor_pixels":          10334500,
  "seafloor_area_m2":         258.36,
  "nodule_area_m2":           32.11,
  "nodule_density_per_m2":    31.52,
  "nodule_px_density_per_m2": 4970.9,
  "meters_per_pixel":         0.005,
  "min_nodule_px":            20
}
```

`coverage_pct` and `nodule_density_per_m2` are the headline numbers. Lat/long of the mosaic's top-left corner is also included when the GeoTIFF has it.

To reset outputs, delete `deploy/predictions/` — the next run recreates it.

---

## Annotation tool

A GUI for painting and erasing nodule masks on individual 256×256 patches. Two roles:

- **Lead** — exports batches as zips, imports corrections, retrains.
- **Collaborator** — receives a zip, edits masks, sends a zip back.

**[ANNOTATION_GUIDE.txt](ANNOTATION_GUIDE.txt) is the canonical reference.** The rest of this section is a summary.

### Keybinds

| Key | Action |
|---|---|
| Left-click drag | Paint |
| Right-click drag | Erase |
| `+` / `-` / scroll | Zoom |
| Arrow keys | Pan |
| `N` / `P` | Next / previous patch (auto-saves) |
| `SPACE` (hold) | Peek raw image under the overlay |
| `C` | Toggle outline mode |
| `O` | Toggle overlay |
| `[` / `]` | Decrease / increase overlay opacity |
| `Z` / `Y` | Undo / redo |
| `R` (twice within 2 s) | Reset mask to original |
| `S` | Save |

Brush-size slider at the bottom of the window. Use 0–2 px for grain nodules.

Overlay colors: **green** = original proxy label, **cyan** = pixels added, **red** = pixels removed.

### Lead workflow (summary)

```bash
# Export one mosaic per collaborator (split by mosaic to avoid overlap):
python 4_annotate.py export --mosaic CameraMosaic_D1_Node3_L8 \
    --output batch_kailash.zip --annotator Brian --notes "Kailash: D1 Node3"

# Annotate directly (no bundle):
python 4_annotate.py edit --mosaic CameraMosaic_Node6_CAM_L1 --annotator Brian

# Correct patches flagged by the audit (Step 5):
python 4_annotate.py edit --audit-queue --annotator Brian

# Drop returned zips into outputs/annotation_inbox/, then:
python 4_annotate.py import-all
python 4_annotate.py status

# Next round:
python 4_annotate.py export --unannotated --output round2.zip --annotator Brian

# Retrain (corrected masks picked up automatically):
python 2_train.py
```

**Always split exports by mosaic** to prevent two annotators editing the same patches.

### Collaborator workflow (summary)

```bash
# One-time:
git clone <repo> && cd BOEM_CV_V2 && pip install -r requirements.txt

# Open the bundle the lead sent you (extracts _work/ next to the zip; resumes on re-run):
python 4_annotate.py edit --bundle ~/Downloads/batch_for_bob.zip --annotator bob

# When finished, repack and send back:
python 4_annotate.py repack \
    --work-dir ~/Downloads/batch_for_bob_work \
    --output ~/Downloads/corrected_by_bob.zip \
    --annotator bob
```

Autosave is on. Once you leave a patch and close the tool, you cannot return to it in the same bundle. Skipped patches are released back to the pool for someone else.

### CLI reference

| Command | Purpose |
|---|---|
| `edit` | Open the editor. Filters: `--split`, `--mosaic`, `--worst N`, `--unannotated`, `--audit-queue [CSV]`, `--bundle`, `--start N`. Required: `--annotator`. |
| `export` | Package patches into a `.zip`. Filters: `--split`, `--mosaic`, `--unannotated`, `--max-patches`, `--notes`. Required: `--output`, `--annotator`. |
| `import` | Import one returned bundle. `--bundle FILE.zip`, `--strategy {newest,overwrite,skip}` (default `newest`). |
| `import-all` | Import every zip in `outputs/annotation_inbox/`; processed zips move to `annotation_inbox/imported/`. |
| `repack` | Package a `_work/` dir back into a zip. `--work-dir`, `--output`, `--annotator`. |
| `inspect` | Show bundle metadata. `--bundle FILE.zip`. |
| `status` | Totals, per-annotator counts, percent complete. |

---

## Directory layout

```
data/
  raw_mosaics/             input GeoTIFFs (lead only)
  manual_labels/           optional external ground truth
outputs/
  patches/<mosaic>/        tiled patches + patch_manifest.json
  preprocessed/            preprocessed image artifacts
  proxy_labels/            proxy mask artifacts
  step_by_step_logs/       per-stage debug views
  checkpoints/             model weights, split_info, training_history
  results/inference/       Step 3 outputs (patch_metrics.json, overlays)
  corrected_masks/         human-corrected masks (override proxy labels on train)
  annotation_inbox/        queued patches for Step 4 / returned zips / audit outputs
  logs/                    per-stage log files
  pipeline_manifest.json   stage-by-stage run metadata

deploy/                    standalone prediction package (no pipeline required)
  predict.py               single entry point: raw mosaic -> metrics
  inference.py             model loader + sliding-window inference
  metrics.py               coverage % + nodule density per m^2
  geo_resolution.py        reads pixel size from GeoTIFF metadata
  model_config.json        architecture + patch geometry sidecar
  requirements.txt         lightweight deps (torch, smp, opencv, tifffile)
  HOW_TO_USE.txt           canonical deploy walkthrough
  checkpoints/
    checkpoint_best.pt           trained weights (not in repo — download separately)
    engineered_norm_stats.json   normalization stats
  input/                   drop raw mosaics here
  predictions/             all outputs land here

annotation/                annotation tool source (editor, tracker, collaborator)
preprocessing/             filter chain, auto-tuner, patcher, geo resolution
training/                  dataset, model, trainer, splits, confident learning

ANNOTATION_GUIDE.txt       read before touching the annotation tool
AUDIT_PIPELINE.txt         read before running 5_audit_labels.py
```

## Configuration

Everything tunable is in [config.py](config.py), grouped by stage: `PATCHING`, `AUTO_TUNER`, `PREPROCESSING`, `PROXY_LABEL`, `MODEL`, `TRAINING`, `INFERENCE`, `METRICS`, `VISUALIZATION`. Each downstream module reads only its own section — add a parameter by touching this file plus the one function that uses it.

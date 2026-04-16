# BOEM CV — Polymetallic Nodule Segmentation

End-to-end pipeline for detecting and segmenting polymetallic nodules in seafloor mosaics. Each numbered script is a stage; run them in order.

## Setup

```bash
git clone https://github.com/WGoogle/BOEM_CV_V2.git
cd BOEM_CV_V2
pip install -r requirements.txt
```

Drop raw mosaic GeoTIFFs into `data/raw_mosaics/`. All paths, hyperparameters, and feature flags live in [config.py](config.py).

## Pipeline

### 1. Preprocess and label — [1_preprocess_and_label.py](1_preprocess_and_label.py)

```bash
python 1_preprocess_and_label.py          # normal run
python 1_preprocess_and_label.py --force  # reprocess everything
```

For each mosaic:
- Tiles into 256px patches with 32px overlap.
- Rejects featureless/black-border patches (`min_std`, `max_black_fraction` in `PATCHING`).
- Runs the preprocessing filter chain (gray-world white balance, multi-scale Retinex, CLAHE, bilateral denoise, sediment fade, unsharp mask). Per-patch parameters are auto-tuned from local contrast/noise.
- Generates proxy labels via multi-scale black top-hat + local contrast ratio gating + shape filters.

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
- Prefers corrected masks from `outputs/corrected_masks/` over proxy labels when available.
- UNet/`resnet34` + ImageNet weights by default ([config.MODEL](config.py)).
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

Loads `checkpoint_best.pt`, runs predictions on the test split, and writes probability maps, binary masks, and overlay visualizations to `outputs/results/inference/`. Use this to eyeball model quality against proxy labels.

### 4. Annotate — [4_annotate.py](4_annotate.py)

Correct proxy labels (or labels flagged by auditing in Step 5). See the [Annotation tool](#annotation-tool) section below.

Saved corrections land in `outputs/corrected_masks/` and are picked up automatically on the next retrain.

### 5. Audit labels — [5_audit_labels.py](5_audit_labels.py)

```bash
python 5_audit_labels.py                  # top 50 disagreements, default checkpoint
python 5_audit_labels.py --top-k 100
python 5_audit_labels.py --adaptive       # Northcutt-style confident-learning thresholds
python 5_audit_labels.py --no-viz         # skip visualization rendering
```

Runs confident-learning-style scoring: ranks patches by disagreement between proxy labels and model predictions. The worst-K patches are exported to `outputs/annotation_inbox/` for correction. Patches already in `outputs/corrected_masks/` are skipped.

### Loop

After Step 5, run Step 4 to correct the queued patches, then rerun Step 2. Repeat.

---

## Annotation tool

A GUI for painting/erasing nodule masks on individual patches. Supports two roles:
- **Lead** — exports batches, imports corrections, retrains.
- **Collaborator** — receives a `.zip`, edits, sends it back.

### Keybinds

| Key | Action |
|---|---|
| Left-click drag | Paint |
| Right-click drag | Erase |
| `+` / `-` / scroll | Zoom |
| Arrow keys | Pan |
| `N` / `P` | Next / previous patch (auto-saves) |
| `SPACE` (hold) | Peek raw image under overlay |
| `C` | Toggle outline mode |
| `O` | Toggle overlay |
| `[` / `]` | Decrease / increase overlay opacity |
| `Z` / `Y` | Undo / redo |
| `R` (twice within 2s) | Reset mask to original |
| `S` | Save |

Brush size slider at the bottom of the window. Use 1–3 px for grain nodules.

Overlay colors: **green** = original proxy label, **cyan** = pixels added, **red** = pixels removed.

### Lead workflow

```bash
# Export one mosaic to each collaborator
python 4_annotate.py export --mosaic CameraMosaic_D1_Node3_L8 \
    --output batch_kailash.zip --annotator Brian --notes "Kailash: D1 Node3"

# Annotate directly (no bundle)
python 4_annotate.py edit --split test --annotator Brian
python 4_annotate.py edit --worst 20 --annotator Brian
python 4_annotate.py edit --mosaic CameraMosaic_Node6_CAM_L1 --unannotated --annotator Brian

# Drop returned zips into outputs/annotation_inbox/, then:
python 4_annotate.py import-all
python 4_annotate.py status

# Next round
python 4_annotate.py export --unannotated --output round2.zip --annotator Brian

# Retrain (corrected masks picked up automatically)
python 2_train.py
```

Split exports **by mosaic** to prevent annotator overlap.

### Collaborator workflow

```bash
# One-time
git clone <repo> && cd BOEM_CV_V2 && pip install -r requirements.txt

# Open the bundle (extracts _work/ next to the zip, resumes on re-run)
python 4_annotate.py edit --bundle ~/Downloads/batch_for_bob.zip --annotator bob

# When finished, repack and send back
python 4_annotate.py repack \
    --work-dir ~/Downloads/batch_for_bob_work \
    --output ~/Downloads/corrected_by_bob.zip \
    --annotator bob
```

Autosave is on. Closing the tool commits all annotated patches — you cannot return to previously-annotated patches in the same bundle. Skipped patches are released back to the pool.

### CLI reference

| Command | Purpose |
|---|---|
| `edit` | Open the editor. Filters: `--split`, `--mosaic`, `--worst N`, `--unannotated`, `--bundle`, `--start N`. Required: `--annotator`. |
| `export` | Package patches into a `.zip`. Filters: `--split`, `--mosaic`, `--unannotated`, `--max-patches`, `--notes`. Required: `--output`, `--annotator`. |
| `import` | Import one returned bundle. `--bundle FILE.zip` `--strategy {newest,overwrite,skip}` (default `newest`). |
| `import-all` | Import every zip in `outputs/annotation_inbox/`; processed zips move to `annotation_inbox/imported/`. |
| `repack` | Package a `_work/` dir back into a zip. `--work-dir`, `--output`, `--annotator`. |
| `inspect` | Show bundle metadata. `--bundle FILE.zip`. |
| `status` | Totals, per-annotator counts, percent complete. |

---

## Directory layout

```
data/
  raw_mosaics/           input GeoTIFFs
  manual_labels/         optional external ground truth
outputs/
  patches/<mosaic>/      tiled patches + patch_manifest.json
  preprocessed/          preprocessed image artifacts
  proxy_labels/          proxy mask artifacts
  step_by_step_logs/     per-stage debug views
  checkpoints/           model weights, split_info, training_history
  results/inference/     Step 3 outputs
  corrected_masks/       human-corrected masks (override proxy labels on train)
  annotation_inbox/      queued patches for Step 4 / returned zips
  logs/                  per-stage log files
  pipeline_manifest.json stage-by-stage run metadata
```

## Configuration

Everything tunable is in [config.py](config.py), grouped by stage: `PATCHING`, `AUTO_TUNER`, `PREPROCESSING`, `PROXY_LABEL`, `MODEL`, `TRAINING`, `INFERENCE`, `METRICS`, `VISUALIZATION`. Downstream modules read their section and nothing else — add a parameter by touching this file plus the one function that uses it.

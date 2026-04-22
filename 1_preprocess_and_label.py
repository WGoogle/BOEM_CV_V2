"""
Step 1 — Preprocess and Label

Helpful Shortcuts I use:
    python 1_preprocess_and_label.py                  # full run, for new data -- avoids re running already preprocessed ones (old data)
    python 1_preprocess_and_label.py --force          # re-process everything, when debugging, so no need to delete whole outputs folder :)

"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
import cv2
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from preprocessing.patcher import MosaicPatcher, PatchInfo
from preprocessing.auto_tuner import PatchAutoTuner
from preprocessing.filters import FilterPipeline, generate_proxy_label
from preprocessing.geo_resolution import extract_meters_per_pixel

def _setup_logging():
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOGS_DIR / "preprocess.log"

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

logger = logging.getLogger(__name__)

# Inspired by CoralNet pattern: audit trail per file
MANIFEST_PATH = config.OUTPUT_DIR / "pipeline_manifest.json"

def _load_manifest():
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"version": "2.0", "mosaics": {}}

def _save_manifest(manifest):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

def _find_mosaics():
    """Return sorted list of raw mosaic files."""
    exts = ["*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.png", "*.PNG"]
    files = []
    for ext in exts:
        files.extend(config.RAW_MOSAICS_DIR.glob(ext))
    return sorted(set(files))

# CORE PIPELINE
def process_mosaic(mosaic_path, manifest, *, force = False):
    """
    Full Step-1 pipeline for a single mosaic.

    1. Load the mosaic
    2. Split into overlapping patches  (patcher.py)
    3. For each valid patch:
       a. Auto-tune parameters         (auto_tuner.py)
       b. Run filter chain              (filters.py)
       c. Generate proxy label          (filters.py)
       d. Save intermediate images      (filters.py)
    4. Save preprocessed patches + proxy masks to disk
    5. Update the manifest
    """
    key = mosaic_path.stem
    entry = manifest["mosaics"].setdefault(key, {})

    # Idempotency check
    if not force and entry.get("completed"):
        logger.info(f"  ⤳ {key}: already processed — skipping (use --force to redo)")
        return entry

    t0 = time.time()
    logger.info("─" * 70)
    logger.info(f"Processing mosaic: {mosaic_path.name}")
    logger.info("─" * 70)

    patcher  = MosaicPatcher(**config.PATCHING)
    tuner    = PatchAutoTuner(config.AUTO_TUNER)
    pipeline = FilterPipeline(config.PREPROCESSING)

    # Load
    mosaic = patcher.load_mosaic(mosaic_path)
    mosaic_h, mosaic_w = mosaic.shape[:2]

    # Extract spatial resolution from GeoTIFF metadata
    mpp = extract_meters_per_pixel(
        mosaic_path, fallback=config.METRICS["meters_per_pixel"],
    )
    if mpp is not None:
        logger.info(
            f"  Spatial resolution: {mpp:.6f} m/px ({mpp * 1000:.2f} mm/px)  "
            f"— image covers {mosaic_w * mpp:.1f} m × {mosaic_h * mpp:.1f} m"
        )

    # Patch
    patches, infos = patcher.extract_patches(mosaic)
    n_valid = len(patches)

    if n_valid == 0:
        logger.warning(f"  No valid patches extracted from {key} — skipping")
        entry["error"] = "no_valid_patches"
        _save_manifest(manifest)
        return entry

    # Prepare output dirs
    mosaic_preproc_dir = config.PREPROCESSED_DIR / key
    mosaic_preproc_dir.mkdir(parents=True, exist_ok=True)

    mosaic_proxy_dir = config.PROXY_LABELS_DIR / key
    mosaic_proxy_dir.mkdir(parents=True, exist_ok=True)

    mosaic_patches_img_dir = config.PATCHES_DIR / key / "images"
    mosaic_patches_msk_dir = config.PATCHES_DIR / key / "masks"
    mosaic_patches_img_dir.mkdir(parents=True, exist_ok=True)
    mosaic_patches_msk_dir.mkdir(parents=True, exist_ok=True)

    mosaic_log_dir = config.STEP_BY_STEP_DIR / key
    mosaic_log_dir.mkdir(parents=True, exist_ok=True)

    # Per-patch processing
    patch_records = []          # raw BGR crops paired with preprocessed proxy masks
    preprocessed_patches: list[np.ndarray] = []
    total_nodules = 0
    valid_idx = 0

    # Pick a random subset of patches to log step-by-step images for
    import random
    max_step_logs = config.LOGGING.get("max_step_log_patches", 5)
    valid_indices = [i for i, inf in enumerate(infos) if inf.is_valid]
    logged_set = set(random.sample(valid_indices, min(max_step_logs, len(valid_indices))))

    for info in infos:
        if not info.is_valid:
            continue

        patch_bgr = patches[valid_idx]
        valid_idx += 1
        pid = f"{key}_patch_{info.patch_index:04d}"

        # All steps here
        params = tuner.analyse(patch_bgr)
        preprocessed, filter_steps = pipeline.run(patch_bgr, params)
        proxy_mask, label_steps, label_stats = generate_proxy_label(
            preprocessed, params, config.PROXY_LABEL,
        )
        total_nodules += label_stats["nodules_after_filter"]
        should_log = (
            config.LOGGING["save_intermediate_steps"]
            and info.patch_index in logged_set
        )
        if should_log:
            # Combine filter steps + label steps into one sequence
            all_steps = filter_steps + label_steps
            FilterPipeline.save_step_images(
                all_steps,
                output_dir=mosaic_log_dir / pid,
                prefix=pid,
            )

        img_path = mosaic_patches_img_dir / f"{pid}.png"
        msk_path = mosaic_patches_msk_dir / f"{pid}.png"
        # Save raw BGR crop as the training image; mask is the proxy label
        # derived from the preprocessed version.
        cv2.imwrite(str(img_path), patch_bgr)
        cv2.imwrite(str(msk_path), proxy_mask)

        preprocessed_patches.append(preprocessed)
        patch_records.append({
            "patch_id":     pid,
            "image_path":   str(img_path),
            "mask_path":    str(msk_path),
            "grid_row":     info.row,
            "grid_col":     info.col,
            "origin_y":     info.y,
            "origin_x":     info.x,
            "tuned_params": params.as_dict(),
            "label_stats":  label_stats,
        })

        # Progress tick
        if valid_idx % 25 == 0 or valid_idx == n_valid:
            logger.info(
                f"  [{valid_idx}/{n_valid}] patches processed  "
                f"({total_nodules} nodules detected so far)"
            )

    patch_masks = []
    vi = 0
    for info in infos:
        if not info.is_valid:
            continue
        msk = cv2.imread(
            str(mosaic_patches_msk_dir / f"{key}_patch_{info.patch_index:04d}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        patch_masks.append(msk)
        vi += 1

    full_mask = patcher.reassemble(
        patch_masks, infos, (mosaic_h, mosaic_w), dtype=np.float32,
    )
    full_mask_binary = (full_mask > 127).astype(np.uint8) * 255

    full_mask_path = mosaic_proxy_dir / f"{key}_full_proxy_mask.png"
    cv2.imwrite(str(full_mask_path), full_mask_binary)

    # Overlay visualisation
    overlay = mosaic.copy()
    overlay[full_mask_binary > 0] = [0, 255, 0]
    blended = cv2.addWeighted(mosaic, 0.6, overlay, 0.4, 0)
    overlay_path = mosaic_proxy_dir / f"{key}_overlay.png"
    cv2.imwrite(str(overlay_path), blended)

    # For the deploy/ folder, preprocessing mosaic code here
    accum_bgr = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.float64)
    count_bgr = np.zeros((mosaic_h, mosaic_w),    dtype=np.float64)
    pi = 0
    for info in infos:
        if not info.is_valid:
            continue
        patch = preprocessed_patches[pi]
        pi += 1
        y, x = info.y, info.x
        ph = min(info.height, mosaic_h - y)
        pw = min(info.width,  mosaic_w - x)
        accum_bgr[y:y + ph, x:x + pw] += patch[:ph, :pw].astype(np.float64)
        count_bgr[y:y + ph, x:x + pw] += 1.0

    covered = count_bgr > 0
    full_preproc = mosaic.copy()
    full_preproc[covered] = (
        accum_bgr[covered] / count_bgr[covered, None]
    ).clip(0, 255).astype(np.uint8)

    full_preproc_path = mosaic_preproc_dir / "preprocessed.png"
    cv2.imwrite(str(full_preproc_path), full_preproc)
    logger.info(
        f"  Saved full preprocessed mosaic → {full_preproc_path}  "
        f"(coverage {100.0 * covered.mean():.1f}%; rejected tiles filled from raw)"
    )

    # Manifest stuff
    patch_manifest_path = config.PATCHES_DIR / key / "patch_manifest.json"
    with open(patch_manifest_path, "w") as f:
        json.dump(patch_records, f, indent=2, default=str)
    elapsed = time.time() - t0
    entry.update({
        "source":           str(mosaic_path),
        "shape":            [mosaic_h, mosaic_w, mosaic.shape[2]],
        "meters_per_pixel": mpp,
        "coverage_meters":  [round(mosaic_w * mpp, 2), round(mosaic_h * mpp, 2)] if mpp else None,
        "total_patches":    len(infos),
        "valid_patches":    n_valid,
        "total_nodules":    total_nodules,
        "full_proxy_mask":  str(full_mask_path),
        "preprocessed_mosaic": str(full_preproc_path),
        "overlay":          str(overlay_path),
        "patch_manifest":   str(patch_manifest_path),
        "completed":        True,
        "completed_at":     datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds":  round(elapsed, 1),
    })
    _save_manifest(manifest)

    logger.info(
        f"  ✓ {key}: {n_valid} patches, {total_nodules} nodules, "
        f"{elapsed:.1f}s"
    )
    return entry

# CLI ENTRY POINT
def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Preprocess seafloor mosaics + generate proxy labels",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process mosaics even if already completed",
    )
    args = parser.parse_args()

    _setup_logging()

    logger.info("=" * 70)
    logger.info("BOEM CV  —  Step 1: Preprocess + Proxy Label Pipeline")
    logger.info("=" * 70)
    logger.info(f"  Raw mosaics dir : {config.RAW_MOSAICS_DIR}")
    logger.info(f"  Output dir      : {config.OUTPUT_DIR}")
    logger.info(f"  Patch size      : {config.PATCHING['patch_size']}px  "
                f"(overlap {config.PATCHING['overlap']}px)")
    logger.info(f"  Filter chain    : {config.PREPROCESSING['filter_chain']}")
    logger.info(f"  Force re-run    : {args.force}")

    mosaic_files = _find_mosaics()
    if not mosaic_files:
        logger.error(
            f"No mosaic files found in {config.RAW_MOSAICS_DIR}\n"
            f"Place .tif / .tiff / .png files there and re-run."
        )
        sys.exit(1)

    logger.info(f"  Mosaics found   : {len(mosaic_files)}")

    manifest = _load_manifest()
    summaries = []

    for mosaic_path in mosaic_files:
        try:
            summary = process_mosaic(
                mosaic_path, manifest,
                force=args.force,
            )
            summaries.append(summary)
        except Exception as exc:
            logger.error(f"FAILED {mosaic_path.name}: {exc}", exc_info=True)
            manifest["mosaics"].setdefault(mosaic_path.stem, {})["error"] = str(exc)
            _save_manifest(manifest)

    # Final Summary
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    total_patches = sum(s.get("valid_patches", 0) for s in summaries)
    total_nodules = sum(s.get("total_nodules", 0) for s in summaries)
    total_time = sum(s.get("elapsed_seconds", 0) for s in summaries)
    logger.info(f"  Mosaics processed : {len(summaries)}")
    logger.info(f"  Total patches     : {total_patches}")
    logger.info(f"  Total nodules     : {total_nodules}")
    logger.info(f"  Total time        : {total_time:.1f}s")
    logger.info(f"  Manifest          : {MANIFEST_PATH}")
    logger.info(f"  Step-by-step logs : {config.STEP_BY_STEP_DIR}")
    logger.info("")
    logger.info("Next step: python 2_train.py")

if __name__ == "__main__":
    main()
"""
Step 3 — Inference & Visualisation
Loads the best checkpoint from Step 2, runs predictions on test patches, and produces visual outputs so you can inspect model performance
Check the outputs/results folder, especially the overlays/ subfolder, to see the predictions overlaid on the original seafloor images. 

    Run python 3_inference.py       
"""
from __future__ import annotations
import json
import logging
import os
import shutil
import sys
from pathlib import Path

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_erosion
from torch.utils.data import DataLoader
import config
from training.model import build_model
from training.dataset import (
    NoduleSegmentationDataset,
    get_val_augmentations,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-12s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

INFERENCE_DIR = config.RESULTS_DIR / "inference"
OVERLAYS_DIR  = INFERENCE_DIR / "overlays"

def load_model(checkpoint_path, device):
    # Load model from checkpoint, set to eval mode.

    model = build_model(config.MODEL)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    val_dice = ckpt.get("val_dice", 0)
    best_threshold = ckpt.get("best_threshold", None)
    if best_threshold is not None:
        logger.info(
            f"Loaded checkpoint: epoch {epoch}, val_dice {val_dice:.4f}, "
            f"best_threshold {best_threshold:.2f}"
        )
    else:
        logger.info(f"Loaded checkpoint: epoch {epoch}, val_dice {val_dice:.4f}")
    return model, best_threshold

def collect_all_records():
    # Load all patch records from every mosaic's manifest
    records = []
    for mosaic_dir in sorted(config.PATCHES_DIR.iterdir()):
        manifest = mosaic_dir / "patch_manifest.json"
        if not manifest.exists():
            continue
        with open(manifest) as f:
            patches = json.load(f)
        records.extend(patches)
    logger.info(f"Loaded {len(records)} total patch records")
    return records

def load_raw_rgb(rec):
    # Load the original seafloor patch as RGB for display (independent of input_mode).
    bgr = cv2.imread(rec["image_path"], cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {rec['image_path']}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def make_contour_overlay(image_rgb, gt_mask, pred_mask):
    # Draw GT and prediction as thin contours on top of the raw image so the seafloor stays visible. Yellow = ground truth, cyan = prediction.
    overlay = image_rgb.copy()
    gt_u8   = (gt_mask   > 0.5).astype(np.uint8) * 255
    pred_u8 = (pred_mask > 0.5).astype(np.uint8) * 255

    gt_contours, _   = cv2.findContours(gt_u8,   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(overlay, gt_contours,   -1, (255, 255, 0), 1)   # yellow GT
    cv2.drawContours(overlay, pred_contours, -1, (0, 255, 255), 1)   # cyan pred
    return overlay

def predict_batch(model, images, device, threshold):
    with torch.no_grad():
        images = images.to(device)
        _autocast_device = "cuda" if device == "cuda" else "cpu"
        with torch.autocast(device_type=_autocast_device, enabled=(device == "cuda")):
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
    return (probs > threshold).astype(np.float32), probs

# Boundary-aware metrics (ASD, NSD)
def _surface_distances(pred_mask, gt_mask):
    pred_bool = pred_mask > 0.5
    gt_bool = gt_mask > 0.5

    if not pred_bool.any() or not gt_bool.any():
        return None, None

    # Extract boundaries via erosion
    pred_boundary = pred_bool ^ binary_erosion(pred_bool)
    gt_boundary = gt_bool ^ binary_erosion(gt_bool)

    # Handle single-pixel masks where erosion removes everything
    if not pred_boundary.any():
        pred_boundary = pred_bool
    if not gt_boundary.any():
        gt_boundary = gt_bool

    # Distance transform of the complement
    dt_gt = distance_transform_edt(~gt_bool)
    dt_pred = distance_transform_edt(~pred_bool)

    # Surface distances
    pred_to_gt = dt_gt[pred_boundary]
    gt_to_pred = dt_pred[gt_boundary]

    return pred_to_gt, gt_to_pred


def _compute_boundary_metrics(pred_mask, gt_mask, nsd_tolerance=2.0):
    pred_to_gt, gt_to_pred = _surface_distances(pred_mask, gt_mask)

    if pred_to_gt is None:
        return {"asd": float("nan"), "nsd": float("nan")}

    all_distances = np.concatenate([pred_to_gt, gt_to_pred])
    asd = float(np.mean(all_distances))

    # Normalized Surface Dice: fraction of boundary pixels within tolerance
    pred_within = np.sum(pred_to_gt <= nsd_tolerance)
    gt_within = np.sum(gt_to_pred <= nsd_tolerance)
    nsd = float((pred_within + gt_within) / (len(pred_to_gt) + len(gt_to_pred)))

    return {"asd": asd, "nsd": nsd}

# Metrics of all patches
def compute_all_metrics(model, records, device, threshold, input_mode = "rgb",
                        pred_masks_dir = None):

    dataset = NoduleSegmentationDataset(
        records, transform=get_val_augmentations(input_mode=input_mode),
        input_mode=input_mode,
        corrected_masks_dir=config.CORRECTED_MASKS_DIR,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        num_workers=config.TRAINING["num_workers"])

    if pred_masks_dir is not None:
        pred_masks_dir = Path(pred_masks_dir)
        pred_masks_dir.mkdir(parents=True, exist_ok=True)

    results = []
    patch_idx = 0

    for images, masks in loader:
        preds_binary, _ = predict_batch(model, images, device, threshold)
        masks_np = masks.numpy()[:, 0]

        for i in range(images.size(0)):
            rec = records[patch_idx]
            pred_mask = preds_binary[i]
            gt_mask = masks_np[i]

            pred_sum = pred_mask.sum()
            gt_sum = gt_mask.sum()
            if pred_sum == 0 and gt_sum == 0:
                dice = float("nan")
            else:
                smooth = 1e-6
                inter = (pred_mask * gt_mask).sum()
                dice = (2 * inter + smooth) / (pred_sum + gt_sum + smooth)

            boundary = _compute_boundary_metrics(pred_mask, gt_mask)

            patch_id = rec.get("patch_id", f"patch_{patch_idx:04d}")
            results.append({
                "patch_id": patch_id,
                "dice": float(dice),
                "asd": boundary["asd"],
                "nsd": boundary["nsd"],
                "image_path": rec["image_path"],
                "mask_path": rec["mask_path"],
                "mosaic_id": rec.get("mosaic_id", "unknown"),
            })

            if pred_masks_dir is not None:
                cv2.imwrite(
                    str(pred_masks_dir / f"{patch_id}.png"),
                    (pred_mask * 255).astype(np.uint8),
                )

            patch_idx += 1

    logger.info(f"Computed metrics for {len(results)} patches")
    if pred_masks_dir is not None:
        logger.info(f"Saved prediction masks → {pred_masks_dir}")
    return results


def save_patch_metrics(results, output_path):
    def _clean(v):
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    payload = [{k: _clean(v) for k, v in r.items()} for r in results]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved per-patch metrics → {output_path}")

# Visualizations
def generate_overlays(model, records, device, threshold, per_mosaic = 10, input_mode = "rgb", seed = 0):
    if OVERLAYS_DIR.exists():
        shutil.rmtree(OVERLAYS_DIR)
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    by_mosaic: dict[str, list] = {}
    for rec in records:
        by_mosaic.setdefault(rec.get("mosaic_id", "unknown"), []).append(rec)

    sampled = []
    for recs in by_mosaic.values():
        k = min(per_mosaic, len(recs))
        picks = rng.choice(len(recs), size=k, replace=False)
        sampled.extend(recs[i] for i in picks)
    logger.info(f"Sampled {len(sampled)} patches across {len(by_mosaic)} mosaics for overlays")

    dataset = NoduleSegmentationDataset(
        sampled, transform=get_val_augmentations(input_mode=input_mode),
        input_mode=input_mode,
        corrected_masks_dir=config.CORRECTED_MASKS_DIR,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        num_workers=config.TRAINING["num_workers"])

    saved = 0
    patch_idx = 0

    for images, masks in loader:
        preds_binary, preds_prob = predict_batch(model, images, device, threshold)
        masks_np = masks.numpy()[:, 0]

        for i in range(images.size(0)):
            rec = sampled[patch_idx]
            img_rgb = load_raw_rgb(rec)
            gt_mask = masks_np[i]
            pred_mask = preds_binary[i]
            prob_map = preds_prob[i]

            smooth = 1e-6
            inter = (pred_mask * gt_mask).sum()
            dice = (2 * inter + smooth) / (pred_mask.sum() + gt_mask.sum() + smooth)

            contour_overlay = make_contour_overlay(img_rgb, gt_mask, pred_mask)

            heatmap = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            font = cv2.FONT_HERSHEY_SIMPLEX
            panels = [img_rgb.copy(), contour_overlay, heatmap]
            labels = ["Input", f"GT=Yellow Pred=Cyan (D:{dice:.2f})", "Probability"]
            for panel, label in zip(panels, labels):
                cv2.putText(panel, label, (5, 15), font, 0.4, (255, 255, 255), 1,
                            cv2.LINE_AA)

            composite = np.hstack(panels)
            patch_id = rec.get("patch_id", f"patch_{patch_idx:04d}")
            out_path = OVERLAYS_DIR / f"{patch_id}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

            saved += 1
            patch_idx += 1

    logger.info(f"Saved {saved} overlay images → {OVERLAYS_DIR}")
    return saved

def print_metrics_summary(results):
    if not results:
        return
    scored = [r for r in results if not np.isnan(r["dice"])]

    # Boundary metrics — filter NaN (empty masks)
    asds  = [r["asd"]  for r in results if not np.isnan(r["asd"])]
    nsds  = [r["nsd"]  for r in results if not np.isnan(r["nsd"])]

    logger.info("=" * 60)
    logger.info(f"  Patches evaluated : {len(results)}  "
                f"(scored: {len(scored)}, true-neg skipped: {len(results) - len(scored)})")
    logger.info("-" * 60)
    logger.info("  Boundary metrics (tolerant of tight-vs-loose annotations):")
    if asds:
        logger.info(f"    Mean ASD   (px)   : {np.mean(asds):.2f}  (std {np.std(asds):.2f})")
        logger.info(f"    Mean NSD (tol=2px): {np.mean(nsds):.4f}  (std {np.std(nsds):.4f})")
    else:
        logger.info("    (no valid boundary metrics — all masks empty)")
    logger.info("=" * 60)

    # Report worst patches for inspection
    sorted_results = sorted(scored, key=lambda r: r["dice"])
    logger.info("  Worst 5 patches (lowest Dice):")
    for r in sorted_results[:5]:
        nsd_str = f"  NSD={r['nsd']:.4f}" if not np.isnan(r["nsd"]) else ""
        logger.info(f"    {r['patch_id']}  Dice={r['dice']:.4f}{nsd_str}")

def main():
    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")

    # Checkpoint
    ckpt_path = config.CHECKPOINTS_DIR / "checkpoint_best.pt"
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        logger.error("Run Step 2 (2_train.py) first.")
        sys.exit(1)

    # Load model — also picks up the trainer's per-checkpoint threshold
    model, best_threshold = load_model(ckpt_path, device)
    override = config.INFERENCE.get("threshold_override")
    if override is not None:
        threshold = float(override)
        logger.info(f"  Using threshold={threshold:.2f} (hardcoded override from config)")
    elif best_threshold is not None:
        threshold = best_threshold
        logger.info(f"  Using threshold={threshold:.2f} (adaptive, from checkpoint)")
    else:
        threshold = config.INFERENCE["probability_threshold"]
        logger.info(f"  Using threshold={threshold:.2f} (config default)")

    records = collect_all_records()
    logger.info(f"Running inference on {len(records)} patches (threshold={threshold})")

    input_mode = config.MODEL.get("input_mode", "rgb")

    pred_masks_dir = config.ANNOTATION_INBOX / "pred_masks"
    all_metrics = compute_all_metrics(model, records, device, threshold,
                                      input_mode=input_mode,
                                      pred_masks_dir=pred_masks_dir)

    # Persist metrics so 5_audit_labels.py can rank by DICE without re-running
    metrics_path = INFERENCE_DIR / "patch_metrics.json"
    save_patch_metrics(all_metrics, metrics_path)

    # Generate visual outputs
    generate_overlays(model, records, device, threshold,
                      per_mosaic=10, input_mode=input_mode)

    print_metrics_summary(all_metrics)
    logger.info(f"\nOutputs saved to: {INFERENCE_DIR}")
    logger.info(f"  Overlays:      {OVERLAYS_DIR}")
    logger.info(f"  Patch metrics: {metrics_path}")
    logger.info("\nNext: open the overlay images to inspect predictions visually.")

if __name__ == "__main__":
    main()
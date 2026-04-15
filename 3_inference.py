"""
Step 3 — Inference & Visualisation
Loads the best checkpoint from Step 2, runs predictions on test patches, and produces visual outputs so you can inspect model performance:

Usages:
    python 3_inference.py            # basic run
"""
from __future__ import annotations
import json
import logging
import sys
import cv2
import numpy as np
import torch
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
GRIDS_DIR     = INFERENCE_DIR / "grids"

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
    # Draw GT and prediction as thin contours on top of the raw image so the
    # seafloor stays visible. Yellow = ground truth, cyan = prediction.
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
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
    return (probs > threshold).astype(np.float32), probs

# Metrics of all patches 
def compute_all_metrics(model, records, device, threshold, input_mode = "rgb"):

    dataset = NoduleSegmentationDataset(
        records, transform=get_val_augmentations(input_mode=input_mode),
        input_mode=input_mode,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        num_workers=config.TRAINING["num_workers"])

    results = []
    patch_idx = 0

    for images, masks in loader:
        preds_binary, _ = predict_batch(model, images, device, threshold)
        masks_np = masks.numpy()[:, 0]

        for i in range(images.size(0)):
            rec = records[patch_idx]
            pred_mask = preds_binary[i]
            gt_mask = masks_np[i]

            smooth = 1e-6
            inter = (pred_mask * gt_mask).sum()
            dice = (2 * inter + smooth) / (pred_mask.sum() + gt_mask.sum() + smooth)
            iou = (inter + smooth) / (pred_mask.sum() + gt_mask.sum() - inter + smooth)

            patch_id = rec.get("patch_id", f"patch_{patch_idx:04d}")
            results.append({
                "patch_id": patch_id,
                "dice": float(dice),
                "iou": float(iou),
            })
            patch_idx += 1

    logger.info(f"Computed metrics for {len(results)} patches")
    return results

# Visualizations 
def generate_overlays(model, records, device, threshold, max_overlays = 50, input_mode = "rgb"):
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = NoduleSegmentationDataset(
        records, transform=get_val_augmentations(input_mode=input_mode),
        input_mode=input_mode,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        num_workers=config.TRAINING["num_workers"])

    saved = 0
    patch_idx = 0

    for images, masks in loader:
        preds_binary, preds_prob = predict_batch(model, images, device, threshold)
        masks_np = masks.numpy()[:, 0]

        for i in range(images.size(0)):
            if saved >= max_overlays:
                break

            rec = records[patch_idx]
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

        if saved >= max_overlays:
            break

    logger.info(f"Saved {saved} overlay images → {OVERLAYS_DIR}")
    return saved

def generate_summary_grid(model, records, device, threshold, n_samples = 16, cols = 4, input_mode = "rgb"):
    GRIDS_DIR.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(records) - 1, min(n_samples, len(records)),
                          dtype=int)
    sampled = [records[i] for i in indices]

    dataset = NoduleSegmentationDataset(
        sampled, transform=get_val_augmentations(input_mode=input_mode),
        input_mode=input_mode,
    )
    loader = DataLoader(dataset, batch_size=min(8, len(sampled)), shuffle=False,
                        num_workers=config.TRAINING["num_workers"])

    all_images, all_masks = [], []
    all_preds_binary, all_preds_prob = [], []
    for images, masks in loader:
        preds_binary, preds_prob = predict_batch(model, images, device, threshold)
        all_images.append(images)
        all_masks.append(masks.numpy()[:, 0])
        all_preds_binary.append(preds_binary)
        all_preds_prob.append(preds_prob)

    images = torch.cat(all_images, dim=0)
    masks_np = np.concatenate(all_masks, axis=0)
    preds_binary = np.concatenate(all_preds_binary, axis=0)

    cells = []
    for i in range(len(sampled)):
        img_rgb = load_raw_rgb(sampled[i])
        contour_overlay = make_contour_overlay(img_rgb, masks_np[i], preds_binary[i])

        smooth = 1e-6
        inter = (preds_binary[i] * masks_np[i]).sum()
        dice = (2 * inter + smooth) / (preds_binary[i].sum() + masks_np[i].sum() + smooth)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(contour_overlay, f"D:{dice:.2f} GT=Y Pred=C", (3, 12), font, 0.35,
                    (255, 255, 255), 1)
        cell = np.vstack([img_rgb.copy(), contour_overlay])
        cells.append(cell)

    # Arrange in grid
    rows_list = []
    for r in range(0, len(cells), cols):
        row_cells = cells[r:r + cols]
        while len(row_cells) < cols:
            row_cells.append(np.zeros_like(cells[0]))
        rows_list.append(np.hstack(row_cells))

    grid = np.vstack(rows_list)
    grid_path = GRIDS_DIR / "summary_grid.png"
    cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    logger.info(f"Summary grid ({len(sampled)} patches) → {grid_path}")
    return grid_path

def print_metrics_summary(results):
    if not results:
        return
    dices = [r["dice"] for r in results]
    ious  = [r["iou"] for r in results]
    logger.info("=" * 50)
    logger.info(f"  Patches evaluated : {len(results)}")
    logger.info(f"  Mean Dice         : {np.mean(dices):.4f}  (std {np.std(dices):.4f})")
    logger.info(f"  Mean IoU          : {np.mean(ious):.4f}  (std {np.std(ious):.4f})")
    logger.info(f"  Min  Dice         : {np.min(dices):.4f}")
    logger.info(f"  Max  Dice         : {np.max(dices):.4f}")
    logger.info("=" * 50)

    # Report worst patches for inspection
    sorted_results = sorted(results, key=lambda r: r["dice"])
    logger.info("  Worst 5 patches (lowest Dice):")
    for r in sorted_results[:5]:
        logger.info(f"    {r['patch_id']}  Dice={r['dice']:.4f}  IoU={r['iou']:.4f}")

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
    threshold = best_threshold if best_threshold is not None else config.INFERENCE["probability_threshold"]
    logger.info(f"  Using threshold={threshold:.2f}")

    records = collect_all_records()
    logger.info(f"Running inference on {len(records)} patches (threshold={threshold})")

    input_mode = config.MODEL.get("input_mode", "rgb")

    # Compute metrics on ALL patches
    all_metrics = compute_all_metrics(model, records, device, threshold,
                                      input_mode=input_mode)

    # Generate visual outputs
    generate_overlays(model, records, device, threshold,
                      input_mode=input_mode)
    grid_path = generate_summary_grid(model, records, device, threshold,
                                      input_mode=input_mode)

    print_metrics_summary(all_metrics)
    logger.info(f"\nOutputs saved to: {INFERENCE_DIR}")
    logger.info(f"  Overlays:      {OVERLAYS_DIR}")
    logger.info(f"  Summary grid:  {grid_path}")
    logger.info("\nNext: open the overlay images to inspect predictions visually.")

if __name__ == "__main__":
    main()
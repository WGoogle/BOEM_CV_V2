#!/usr/bin/env python3
"""
Step 3 — Inference & Visualisation
====================================
Loads the best checkpoint from Step 2, runs predictions on test patches,
and produces visual outputs so you can inspect model performance:

  1. Per-patch overlays  (image + ground-truth mask + prediction side-by-side)
  2. Summary grid         (random sample of test patches in one image)
  3. Full-mosaic stitched prediction maps (reassembled from patches)

Usage:
    python 3_inference.py                     # defaults: test split, best checkpoint
    python 3_inference.py --split all         # run on ALL patches
    python 3_inference.py --threshold 0.3     # lower confidence threshold
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from training.model import build_model
from training.dataset import (
    NoduleSegmentationDataset,
    get_val_augmentations,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from training.splits import split_dataset

# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-12s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Output directory ────────────────────────────────────────────────────
INFERENCE_DIR = config.RESULTS_DIR / "inference"
OVERLAYS_DIR  = INFERENCE_DIR / "overlays"
GRIDS_DIR     = INFERENCE_DIR / "grids"


# ── Helpers ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: str) -> nn.Module:
    """Load model from checkpoint, set to eval mode."""
    model = build_model(config.MODEL)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    val_dice = ckpt.get("val_dice", 0)
    logger.info(f"Loaded checkpoint: epoch {epoch}, val_dice {val_dice:.4f}")
    return model


def collect_all_records() -> list[dict]:
    """Load all patch records from every mosaic's manifest."""
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


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised (3,H,W) tensor back to uint8 RGB image."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return img


def make_overlay(image_rgb: np.ndarray, mask: np.ndarray,
                 color: tuple = (0, 255, 0), alpha: float = 0.4) -> np.ndarray:
    """Blend a binary mask onto an RGB image."""
    overlay = image_rgb.copy()
    mask_bool = mask > 0.5
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + alpha * np.array(color)
    ).astype(np.uint8)
    return overlay


def predict_batch(model: nn.Module, images: torch.Tensor,
                  device: str, threshold: float) -> np.ndarray:
    """Run inference on a batch, return binary masks (N, H, W)."""
    with torch.no_grad():
        images = images.to(device)
        with torch.autocast(device_type=device if device == "cuda" else "cpu",
                            enabled=(device == "cuda")):
            logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()[:, 0]  # (N, H, W)
    return (probs > threshold).astype(np.float32), probs


# ── Main visualisation routines ─────────────────────────────────────────

def generate_overlays(
    model: nn.Module,
    records: list[dict],
    device: str,
    threshold: float,
    max_overlays: int = 50,
) -> list[dict]:
    """Generate side-by-side overlay images for each patch.

    Returns list of result dicts with paths and metrics.
    """
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = NoduleSegmentationDataset(records, transform=get_val_augmentations())
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    results = []
    patch_idx = 0

    for images, masks in loader:
        preds_binary, preds_prob = predict_batch(model, images, device, threshold)
        masks_np = masks.numpy()[:, 0]  # (N, H, W)

        for i in range(images.size(0)):
            if patch_idx >= max_overlays:
                break

            rec = records[patch_idx]
            img_rgb = denormalize_image(images[i])
            gt_mask = masks_np[i]
            pred_mask = preds_binary[i]
            prob_map = preds_prob[i]

            # Compute per-patch metrics
            smooth = 1e-6
            inter = (pred_mask * gt_mask).sum()
            dice = (2 * inter + smooth) / (pred_mask.sum() + gt_mask.sum() + smooth)
            iou = (inter + smooth) / (pred_mask.sum() + gt_mask.sum() - inter + smooth)

            # Build side-by-side: image | GT overlay (green) | pred overlay (red) | probability heatmap
            gt_overlay = make_overlay(img_rgb, gt_mask, color=(0, 255, 0), alpha=0.4)
            pred_overlay = make_overlay(img_rgb, pred_mask, color=(255, 0, 0), alpha=0.4)

            # Probability heatmap
            heatmap = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # Labels
            h, w = img_rgb.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            panels = [img_rgb.copy(), gt_overlay, pred_overlay, heatmap]
            labels = ["Input", "Ground Truth", f"Prediction (D:{dice:.2f})", "Probability"]
            for panel, label in zip(panels, labels):
                cv2.putText(panel, label, (5, 15), font, 0.4, (255, 255, 255), 1,
                            cv2.LINE_AA)

            composite = np.hstack(panels)
            patch_id = rec.get("patch_id", f"patch_{patch_idx:04d}")
            out_path = OVERLAYS_DIR / f"{patch_id}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

            results.append({
                "patch_id": patch_id,
                "dice": float(dice),
                "iou": float(iou),
                "overlay_path": str(out_path),
            })
            patch_idx += 1

        if patch_idx >= max_overlays:
            break

    logger.info(f"Saved {len(results)} overlay images → {OVERLAYS_DIR}")
    return results


def generate_summary_grid(
    model: nn.Module,
    records: list[dict],
    device: str,
    threshold: float,
    n_samples: int = 16,
    cols: int = 4,
) -> Path:
    """Create a grid of sample predictions for quick visual inspection."""
    GRIDS_DIR.mkdir(parents=True, exist_ok=True)

    # Sample evenly across the records
    indices = np.linspace(0, len(records) - 1, min(n_samples, len(records)),
                          dtype=int)
    sampled = [records[i] for i in indices]

    dataset = NoduleSegmentationDataset(sampled, transform=get_val_augmentations())
    loader = DataLoader(dataset, batch_size=len(sampled), shuffle=False, num_workers=0)

    images, masks = next(iter(loader))
    preds_binary, preds_prob = predict_batch(model, images, device, threshold)
    masks_np = masks.numpy()[:, 0]

    cells = []
    for i in range(len(sampled)):
        img_rgb = denormalize_image(images[i])
        gt_overlay = make_overlay(img_rgb, masks_np[i], color=(0, 255, 0), alpha=0.4)
        pred_overlay = make_overlay(img_rgb, preds_binary[i], color=(255, 0, 0), alpha=0.4)

        # Stack vertically: GT on top, prediction on bottom
        smooth = 1e-6
        inter = (preds_binary[i] * masks_np[i]).sum()
        dice = (2 * inter + smooth) / (preds_binary[i].sum() + masks_np[i].sum() + smooth)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(gt_overlay, "GT", (3, 12), font, 0.35, (255, 255, 255), 1)
        cv2.putText(pred_overlay, f"Pred D:{dice:.2f}", (3, 12), font, 0.35,
                    (255, 255, 255), 1)
        cell = np.vstack([gt_overlay, pred_overlay])
        cells.append(cell)

    # Arrange in grid
    rows_list = []
    for r in range(0, len(cells), cols):
        row_cells = cells[r:r + cols]
        # Pad incomplete row
        while len(row_cells) < cols:
            row_cells.append(np.zeros_like(cells[0]))
        rows_list.append(np.hstack(row_cells))

    grid = np.vstack(rows_list)
    grid_path = GRIDS_DIR / "summary_grid.png"
    cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    logger.info(f"Summary grid ({len(sampled)} patches) → {grid_path}")
    return grid_path


def print_metrics_summary(results: list[dict]) -> None:
    """Print aggregate metrics from overlay results."""
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


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Step 3: Inference & Visualisation")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint .pt file (default: outputs/checkpoints/checkpoint_best.pt)",
    )
    parser.add_argument(
        "--split", choices=["test", "val", "all"], default="test",
        help="Which split to run inference on (default: test)",
    )
    parser.add_argument(
        "--threshold", type=float, default=config.INFERENCE["probability_threshold"],
        help=f"Probability threshold (default: {config.INFERENCE['probability_threshold']})",
    )
    parser.add_argument(
        "--max-overlays", type=int, default=50,
        help="Max number of per-patch overlay images to save (default: 50)",
    )
    parser.add_argument(
        "--grid-samples", type=int, default=16,
        help="Number of patches in summary grid (default: 16)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")

    # Checkpoint
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (
        config.CHECKPOINTS_DIR / "checkpoint_best.pt"
    )
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        logger.error("Run Step 2 (2_train.py) first.")
        sys.exit(1)

    # Load model
    model = load_model(ckpt_path, device)

    # Load records and select split
    all_records = collect_all_records()
    if args.split == "all":
        records = all_records
        split_name = "all"
    else:
        train_rec, val_rec, test_rec = split_dataset(
            all_records,
            train_frac=config.TRAINING["train_split"],
            val_frac=config.TRAINING["val_split"],
            test_frac=config.TRAINING["test_split"],
            seed=config.TRAINING["random_seed"],
        )
        if args.split == "test":
            records = test_rec
            split_name = "test"
        else:
            records = val_rec
            split_name = "val"

    logger.info(f"Running inference on {len(records)} {split_name} patches "
                f"(threshold={args.threshold})")

    # Generate outputs
    results = generate_overlays(model, records, device, args.threshold,
                                max_overlays=args.max_overlays)
    grid_path = generate_summary_grid(model, records, device, args.threshold,
                                      n_samples=args.grid_samples)

    # Print summary
    print_metrics_summary(results)

    logger.info(f"\nOutputs saved to: {INFERENCE_DIR}")
    logger.info(f"  Overlays:      {OVERLAYS_DIR}")
    logger.info(f"  Summary grid:  {grid_path}")
    logger.info("\nNext: open the overlay images to inspect predictions visually.")


if __name__ == "__main__":
    main()

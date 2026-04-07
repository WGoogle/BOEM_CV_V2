#!/usr/bin/env python3
"""
Step 2 — Train Nodule Segmentation Model
==========================================
This script handles ONLY:
  - CLI argument parsing
  - Logging setup (file + console)
  - Pipeline manifest tracking (CoralNet-inspired idempotency)
  - Progress reporting and summary statistics

All training logic lives in the ``training/`` package:
  - training/dataset.py   — Dataset + augmentations
  - training/model.py     — U-Net factory + combined BCE+Dice loss
  - training/splits.py    — Stratified train/val/test splitting
  - training/trainer.py   — Training loop (AMP, early stopping, checkpoints)

Helpful Shortcuts:
    python 2_train.py                     # train on all preprocessed patches
    python 2_train.py --resume            # resume from last checkpoint
    python 2_train.py --epochs 50         # override epoch count
    python 2_train.py --batch-size 8      # override batch size (for small GPUs)
    python 2_train.py --no-augmentation   # disable augmentation (ablation)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import config
from training import (
    NoduleSegmentationDataset,
    build_model,
    CombinedLoss,
    split_dataset,
    save_split_info,
    Trainer,
    EpochLog,
    get_train_augmentations,
    get_val_augmentations,
)


# ── Logging ──────────────────────────────────────────────────────────────

def _setup_logging() -> None:
    """Configure root logger to write to console + log file."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOGS_DIR / "train.log"

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


# ── Pipeline Manifest ────────────────────────────────────────────────────

MANIFEST_PATH = config.OUTPUT_DIR / "pipeline_manifest.json"


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"version": "2.0", "mosaics": {}}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


# ── Data Loading ─────────────────────────────────────────────────────────

def _collect_patch_records() -> list[dict]:
    """Gather all patch records from every mosaic's patch_manifest.json."""
    records = []
    for manifest_file in sorted(config.PATCHES_DIR.glob("*/patch_manifest.json")):
        with open(manifest_file) as f:
            mosaic_records = json.load(f)
        # Validate paths exist
        valid = [
            r for r in mosaic_records
            if Path(r["image_path"]).exists() and Path(r["mask_path"]).exists()
        ]
        if len(valid) < len(mosaic_records):
            logger.warning(
                f"  {manifest_file.parent.name}: "
                f"{len(mosaic_records) - len(valid)} patches have missing files — skipped"
            )
        records.extend(valid)
    return records


# ── Epoch Logging Callback ───────────────────────────────────────────────

def _make_epoch_callback(history_path: Path):
    """Return a callback that logs each epoch and appends to a JSON history."""
    history = []

    def callback(log: EpochLog) -> None:
        logger.info(
            f"  Epoch {log.epoch:3d}  │  "
            f"loss {log.train_loss:.4f} / {log.val_loss:.4f}  │  "
            f"dice {log.train_dice:.4f} / {log.val_dice:.4f}  │  "
            f"iou {log.train_iou:.4f} / {log.val_iou:.4f}  │  "
            f"lr {log.lr:.2e}"
        )
        history.append({
            "epoch":      log.epoch,
            "train_loss": round(log.train_loss, 6),
            "val_loss":   round(log.val_loss, 6),
            "train_dice": round(log.train_dice, 6),
            "val_dice":   round(log.val_dice, 6),
            "train_iou":  round(log.train_iou, 6),
            "val_iou":    round(log.val_iou, 6),
            "lr":         log.lr,
        })
        # Write after every epoch so progress is never lost
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    return callback


# ── CLI Entry Point ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: Train U-Net segmentation model on preprocessed patches",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the last checkpoint",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--no-augmentation", action="store_true",
        help="Disable training augmentations (for ablation studies)",
    )
    args = parser.parse_args()

    _setup_logging()

    # ── Device selection ─────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info(f"  GPU: {gpu_name}  ({gpu_mem:.1f} GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("  Device: Apple MPS")
    else:
        device = "cpu"
        logger.info("  Device: CPU (training will be slow)")

    # ── Build effective config (CLI overrides) ───────────────────────
    train_cfg = dict(config.TRAINING)  # copy
    if args.epochs is not None:
        train_cfg["num_epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["learning_rate"] = args.lr
    if args.no_augmentation:
        train_cfg["augmentation"] = False

    # ── Banner ───────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("BOEM CV  —  Step 2: Train Segmentation Model")
    logger.info("=" * 70)
    logger.info(f"  Architecture    : {config.MODEL['architecture']} + "
                f"{config.MODEL['encoder_name']} "
                f"({config.MODEL['encoder_weights']} weights)")
    logger.info(f"  Batch size      : {train_cfg['batch_size']}")
    logger.info(f"  Epochs          : {train_cfg['num_epochs']}")
    logger.info(f"  Learning rate   : {train_cfg['learning_rate']}")
    logger.info(f"  Early stopping  : {train_cfg['early_stopping_patience']} epochs")
    logger.info(f"  Augmentation    : {train_cfg.get('augmentation', True)}")
    logger.info(f"  Device          : {device}")
    logger.info(f"  Resume          : {args.resume}")

    # ── Collect patches from Step 1 ──────────────────────────────────
    logger.info("─" * 70)
    logger.info("Loading patch records from Step 1 ...")
    records = _collect_patch_records()
    if not records:
        logger.error(
            "No patch records found.  Run Step 1 first:\n"
            "  python 1_preprocess_and_label.py"
        )
        sys.exit(1)
    logger.info(f"  Total patches: {len(records)}")

    # ── Split dataset ────────────────────────────────────────────────
    logger.info("Splitting dataset (stratified by nodule density) ...")
    train_recs, val_recs, test_recs = split_dataset(
        records,
        train_frac=train_cfg["train_split"],
        val_frac=train_cfg["val_split"],
        test_frac=train_cfg["test_split"],
        seed=train_cfg["random_seed"],
    )

    # Save split composition for reproducibility
    split_path = config.CHECKPOINTS_DIR / "split_info.json"
    save_split_info(train_recs, val_recs, test_recs, split_path, train_cfg["random_seed"])

    # ── Build datasets and loaders ───────────────────────────────────
    logger.info("Building datasets and data loaders ...")
    patch_size = config.PATCHING["patch_size"]
    use_aug = train_cfg.get("augmentation", True)

    train_transform = get_train_augmentations(patch_size) if use_aug else get_val_augmentations()
    val_transform   = get_val_augmentations()

    # Use manually corrected masks when available (from Step 4)
    corrected_dir = str(config.CORRECTED_MASKS_DIR) if config.CORRECTED_MASKS_DIR.exists() else None
    if corrected_dir:
        n_corrected = len(list(config.CORRECTED_MASKS_DIR.glob("*.png")))
        logger.info(f"  Found {n_corrected} manually corrected masks — will prefer over proxy labels")

    train_ds = NoduleSegmentationDataset(train_recs, transform=train_transform,
                                         corrected_masks_dir=corrected_dir)
    val_ds   = NoduleSegmentationDataset(val_recs,   transform=val_transform,
                                         corrected_masks_dir=corrected_dir)
    test_ds  = NoduleSegmentationDataset(test_recs,  transform=val_transform,
                                         corrected_masks_dir=corrected_dir)

    num_workers = train_cfg.get("num_workers", 4)
    pin_memory  = (device == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(
        f"  Train: {len(train_ds)} patches ({len(train_loader)} batches)  │  "
        f"Val: {len(val_ds)} patches  │  Test: {len(test_ds)} patches"
    )

    # ── Build model + loss ───────────────────────────────────────────
    logger.info("Building model ...")
    model = build_model(config.MODEL)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"  Parameters: {n_params:.1f}M")

    criterion = CombinedLoss(
        bce_weight=train_cfg.get("bce_weight", 0.5),
        dice_weight=train_cfg.get("dice_weight", 0.5),
    )

    # ── Build trainer ────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_cfg=train_cfg,
        checkpoint_dir=config.CHECKPOINTS_DIR,
        device=device,
    )

    # Resume if requested
    start_epoch = 0
    if args.resume:
        last_ckpt = config.CHECKPOINTS_DIR / "checkpoint_last.pt"
        if last_ckpt.exists():
            start_epoch = trainer.load_checkpoint(last_ckpt)
        else:
            logger.warning("  No checkpoint found to resume from — starting fresh")

    # ── Train ────────────────────────────────────────────────────────
    logger.info("─" * 70)
    logger.info("Starting training ...")
    logger.info(
        f"  {'Epoch':>7}  │  {'Train Loss / Val Loss':^23}  │  "
        f"{'Train Dice / Val Dice':^23}  │  "
        f"{'Train IoU / Val IoU':^23}  │  LR"
    )
    logger.info("  " + "─" * 100)

    history_path = config.CHECKPOINTS_DIR / "training_history.json"
    epoch_callback = _make_epoch_callback(history_path)

    t0 = time.time()
    result = trainer.fit(
        train_loader,
        val_loader,
        start_epoch=start_epoch,
        epoch_callback=epoch_callback,
    )
    elapsed = time.time() - t0

    # ── Evaluate on test set ─────────────────────────────────────────
    logger.info("─" * 70)
    logger.info("Evaluating best model on test set ...")
    best_ckpt = Path(result.best_checkpoint_path)
    if best_ckpt.exists():
        trainer.load_checkpoint(best_ckpt)
    test_loss, test_iou, test_dice = trainer._validate(test_loader)
    logger.info(f"  Test loss: {test_loss:.4f}  │  Test Dice: {test_dice:.4f}  │  Test IoU: {test_iou:.4f}")

    # ── Update pipeline manifest ─────────────────────────────────────
    manifest = _load_manifest()
    manifest.setdefault("training", {}).update({
        "model":                config.MODEL,
        "training_config":      train_cfg,
        "device":               device,
        "total_patches":        len(records),
        "split": {
            "train": len(train_recs),
            "val":   len(val_recs),
            "test":  len(test_recs),
        },
        "epochs_run":           result.epochs_run,
        "best_epoch":           result.best_epoch,
        "best_val_dice":        round(result.best_val_dice, 6),
        "test_metrics": {
            "loss":  round(test_loss, 6),
            "dice":  round(test_dice, 6),
            "iou":   round(test_iou, 6),
        },
        "best_checkpoint":      result.best_checkpoint_path,
        "last_checkpoint":      result.last_checkpoint_path,
        "training_history":     str(history_path),
        "completed":            True,
        "completed_at":         datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds":      round(elapsed, 1),
    })
    _save_manifest(manifest)

    # ── Final Summary ────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Epochs run        : {result.epochs_run}")
    logger.info(f"  Best epoch        : {result.best_epoch}")
    logger.info(f"  Best val Dice     : {result.best_val_dice:.4f}")
    logger.info(f"  Test Dice         : {test_dice:.4f}")
    logger.info(f"  Test IoU          : {test_iou:.4f}")
    logger.info(f"  Total time        : {elapsed:.1f}s")
    logger.info(f"  Best checkpoint   : {result.best_checkpoint_path}")
    logger.info(f"  Training history  : {history_path}")
    logger.info(f"  Manifest          : {MANIFEST_PATH}")
    logger.info("")
    logger.info("Next step: python 3_inference.py")


if __name__ == "__main__":
    main()

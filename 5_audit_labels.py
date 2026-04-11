"""
Step 5 — Confident-Learning Label Audit
=========================================
Ranks proxy-labelled patches by disagreement with a trained model and
writes the top-K worst patches into ``outputs/annotation_inbox/`` for
manual correction. Patches that already have a corrected mask in
``outputs/corrected_masks/`` are skipped automatically — they're
treated as ground truth.

All scoring logic lives in ``training/confident_learning.py``; this
script is a thin runner:
  - CLI parsing
  - Logging setup
  - Checkpoint loading
  - Progress reporting

Helpful shortcuts:
    python 5_audit_labels.py
    python 5_audit_labels.py --top-k 100
    python 5_audit_labels.py --checkpoint outputs/checkpoints/checkpoint_best.pt
    python 5_audit_labels.py --adaptive      # Northcutt-style thresholds
    python 5_audit_labels.py --no-viz        # skip visualization rendering
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import config
from training import (
    ConfidentLabelAuditor,
    export_audit_queue,
    load_model_from_checkpoint,
)


def _setup_logging() -> None:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOGS_DIR / "audit.log"

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


def _collect_patch_records() -> list[dict]:
    """Gather all patch records from every mosaic's patch_manifest.json."""
    records = []
    for manifest_file in sorted(config.PATCHES_DIR.glob("*/patch_manifest.json")):
        with open(manifest_file) as f:
            mosaic_records = json.load(f)
        valid = [
            r for r in mosaic_records
            if Path(r["image_path"]).exists() and Path(r["mask_path"]).exists()
        ]
        records.extend(valid)
    return records


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 5: Confident-Learning label audit for proxy masks",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Path to model checkpoint. Default: outputs/checkpoints/checkpoint_best.pt",
    )
    parser.add_argument(
        "--top-k", type=int, default=200,
        help="Number of worst-disagreement patches to queue (default: 200)",
    )
    parser.add_argument(
        "--low-thresh", type=float, default=0.15,
        help="Probability below which model is 'confidently negative' (default 0.15)",
    )
    parser.add_argument(
        "--high-thresh", type=float, default=0.85,
        help="Probability above which model is 'confidently positive' (default 0.85)",
    )
    parser.add_argument(
        "--w-fp", type=float, default=0.6,
        help="Combined-score weight on confident FP fraction (default 0.6)",
    )
    parser.add_argument(
        "--w-fn", type=float, default=0.3,
        help="Combined-score weight on confident FN fraction (default 0.3)",
    )
    parser.add_argument(
        "--w-dd", type=float, default=0.1,
        help="Combined-score weight on Dice disagreement (default 0.1)",
    )
    parser.add_argument(
        "--adaptive", action="store_true",
        help="Use cleanlab-style adaptive class-conditional thresholds",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Inference batch size for the audit pass (default 16)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader workers (default 4)",
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip visualization rendering (CSV/JSON only)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory (default: config.ANNOTATION_INBOX)",
    )
    args = parser.parse_args()

    _setup_logging()

    # Resolve checkpoint
    ckpt_path = args.checkpoint or (config.CHECKPOINTS_DIR / "checkpoint_best.pt")
    if not ckpt_path.exists():
        logger.error(
            f"Checkpoint not found at {ckpt_path}. "
            f"Train a model first with `python 2_train.py`."
        )
        sys.exit(1)

    device = _pick_device()
    patch_size = config.PATCHING["patch_size"]
    output_dir = args.output_dir or config.ANNOTATION_INBOX

    logger.info("=" * 70)
    logger.info("BOEM CV  —  Step 5: Confident-Learning Label Audit")
    logger.info("=" * 70)
    logger.info(f"  Checkpoint    : {ckpt_path}")
    logger.info(f"  Device        : {device}")
    logger.info(f"  Patch size    : {patch_size}")
    logger.info(f"  Top-K         : {args.top_k}")
    logger.info(f"  low/high      : {args.low_thresh}/{args.high_thresh}")
    logger.info(f"  weights (fp,fn,dd): {args.w_fp}/{args.w_fn}/{args.w_dd}")
    logger.info(f"  Adaptive      : {args.adaptive}")
    logger.info(f"  Output dir    : {output_dir}")

    # Load model
    logger.info("─" * 70)
    logger.info("Loading model ...")
    model = load_model_from_checkpoint(
        ckpt_path, config.MODEL, device=device, patch_size=patch_size,
    )

    # Collect patch records
    logger.info("Collecting patch records from Step 1 ...")
    records = _collect_patch_records()
    if not records:
        logger.error("No patch records found. Run Step 1 first.")
        sys.exit(1)
    logger.info(f"  Total patches: {len(records)}")

    corrected_dir = config.CORRECTED_MASKS_DIR if config.CORRECTED_MASKS_DIR.exists() else None
    if corrected_dir:
        n_corrected = len(list(corrected_dir.glob("*.png")))
        logger.info(f"  Found {n_corrected} already-corrected patches — will be skipped")

    # Build auditor — input_mode must match the training config so the
    # checkpoint sees the channel representation it was trained on.
    input_mode = config.MODEL.get("input_mode", "rgb")
    logger.info(f"  Input mode    : {input_mode}")
    auditor = ConfidentLabelAuditor(
        model=model,
        device=device,
        low_thresh=args.low_thresh,
        high_thresh=args.high_thresh,
        w_fp=args.w_fp,
        w_fn=args.w_fn,
        w_dd=args.w_dd,
        adaptive_thresholds=args.adaptive,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_mode=input_mode,
    )

    # Score
    logger.info("─" * 70)
    logger.info("Scoring patches ...")
    scores = auditor.score(records, corrected_masks_dir=corrected_dir)

    if not scores:
        logger.info("Nothing to audit. Exiting.")
        return

    # Summary stats
    n = len(scores)
    median = scores[n // 2].combined_score
    top10 = scores[: max(1, n // 10)]
    top10_mean = sum(s.combined_score for s in top10) / len(top10)
    mean_fp = sum(s.confident_fp_frac for s in scores) / n
    mean_fn = sum(s.confident_fn_frac for s in scores) / n

    logger.info(
        f"  Scored {n} patches  │  "
        f"median_score={median:.4f}  │  "
        f"top-10%_mean={top10_mean:.4f}  │  "
        f"mean_fp_frac={mean_fp:.4f}  │  "
        f"mean_fn_frac={mean_fn:.4f}"
    )

    # Export top-K
    logger.info("─" * 70)
    logger.info("Writing audit queue ...")
    csv_path = export_audit_queue(
        scores,
        output_dir=output_dir,
        top_k=args.top_k,
        save_visualizations=(not args.no_viz),
    )

    # Append a small manifest entry
    audit_manifest = output_dir / "audit_manifest.json"
    existing = []
    if audit_manifest.exists():
        with open(audit_manifest) as f:
            existing = json.load(f)
    existing.append({
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "checkpoint":    str(ckpt_path),
        "total_scored":  n,
        "top_k":         args.top_k,
        "low_thresh":    auditor.low_thresh,
        "high_thresh":   auditor.high_thresh,
        "weights":       {"fp": args.w_fp, "fn": args.w_fn, "dd": args.w_dd},
        "adaptive":      args.adaptive,
        "queue_csv":     str(csv_path),
    })
    with open(audit_manifest, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info("=" * 70)
    logger.info("AUDIT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Queue CSV     : {csv_path}")
    logger.info(f"  Visualizations: {output_dir / 'visualizations'}")
    logger.info(f"  Manifest      : {audit_manifest}")
    logger.info("")
    logger.info("Next step: correct the queued patches, place corrected masks in")
    logger.info(f"  {config.CORRECTED_MASKS_DIR}")
    logger.info("then re-run `python 2_train.py` to train on the improved labels.")


if __name__ == "__main__":
    main()

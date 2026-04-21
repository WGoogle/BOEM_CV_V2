"""
Step 5 — DICE-Based Label Audit
Please refer to AUDIT_PIPELINE.txt for instructions on how to use. 

Helpful shortcuts:
    python 5_audit_labels.py --top-k 150 (USE IF ONLY WANT TO AUDIT PATCHES THAT HAVE YET TO BE MANUALLY CORRECTED)
    python 5_audit_labels.py --include-corrected --top-k 150 (USE IF WANT TO RE-AUDIT ALL PATCHES INCLUDING ONES THAT HAVE BEEN MANUALLY CORRECTED — HELPFUL FOR CATCHING HUMAN ANNOTATION ERRORS)
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import config
from training import export_dice_audit_queue

def _setup_logging():
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

INFERENCE_METRICS_PATH = config.RESULTS_DIR / "inference" / "patch_metrics.json"

def _load_patch_metrics():
    if not INFERENCE_METRICS_PATH.exists():
        logger.error(
            f"Patch metrics not found at {INFERENCE_METRICS_PATH}. "
            f"Run `python 3_inference.py` first to generate them."
        )
        sys.exit(1)
    with open(INFERENCE_METRICS_PATH) as f:
        metrics = json.load(f)
    logger.info(f"  Loaded {len(metrics)} patch metrics from {INFERENCE_METRICS_PATH}")
    return metrics


def _filter_metrics(metrics, include_corrected):
    corrected_ids = set()
    if not include_corrected and config.CORRECTED_MASKS_DIR.exists():
        corrected_ids = {p.stem for p in config.CORRECTED_MASKS_DIR.glob("*.png")}
        logger.info(f"  Found {len(corrected_ids)} already-corrected patches — will be skipped")
    elif include_corrected:
        logger.info("  --include-corrected set: corrected patches will be re-ranked too")

    kept = []
    skipped_corrected = 0
    skipped_missing = 0
    skipped_true_negative = 0
    for m in metrics:
        if m["patch_id"] in corrected_ids:
            skipped_corrected += 1
            continue
        if not Path(m["image_path"]).exists() or not Path(m["mask_path"]).exists():
            skipped_missing += 1
            continue
        # Dice is None (serialized NaN) for true-negative patches where both
        # GT and prediction are empty — nothing to audit, no real disagreement.
        if m.get("dice") is None:
            skipped_true_negative += 1
            continue
        kept.append(m)

    logger.info(
        f"  Filtered metrics: kept {len(kept)}  │  "
        f"skipped_corrected={skipped_corrected}  "
        f"skipped_missing={skipped_missing}  "
        f"skipped_true_negative={skipped_true_negative}"
    )
    return kept

def main():
    parser = argparse.ArgumentParser(
        description="Step 5: DICE-based label audit (worst patches first)",
    )
    parser.add_argument(
        "--top-k", type=int, default=150,
        help="Number of worst-DICE patches to queue (default: 150)",
    )
    parser.add_argument(
        "--include-corrected", action="store_true",
        help="Also audit patches that already have a corrected mask "
             "(useful for catching human annotation errors)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory (default: config.ANNOTATION_INBOX)",
    )
    args = parser.parse_args()

    _setup_logging()
    output_dir = args.output_dir or config.ANNOTATION_INBOX
    logger.info("=" * 70)
    logger.info("BOEM CV  —  Step 5: DICE-Based Label Audit")
    logger.info("=" * 70)
    logger.info(f"  Metrics file  : {INFERENCE_METRICS_PATH}")
    logger.info(f"  Top-K         : {args.top_k}")
    logger.info(f"  Output dir    : {output_dir}")
    logger.info("─" * 70)
    metrics = _load_patch_metrics()
    kept = _filter_metrics(metrics, include_corrected=args.include_corrected)

    if not kept:
        logger.info("Nothing to audit. Exiting.")
        return

    # Rank ascending by DICE — worst patches first
    kept.sort(key=lambda m: m["dice"])

    n = len(kept)
    dices = [m["dice"] for m in kept]
    worst = dices[0]
    median = dices[n // 2]
    top_k_mean = sum(dices[: min(args.top_k, n)]) / min(args.top_k, n)
    logger.info(
        f"  Ranked {n} patches by DICE  │  "
        f"worst={worst:.4f}  │  "
        f"median={median:.4f}  │  "
        f"top-{args.top_k}_mean_dice={top_k_mean:.4f}"
    )

    logger.info("─" * 70)
    logger.info("Writing audit queue ...")
    pred_masks_dir = output_dir / "pred_masks"
    csv_path = export_dice_audit_queue(
        kept,
        output_dir=output_dir,
        top_k=args.top_k,
        pred_masks_dir=pred_masks_dir if pred_masks_dir.exists() else None,
    )

    audit_manifest = output_dir / "audit_manifest.json"
    existing = []
    if audit_manifest.exists():
        with open(audit_manifest) as f:
            existing = json.load(f)
    existing.append({
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "metrics_file":  str(INFERENCE_METRICS_PATH),
        "total_scored":  n,
        "top_k":         args.top_k,
        "ranking":       "dice_ascending",
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
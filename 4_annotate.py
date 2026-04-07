#!/usr/bin/env python3
"""
Step 4 — Manual Annotation & Collaboration
============================================
Correct proxy-label masks by hand, export batches for collaborators,
and import their corrections back into the pipeline.

Usage:
    # ── Annotate (lead annotator, has full repo) ─────────────────
    python 4_annotate.py edit --split test --annotator alice
    python 4_annotate.py edit --worst 20 --annotator alice
    python 4_annotate.py edit --unannotated --annotator alice
    python 4_annotate.py edit --mosaic CameraMosaic_D1_Node3_L8 --annotator alice

    # ── Annotate from bundle (collaborator, no full repo needed) ─
    python 4_annotate.py edit --bundle batch_for_bob.zip --annotator bob

    # ── Collaborate ──────────────────────────────────────────────
    # Export patches for a collaborator
    python 4_annotate.py export --split test --output batch_for_bob.zip --annotator alice

    # Bob annotates, then repacks his corrections
    python 4_annotate.py repack --work-dir batch_for_bob_work --output corrected_by_bob.zip --annotator bob

    # Alice imports Bob's corrections
    python 4_annotate.py import --bundle corrected_by_bob.zip

    # Inspect a bundle without importing
    python 4_annotate.py inspect --bundle corrected_by_bob.zip

    # ── Status ───────────────────────────────────────────────────
    python 4_annotate.py status
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

import config
from annotation.editor import AnnotationEditor
from annotation.collaborate import (
    export_bundle, import_bundle, list_bundle_contents,
    open_bundle, repack_bundle,
)
from annotation.tracker import AnnotationTracker

# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-12s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────
CORRECTED_MASKS_DIR = config.OUTPUT_DIR / "corrected_masks"
TRACKER_PATH        = config.OUTPUT_DIR / "annotation_tracker.json"
BUNDLES_DIR         = config.OUTPUT_DIR / "annotation_bundles"
INBOX_DIR           = config.OUTPUT_DIR / "annotation_inbox"


# ── Record loading ──────────────────────────────────────────────────────

def load_all_records() -> list[dict]:
    """Load all patch records from every mosaic's manifest."""
    records = []
    for mosaic_dir in sorted(config.PATCHES_DIR.iterdir()):
        manifest = mosaic_dir / "patch_manifest.json"
        if not manifest.exists():
            continue
        with open(manifest) as f:
            records.extend(json.load(f))
    return records


def filter_by_split(records: list[dict], split: str) -> list[dict]:
    """Filter records to a specific train/val/test split."""
    from training.splits import split_dataset
    train_rec, val_rec, test_rec = split_dataset(
        records,
        train_frac=config.TRAINING["train_split"],
        val_frac=config.TRAINING["val_split"],
        test_frac=config.TRAINING["test_split"],
        seed=config.TRAINING["random_seed"],
    )
    return {"train": train_rec, "val": val_rec, "test": test_rec, "all": records}[split]


def filter_by_mosaic(records: list[dict], mosaic_name: str) -> list[dict]:
    """Filter records to a specific mosaic."""
    return [r for r in records if mosaic_name in r.get("patch_id", "")]


def filter_worst_patches(records: list[dict], n: int) -> list[dict]:
    """Load inference results and return the n worst-performing patches."""
    inference_dir = config.RESULTS_DIR / "inference" / "overlays"
    if not inference_dir.exists():
        logger.warning("No inference results found. Run 3_inference.py first.")
        logger.warning("Falling back to all records.")
        return records[:n]

    # Try to load metrics from inference run
    # Match records to inference overlay files to find worst patches
    # For now, use a heuristic: patches with low coverage tend to be harder
    sorted_recs = sorted(
        records,
        key=lambda r: r.get("label_stats", {}).get("coverage_pct", 0),
    )
    return sorted_recs[:n]


# ── Subcommands ─────────────────────────────────────────────────────────

def cmd_edit(args):
    """Open the annotation editor."""

    # ── Bundle mode: collaborator opens a .zip directly ──────────
    if args.bundle:
        bundle_path = Path(args.bundle)
        if not bundle_path.exists():
            logger.error(f"Bundle not found: {bundle_path}")
            sys.exit(1)

        records, masks_dir = open_bundle(bundle_path)
        logger.info(f"Opened bundle: {len(records)} patches")
        logger.info(f"Corrected masks will be saved to: {masks_dir}")

        if not records:
            logger.info("Bundle is empty!")
            return

        editor = AnnotationEditor(
            records=records,
            output_dir=masks_dir,
            annotator=args.annotator,
        )
        editor.launch(start_idx=args.start)

        # After editor closes, tell user how to send corrections back
        work_dir = masks_dir.parent
        n_modified = len(editor.modified_patches)
        logger.info(f"\nAnnotated {n_modified} patches.")
        if n_modified > 0:
            logger.info(f"To send corrections back, run:")
            logger.info(f"  python 4_annotate.py repack "
                        f"--work-dir {work_dir} "
                        f"--output corrected_by_{args.annotator}.zip "
                        f"--annotator {args.annotator}")
        return

    # ── Normal mode: lead annotator with full repo ───────────────
    all_records = load_all_records()
    logger.info(f"Loaded {len(all_records)} total patches")

    # Filter records based on args
    if args.mosaic:
        records = filter_by_mosaic(all_records, args.mosaic)
        logger.info(f"Filtered to {len(records)} patches from {args.mosaic}")
    elif args.worst:
        records = filter_by_split(all_records, args.split)
        records = filter_worst_patches(records, args.worst)
        logger.info(f"Selected {len(records)} worst-performing patches")
    elif args.unannotated:
        records = filter_by_split(all_records, args.split)
        tracker = AnnotationTracker(TRACKER_PATH)
        all_ids = [r.get("patch_id") for r in records]
        unannotated_ids = set(tracker.get_unannotated_ids(all_ids))
        records = [r for r in records if r.get("patch_id") in unannotated_ids]
        logger.info(f"Filtered to {len(records)} unannotated patches")
    else:
        records = filter_by_split(all_records, args.split)
        logger.info(f"Using {len(records)} patches from '{args.split}' split")

    if not records:
        logger.info("No patches to annotate!")
        return

    CORRECTED_MASKS_DIR.mkdir(parents=True, exist_ok=True)

    editor = AnnotationEditor(
        records=records,
        output_dir=CORRECTED_MASKS_DIR,
        annotator=args.annotator,
    )
    editor.launch(start_idx=args.start)

    # After editor closes, update tracker
    tracker = AnnotationTracker(TRACKER_PATH)
    for patch_id in editor.modified_patches:
        tracker.record_annotation(patch_id, args.annotator)
    logger.info(f"Updated tracker: {len(editor.modified_patches)} patches annotated")


def cmd_export(args):
    """Export patches as a shareable bundle."""
    all_records = load_all_records()

    if args.unannotated:
        records = filter_by_split(all_records, args.split)
        tracker = AnnotationTracker(TRACKER_PATH)
        all_ids = [r.get("patch_id") for r in records]
        unannotated_ids = set(tracker.get_unannotated_ids(all_ids))
        records = [r for r in records if r.get("patch_id") in unannotated_ids]
        logger.info(f"Exporting {len(records)} unannotated patches")
    elif args.mosaic:
        records = filter_by_mosaic(all_records, args.mosaic)
    else:
        records = filter_by_split(all_records, args.split)

    if args.max_patches and len(records) > args.max_patches:
        records = records[:args.max_patches]
        logger.info(f"Capped to {args.max_patches} patches")

    if not records:
        logger.info("No patches to export!")
        return

    BUNDLES_DIR.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.output else (
        BUNDLES_DIR / f"annotation_bundle_{args.split}.zip"
    )

    export_bundle(
        records=records,
        output_path=output,
        corrected_masks_dir=CORRECTED_MASKS_DIR if CORRECTED_MASKS_DIR.exists() else None,
        annotator=args.annotator,
        notes=args.notes or "",
    )


def cmd_import(args):
    """Import corrected masks from a collaborator's bundle."""
    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        logger.error(f"Bundle not found: {bundle_path}")
        sys.exit(1)

    summary = import_bundle(
        bundle_path=bundle_path,
        corrected_masks_dir=CORRECTED_MASKS_DIR,
        merge_strategy=args.strategy,
    )

    # Update tracker for imported patches
    tracker = AnnotationTracker(TRACKER_PATH)
    # Re-read the imported metadata to find patch IDs
    from annotation.collaborate import list_bundle_contents
    meta = list_bundle_contents(bundle_path)
    annotator = meta.get("created_by", "unknown")
    for pid in meta.get("actual_patches_found", []):
        tracker.record_annotation(pid, annotator)

    logger.info(f"Import summary: {json.dumps(summary, indent=2)}")


def cmd_import_all(args):
    """Import all .zip bundles from the annotation_inbox folder."""
    zips = sorted(INBOX_DIR.glob("*.zip"))
    if not zips:
        logger.info(f"No .zip files found in {INBOX_DIR}")
        logger.info("Ask collaborators to drop their bundles there.")
        return

    logger.info(f"Found {len(zips)} bundle(s) in inbox")
    imported_dir = INBOX_DIR / "imported"
    imported_dir.mkdir(exist_ok=True)

    tracker = AnnotationTracker(TRACKER_PATH)
    total_imported = 0

    for zip_path in zips:
        logger.info(f"\nProcessing: {zip_path.name}")
        summary = import_bundle(
            bundle_path=zip_path,
            corrected_masks_dir=CORRECTED_MASKS_DIR,
            merge_strategy=args.strategy,
        )
        total_imported += summary["imported"]

        # Update tracker
        meta = list_bundle_contents(zip_path)
        annotator = meta.get("created_by", "unknown")
        for pid in meta.get("actual_patches_found", []):
            tracker.record_annotation(pid, annotator)

        # Move processed bundle to imported/
        dest = imported_dir / zip_path.name
        zip_path.rename(dest)
        logger.info(f"  Moved → {dest}")

    logger.info(f"\nDone. Imported {total_imported} masks from {len(zips)} bundle(s).")


def cmd_inspect(args):
    """Inspect a bundle without importing."""
    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        logger.error(f"Bundle not found: {bundle_path}")
        sys.exit(1)

    meta = list_bundle_contents(bundle_path)
    print(json.dumps(meta, indent=2))


def cmd_repack(args):
    """Re-pack a working directory into a bundle for return to lead annotator."""
    work_dir = Path(args.work_dir)
    if not work_dir.exists():
        logger.error(f"Working directory not found: {work_dir}")
        sys.exit(1)

    output = Path(args.output) if args.output else (
        work_dir.parent / f"corrected_by_{args.annotator}.zip"
    )

    repack_bundle(
        work_dir=work_dir,
        output_path=output,
        annotator=args.annotator,
    )
    logger.info(f"Send this file back to the lead annotator: {output}")


def cmd_status(args):
    """Show annotation progress."""
    all_records = load_all_records()
    tracker = AnnotationTracker(TRACKER_PATH)
    status = tracker.get_status(len(all_records))

    # Check corrected masks on disk
    n_corrected_files = 0
    if CORRECTED_MASKS_DIR.exists():
        n_corrected_files = len(list(CORRECTED_MASKS_DIR.glob("*.png")))

    print("\n" + "=" * 50)
    print("  ANNOTATION PROGRESS")
    print("=" * 50)
    print(f"  Total patches       : {status['total_patches']}")
    print(f"  Annotated (tracked) : {status['annotated']}")
    print(f"  Corrected on disk   : {n_corrected_files}")
    print(f"  Remaining           : {status['remaining']}")
    print(f"  Progress            : {status['percent_complete']:.1f}%")
    print()
    if status["annotators"]:
        print("  Annotators:")
        for name, count in status["annotators"].items():
            print(f"    {name:20s} : {count} patches")
    print("=" * 50)


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Manual Annotation & Collaboration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── edit ─────────────────────────────────────────────────────────
    p_edit = sub.add_parser("edit", help="Open the annotation editor")
    p_edit.add_argument("--bundle", type=str, default=None,
                        help="Open a .zip bundle directly (for collaborators)")
    p_edit.add_argument("--split", choices=["train", "val", "test", "all"],
                        default="test")
    p_edit.add_argument("--mosaic", type=str, default=None,
                        help="Filter to a specific mosaic name")
    p_edit.add_argument("--worst", type=int, default=None,
                        help="Select N worst-performing patches")
    p_edit.add_argument("--unannotated", action="store_true",
                        help="Only show patches not yet annotated")
    p_edit.add_argument("--annotator", type=str, default="anonymous",
                        help="Your name/ID for provenance tracking")
    p_edit.add_argument("--start", type=int, default=0,
                        help="Index of first patch to display")

    # ── export ───────────────────────────────────────────────────────
    p_export = sub.add_parser("export", help="Export patches for collaboration")
    p_export.add_argument("--split", choices=["train", "val", "test", "all"],
                          default="test")
    p_export.add_argument("--mosaic", type=str, default=None)
    p_export.add_argument("--unannotated", action="store_true")
    p_export.add_argument("--max-patches", type=int, default=None,
                          help="Limit number of patches in bundle")
    p_export.add_argument("--output", "-o", type=str, default=None,
                          help="Output .zip path")
    p_export.add_argument("--annotator", type=str, default="anonymous")
    p_export.add_argument("--notes", type=str, default=None,
                          help="Instructions for collaborators")

    # ── import ───────────────────────────────────────────────────────
    p_import = sub.add_parser("import", help="Import corrected masks from bundle")
    p_import.add_argument("--bundle", "-b", type=str, required=True,
                          help="Path to .zip bundle")
    p_import.add_argument("--strategy", choices=["newest", "overwrite", "skip"],
                          default="newest",
                          help="Merge strategy for conflicts (default: newest)")

    # ── import-all ────────────────────────────────────────────────────
    p_import_all = sub.add_parser("import-all",
                                  help="Import all bundles from annotation_inbox/")
    p_import_all.add_argument("--strategy", choices=["newest", "overwrite", "skip"],
                              default="newest")

    # ── repack ───────────────────────────────────────────────────────
    p_repack = sub.add_parser("repack", help="Re-pack corrections into a bundle")
    p_repack.add_argument("--work-dir", type=str, required=True,
                          help="Working directory created by 'edit --bundle'")
    p_repack.add_argument("--output", "-o", type=str, default=None,
                          help="Output .zip path")
    p_repack.add_argument("--annotator", type=str, default="anonymous")

    # ── inspect ──────────────────────────────────────────────────────
    p_inspect = sub.add_parser("inspect", help="Inspect bundle contents")
    p_inspect.add_argument("--bundle", "-b", type=str, required=True)

    # ── status ───────────────────────────────────────────────────────
    sub.add_parser("status", help="Show annotation progress")

    args = parser.parse_args()

    commands = {
        "edit": cmd_edit,
        "export": cmd_export,
        "import": cmd_import,
        "import-all": cmd_import_all,
        "repack": cmd_repack,
        "inspect": cmd_inspect,
        "status": cmd_status,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()

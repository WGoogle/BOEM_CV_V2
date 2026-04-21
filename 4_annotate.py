"""
Step 4 — Manual Annotation & Collaboration
Reference ANNOATIONS.txt for detailed instructions.
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
import sys
from pathlib import Path
import cv2
import numpy as np
import config
from annotation.editor import AnnotationEditor
from annotation.collaborate import (
    export_bundle, import_bundle, list_bundle_contents,
    open_bundle, repack_bundle,
)
from annotation.tracker import AnnotationTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-12s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
CORRECTED_MASKS_DIR = config.OUTPUT_DIR / "corrected_masks"
TRACKER_PATH        = config.OUTPUT_DIR / "annotation_tracker.json"
BUNDLES_DIR         = config.OUTPUT_DIR / "annotation_bundles"
INBOX_DIR           = config.OUTPUT_DIR / "annotation_inbox"

def load_all_records():
    records = []
    for mosaic_dir in sorted(config.PATCHES_DIR.iterdir()):
        manifest = mosaic_dir / "patch_manifest.json"
        if not manifest.exists():
            continue
        with open(manifest) as f:
            records.extend(json.load(f))
    return records

def filter_by_split(records, split):
    from training.splits import split_dataset
    train_rec, val_rec, test_rec = split_dataset(
        records,
        train_frac=config.TRAINING["train_split"],
        val_frac=config.TRAINING["val_split"],
        test_frac=config.TRAINING["test_split"],
        seed=config.TRAINING["random_seed"],
    )
    return {"train": train_rec, "val": val_rec, "test": test_rec, "all": records}[split]

def filter_by_mosaic(records, mosaic_name):
    return [r for r in records if mosaic_name in r.get("patch_id", "")]

def load_audit_queue(csv_path, all_records):
    # Return records whose patch_id is in audit_queue.csv, ordered by rank.
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error(f"Audit queue CSV not found: {csv_path}")
        logger.error("Run `python 5_audit_labels.py` first.")
        sys.exit(1)

    by_id = {r.get("patch_id"): r for r in all_records}
    ordered = []
    missing = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            pid = row["patch_id"]
            rec = by_id.get(pid)
            if rec is None:
                missing.append(pid)
            else:
                ordered.append(rec)
    if missing:
        logger.warning(
            f"{len(missing)} audit patch_ids not found in patch manifests "
            f"(first few: {missing[:3]})"
        )
    return ordered

def filter_worst_patches(records, n):
    pred_masks_dir = config.OUTPUT_DIR / "annotation_inbox" / "pred_masks"
    if not pred_masks_dir.exists():
        logger.warning(
            "No predicted masks found. Run 5_audit_labels.py first to "
            "generate model predictions."
        )
        logger.warning("Falling back to all records (unranked).")
        return records[:n]

    scored: list[tuple[int, dict]] = []  # (missed_pixels, record)
    for rec in records:
        patch_id = rec.get("patch_id", "")
        pred_path = pred_masks_dir / f"{patch_id}.png"
        gt_path = Path(rec["mask_path"])

        if not pred_path.exists() or not gt_path.exists():
            scored.append((0, rec))
            continue

        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None or pred_mask is None:
            scored.append((0, rec))
            continue

        gt_bin = (gt_mask > 127).astype(np.uint8)
        pred_bin = (pred_mask > 127).astype(np.uint8)
        missed_pixels = int(np.sum(gt_bin != pred_bin))
        scored.append((missed_pixels, rec))

    # Sort descending: most disagreement pixels first
    scored.sort(key=lambda t: t[0], reverse=True)
    return [rec for _, rec in scored[:n]]

def cmd_edit(args):
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

        # Auto-resume: start at the first patch that hasn't been corrected yet
        start = args.start
        if start == 0 and masks_dir.exists():
            corrected_ids = {p.stem for p in masks_dir.glob("*.png")}
            if corrected_ids:
                for i, rec in enumerate(records):
                    if rec.get("patch_id") not in corrected_ids:
                        start = i
                        break
                else:
                    start = 0 
                logger.info(f"Resuming at patch {start + 1}/{len(records)} "
                            f"({len(corrected_ids)} already corrected)")

        editor = AnnotationEditor(
            records=records,
            output_dir=masks_dir,
            annotator=args.annotator,
        )
        editor.launch(start_idx=start)

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

    all_records = load_all_records()
    logger.info(f"Loaded {len(all_records)} total patches")
    pred_masks_dir = None
    if args.audit_queue is not None:
        queue_path = Path(args.audit_queue) if args.audit_queue else (
            INBOX_DIR / "audit_queue.csv"
        )  
        records = load_audit_queue(queue_path, all_records)
        logger.info(f"Loaded {len(records)} patches from audit queue: {queue_path}")
        candidate = queue_path.parent / "pred_masks"
        if candidate.exists():
            pred_masks_dir = candidate
            logger.info(f"Model prediction masks found: {pred_masks_dir}")
        else:
            logger.info("No pred_masks/ folder found next to audit queue — "
                        "re-run 5_audit_labels.py to generate them.")
    elif args.mosaic:
        records = filter_by_mosaic(all_records, args.mosaic)
        logger.info(f"Filtered to {len(records)} patches from {args.mosaic}")
    elif args.worst:
        records = filter_by_split(all_records, args.split)
        records = filter_worst_patches(records, args.worst)
        logger.info(f"Selected {len(records)} worst-performing patches")
    else:
        records = filter_by_split(all_records, args.split)
        logger.info(f"Using {len(records)} patches from '{args.split}' split")
    if args.unannotated:
        tracker = AnnotationTracker(TRACKER_PATH)
        tracked_ids = set(tracker.data.get("patches", {}).keys())
        disk_ids = set()
        if CORRECTED_MASKS_DIR.exists():
            disk_ids = {p.stem for p in CORRECTED_MASKS_DIR.glob("*.png")}
        done_ids = tracked_ids | disk_ids
        before = len(records)
        records = [r for r in records if r.get("patch_id") not in done_ids]
        logger.info(f"Filtered to {len(records)} unannotated patches "
                    f"(removed {before - len(records)} already corrected)")

    if not records:
        logger.info("No patches to annotate!")
        return

    CORRECTED_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    editor = AnnotationEditor(
        records=records,
        output_dir=CORRECTED_MASKS_DIR,
        annotator=args.annotator,
        pred_masks_dir=pred_masks_dir,
        show_progress=args.audit_queue is None,
    )
    editor.launch(start_idx=args.start)

    tracker = AnnotationTracker(TRACKER_PATH)
    touched = editor.modified_patches | editor.reviewed_patches
    for patch_id in touched:
        tracker.record_annotation(patch_id, args.annotator)
    reviewed_only = len(touched) - len(editor.modified_patches)
    logger.info(
        f"Updated tracker: {len(editor.modified_patches)} modified, "
        f"{reviewed_only} reviewed as-is"
    )

def cmd_export(args):
    all_records = load_all_records()
    if args.mosaic:
        records = filter_by_mosaic(all_records, args.mosaic)
    else:
        records = filter_by_split(all_records, args.split)
    if args.unannotated:
        tracker = AnnotationTracker(TRACKER_PATH)
        tracked_ids = set(tracker.data.get("patches", {}).keys())
        disk_ids = set()
        if CORRECTED_MASKS_DIR.exists():
            disk_ids = {p.stem for p in CORRECTED_MASKS_DIR.glob("*.png")}
        done_ids = tracked_ids | disk_ids
        before = len(records)
        records = [r for r in records if r.get("patch_id") not in done_ids]
        logger.info(f"Exporting {len(records)} unannotated patches "
                    f"(removed {before - len(records)} already corrected)")
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
    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        logger.error(f"Bundle not found: {bundle_path}")
        sys.exit(1)
    summary = import_bundle(
        bundle_path=bundle_path,
        corrected_masks_dir=CORRECTED_MASKS_DIR,
        merge_strategy=args.strategy,
    )
    tracker = AnnotationTracker(TRACKER_PATH)
    meta = list_bundle_contents(bundle_path)
    annotator = meta.get("created_by", "unknown")
    corrected_ids = meta.get("corrected_ids", [])
    for pid in corrected_ids:
        tracker.record_annotation(pid, annotator)

    logger.info(f"Import summary: {json.dumps(summary, indent=2)}")

def cmd_import_all(args):
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

        meta = list_bundle_contents(zip_path)
        annotator = meta.get("created_by", "unknown")
        corrected_ids = meta.get("corrected_ids", [])
        for pid in corrected_ids:
            tracker.record_annotation(pid, annotator)
        dest = imported_dir / zip_path.name
        zip_path.rename(dest)
        logger.info(f"  Moved → {dest}")

    logger.info(f"\nDone. Imported {total_imported} masks from {len(zips)} bundle(s).")

def cmd_inspect(args):
    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        logger.error(f"Bundle not found: {bundle_path}")
        sys.exit(1)

    meta = list_bundle_contents(bundle_path)
    print(json.dumps(meta, indent=2))

def cmd_repack(args):
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
    all_records = load_all_records()
    tracker = AnnotationTracker(TRACKER_PATH)
    status = tracker.get_status(len(all_records))
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

def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Manual Annotation & Collaboration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # editing 
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
    p_edit.add_argument("--audit-queue", nargs="?", const="", default=None,
                        metavar="CSV",
                        help="Load patches from an audit_queue.csv produced by "
                             "5_audit_labels.py. With no value, uses "
                             "outputs/annotation_inbox/audit_queue.csv.")
    p_edit.add_argument("--annotator", type=str, default="anonymous",
                        help="Your name/ID for provenance tracking")
    p_edit.add_argument("--start", type=int, default=0,
                        help="Index of first patch to display")

    # exporting
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

    # importing
    p_import = sub.add_parser("import", help="Import corrected masks from bundle")
    p_import.add_argument("--bundle", "-b", type=str, required=True,
                          help="Path to .zip bundle")
    p_import.add_argument("--strategy", choices=["newest", "overwrite", "skip"],
                          default="newest",
                          help="Merge strategy for conflicts (default: newest)")
    p_import_all = sub.add_parser("import-all",
                                  help="Import all bundles from annotation_inbox/")
    p_import_all.add_argument("--strategy", choices=["newest", "overwrite", "skip"],
                              default="newest")

    # repacking
    p_repack = sub.add_parser("repack", help="Re-pack corrections into a bundle")
    p_repack.add_argument("--work-dir", type=str, required=True,
                          help="Working directory created by 'edit --bundle'")
    p_repack.add_argument("--output", "-o", type=str, default=None,
                          help="Output .zip path")
    p_repack.add_argument("--annotator", type=str, default="anonymous")

    # inspecting
    p_inspect = sub.add_parser("inspect", help="Inspect bundle contents")
    p_inspect.add_argument("--bundle", "-b", type=str, required=True)

    # status
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
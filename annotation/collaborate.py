"""
The script to allow collaboration between annotators.
"""
from __future__ import annotations
import json
import logging
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import cv2

logger = logging.getLogger(__name__)

def export_bundle(records, output_path, corrected_masks_dir = None, annotator = "anonymous", notes = ""):
    
    output_path = Path(output_path)
    if not output_path.suffix == ".zip":
        output_path = output_path.with_suffix(".zip")

    corrected_dir = Path(corrected_masks_dir) if corrected_masks_dir else None
    timestamp = datetime.now(timezone.utc).isoformat()

    bundle_meta = {
        "created_by": annotator,
        "created_at": timestamp,
        "notes": notes,
        "patch_count": len(records),
        "patch_ids": [r.get("patch_id", f"patch_{i}") for i, r in enumerate(records)],
        "format_version": "1.0",
    }

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Bundle metadata
        zf.writestr("bundle_metadata.json", json.dumps(bundle_meta, indent=2))

        for i, rec in enumerate(records):
            patch_id = rec.get("patch_id", f"patch_{i:04d}")
            prefix = f"patches/{patch_id}"

            # Image
            img_path = rec["image_path"]
            if Path(img_path).exists():
                zf.write(img_path, f"{prefix}/image.png")

            # Mask: prefer corrected, failsafe is original
            mask_path = rec["mask_path"]
            if corrected_dir:
                corrected = corrected_dir / f"{patch_id}.png"
                if corrected.exists():
                    mask_path = str(corrected)

            if Path(mask_path).exists():
                zf.write(mask_path, f"{prefix}/mask.png")

            # Per-patch metadata
            patch_meta = {
                "patch_id": patch_id,
                "original_image_path": rec["image_path"],
                "original_mask_path": rec["mask_path"],
                "mask_source": "corrected" if (
                    corrected_dir and (corrected_dir / f"{patch_id}.png").exists()
                ) else "proxy_label",
                "label_stats": rec.get("label_stats", {}),
                "annotation_history": [{
                    "action": "exported",
                    "annotator": annotator,
                    "timestamp": timestamp,
                }],
            }
            zf.writestr(
                f"{prefix}/metadata.json",
                json.dumps(patch_meta, indent=2),
            )

    logger.info(f"Exported {len(records)} patches → {output_path}")
    logger.info(f"  Bundle size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    return output_path

def import_bundle(bundle_path, corrected_masks_dir, merge_strategy = "newest"):

    bundle_path = Path(bundle_path)
    corrected_dir = Path(corrected_masks_dir)
    corrected_dir.mkdir(parents=True, exist_ok=True)
    imported = 0
    skipped = 0
    conflicts = 0
    untouched = 0

    with zipfile.ZipFile(bundle_path, "r") as zf:
        # Read bundle metadata
        bundle_meta = json.loads(zf.read("bundle_metadata.json"))
        collaborator = bundle_meta.get("created_by", "unknown")

        # Determine which patches were actually corrected
        corrected_set = set(bundle_meta.get("corrected_ids", []))
        total = bundle_meta.get("patch_count", 0)
        n_corrected = len(corrected_set)
        logger.info(
            f"Importing bundle from '{collaborator}' "
            f"({total} patches total, {n_corrected} corrected)"
        )

        has_corrected_list = bool(corrected_set)

        # Find all mask files in the bundle
        mask_entries = [
            name for name in zf.namelist()
            if name.startswith("patches/") and name.endswith("/mask.png")
        ]

        for mask_entry in mask_entries:
            # Extract patch_id from path: patches/<patch_id>/mask.png
            parts = mask_entry.split("/")
            if len(parts) < 3:
                continue
            patch_id = parts[1]

            # Skip patches that were not corrected
            if has_corrected_list:
                if patch_id not in corrected_set:
                    untouched += 1
                    continue
            else:
                meta_entry = f"patches/{patch_id}/metadata.json"
                if meta_entry in zf.namelist():
                    patch_meta = json.loads(zf.read(meta_entry))
                    if patch_meta.get("mask_source") == "untouched":
                        untouched += 1
                        continue

            dest_path = corrected_dir / f"{patch_id}.png"

            if dest_path.exists():
                if merge_strategy == "skip":
                    skipped += 1
                    continue
                elif merge_strategy == "newest":
                    meta_entry = f"patches/{patch_id}/metadata.json"
                    if meta_entry in zf.namelist():
                        patch_meta = json.loads(zf.read(meta_entry))
                        history = patch_meta.get("annotation_history", [])
                        if history:
                            bundle_time = history[-1].get("timestamp", "")
                            existing_time = datetime.fromtimestamp(
                                dest_path.stat().st_mtime, tz=timezone.utc
                            ).isoformat()
                            if bundle_time < existing_time:
                                skipped += 1
                                continue
                conflicts += 1

            # Extract mask
            mask_data = zf.read(mask_entry)
            dest_path.write_bytes(mask_data)

            imported += 1

    summary = {
        "imported": imported,
        "untouched": untouched,
        "skipped": skipped,
        "conflicts_resolved": conflicts,
        "bundle_annotator": collaborator,
        "bundle_created_at": bundle_meta.get("created_at", ""),
        "bundle_notes": bundle_meta.get("notes", ""),
    }

    logger.info(f"  Imported: {imported} | Untouched (ignored): {untouched} | "
                f"Skipped: {skipped} | Conflicts resolved: {conflicts}")
    return summary

def open_bundle(bundle_path, work_dir = None):

    bundle_path = Path(bundle_path)
    if work_dir is None:
        work_dir = bundle_path.parent / f"{bundle_path.stem}_work"
    work_dir = Path(work_dir)
    masks_dir = work_dir / "corrected_masks"

    # If work dir already exists, reuse it (resume session)
    already_extracted = (work_dir / "patches").exists()
    if already_extracted:
        n_existing = len(list(masks_dir.glob("*.png"))) if masks_dir.exists() else 0
        logger.info(f"Resuming from existing work directory: {work_dir}")
        logger.info(f"  Already corrected: {n_existing} patches")
    else:
        masks_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle_path, "r") as zf:
            zf.extractall(work_dir)

    # Build records from extracted files
    patches_dir = work_dir / "patches"
    records = []
    for patch_dir in sorted(patches_dir.iterdir()):
        if not patch_dir.is_dir():
            continue
        image_path = patch_dir / "image.png"
        mask_path = patch_dir / "mask.png"
        if not image_path.exists() or not mask_path.exists():
            continue

        # Load metadata if available
        meta_path = patch_dir / "metadata.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        records.append({
            "patch_id": meta.get("patch_id", patch_dir.name),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "label_stats": meta.get("label_stats", {}),
        })

    logger.info(f"Opened bundle: {len(records)} patches extracted → {work_dir}")
    return records, masks_dir

def repack_bundle(work_dir, output_path, annotator = "anonymous"):

    work_dir = Path(work_dir)
    output_path = Path(output_path)
    if not output_path.suffix == ".zip":
        output_path = output_path.with_suffix(".zip")

    patches_dir = work_dir / "patches"
    masks_dir = work_dir / "corrected_masks"
    timestamp = datetime.now(timezone.utc).isoformat()

    # Collect all patch IDs and separate corrected from untouched
    all_patch_ids = [d.name for d in sorted(patches_dir.iterdir()) if d.is_dir()]
    corrected_ids = [
        pid for pid in all_patch_ids
        if (masks_dir / f"{pid}.png").exists()
    ]
    untouched_ids = [pid for pid in all_patch_ids if pid not in set(corrected_ids)]

    bundle_meta = {
        "created_by": annotator,
        "created_at": timestamp,
        "notes": f"Corrections by {annotator}",
        "patch_count": len(all_patch_ids),
        "corrected_count": len(corrected_ids),
        "untouched_count": len(untouched_ids),
        "corrected_ids": corrected_ids,
        "untouched_ids": untouched_ids,
        "patch_ids": all_patch_ids,
        "format_version": "1.1",
    }

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bundle_metadata.json", json.dumps(bundle_meta, indent=2))

        for patch_id in all_patch_ids:
            src_dir = patches_dir / patch_id
            prefix = f"patches/{patch_id}"
            is_corrected = patch_id in set(corrected_ids)

            # Image (unchanged)
            img = src_dir / "image.png"
            if img.exists():
                zf.write(str(img), f"{prefix}/image.png")

            # Mask: use corrected if it exists, otherwise original
            corrected_mask = masks_dir / f"{patch_id}.png"
            original_mask = src_dir / "mask.png"
            mask_to_use = corrected_mask if is_corrected else original_mask
            mask_source = "corrected" if is_corrected else "untouched"

            if mask_to_use.exists():
                zf.write(str(mask_to_use), f"{prefix}/mask.png")

            # Metadata: update history
            meta_path = src_dir / "metadata.json"
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            history = meta.get("annotation_history", [])
            if is_corrected:
                history.append({
                    "action": "corrected",
                    "annotator": annotator,
                    "timestamp": timestamp,
                })
            meta["annotation_history"] = history
            meta["mask_source"] = mask_source
            zf.writestr(f"{prefix}/metadata.json", json.dumps(meta, indent=2))

    logger.info(f"Repacked bundle: {len(corrected_ids)} corrected, "
                f"{len(untouched_ids)} untouched → {output_path}")
    return output_path

def list_bundle_contents(bundle_path):
    # Inspect a bundle without extracting it.

    bundle_path = Path(bundle_path)
    with zipfile.ZipFile(bundle_path, "r") as zf:
        meta = json.loads(zf.read("bundle_metadata.json"))
        mask_entries = [
            name.split("/")[1]
            for name in zf.namelist()
            if name.startswith("patches/") and name.endswith("/mask.png")
        ]
    meta["actual_patches_found"] = mask_entries
    return meta

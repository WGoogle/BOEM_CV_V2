"""
annotation.collaborate — Export & Import Annotation Bundles
============================================================
Enables multi-user annotation workflows:

  1. Lead annotator runs export_bundle() to create a .zip with
     selected patches (images + masks + metadata).
  2. .zip is shared with collaborators (email, Drive, Slack, etc.).
  3. Collaborators open the bundle with the annotation editor,
     correct masks, and re-zip.
  4. Lead annotator runs import_bundle() to merge corrections
     back into the pipeline.

Bundle format (.zip):
    bundle_metadata.json          # annotator info, patch list, timestamps
    patches/
        <patch_id>/
            image.png             # original image (for reference)
            mask.png              # current mask (proxy or corrected)
            metadata.json         # patch record + annotation history
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


def export_bundle(
    records: list[dict],
    output_path: Path | str,
    corrected_masks_dir: Path | str | None = None,
    annotator: str = "anonymous",
    notes: str = "",
) -> Path:
    """Package patches into a shareable .zip bundle.

    Parameters
    ----------
    records : list[dict]
        Patch records to include.  Each must have
        ``patch_id``, ``image_path``, ``mask_path``.
    output_path : Path
        Where to write the .zip file.
    corrected_masks_dir : Path | None
        If provided, use corrected masks from this directory
        (falling back to original mask_path if no correction exists).
    annotator : str
        Name/ID of the person creating the bundle.
    notes : str
        Free-text instructions for collaborators.

    Returns
    -------
    Path
        Path to the created .zip file.
    """
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

            # Mask: prefer corrected, fall back to original
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


def import_bundle(
    bundle_path: Path | str,
    corrected_masks_dir: Path | str,
    merge_strategy: str = "newest",
) -> dict:
    """Import corrected masks from a collaborator's bundle.

    Parameters
    ----------
    bundle_path : Path
        Path to the .zip bundle.
    corrected_masks_dir : Path
        Directory to save imported corrected masks.
    merge_strategy : str
        How to handle conflicts when a mask already exists:
        - "newest"    : keep whichever was modified more recently
        - "overwrite" : always use the imported mask
        - "skip"      : never overwrite existing corrected masks

    Returns
    -------
    dict
        Summary: imported count, skipped count, bundle metadata.
    """
    bundle_path = Path(bundle_path)
    corrected_dir = Path(corrected_masks_dir)
    corrected_dir.mkdir(parents=True, exist_ok=True)

    imported = 0
    skipped = 0
    conflicts = 0

    with zipfile.ZipFile(bundle_path, "r") as zf:
        # Read bundle metadata
        bundle_meta = json.loads(zf.read("bundle_metadata.json"))
        collaborator = bundle_meta.get("created_by", "unknown")
        logger.info(
            f"Importing bundle from '{collaborator}' "
            f"({bundle_meta['patch_count']} patches)"
        )

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

            dest_path = corrected_dir / f"{patch_id}.png"

            # Conflict resolution
            if dest_path.exists():
                if merge_strategy == "skip":
                    skipped += 1
                    continue
                elif merge_strategy == "newest":
                    # Read metadata to compare timestamps
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

            # Update metadata: append import record
            meta_entry = f"patches/{patch_id}/metadata.json"
            if meta_entry in zf.namelist():
                patch_meta = json.loads(zf.read(meta_entry))
                history = patch_meta.get("annotation_history", [])
                history.append({
                    "action": "imported",
                    "from_bundle": bundle_path.name,
                    "from_annotator": collaborator,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "merge_strategy": merge_strategy,
                })
                # Save updated metadata alongside mask
                meta_dest = corrected_dir / f"{patch_id}_meta.json"
                with open(meta_dest, "w") as f:
                    json.dump(patch_meta, f, indent=2)

            imported += 1

    summary = {
        "imported": imported,
        "skipped": skipped,
        "conflicts_resolved": conflicts,
        "bundle_annotator": collaborator,
        "bundle_created_at": bundle_meta.get("created_at", ""),
        "bundle_notes": bundle_meta.get("notes", ""),
    }

    logger.info(f"  Imported: {imported} | Skipped: {skipped} | "
                f"Conflicts resolved: {conflicts}")
    return summary


def open_bundle(
    bundle_path: Path | str,
    work_dir: Path | str | None = None,
) -> tuple[list[dict], Path]:
    """Extract a bundle into a working directory for annotation.

    Returns records the editor can use (image_path, mask_path, patch_id)
    and the directory where corrected masks should be saved.

    Parameters
    ----------
    bundle_path : Path
        Path to the .zip bundle.
    work_dir : Path | None
        Where to extract.  Defaults to a sibling directory of the bundle
        named ``<bundle_stem>_work/``.

    Returns
    -------
    records : list[dict]
        Patch records with local image_path and mask_path.
    masks_dir : Path
        Directory where corrected masks will be saved (editor output_dir).
    """
    bundle_path = Path(bundle_path)
    if work_dir is None:
        work_dir = bundle_path.parent / f"{bundle_path.stem}_work"
    work_dir = Path(work_dir)
    masks_dir = work_dir / "corrected_masks"
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


def repack_bundle(
    work_dir: Path | str,
    output_path: Path | str,
    annotator: str = "anonymous",
) -> Path:
    """Re-pack a working directory into a bundle .zip for return.

    Merges any corrected masks back into the bundle so the lead
    annotator can import them.

    Parameters
    ----------
    work_dir : Path
        The working directory created by open_bundle().
    output_path : Path
        Where to write the new .zip.
    annotator : str
        Name of the person who made corrections.

    Returns
    -------
    Path
        Path to the created .zip file.
    """
    work_dir = Path(work_dir)
    output_path = Path(output_path)
    if not output_path.suffix == ".zip":
        output_path = output_path.with_suffix(".zip")

    patches_dir = work_dir / "patches"
    masks_dir = work_dir / "corrected_masks"
    timestamp = datetime.now(timezone.utc).isoformat()

    # Collect patch IDs
    patch_ids = [d.name for d in sorted(patches_dir.iterdir()) if d.is_dir()]

    bundle_meta = {
        "created_by": annotator,
        "created_at": timestamp,
        "notes": f"Corrections by {annotator}",
        "patch_count": len(patch_ids),
        "patch_ids": patch_ids,
        "format_version": "1.0",
    }

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bundle_metadata.json", json.dumps(bundle_meta, indent=2))

        for patch_id in patch_ids:
            src_dir = patches_dir / patch_id
            prefix = f"patches/{patch_id}"

            # Image (unchanged)
            img = src_dir / "image.png"
            if img.exists():
                zf.write(str(img), f"{prefix}/image.png")

            # Mask: use corrected if it exists, otherwise original
            corrected_mask = masks_dir / f"{patch_id}.png"
            original_mask = src_dir / "mask.png"
            mask_to_use = corrected_mask if corrected_mask.exists() else original_mask
            mask_source = "corrected" if corrected_mask.exists() else "proxy_label"

            if mask_to_use.exists():
                zf.write(str(mask_to_use), f"{prefix}/mask.png")

            # Metadata: update history
            meta_path = src_dir / "metadata.json"
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            history = meta.get("annotation_history", [])
            history.append({
                "action": "corrected" if corrected_mask.exists() else "reviewed",
                "annotator": annotator,
                "timestamp": timestamp,
            })
            meta["annotation_history"] = history
            meta["mask_source"] = mask_source
            zf.writestr(f"{prefix}/metadata.json", json.dumps(meta, indent=2))

    logger.info(f"Repacked bundle: {len(patch_ids)} patches → {output_path}")
    return output_path


def list_bundle_contents(bundle_path: Path | str) -> dict:
    """Inspect a bundle without extracting it.

    Returns
    -------
    dict
        Bundle metadata + list of contained patch IDs.
    """
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

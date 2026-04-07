"""
annotation.tracker — Annotation Status & Provenance
=====================================================
Tracks which patches have been manually corrected, by whom,
and when.  Persists state to a JSON file so progress survives
across sessions.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class AnnotationTracker:
    """Track annotation progress and provenance.

    Parameters
    ----------
    tracker_path : Path
        Path to the JSON file that stores annotation state.
    """

    def __init__(self, tracker_path: Path) -> None:
        self.path = Path(tracker_path)
        self.data: dict = {"patches": {}, "annotators": {}}
        if self.path.exists():
            with open(self.path) as f:
                self.data = json.load(f)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def record_annotation(self, patch_id: str, annotator: str) -> None:
        """Record that a patch was annotated."""
        now = datetime.now(timezone.utc).isoformat()
        patches = self.data.setdefault("patches", {})
        entry = patches.setdefault(patch_id, {"history": []})
        entry["last_annotator"] = annotator
        entry["last_modified"] = now
        entry["history"].append({
            "annotator": annotator,
            "timestamp": now,
        })

        # Update annotator stats
        annotators = self.data.setdefault("annotators", {})
        astats = annotators.setdefault(annotator, {"count": 0, "first_seen": now})
        astats["count"] += 1
        astats["last_active"] = now

        self.save()

    def get_status(self, total_patches: int) -> dict:
        """Get summary of annotation progress."""
        patches = self.data.get("patches", {})
        annotators = self.data.get("annotators", {})
        return {
            "total_patches": total_patches,
            "annotated": len(patches),
            "remaining": total_patches - len(patches),
            "percent_complete": (len(patches) / max(total_patches, 1)) * 100,
            "annotators": {
                name: stats["count"]
                for name, stats in annotators.items()
            },
        }

    def get_unannotated_ids(self, all_patch_ids: list[str]) -> list[str]:
        """Return patch IDs that haven't been corrected yet."""
        done = set(self.data.get("patches", {}).keys())
        return [pid for pid in all_patch_ids if pid not in done]

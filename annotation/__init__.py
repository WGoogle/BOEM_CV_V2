"""
annotation — Manual Annotation & Collaboration Tools
=====================================================
Provides a GUI editor for correcting proxy-label masks and a
collaboration system for distributing annotation work across
multiple users.

Public API:
  - AnnotationEditor   : matplotlib-based mask editor
  - export_bundle      : package patches for sharing
  - import_bundle      : ingest corrected masks from collaborators
  - AnnotationTracker  : track annotation status and provenance
"""

from annotation.editor import AnnotationEditor
from annotation.collaborate import export_bundle, import_bundle, open_bundle, repack_bundle
from annotation.tracker import AnnotationTracker

__all__ = [
    "AnnotationEditor",
    "export_bundle",
    "import_bundle",
    "open_bundle",
    "repack_bundle",
    "AnnotationTracker",
]

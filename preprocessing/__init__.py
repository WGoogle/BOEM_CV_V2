"""
preprocessing — Polymetallic Nodule Preprocessing Module
=========================================================
Public API re-exported here so callers can do::
 
    from preprocessing import MosaicPatcher, PatchAutoTuner, FilterPipeline
"""
 
from .patcher import MosaicPatcher
from .auto_tuner import PatchAutoTuner
from .filters import FilterPipeline
from .geo_resolution import extract_meters_per_pixel

__all__ = ["MosaicPatcher", "PatchAutoTuner", "FilterPipeline", "extract_meters_per_pixel"]
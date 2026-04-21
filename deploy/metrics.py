"""
metrics.py — Nodule density and seafloor coverage from a binary mask.

Coverage %     = fraction of seafloor pixels flagged as nodule * 100.
Density        = connected-component count per m^2, plus nodule-pixels per m^2.
                 Components smaller than MIN_NODULE_PX are treated as noise.
"""
from __future__ import annotations
import cv2
import numpy as np

MIN_NODULE_PX = 20      # matches config.METRICS["min_nodule_size"]
CONNECTIVITY  = 8       # matches config.METRICS["connectivity"]


def compute_metrics(binary_mask, *, meters_per_pixel, seafloor_mask=None):
    """
    Parameters
    ----------
    binary_mask : np.ndarray (H, W) uint8
        0 = background, >0 = nodule.
    meters_per_pixel : float | None
        Physical pixel size. If None, area-based metrics return None.
    seafloor_mask : np.ndarray (H, W) bool | None
        True where the image actually contains seafloor (excludes black borders
        from AUV mosaic reassembly). If None, the full image is counted.

    Returns dict with:
        coverage_pct               : % of seafloor pixels flagged as nodule
        nodule_count               : # of connected components >= MIN_NODULE_PX
        nodule_pixels              : total flagged pixels (after size filter)
        seafloor_pixels            : denominator used for coverage
        seafloor_area_m2           : physical area (or None if mpp missing)
        nodule_area_m2             : physical nodule area (or None)
        nodule_density_per_m2      : nodules / m^2 (or None)
        nodule_px_density_per_m2   : flagged pixels / m^2 (or None)
        meters_per_pixel           : echo of input
        min_nodule_px              : size floor used
    """
    if binary_mask.ndim != 2:
        raise ValueError(f"binary_mask must be 2D, got {binary_mask.shape}")

    mask_u8 = (binary_mask > 0).astype(np.uint8)

    if seafloor_mask is None:
        seafloor = np.ones(mask_u8.shape, dtype=bool)
    else:
        seafloor = seafloor_mask.astype(bool)
        # Nodule predictions outside the seafloor region are ignored.
        mask_u8 = (mask_u8 & seafloor.astype(np.uint8))

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=CONNECTIVITY,
    )
    # label 0 is background; filter the rest by size
    areas = stats[1:, cv2.CC_STAT_AREA] if n_labels > 1 else np.zeros(0, dtype=np.int64)
    kept = areas[areas >= MIN_NODULE_PX]

    nodule_count  = int(kept.size)
    nodule_pixels = int(kept.sum()) if kept.size else 0
    seafloor_pixels = int(seafloor.sum())

    coverage_pct = (
        100.0 * nodule_pixels / seafloor_pixels if seafloor_pixels else 0.0
    )

    if meters_per_pixel is not None and meters_per_pixel > 0:
        px_area_m2 = float(meters_per_pixel) ** 2
        seafloor_area_m2 = seafloor_pixels * px_area_m2
        nodule_area_m2   = nodule_pixels   * px_area_m2
        if seafloor_area_m2 > 0:
            nodule_density_per_m2    = nodule_count  / seafloor_area_m2
            nodule_px_density_per_m2 = nodule_pixels / seafloor_area_m2
        else:
            nodule_density_per_m2 = nodule_px_density_per_m2 = 0.0
    else:
        seafloor_area_m2 = nodule_area_m2 = None
        nodule_density_per_m2 = nodule_px_density_per_m2 = None

    return {
        "coverage_pct":             round(coverage_pct, 4),
        "nodule_count":             nodule_count,
        "nodule_pixels":            nodule_pixels,
        "seafloor_pixels":          seafloor_pixels,
        "seafloor_area_m2":         None if seafloor_area_m2 is None else round(seafloor_area_m2, 4),
        "nodule_area_m2":           None if nodule_area_m2   is None else round(nodule_area_m2,   4),
        "nodule_density_per_m2":    None if nodule_density_per_m2    is None else round(nodule_density_per_m2,    4),
        "nodule_px_density_per_m2": None if nodule_px_density_per_m2 is None else round(nodule_px_density_per_m2, 2),
        "meters_per_pixel":         meters_per_pixel,
        "min_nodule_px":            MIN_NODULE_PX,
    }


def seafloor_mask_from_raw(raw_bgr, border_threshold=10):
    """Valid-seafloor mask: True where the raw mosaic has real pixels (not black border)."""
    gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    return gray > border_threshold


def format_metrics_report(metrics, mosaic_name):
    mpp = metrics["meters_per_pixel"]
    lines = [
        f"  [{mosaic_name}]",
        f"    Coverage            : {metrics['coverage_pct']:.2f}% of seafloor",
        f"    Nodules detected    : {metrics['nodule_count']}  "
        f"(>= {metrics['min_nodule_px']} px each)",
        f"    Nodule pixels       : {metrics['nodule_pixels']:,} / "
        f"{metrics['seafloor_pixels']:,} seafloor px",
    ]
    if mpp is not None:
        lines += [
            f"    Pixel size          : {mpp:.6f} m/px ({mpp * 1000:.2f} mm/px)",
            f"    Seafloor area      : {metrics['seafloor_area_m2']:.2f} m^2",
            f"    Nodule density     : {metrics['nodule_density_per_m2']:.3f} nodules/m^2",
            f"    Nodule px density  : {metrics['nodule_px_density_per_m2']:.1f} px/m^2",
        ]
    else:
        lines.append("    (no meters_per_pixel available — area metrics skipped)")
    return "\n".join(lines)

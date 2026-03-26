"""
filters.py — Individual CV Preprocessing Steps + Proxy Label Generation
=========================================================================
Every public function in this module implements a single, testable
preprocessing step.  The ``FilterPipeline`` class chains them together
and optionally saves intermediate images to ``step_by_step_logs/`` for
visual debugging.

Design principles
-----------------
* Each filter takes a BGR ``np.ndarray`` and a ``TunedParams`` (from
  ``auto_tuner.py``) and returns a BGR ``np.ndarray``.
* Filters are stateless — no hidden instance variables.
* The pipeline log is a flat list of ``(step_name, image)`` tuples that
  the runner can persist to disk.

Steps ported from the original BOEM_CV repository
--------------------------------------------------
1. Gray-world white balance
2. CLAHE in LAB colour space
3. Bilateral denoising
4. Nodule boost (bottom-hat + texture gate)
5. Sediment fade
6. Unsharp mask

Proxy-label generation (also here to keep all CV logic in one module)
---------------------------------------------------------------------
7.  Gaussian blur
8.  Multi-scale black top-hat
9.  Percentile thresholding
10. Absolute intensity gate
11. Morphological cleanup
12. Contour shape filtering
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .auto_tuner import TunedParams

logger = logging.getLogger(__name__)


# ----------- INDIVIDUAL FILTER FUNCTIONS -----------

def gray_world_white_balance(
    image: np.ndarray, params: TunedParams
) -> np.ndarray:
    """Gray-world white balance to compensate AUV/ROV lighting bias.

    Scales each BGR channel so that the per-channel means are equal,
    approximating the assumption that the average scene reflectance
    is achromatic.
    """
    img = image.astype(np.float32)
    means = img.mean(axis=(0, 1))                      # (B, G, R)
    gray_mean = means.mean()

    if np.any(means < 1.0):
        logger.warning("Near-zero channel mean — skipping white balance")
        return image

    scale = gray_mean / means                           # per-channel scale
    balanced = np.clip(img * scale[np.newaxis, np.newaxis, :], 0, 255)
    return balanced.astype(np.uint8)


def illumination_normalize(
    image: np.ndarray, params: TunedParams,
    preprocess_cfg: dict,
) -> np.ndarray:
    """Flatten the AUV light-cone gradient by dividing out low-frequency illumination.

    Estimates the illumination field with a large Gaussian blur (σ≈51px,
    capturing gradients over ~300px) then divides L by it and rescales to
    the original mean.  Black mosaic borders are filled before the blur
    and restored afterward so they don't drag down the estimate at edges.
    """
    sigma = preprocess_cfg.get("illum_norm_sigma", 51.0)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_f = l_ch.astype(np.float32)

    # Mask out black borders
    valid_mask = l_f > 8
    valid_px = l_f[valid_mask]
    if valid_px.size < 100:
        return image  # patch is nearly empty, skip

    valid_mean = float(np.mean(valid_px))

    # Fill borders with valid mean so they don't create edge artefacts in blur
    l_filled = l_f.copy()
    l_filled[~valid_mask] = valid_mean

    # Estimate illumination field
    illum = cv2.GaussianBlur(l_filled, (0, 0), sigma)
    illum = np.clip(illum, 1.0, None)  # prevent division by zero

    # Divide and rescale to original mean brightness
    l_norm = (l_f / illum) * valid_mean
    l_norm[~valid_mask] = 0  # restore black borders
    l_norm = np.clip(l_norm, 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([l_norm, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def clahe_lab(
    image: np.ndarray, params: TunedParams
) -> np.ndarray:
    """CLAHE on the L channel of LAB colour space.

    Clip limit and tile grid are taken from *params* (set by auto-tuner).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=params.clahe_clip_limit,
        tileGridSize=params.clahe_tile_grid,
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def bilateral_denoise(
    image: np.ndarray, params: TunedParams
) -> np.ndarray:
    """Bilateral filter — edge-preserving noise suppression.

    Sigma values are auto-tuned from the patch's noise estimate.
    """
    return cv2.bilateralFilter(
        image,
        d=params.bilateral_d,
        sigmaColor=params.bilateral_sigma_color,
        sigmaSpace=params.bilateral_sigma_space,
    )


def nodule_boost(
    image: np.ndarray,
    params: TunedParams,
    preprocess_cfg: dict,
    l_pre_clahe: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Darken nodule blobs using bottom-hat transform + texture gate.

    Uses the pre-CLAHE L channel when available to avoid halo artefacts.

    The bottom-hat response is soft-thresholded at the 60th percentile
    so that background sediment (which also has a small bottom-hat value
    from grain texture) is left untouched, while actual nodules receive
    strong darkening proportional to their response.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    l_ref = l_pre_clahe if l_pre_clahe is not None else l_ch
    l_ref_u8 = l_ref.astype(np.uint8)

    # Bottom-hat: close(L) − L → large response at dark compact blobs
    se_size = 2 * params.morph_radius + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    closed = cv2.morphologyEx(l_ref_u8, cv2.MORPH_CLOSE, se)
    bottom_hat = cv2.subtract(closed, l_ref_u8).astype(np.float32)

    # Texture gate: suppress response in grainy sediment regions
    blur_fine = cv2.GaussianBlur(
        l_ref_u8.astype(np.float32), (0, 0), params.texture_sigma
    )
    abs_diff = np.abs(l_ref_u8.astype(np.float32) - blur_fine)
    texture_score = cv2.GaussianBlur(abs_diff, (0, 0), params.texture_sigma * 1.5)
    texture_weight = np.clip(1.0 - texture_score / params.texture_threshold, 0, 1)

    # Soft-threshold: subtract the background bottom-hat floor so only
    # pixels with a strong response (actual nodules) get darkened.
    nonzero = bottom_hat[bottom_hat > 0]
    if nonzero.size > 0:
        bh_floor = float(np.percentile(nonzero, 60))
    else:
        bh_floor = 5.0
    boosted_bh = np.clip(bottom_hat - bh_floor, 0, None)

    darkening = np.clip(
        params.nodule_boost_factor * boosted_bh * texture_weight,
        0, params.max_darkening,
    )
    l_boosted = np.clip(l_ch.astype(np.float32) - darkening, 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([l_boosted, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def sediment_fade(
    image: np.ndarray,
    params: TunedParams,
    preprocess_cfg: dict,
) -> np.ndarray:
    """Fade bright sediment toward a smooth background without touching nodules.

    The L-threshold is computed adaptively from the patch's own brightness
    distribution: only pixels above the median L value can be faded.  This
    guarantees that dark compact blobs (nodules) are never brightened.  A
    wider ramp (40 levels) and reduced mask blur prevent bleed into dark
    regions adjacent to bright sediment.
    """
    fade_sigma = preprocess_cfg.get("sediment_fade_blur_sigma", 15.0)
    fade_strength = preprocess_cfg.get("sediment_fade_strength", 0.6)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_f = l_ch.astype(np.float32)

    l_smooth = cv2.GaussianBlur(l_f, (0, 0), fade_sigma)

    # Adaptive threshold: only fade pixels brighter than the patch median.
    # Exclude near-black border pixels (L < 10) from the median calculation.
    valid_L = l_f[l_f > 10]
    if valid_L.size > 0:
        l_threshold = float(np.median(valid_L))
    else:
        l_threshold = preprocess_cfg.get("sediment_l_threshold", 80)

    # Soft sediment mask: ramps from 0 at l_threshold to 1 at l_threshold+40.
    # The wider ramp (40 vs old 20) + higher adaptive threshold keeps dark
    # nodule pixels firmly at 0.  Minimal mask blur (σ=2) prevents bleed.
    sediment_mask = np.clip((l_f - l_threshold) / 40.0, 0, 1)
    sediment_mask = cv2.GaussianBlur(sediment_mask, (0, 0), 2.0)

    blend_w = sediment_mask * fade_strength
    l_faded = l_f * (1.0 - blend_w) + l_smooth * blend_w
    l_faded = np.clip(l_faded, 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([l_faded, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def unsharp_mask(
    image: np.ndarray,
    params: TunedParams,
    preprocess_cfg: dict,
) -> np.ndarray:
    """Edge-selective unsharp mask — sharpens nodule boundaries, not grain.

    Uses a two-sigma approach: the sharpening kernel targets nodule-edge
    scale (σ≈2px), while a grain-suppression guard computed at fine scale
    (σ≈0.7px) prevents amplification of sediment texture.  Only edges
    with a gradient magnitude above the median get the full sharpening.
    """
    sigma = preprocess_cfg.get("unsharp_sigma", 2.0)
    strength = preprocess_cfg.get("unsharp_strength", 0.5)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_f = l_ch.astype(np.float32)

    blur = cv2.GaussianBlur(l_f, (0, 0), sigma)
    detail = l_f - blur  # high-pass at nodule-edge scale

    # Edge magnitude — stronger at real boundaries, weaker at grain
    grad_x = cv2.Sobel(l_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(l_f, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Soft gate: ramp from 0 at low edges to 1 at strong edges.
    # Threshold at the median edge magnitude of non-black pixels.
    valid = l_ch > 8
    if valid.any():
        edge_thresh = float(np.median(edge_mag[valid]))
    else:
        edge_thresh = 1.0
    edge_weight = np.clip(edge_mag / (edge_thresh * 2.0 + 1e-6), 0, 1)

    l_sharp = np.clip(l_f + strength * detail * edge_weight, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l_sharp, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


# ----------- PROXY LABEL GENERATION -----------

def generate_proxy_label(
    patch_bgr: np.ndarray,
    params: TunedParams,
    proxy_cfg: dict,
) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]], Dict]:
    """Generate a binary nodule mask for a single patch.

    Returns
    -------
    mask : np.ndarray  (H, W) uint8, 0 or 255
    steps : list[(name, image)]   intermediate images for debugging
    stats : dict                  generation statistics
    """
    steps: List[Tuple[str, np.ndarray]] = []

    def _log(name: str, img: np.ndarray):
        # Store a copy so later mutations don't corrupt the log
        steps.append((name, img.copy() if img.ndim == 2 else img.copy()))

    # 1. Grayscale
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    _log("01_grayscale", gray)

    # 2. Gaussian blur (smooth sediment grain texture)
    if proxy_cfg.get("apply_gaussian_blur", True):
        ksize = proxy_cfg.get("gaussian_kernel_size", 15)
        sigma = proxy_cfg.get("gaussian_sigma", 5.0)
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    else:
        blurred = gray.copy()
    _log("02_gaussian_blur", blurred)

    gray_raw = gray.copy()  # keep pre-blur version for intensity gate

    # 3. Multi-scale black top-hat
    combined = np.zeros(blurred.shape, dtype=np.float32)
    for r in params.tophat_radii:
        se_size = 2 * r + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
        closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, se)
        tophat = cv2.subtract(closed, blurred).astype(np.float32)
        combined = np.maximum(combined, tophat)

    # Texture gate
    tex_sigma = params.texture_sigma
    tex_thresh = params.texture_threshold
    blur_fine = cv2.GaussianBlur(blurred.astype(np.float32), (0, 0), tex_sigma)
    local_diff = np.abs(blurred.astype(np.float32) - blur_fine)
    texture_score = cv2.GaussianBlur(local_diff, (0, 0), tex_sigma * 2.0)
    texture_weight = np.clip(1.0 - texture_score / tex_thresh, 0, 1)
    combined *= texture_weight

    _log("03_tophat_response", np.clip(combined, 0, 255).astype(np.uint8))

    # 4. Percentile thresholding with hard floor
    pos = combined[combined > 0]
    if pos.size > 0:
        threshold = float(np.percentile(pos, params.tophat_percentile))
    else:
        threshold = params.tophat_threshold_floor
    threshold = max(threshold, params.tophat_threshold_floor)

    binary = (combined >= threshold).astype(np.uint8) * 255
    _log("04_thresholded", binary)

    # 5. Absolute intensity gate — only the darkest pixels can be nodules
    if proxy_cfg.get("adaptive_abs_intensity", True):
        pct = proxy_cfg.get("adaptive_abs_percentile", 8)
        abs_max = float(np.percentile(gray_raw, pct))
    else:
        abs_max = proxy_cfg.get("absolute_intensity_max", 85)

    abs_gate = (gray_raw <= abs_max).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, abs_gate)
    _log("05_intensity_gated", binary)

    # 6. Morphological cleanup
    open_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params.morph_open_k, params.morph_open_k)
    )
    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params.morph_close_k, params.morph_close_k)
    )
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_k)
    _log("06_morph_cleaned", cleaned)

    # 7. Contour shape filtering
    contours_raw, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    reject_counts = {
        "area_too_small": 0, "area_too_large": 0,
        "low_solidity": 0, "eccentricity": 0, "low_circularity": 0,
    }
    filtered_contours = []

    for c in contours_raw:
        area = cv2.contourArea(c)
        if area < params.min_contour_area:
            reject_counts["area_too_small"] += 1
            continue
        if area > params.max_contour_area:
            reject_counts["area_too_large"] += 1
            continue

        # Solidity
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < params.min_solidity:
            reject_counts["low_solidity"] += 1
            continue

        # Eccentricity
        if len(c) >= 5:
            try:
                _, (MA, ma), _ = cv2.fitEllipse(c)
                long_ax, short_ax = max(MA, ma), min(MA, ma)
                ecc = float(np.sqrt(1.0 - (short_ax / long_ax) ** 2)) if long_ax > 0 else 0
            except Exception:
                ecc = 0.0
        else:
            ecc = 0.0
        if ecc > params.max_eccentricity:
            reject_counts["eccentricity"] += 1
            continue

        # Circularity
        perimeter = cv2.arcLength(c, True)
        circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        if circularity < params.min_circularity:
            reject_counts["low_circularity"] += 1
            continue

        filtered_contours.append(c)

    mask = np.zeros_like(cleaned)
    if filtered_contours:
        cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)
    _log("07_proxy_mask", mask)

    # Overlay for visual verification
    overlay = patch_bgr.copy()
    overlay[mask > 0] = [0, 255, 0]     # green for detected nodules
    blended = cv2.addWeighted(patch_bgr, 0.6, overlay, 0.4, 0)
    _log("08_overlay", blended)

    stats = {
        "candidates_before_filter": len(contours_raw),
        "nodules_after_filter": len(filtered_contours),
        "coverage_pct": 100.0 * np.sum(mask > 0) / mask.size,
        "threshold_used": threshold,
        "abs_intensity_gate": abs_max,
        "rejection_counts": reject_counts,
    }

    return mask, steps, stats


# ----------- FILTER PIPELINE — chains filters + logs intermediates -----------

# Registry mapping step names (from config.PREPROCESSING["filter_chain"])
# to callables.  Each callable has signature:
#   (image, params, [preprocess_cfg]) -> image
_FILTER_REGISTRY = {
    "gray_world_white_balance": gray_world_white_balance,
    "illumination_normalize":   illumination_normalize,
    "clahe_lab":                clahe_lab,
    "bilateral_denoise":        bilateral_denoise,
    "nodule_boost":             nodule_boost,
    "sediment_fade":            sediment_fade,
    "unsharp_mask":             unsharp_mask,
}


class FilterPipeline:
    """Execute the configured filter chain on a patch, logging each step.

    Parameters
    ----------
    preprocess_cfg : dict
        ``config.PREPROCESSING`` — contains the filter chain list and
        parameters for sediment_fade / unsharp_mask.
    """

    def __init__(self, preprocess_cfg: dict):
        self.cfg = preprocess_cfg
        self.chain = preprocess_cfg.get("filter_chain", list(_FILTER_REGISTRY.keys()))

    def run(
        self,
        patch_bgr: np.ndarray,
        params: TunedParams,
    ) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
        """Apply every filter in the chain to *patch_bgr*.

        Returns
        -------
        result : np.ndarray   — fully preprocessed patch (BGR uint8)
        steps  : list[(name, image)]  — intermediate snapshots
        """
        steps: List[Tuple[str, np.ndarray]] = []
        steps.append(("00_original", patch_bgr.copy()))

        image = patch_bgr.copy()

        # Snapshot L-channel before CLAHE for nodule_boost
        l_pre_clahe = None

        for i, step_name in enumerate(self.chain, start=1):
            fn = _FILTER_REGISTRY.get(step_name)
            if fn is None:
                logger.warning(f"Unknown filter step '{step_name}' — skipping")
                continue

            # Capture pre-CLAHE L channel
            if step_name == "clahe_lab":
                lab_tmp = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l_pre_clahe = lab_tmp[:, :, 0].copy()

            # Call the filter with the right signature
            if step_name in ("nodule_boost",):
                image = fn(image, params, self.cfg, l_pre_clahe)
            elif step_name in ("illumination_normalize", "sediment_fade", "unsharp_mask"):
                image = fn(image, params, self.cfg)
            else:
                image = fn(image, params)

            steps.append((f"{i:02d}_{step_name}", image.copy()))

        return image, steps

    @staticmethod
    def save_step_images(
        steps: List[Tuple[str, np.ndarray]],
        output_dir: Path,
        prefix: str = "patch",
    ) -> Path:
        """Persist each intermediate step as a numbered PNG.

        Also generates a single composite grid image for quick comparison.

        Returns the path to the composite image.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images_for_grid = []

        for name, img in steps:
            fname = f"{prefix}__{name}.png"
            cv2.imwrite(str(output_dir / fname), img)

            # Convert grayscale → BGR for the grid
            if img.ndim == 2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img
            images_for_grid.append((name, img_bgr))

        # Build composite grid
        composite_path = output_dir / f"{prefix}__composite.png"
        _build_composite(images_for_grid, composite_path)

        return composite_path


# ----------- Composite grid builder -----------

def _build_composite(
    named_images: List[Tuple[str, np.ndarray]],
    output_path: Path,
    thumb_size: int = 384,
    cols: int = 4,
    padding: int = 4,
    label_height: int = 28,
) -> None:
    """Create a labelled thumbnail grid of all intermediate steps."""
    n = len(named_images)
    if n == 0:
        return
    rows = (n + cols - 1) // cols

    cell_w = thumb_size + padding
    cell_h = thumb_size + label_height + padding
    canvas_w = cols * cell_w + padding
    canvas_h = rows * cell_h + padding

    canvas = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8)   # dark gray bg

    for idx, (name, img) in enumerate(named_images):
        r, c = divmod(idx, cols)
        # Resize to thumbnail
        h, w = img.shape[:2]
        scale = thumb_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        thumb = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Paste into cell
        x0 = c * cell_w + padding
        y0 = r * cell_h + padding

        # Center the thumbnail
        dx = (thumb_size - new_w) // 2
        dy = (thumb_size - new_h) // 2
        canvas[y0 + dy : y0 + dy + new_h, x0 + dx : x0 + dx + new_w] = thumb

        # Label
        label_y = y0 + thumb_size + 2
        cv2.putText(
            canvas, name, (x0 + 4, label_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)
    logger.debug(f"Saved composite grid: {output_path}")
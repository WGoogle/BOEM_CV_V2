"""
Individual CV Preprocessing Steps + Proxy Label Generation

Each filter takes a BGR np.ndarray and a TunedParams and returns a BGR np.ndarray.
"""
from __future__ import annotations
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .auto_tuner import TunedParams

logger = logging.getLogger(__name__)

def gray_world_white_balance(image, params):
   # Gray-world white balance to compensate AUV/ROV lighting bias.

    img = image.astype(np.float32)
    means = img.mean(axis=(0, 1))  # (B, G, R)
    gray_mean = means.mean()

    if np.any(means < 1.0):
        logger.warning("Near-zero channel mean — skipping white balance")
        return image

    scale = gray_mean / means  # per-channel scale
    balanced = np.clip(img * scale[np.newaxis, np.newaxis, :], 0, 255)
    return balanced.astype(np.uint8)

def illumination_normalize(image, params, preprocess_cfg):
    # Flatten the AUV light-cone gradient by dividing out low-frequency illumination.
    sigma = preprocess_cfg.get("illum_norm_sigma", 51.0)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_f = l_ch.astype(np.float32)

    # Mask out black borders
    valid_mask = l_f > 8
    valid_px = l_f[valid_mask]
    if valid_px.size < 100:
        return image  

    valid_mean = float(np.mean(valid_px))

    # Fill borders with valid mean so they don't create edge artefacts in blur
    l_filled = l_f.copy()
    l_filled[~valid_mask] = valid_mean

    illum = cv2.GaussianBlur(l_filled, (0, 0), sigma)
    illum = np.clip(illum, 1.0, None) 

    l_norm = (l_f / illum) * valid_mean
    l_norm[~valid_mask] = 0  
    l_norm = np.clip(l_norm, 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([l_norm, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

def clahe_lab(image, params):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_orig = lab[:, :, 0].copy()
    clahe = cv2.createCLAHE(
        clipLimit=params.clahe_clip_limit,
        tileGridSize=params.clahe_tile_grid,
    )
    l_clahe = clahe.apply(l_orig)
    alpha = params.clahe_blend
    lab[:, :, 0] = cv2.addWeighted(l_clahe, alpha, l_orig, 1.0 - alpha, 0)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def bilateral_denoise(image, params):
    return cv2.bilateralFilter(
        image,
        d=params.bilateral_d,
        sigmaColor=params.bilateral_sigma_color,
        sigmaSpace=params.bilateral_sigma_space,
    )

def multi_scale_retinex(image, params, preprocess_cfg):
    """
    Multi-Scale Retinex on the L channel — reflectance/illumination separation.
    This intrinsically suppresses:
    - AUV light-cone gradients (large-scale illumination)
    - Shadow artifacts from sediment bumps/divots (medium-scale shading)
    """
    sigmas = preprocess_cfg.get("msr_sigmas", [5, 20, 80])
    gain = preprocess_cfg.get("msr_gain", 1.0)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_f = l_ch.astype(np.float32)

    # Mask black borders
    valid_mask = l_f > 8
    valid_px = l_f[valid_mask]
    if valid_px.size < 100:
        return image

    valid_mean = float(np.mean(valid_px))
    valid_std = float(np.std(valid_px))

    # Fill borders with valid mean to prevent edge artifacts in blurs
    l_filled = l_f.copy()
    l_filled[~valid_mask] = valid_mean
    l_log = np.log1p(l_filled)  

    # Multi-scale Retinex: average of (log(L) - log(blur(L))) across scales
    retinex = np.zeros_like(l_log)
    for sigma in sigmas:
        l_blur = cv2.GaussianBlur(l_filled, (0, 0), sigma)
        l_blur_log = np.log1p(l_blur)
        retinex += (l_log - l_blur_log)
    retinex /= len(sigmas)

    # Normalize retinex to [0, 255] using only valid pixels
    r_valid = retinex[valid_mask]
    r_min = float(np.percentile(r_valid, 1))
    r_max = float(np.percentile(r_valid, 99))
    if r_max - r_min < 1e-6:
        return image

    # Linear stretch to 0-255, then match original brightness statistics
    l_retinex = (retinex - r_min) / (r_max - r_min) * 255.0

    # Apply gain
    if gain != 1.0:
        lr_valid = l_retinex[valid_mask]
        lr_mean = float(np.mean(lr_valid))
        l_retinex = (l_retinex - lr_mean) * gain + lr_mean

    # Match the original valid-pixel mean and std so downstream filters
    lr_valid = l_retinex[valid_mask]
    lr_mean = float(np.mean(lr_valid))
    lr_std = float(np.std(lr_valid))
    if lr_std > 1e-6:
        l_retinex = (l_retinex - lr_mean) / lr_std * valid_std + valid_mean

    # Blend retinex with original L to control enhancement strength.
    # Low-contrast (sediment) patches use msr_blend < 1 to avoid amplifying noise.
    alpha = params.msr_blend
    l_retinex = l_retinex * alpha + l_f * (1.0 - alpha)

    # Restore black borders
    l_retinex[~valid_mask] = 0
    l_retinex = np.clip(l_retinex, 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([l_retinex, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def sediment_fade(image, params, preprocess_cfg):
    # Fade bright sediment toward a smooth background without touching nodules.
    fade_sigma = preprocess_cfg.get("sediment_fade_blur_sigma", 15.0)
    fade_strength = preprocess_cfg.get("sediment_fade_strength", 0.6)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_f = l_ch.astype(np.float32)

    l_smooth = cv2.GaussianBlur(l_f, (0, 0), fade_sigma)

    # Adaptive threshold
    valid_L = l_f[l_f > 10]
    if valid_L.size > 0:
        l_threshold = float(np.median(valid_L))
    else:
        l_threshold = preprocess_cfg.get("sediment_l_threshold", 80)

    sediment_mask = np.clip((l_f - l_threshold) / 40.0, 0, 1)
    sediment_mask = cv2.GaussianBlur(sediment_mask, (0, 0), 2.0)

    blend_w = sediment_mask * fade_strength
    l_faded = l_f * (1.0 - blend_w) + l_smooth * blend_w
    l_faded = np.clip(l_faded, 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([l_faded, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

def unsharp_mask(image, params, preprocess_cfg):
    # enhances contrast
    sigma = preprocess_cfg.get("unsharp_sigma", 2.0)
    strength = params.unsharp_strength

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_f = l_ch.astype(np.float32)

    blur = cv2.GaussianBlur(l_f, (0, 0), sigma)
    detail = l_f - blur  

    grad_x = cv2.Sobel(l_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(l_f, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    valid = l_ch > 8
    if valid.any():
        edge_thresh = float(np.median(edge_mag[valid]))
    else:
        edge_thresh = 1.0
    edge_weight = np.clip(edge_mag / (edge_thresh * 2.0 + 1e-6), 0, 1)

    l_sharp = np.clip(l_f + strength * detail * edge_weight, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l_sharp, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

# END OF PREPROCESSING FEATURES

# BELOW ARE PROXY LABELLING FEATURES    

def _feature_tophat(blurred, radii, texture_sigma, texture_threshold):
    # The texture gate suppresses responses in grainy sediment regions where top-hat fires on grain, not nodules.
    combined = np.zeros(blurred.shape, dtype=np.float32)
    for r in radii:
        se_size = 2 * r + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
        closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, se)
        tophat = cv2.subtract(closed, blurred).astype(np.float32)
        combined = np.maximum(combined, tophat)

    blur_fine = cv2.GaussianBlur(blurred.astype(np.float32), (0, 0), texture_sigma)
    local_diff = np.abs(blurred.astype(np.float32) - blur_fine)
    texture_score = cv2.GaussianBlur(local_diff, (0, 0), texture_sigma * 2.0)
    texture_weight = np.clip(1.0 - texture_score / texture_threshold, 0, 1)
    combined *= texture_weight

    return combined

def _feature_local_contrast_ratio(gray_f, bg_sigma, valid_mask):
    # Local contrast ratio — how dark each pixel is relative to its local background.
    gray_filled = gray_f.copy()
    valid_px = gray_f[valid_mask]
    if valid_px.size > 0:
        gray_filled[~valid_mask] = float(np.mean(valid_px))
    local_bg = cv2.GaussianBlur(gray_filled, (0, 0), bg_sigma)
    local_bg = np.clip(local_bg, 1.0, None)  # avoid division by zero

    # LCR: how much darker than local background (0 = same, 1 = completely dark)
    lcr = np.clip((local_bg - gray_f) / local_bg, 0, 1)
    lcr[~valid_mask] = 0
    return lcr

def _feature_dog(gray_f, sigma_pairs):
    """
    Difference of Gaussians (DoG) blob detection — scale-normalized.
    DoG approximates the Laplacian of Gaussian (LoG), the theoretically optimal blob detector.  
    """
    combined = np.zeros_like(gray_f)
    for s_small, s_large in sigma_pairs:
        g_small = cv2.GaussianBlur(gray_f, (0, 0), s_small)
        g_large = cv2.GaussianBlur(gray_f, (0, 0), s_large)
        dog = g_large - g_small
        sigma_avg = (s_small + s_large) / 2.0
        dog_norm = dog * sigma_avg
        combined = np.maximum(combined, np.clip(dog_norm, 0, None))

    return combined


def _feature_smoothness(gray_f, inner_sigma, outer_sigma, valid_mask):
    # Smoothness score — nodules have smooth surfaces vs. grainy sediment.
    mean_fine = cv2.GaussianBlur(gray_f, (0, 0), inner_sigma)
    sq_fine = cv2.GaussianBlur(gray_f ** 2, (0, 0), inner_sigma)
    var_fine = np.clip(sq_fine - mean_fine ** 2, 0, None)

    mean_coarse = cv2.GaussianBlur(gray_f, (0, 0), outer_sigma)
    sq_coarse = cv2.GaussianBlur(gray_f ** 2, (0, 0), outer_sigma)
    var_coarse = np.clip(sq_coarse - mean_coarse ** 2, 0, None)

    smoothness = np.clip(1.0 - var_fine / (var_coarse + 1e-4), 0, 1)
    smoothness[~valid_mask] = 0
    return smoothness

def _normalize_feature(feat, valid_mask):
    # Normalize a feature map to [0, 1] using robust percentile scaling on valid pixels
    valid_px = feat[valid_mask]
    if valid_px.size < 100:
        return np.zeros_like(feat)
    p_low = float(np.percentile(valid_px, 2))
    p_high = float(np.percentile(valid_px, 98))
    if p_high - p_low < 1e-6:
        return np.zeros_like(feat)
    normed = np.clip((feat - p_low) / (p_high - p_low), 0, 1)
    normed[~valid_mask] = 0
    return normed

def _watershed_split(binary, patch_bgr, min_distance):
    # Only for large wack nodule looking things
    if np.sum(binary > 0) < 10:
        return binary

    min_single_area = max(50, int(np.pi * min_distance ** 2))
    min_split_area = min_single_area * 4  # must be ~4x a single nodule

    n_cc, cc_labels = cv2.connectedComponents(binary)
    result = binary.copy()

    for label_id in range(1, n_cc):
        cc_mask = (cc_labels == label_id).astype(np.uint8) * 255
        cc_area = np.sum(cc_mask > 0)

        if cc_area < min_split_area:
            continue

        dist = cv2.distanceTransform(cc_mask, cv2.DIST_L2, 5)
        max_dist = float(dist.max())

        if max_dist < min_distance:
            continue

        peak_kernel = max(min_distance * 2 + 1, 7)
        if peak_kernel % 2 == 0:
            peak_kernel += 1
        dist_dilated = cv2.dilate(
            dist, np.ones((peak_kernel, peak_kernel), np.uint8),
        )
        local_max = (
            (dist == dist_dilated) & (dist > max(2.0, max_dist * 0.3))
        ).astype(np.uint8)

        n_peaks, peak_labels = cv2.connectedComponents(local_max)
        if n_peaks <= 2:
            continue

        markers = peak_labels + 1 
        markers[cc_mask == 0] = 0

        ws_input = (
            patch_bgr.copy()
            if patch_bgr.ndim == 3
            else cv2.cvtColor(patch_bgr, cv2.COLOR_GRAY2BGR)
        )
        markers_ws = markers.astype(np.int32)
        cv2.watershed(ws_input, markers_ws)

        result[cc_labels == label_id] = 0
        result[(markers_ws > 1) & (cc_labels == label_id)] = 255

    return result


# ACTUAL CREATION OF PROXY LABEL
def generate_proxy_label(patch_bgr, params, proxy_cfg):
    # Generate a binary nodule mask using multi-feature blob detection.
    steps: List[Tuple[str, np.ndarray]] = []
    def _log(name, img):
        steps.append((name, img.copy()))

    def _heatmap(feat, valid):
        fv = feat[valid]
        vmax = float(np.percentile(fv, 99.5)) if fv.size > 0 else 1.0
        viz = np.clip(feat / max(vmax, 1e-6) * 255, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(viz, cv2.COLORMAP_JET)
        color[~valid] = 0
        return color

    # ── 01 Grayscale + valid pixel mask ──────────────────────────────────
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)
    valid_mask = gray > 10
    _log("01_grayscale", gray)

    n_valid = int(valid_mask.sum())
    if n_valid < 100:
        empty = np.zeros(gray.shape, dtype=np.uint8)
        _log("02_tophat_response", empty)
        _log("03_local_contrast", empty)
        _log("04_combined_score", empty)
        _log("05_thresholded", empty)
        _log("06_morph_cleaned", empty)
        _log("07_proxy_mask", empty)
        _log("08_overlay", patch_bgr)
        return empty, steps, {
            "candidates_before_filter": 0, "nodules_after_filter": 0,
            "coverage_pct": 0.0, "threshold_used": 0.0,
            "rejection_counts": {},
        }

    # ── 02 Multi-scale black top-hat ─────────────────────────────────────
    tophat_radii = proxy_cfg.get("tophat_radii", [1, 2, 4, 8, 12])
    texture_sigma = proxy_cfg.get("texture_sigma", 2.0)
    texture_threshold = proxy_cfg.get("texture_threshold", 18.0)
    tophat = _feature_tophat(gray, tophat_radii, texture_sigma, texture_threshold)
    _log("02_tophat_response", _heatmap(tophat, valid_mask))

    # ── 03 Local contrast ratio ──────────────────────────────────────────
    lcr_sigma = proxy_cfg.get("lcr_bg_sigma", 30.0)
    lcr = _feature_local_contrast_ratio(gray_f, lcr_sigma, valid_mask)
    _log("03_local_contrast", _heatmap(lcr, valid_mask))

    # ── 04 Combined score ────────────────────────────────────────────────
    score = tophat * lcr
    score[~valid_mask] = 0
    _log("04_combined_score", _heatmap(score, valid_mask))

    # ── 05 Threshold (density-adaptive, two-pass) ─────────────────────
    abs_threshold = proxy_cfg.get("score_threshold", 5.0)
    pct_lo, pct_hi = proxy_cfg.get("score_percentile_range", (70, 90))
    dense_abs_min = proxy_cfg.get("dense_score_threshold_min", 3.0)
    dense_frac = proxy_cfg.get("dense_frac_trigger", 0.03)

    valid_scores = score[valid_mask]
    pos = valid_scores[valid_scores > 0]
    if pos.size > 100:
        frac_above = float(np.sum(valid_scores > abs_threshold)) / n_valid
        density_t = np.clip(frac_above / dense_frac, 0.0, 1.0)
        noise_suppress = np.clip((params.noise_estimate - 4.0) / 4.0, 0.0, 1.0)
        density_t = density_t * (1.0 - noise_suppress)
        effective_abs = float(abs_threshold - density_t * (abs_threshold - dense_abs_min))
        adaptive_pct = float(pct_hi - density_t * (pct_hi - pct_lo))
        pct_threshold = float(np.percentile(pos, adaptive_pct))
        threshold = effective_abs + (1.0 - density_t) * max(0.0, pct_threshold - effective_abs)
    else:
        pct_threshold = abs_threshold
        threshold = abs_threshold

    binary = (score >= threshold).astype(np.uint8) * 255
    binary[~valid_mask] = 0
    _log("05_thresholded", binary)

    # ── 06 Morphological cleanup ─────────────────────────────────────────
    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params.morph_close_k, params.morph_close_k),
    )
    open_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params.morph_open_k, params.morph_open_k),
    )
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_k)
    _log("06_morph_cleaned", cleaned)

    # ── Contour shape filtering (score-gated) ──────────────────────────
    contours_raw, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    reject_counts = {
        "area_too_small": 0, "area_too_large": 0,
        "low_solidity": 0, "eccentricity": 0, "low_circularity": 0,
        "weak_local_contrast": 0,
    }
    filtered_contours = []

    gray_filled = gray_f.copy()
    gray_filled[~valid_mask] = float(np.mean(gray_f[valid_mask]))
    local_bg_map = cv2.GaussianBlur(gray_filled, (0, 0), lcr_sigma)

    min_local_contrast = proxy_cfg.get("min_local_contrast", 0.02)
    shape_bypass_mult = proxy_cfg.get("shape_bypass_score_mult", 2.0)
    shape_bypass_threshold = threshold * shape_bypass_mult

    for c in contours_raw:
        area = cv2.contourArea(c)
        if area > params.max_contour_area:
            reject_counts["area_too_large"] += 1
            continue
        contour_mask_tmp = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask_tmp, [c], 0, 255, cv2.FILLED)
        contour_scores = score[contour_mask_tmp > 0]
        mean_score = float(np.mean(contour_scores)) if contour_scores.size > 0 else 0.0
        high_confidence = mean_score >= shape_bypass_threshold

        area_floor = 1 if high_confidence else params.min_contour_area
        if area < area_floor:
            reject_counts["area_too_small"] += 1
            continue

        skip_shape = area < 20 or high_confidence
        if not skip_shape:
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < params.min_solidity:
                reject_counts["low_solidity"] += 1
                continue

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

            perimeter = cv2.arcLength(c, True)
            circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            if circularity < params.min_circularity:
                reject_counts["low_circularity"] += 1
                continue

        interior = gray_f[contour_mask_tmp > 0]
        bg_vals = local_bg_map[contour_mask_tmp > 0]
        if interior.size > 0 and bg_vals.size > 0:
            mean_interior = float(np.mean(interior))
            mean_bg = float(np.mean(bg_vals))
            if mean_bg > 1.0:
                local_cr = (mean_bg - mean_interior) / mean_bg
                if local_cr < min_local_contrast:
                    reject_counts["weak_local_contrast"] += 1
                    continue

        filtered_contours.append(c)

    mask = np.zeros_like(cleaned)
    if filtered_contours:
        cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)
    _log("07_proxy_mask", mask)

    overlay = patch_bgr.copy()
    overlay[mask > 0] = [0, 255, 0]
    blended = cv2.addWeighted(patch_bgr, 0.6, overlay, 0.4, 0)
    _log("08_overlay", blended)

    stats = {
        "candidates_before_filter": len(contours_raw),
        "nodules_after_filter": len(filtered_contours),
        "coverage_pct": 100.0 * np.sum(mask > 0) / mask.size,
        "threshold_used": float(threshold),
        "rejection_counts": reject_counts,
    }

    return mask, steps, stats

# FILTER PIPELINE
_FILTER_REGISTRY = {
    "gray_world_white_balance": gray_world_white_balance,
    "illumination_normalize":   illumination_normalize,
    "clahe_lab":                clahe_lab,
    "bilateral_denoise":        bilateral_denoise,
    "multi_scale_retinex":      multi_scale_retinex,
    "sediment_fade":            sediment_fade,
    "unsharp_mask":             unsharp_mask,
}

class FilterPipeline:
    def __init__(self, preprocess_cfg):
        self.cfg = preprocess_cfg
        self.chain = preprocess_cfg.get("filter_chain", list(_FILTER_REGISTRY.keys()))

    def run(self, patch_bgr, params):
        steps: List[Tuple[str, np.ndarray]] = []
        steps.append(("00_original", patch_bgr.copy()))

        image = patch_bgr.copy()

        for i, step_name in enumerate(self.chain, start=1):
            fn = _FILTER_REGISTRY.get(step_name)
            if fn is None:
                logger.warning(f"Unknown filter step '{step_name}' — skipping")
                continue

            if step_name in (
                "illumination_normalize", "multi_scale_retinex", "sediment_fade", "unsharp_mask"):
                image = fn(image, params, self.cfg)
            else:
                image = fn(image, params)

            steps.append((f"{i:02d}_{step_name}", image.copy()))

        return image, steps

    @staticmethod
    def save_step_images(steps,output_dir, prefix = "patch"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images_for_grid = []

        for name, img in steps:
            fname = f"{prefix}__{name}.png"
            cv2.imwrite(str(output_dir / fname), img)
            if img.ndim == 2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img
            images_for_grid.append((name, img_bgr))

        composite_path = output_dir / f"{prefix}__composite.png"
        _build_composite(images_for_grid, composite_path)

        return composite_path

def _build_composite(named_images, output_path, thumb_size = 384, cols = 4, padding = 4, label_height = 28):
    n = len(named_images)
    if n == 0:
        return
    rows = (n + cols - 1) // cols

    cell_w = thumb_size + padding
    cell_h = thumb_size + label_height + padding
    canvas_w = cols * cell_w + padding
    canvas_h = rows * cell_h + padding

    canvas = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8)  

    for idx, (name, img) in enumerate(named_images):
        r, c = divmod(idx, cols)
        h, w = img.shape[:2]
        scale = thumb_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        thumb = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        x0 = c * cell_w + padding
        y0 = r * cell_h + padding
        dx = (thumb_size - new_w) // 2
        dy = (thumb_size - new_h) // 2
        canvas[y0 + dy : y0 + dy + new_h, x0 + dx : x0 + dx + new_w] = thumb

        label_y = y0 + thumb_size + 2
        cv2.putText(
            canvas, name, (x0 + 4, label_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), canvas)
    logger.debug(f"Saved composite grid: {output_path}")
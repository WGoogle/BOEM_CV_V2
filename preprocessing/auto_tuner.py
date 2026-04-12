"""
Per-Patch Adaptive Parameters

Essentially, a single set of preprocessing parameters cannot work across an entire seafloor mosaic.
So I use these four signal below to customzie parmaters of each patch: 
1. Contrast ratio: (inter-quartile range of L-channel) alters CLAHE clip limit.

2. Noise estimate: (median absolute deviation of Laplacian) alters bilateral filter sigmas.

3. Brightness / darkness balance: (histogram skewness) alters adaptive threshold C-offset and morphological kernel sizes.

4. Illumination uniformity (std of large-scale blur) alters whether to apply local normalisation before thresholding.
"""
from __future__ import annotations
import logging
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

@dataclass
class TunedParams:
    # Parameters computed for a single patch
     
    msr_blend: float = 1.0  # 0 = keep original L, 1 = full retinex

    clahe_clip_limit: float = 2.0
    clahe_blend: float = 1.0 # 0 = keep original, 1 = full CLAHE
    clahe_tile_grid: Tuple[int, int] = (8, 8)

    bilateral_d: int = 7
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0

    adaptive_block_size: int = 31
    adaptive_c_offset: float = 8.0

    morph_open_k: int = 5
    morph_close_k: int = 9

    unsharp_strength: float = 0.5 

    min_contour_area: int = 5
    max_contour_area: int = 5000
    max_eccentricity: float = 0.85
    min_solidity: float = 0.50
    min_circularity: float = 0.30

    contrast_ratio: float = 0.0
    noise_estimate: float = 0.0
    brightness_skew: float = 0.0
    illumination_uniformity: float = 0.0

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}

class PatchAutoTuner:
    # Compute per-patch preprocessing parameters from image statistics.

    def __init__(self, config):
        self.cfg = config

    # Pixels below this threshold are treated as black border / no-data and excluded from all diagnostic calculations.
    _BORDER_THRESHOLD = 10

    def analyse(self, patch_bgr):
        # Extract L channel from LAB
        lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

        # Build valid-pixel mask — exclude black borders from AUV mosaic
        valid_mask = gray > self._BORDER_THRESHOLD

        # Diagnostic signals (computed on valid pixels only)
        contrast   = self._contrast_ratio(L, valid_mask)
        noise      = self._noise_estimate(gray, valid_mask)
        skew       = self._brightness_skew(L, valid_mask)
        uniformity = self._illumination_uniformity(L, valid_mask)
        mean = cv2.blur(gray, (5, 5))
        mean_sq = cv2.blur(gray**2, (5, 5))
        variance = np.maximum(0, mean_sq - (mean**2))
        avg_variance = np.median(variance)
        std_dev = np.sqrt(avg_variance)

        params = TunedParams()
        t = np.clip(contrast / 40.0, 0.0, 1.0)
        n = np.clip(noise / 20.0, 0.0, 1.0)
        msr_lo, msr_hi = self.cfg.get("msr_blend_range", (0.3, 1.0))
        params.msr_blend = float(msr_lo + t * (msr_hi - msr_lo))
        cmin, cmax = self.cfg["clahe_clip_range"]
        base_clip = float(cmin + t * (cmax - cmin))
        bmin, bmax = self.cfg.get("clahe_blend_range", (0.3, 1.0))
        base_blend = float(bmin + t * (bmax - bmin))
        noise_penalty = np.clip(noise / 15.0, 0.0, 1.0)
        params.clahe_clip_limit = float(base_clip * (1.0 - 0.6 * noise_penalty))
        params.clahe_blend = float(base_blend * (1.0 - 0.5 * noise_penalty))
        params.clahe_tile_grid = tuple(self.cfg["clahe_tile_grid"])
        min_d = 5
        max_d = 15
        max_v = np.percentile(variance, 90)
        if max_v == 0: max_v = 1
        normalized_v = np.clip(variance / max_v, 0, 1)
        d_map = max_d - (normalized_v * (max_d - min_d))
        optimal_d = int(np.median(d_map))
        if optimal_d % 2 == 0:
            optimal_d += 1
        params.bilateral_d = optimal_d

        sc_lo, sc_hi = self.cfg.get("bilateral_sigma_color_range", (30, 60))
        ss_lo, ss_hi = self.cfg.get("bilateral_sigma_space_range", (30, 60))

        noise_extend = 1.0 + 0.5 * noise_penalty
        sc_hi_eff = int(sc_hi * noise_extend)
        ss_hi_eff = int(ss_hi * noise_extend)
        sigma_colors = np.arange(sc_lo, sc_hi_eff + 1, 10).tolist()
        sigma_spaces = np.arange(ss_lo, ss_hi_eff + 1, 10).tolist()
        combinations = []
        variances = []
        for color in sigma_colors:
            for space in sigma_spaces:
                combinations.append({'sigma_color': color, 'sigma_space': space})
                filtered = cv2.bilateralFilter(gray, d=optimal_d, sigmaColor=color, sigmaSpace=space)
                mean_filtered = cv2.blur(filtered, (5, 5))
                mean_sq_filtered = cv2.blur(filtered**2, (5, 5))
                variance_filtered = np.maximum(0, mean_sq_filtered - (mean_filtered**2))
                variances.append(np.median(variance_filtered))
        for i in range(len(variances)):
            if variances[i] == min(variances):
                params.bilateral_sigma_color = combinations[i]['sigma_color']
                params.bilateral_sigma_space = combinations[i]['sigma_space']

        bmin, bmax = self.cfg["block_size_range"]
        coff_min, coff_max = self.cfg["c_offset_range"]
        # Normalise skewness into [0, 1] — typical range -1 to +1
        s = np.clip((skew + 1.0) / 2.0, 0.0, 1.0)
        raw_block = int(bmin + s * (bmax - bmin))
        params.adaptive_block_size = raw_block if raw_block % 2 == 1 else raw_block + 1
        params.adaptive_c_offset = float(coff_min + s * (coff_max - coff_min))

        mo_min, mo_max = self.cfg["morph_open_range"]
        mc_min, mc_max = self.cfg["morph_close_range"]
        u = np.clip((uniformity - 2.0) / 15.0, 0.0, 1.0)
        raw_open = int(mo_min + u * (mo_max - mo_min))
        raw_close = int(mc_min + u * (mc_max - mc_min))
        params.morph_open_k = raw_open if raw_open % 2 == 1 else raw_open + 1
        params.morph_close_k = raw_close if raw_close % 2 == 1 else raw_close + 1

        us_lo, us_hi = self.cfg.get("unsharp_strength_range", (0.1, 0.5))
        params.unsharp_strength = float(us_lo + t * (us_hi - us_lo))

        ca_lo, ca_hi = self.cfg["contour_area_min_range"]
        params.min_contour_area = int(round(ca_lo + n * (ca_hi - ca_lo)))
        params.max_contour_area = self.cfg["max_contour_area"]

        ecc_lo, ecc_hi = self.cfg["eccentricity_range"]
        params.max_eccentricity = float(ecc_lo + n * (ecc_hi - ecc_lo))

        sol_lo, sol_hi = self.cfg["solidity_range"]
        params.min_solidity = float(sol_hi - n * (sol_hi - sol_lo))

        circ_lo, circ_hi = self.cfg["circularity_range"]
        params.min_circularity = float(circ_hi - n * (circ_hi - circ_lo))

        params.contrast_ratio           = float(contrast)
        params.noise_estimate           = float(noise)
        params.brightness_skew          = float(skew)
        params.illumination_uniformity  = float(uniformity)

        logger.debug(
            f"AutoTune: contrast={contrast:.1f}  noise={noise:.1f}  "
            f"skew={skew:.2f}  uniformity={uniformity:.1f}  →  "
            f"MSR blend={params.msr_blend:.2f}  "
            f"CLAHE clip={params.clahe_clip_limit:.2f} blend={params.clahe_blend:.2f}  "
            f"unsharp={params.unsharp_strength:.2f}  "
            f"bilateral σc={params.bilateral_sigma_color:.0f}  "
            f"block={params.adaptive_block_size}  "
            f"morph open/close={params.morph_open_k}/{params.morph_close_k}  "
            f"contour area_min={params.min_contour_area}  "
            f"ecc≤{params.max_eccentricity:.2f}  sol≥{params.min_solidity:.2f}  circ≥{params.min_circularity:.2f}"
        )
        return params

    @staticmethod
    def _contrast_ratio(L, valid_mask):
        valid_px = L[valid_mask]
        if valid_px.size < 100:
            return 0.0
        q75, q25 = np.percentile(valid_px, [75, 25])
        return float(q75 - q25)

    @staticmethod
    def _noise_estimate(gray, valid_mask):
        # Inspired from Immerkaer, "Fast noise variance estimation", CVIU 1996.
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        valid_lap = np.abs(laplacian[valid_mask])
        if valid_lap.size < 100:
            return 0.0
        med = np.median(valid_lap)
        return float(med * 1.4826)

    @staticmethod
    def _brightness_skew(L, valid_mask):
        valid_px = L[valid_mask]
        if valid_px.size < 100:
            return 0.0
        mean = np.mean(valid_px)
        std = np.std(valid_px)
        if std < 1e-6:
            return 0.0
        return float(np.mean(((valid_px - mean) / std) ** 3))

    @staticmethod
    def _illumination_uniformity(L, valid_mask):
        # Fill black border regions with the valid-pixel mean so they don't create artificial gradients in the blur.
        L_filled = L.copy()
        valid_px = L[valid_mask]
        if valid_px.size < 100:
            return 0.0
        L_filled[~valid_mask] = np.mean(valid_px)
        blurred = cv2.GaussianBlur(L_filled, (0, 0), sigmaX=30.0)
        return float(np.std(blurred[valid_mask]))
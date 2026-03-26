"""
auto_tuner.py — Per-Patch Adaptive Parameter Calculation
==========================================================
The core problem: a single set of preprocessing parameters cannot work
across an entire seafloor mosaic because lighting, sediment type, depth,
and camera altitude vary continuously along the AUV track.

This module analyzes each patch independently and computes "good"
parameters for that specific region.  The analysis is fast (histograms
and simple statistics) so it adds negligible overhead.

Signals used
------------
1. **Contrast ratio** (inter-quartile range of L-channel) → CLAHE clip limit.
   Low contrast → high clip (aggressive enhancement).
   High contrast → low clip (preserve existing dynamic range).

2. **Noise estimate** (median absolute deviation of Laplacian) → bilateral
   filter sigmas.  Noisy patches get stronger smoothing; clean patches
   get lighter smoothing to preserve fine nodule edges.

3. **Brightness / darkness balance** (histogram skewness) → adaptive
   threshold C-offset and morphological kernel sizes.  A dark patch
   (left-skewed histogram) needs a smaller C-offset; a bright patch
   needs a larger one to avoid false positives from bright sediment.

4. **Illumination uniformity** (std of large-scale blur) → whether to
   apply local normalisation before thresholding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ----------- Output container -----------

@dataclass
class TunedParams:
    """Parameters computed for a single patch."""

    # CLAHE
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)

    # Bilateral filter
    bilateral_d: int = 7
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0

    # Adaptive threshold (proxy-label stage)
    adaptive_block_size: int = 31
    adaptive_c_offset: float = 8.0

    # Morphological kernels
    morph_open_k: int = 5
    morph_close_k: int = 9

    # Nodule boost
    nodule_boost_factor: float = 2.0
    morph_radius: int = 20
    texture_sigma: float = 2.0
    texture_threshold: float = 12.0
    max_darkening: int = 70

    # Top-hat proxy labelling
    tophat_radii: list = field(default_factory=lambda: [12, 20, 30])
    tophat_percentile: float = 96.0
    tophat_threshold_floor: float = 15.0

    # Contour filters
    min_contour_area: int = 50
    max_contour_area: int = 3000
    max_eccentricity: float = 0.80
    min_solidity: float = 0.60
    min_circularity: float = 0.45

    # Diagnostics
    contrast_ratio: float = 0.0
    noise_estimate: float = 0.0
    brightness_skew: float = 0.0
    illumination_uniformity: float = 0.0

    def as_dict(self) -> Dict:
        """Serialisable snapshot for the pipeline manifest."""
        return {k: v for k, v in self.__dict__.items()}


# ----------- Tuner -----------

class PatchAutoTuner:
    """Compute per-patch preprocessing parameters from image statistics.

    Parameters
    ----------
    config : dict
        The ``AUTO_TUNER`` section from ``config.py``.
    """

    def __init__(self, config: dict):
        self.cfg = config

    # ── public API ───────────────────────────────────────────────────────────

    # Pixels below this threshold are treated as black border / no-data
    # and excluded from all diagnostic calculations.
    _BORDER_THRESHOLD = 10

    def analyse(self, patch_bgr: np.ndarray) -> TunedParams:
        """Analyse *patch_bgr* and return tuned parameters.

        This is the single entry point.  Internally it:
        1. Extracts the L channel (perceptual lightness).
        2. Builds a valid-pixel mask (excluding black mosaic borders).
        3. Computes four diagnostic signals on valid pixels only.
        4. Maps each signal to parameter ranges defined in config.
        """
        # Extract L channel from LAB
        lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

        # Build valid-pixel mask — exclude black borders from AUV mosaic
        valid_mask = gray > self._BORDER_THRESHOLD

        # ── Diagnostic signals (computed on valid pixels only) ────────
        contrast   = self._contrast_ratio(L, valid_mask)
        noise      = self._noise_estimate(gray, valid_mask)
        skew       = self._brightness_skew(L, valid_mask)
        uniformity = self._illumination_uniformity(L, valid_mask)

        # ── Map signals → parameters ─────────────────────────────────────
        params = TunedParams()

        # 1. CLAHE clip limit  (inverse of contrast)
        cmin, cmax = self.cfg["clahe_clip_range"]
        # Normalise contrast into [0, 1] — typical IQR for seafloor is 15-80
        t = np.clip((contrast - 15.0) / 65.0, 0.0, 1.0)
        params.clahe_clip_limit = float(cmax - t * (cmax - cmin))
        params.clahe_tile_grid = tuple(self.cfg["clahe_tile_grid"])

        # 2. Bilateral sigmas  (proportional to noise)
        params.bilateral_d = self.cfg["bilateral_d"]
        sc_min, sc_max = self.cfg["bilateral_sigma_color_range"]
        ss_min, ss_max = self.cfg["bilateral_sigma_space_range"]
        # Normalise noise into [0, 1] — typical MAD-Laplacian 2-20
        n = np.clip((noise - 2.0) / 18.0, 0.0, 1.0)
        params.bilateral_sigma_color = float(sc_min + n * (sc_max - sc_min))
        params.bilateral_sigma_space = float(ss_min + n * (ss_max - ss_min))

        # 3. Adaptive threshold block size & C-offset (from skewness)
        bmin, bmax = self.cfg["block_size_range"]
        coff_min, coff_max = self.cfg["c_offset_range"]
        # Normalise skewness into [0, 1] — typical range -1 to +1
        s = np.clip((skew + 1.0) / 2.0, 0.0, 1.0)
        raw_block = int(bmin + s * (bmax - bmin))
        params.adaptive_block_size = raw_block if raw_block % 2 == 1 else raw_block + 1
        params.adaptive_c_offset = float(coff_min + s * (coff_max - coff_min))

        # 4. Morphological kernels  (from uniformity)
        mo_min, mo_max = self.cfg["morph_open_range"]
        mc_min, mc_max = self.cfg["morph_close_range"]
        u = np.clip((uniformity - 2.0) / 15.0, 0.0, 1.0)
        raw_open = int(mo_min + u * (mo_max - mo_min))
        raw_close = int(mc_min + u * (mc_max - mc_min))
        params.morph_open_k = raw_open if raw_open % 2 == 1 else raw_open + 1
        params.morph_close_k = raw_close if raw_close % 2 == 1 else raw_close + 1

        # 5. Static pass-through from config (not adaptive yet)
        params.nodule_boost_factor  = self.cfg["nodule_boost_factor"]
        params.morph_radius         = self.cfg["morph_radius"]
        params.texture_sigma        = self.cfg["texture_sigma"]
        params.texture_threshold    = self.cfg["texture_threshold"]
        params.max_darkening        = self.cfg["max_darkening"]
        params.tophat_radii         = list(self.cfg["tophat_radii"])
        params.tophat_percentile    = self.cfg["tophat_percentile"]
        params.tophat_threshold_floor = self.cfg["tophat_threshold_floor"]
        params.min_contour_area     = self.cfg["min_contour_area"]
        params.max_contour_area     = self.cfg["max_contour_area"]
        params.max_eccentricity     = self.cfg["max_eccentricity"]
        params.min_solidity         = self.cfg["min_solidity"]
        params.min_circularity      = self.cfg["min_circularity"]

        # Store diagnostics
        params.contrast_ratio           = float(contrast)
        params.noise_estimate           = float(noise)
        params.brightness_skew          = float(skew)
        params.illumination_uniformity  = float(uniformity)

        logger.debug(
            f"AutoTune: contrast={contrast:.1f}  noise={noise:.1f}  "
            f"skew={skew:.2f}  uniformity={uniformity:.1f}  →  "
            f"CLAHE clip={params.clahe_clip_limit:.2f}  "
            f"bilateral σc={params.bilateral_sigma_color:.0f}  "
            f"block={params.adaptive_block_size}  "
            f"morph open/close={params.morph_open_k}/{params.morph_close_k}"
        )
        return params

    # ── diagnostic signal extractors ─────────────────────────────────────────
    # All methods accept a valid_mask to exclude black mosaic borders.

    @staticmethod
    def _contrast_ratio(L: np.ndarray, valid_mask: np.ndarray) -> float:
        """Inter-quartile range of L-channel valid pixels.

        A high IQR means the patch already has good contrast between
        nodules and sediment; a low IQR means everything looks flat.
        """
        valid_px = L[valid_mask]
        if valid_px.size < 100:
            return 0.0
        q75, q25 = np.percentile(valid_px, [75, 25])
        return float(q75 - q25)

    @staticmethod
    def _noise_estimate(gray: np.ndarray, valid_mask: np.ndarray) -> float:
        """Estimate sensor noise using the Median Absolute Deviation of
        the Laplacian on valid pixels only.

        Ref: Immerkaer, "Fast noise variance estimation", CVIU 1996.
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        valid_lap = np.abs(laplacian[valid_mask])
        if valid_lap.size < 100:
            return 0.0
        med = np.median(valid_lap)
        return float(med * 1.4826)

    @staticmethod
    def _brightness_skew(L: np.ndarray, valid_mask: np.ndarray) -> float:
        """Skewness of the L-channel histogram (valid pixels only).

        Negative skew → dark patch (many dark pixels, few bright).
        Positive skew → bright patch.
        Near zero → balanced.
        """
        valid_px = L[valid_mask]
        if valid_px.size < 100:
            return 0.0
        mean = np.mean(valid_px)
        std = np.std(valid_px)
        if std < 1e-6:
            return 0.0
        return float(np.mean(((valid_px - mean) / std) ** 3))

    @staticmethod
    def _illumination_uniformity(L: np.ndarray, valid_mask: np.ndarray) -> float:
        """Std-dev of a heavily blurred version of L (valid pixels only).

        High value → strong illumination gradient across the patch
        (e.g. AUV light cone falloff).  Low value → uniform lighting.
        """
        # Fill black border regions with the valid-pixel mean so they
        # don't create artificial gradients in the blur.
        L_filled = L.copy()
        valid_px = L[valid_mask]
        if valid_px.size < 100:
            return 0.0
        L_filled[~valid_mask] = np.mean(valid_px)
        blurred = cv2.GaussianBlur(L_filled, (0, 0), sigmaX=30.0)
        return float(np.std(blurred[valid_mask]))
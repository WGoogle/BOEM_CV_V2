"""
patcher.py — Intelligent Mosaic Patching
==========================================
Splits massive seafloor .TIF strips into smaller, manageable patches and
can recombine them later (for inference).  Key design decisions:

* **Tiled reading** via ``cv2.imread`` (with fallback to ``tifffile`` for
  multi-GB TIFs that OpenCV cannot memory-map).
* **Overlap** between patches so nodules on tile edges are never cut.
* **Quality gate** rejects featureless (black border / uniform sediment)
  patches before they enter the preprocessing pipeline.
* Patch metadata is returned as a list of dicts for downstream tracking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ----------- Data classes -----------

@dataclass
class PatchInfo:
    """Metadata for a single extracted patch."""
    patch_index: int
    row: int                        # grid row
    col: int                        # grid column
    y: int                          # top-left y in the full mosaic
    x: int                          # top-left x in the full mosaic
    height: int
    width: int
    is_valid: bool = True           # False → rejected by quality gate
    rejection_reason: str = ""


# ----------- MosaicPatcher -----------

class MosaicPatcher:
    """Split / recombine large seafloor mosaics.

    Parameters
    ----------
    patch_size : int
        Side length of each square patch (pixels).
    overlap : int
        Overlap between adjacent patches (pixels).
    min_std, min_mean, max_black_fraction :
        Quality-gate thresholds.  Patches that fail are still tracked
        (``is_valid=False``) so the grid can be reconstructed, but they
        are excluded from preprocessing.
    """

    def __init__(
        self,
        patch_size: int = 1024,
        overlap: int = 128,
        min_std: float = 5.0,
        min_mean: float = 10.0,
        max_black_fraction: float = 0.25,
    ):
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.min_std = min_std
        self.min_mean = min_mean
        self.max_black_fraction = max_black_fraction

    # ── public ───────────────────────────────────────────────────────────────

    def load_mosaic(self, filepath: Path) -> np.ndarray:
        """Load a mosaic from disk.  Handles TIF, PNG, JPG.

        For very large TIFs (>2 GB) that OpenCV cannot load, falls back
        to ``tifffile`` if installed.
        """
        filepath = Path(filepath)
        image = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)

        if image is None:
            # Fallback: try tifffile for huge TIFs
            try:
                import tifffile
                image = tifffile.imread(str(filepath))
                # tifffile returns RGB; convert to BGR for OpenCV
                if image.ndim == 3 and image.shape[2] >= 3:
                    image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR)
                logger.info(f"Loaded via tifffile: {filepath.name}")
            except ImportError:
                raise RuntimeError(
                    f"OpenCV failed to load {filepath.name} and `tifffile` is "
                    f"not installed.  Install it with:  pip install tifffile"
                )
            except Exception as exc:
                raise RuntimeError(f"Cannot load {filepath.name}: {exc}")

        # Normalise to 3-channel uint8 BGR
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if image.dtype != np.uint8:
            if image.max() > 255:
                image = (image / image.max() * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        logger.info(
            f"Loaded mosaic: {filepath.name}  "
            f"shape={image.shape}  dtype={image.dtype}"
        )
        return image

    def extract_patches(
        self,
        mosaic: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[PatchInfo]]:
        """Divide *mosaic* into overlapping patches.

        Returns
        -------
        patches : list[np.ndarray]
            Only the patches that pass the quality gate.
        infos : list[PatchInfo]
            Metadata for **all** patches (including invalid ones so the
            grid geometry is preserved for reassembly).
        """
        H, W = mosaic.shape[:2]
        ps, stride = self.patch_size, self.stride

        # Compute grid positions, ensuring edge coverage
        y_starts = self._grid_positions(H, ps, stride)
        x_starts = self._grid_positions(W, ps, stride)

        patches: List[np.ndarray] = []
        infos: List[PatchInfo] = []
        idx = 0

        for ri, y in enumerate(y_starts):
            for ci, x in enumerate(x_starts):
                # Clamp to image bounds (handles edges)
                y_end = min(y + ps, H)
                x_end = min(x + ps, W)
                patch = mosaic[y:y_end, x:x_end]

                # Pad if the patch is smaller than patch_size (edge case)
                if patch.shape[0] < ps or patch.shape[1] < ps:
                    padded = np.zeros((ps, ps, mosaic.shape[2]), dtype=mosaic.dtype)
                    padded[: patch.shape[0], : patch.shape[1]] = patch
                    patch = padded

                info = PatchInfo(
                    patch_index=idx, row=ri, col=ci,
                    y=y, x=x, height=ps, width=ps,
                )

                # Quality gate
                valid, reason = self._quality_check(patch)
                info.is_valid = valid
                info.rejection_reason = reason

                infos.append(info)
                if valid:
                    patches.append(patch.copy())
                idx += 1

        n_total = len(infos)
        n_valid = len(patches)
        logger.info(
            f"Patching: {n_valid}/{n_total} patches passed quality gate  "
            f"(grid {len(y_starts)}×{len(x_starts)}, patch={ps}px, "
            f"overlap={self.overlap}px)"
        )
        return patches, infos

    def reassemble(
        self,
        patch_outputs: List[np.ndarray],
        infos: List[PatchInfo],
        mosaic_shape: Tuple[int, int],
        dtype=np.float32,
    ) -> np.ndarray:
        """Recombine patch-level predictions into a full-mosaic map.

        Uses simple averaging in overlapping regions.

        Parameters
        ----------
        patch_outputs : list of 2-D arrays (H_patch, W_patch)
            One per **valid** patch, same order as ``extract_patches`` output.
        infos : list[PatchInfo]
            Full info list (including invalid patches).
        mosaic_shape : (H, W) of the original mosaic.
        """
        H, W = mosaic_shape
        accum = np.zeros((H, W), dtype=np.float64)
        count = np.zeros((H, W), dtype=np.float64)

        valid_idx = 0
        for info in infos:
            if not info.is_valid:
                continue
            pred = patch_outputs[valid_idx].astype(np.float64)
            valid_idx += 1

            y, x = info.y, info.x
            ph = min(info.height, H - y)
            pw = min(info.width, W - x)
            accum[y : y + ph, x : x + pw] += pred[:ph, :pw]
            count[y : y + ph, x : x + pw] += 1.0

        count[count == 0] = 1.0
        return (accum / count).astype(dtype)

    # ── private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _grid_positions(length: int, patch_size: int, stride: int) -> List[int]:
        """Generate start positions that cover the full length."""
        if length <= patch_size:
            return [0]
        positions = list(range(0, length - patch_size + 1, stride))
        # Ensure the very last pixel is covered
        if positions[-1] + patch_size < length:
            positions.append(length - patch_size)
        return sorted(set(positions))

    def _quality_check(self, patch: np.ndarray) -> Tuple[bool, str]:
        """Return (is_valid, rejection_reason)."""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch

        # Black-border check
        black_frac = np.mean(gray < 5)
        if black_frac > self.max_black_fraction:
            return False, f"black_fraction={black_frac:.2f}"

        # Uniform / featureless check
        std = float(np.std(gray))
        if std < self.min_std:
            return False, f"std={std:.1f}"

        mean = float(np.mean(gray))
        if mean < self.min_mean:
            return False, f"mean={mean:.1f}"

        return True, ""
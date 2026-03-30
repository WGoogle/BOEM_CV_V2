"""
patcher.py description:

Splits seafloor .TIF strips into smaller patches and can recombine them later (for inference). 
"""
import cv2
import numpy as np
import logging

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


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


class MosaicPatcher:
    """
    Split then recombine large seafloor mosaics.

    Non Self-Explanatory Parameters:
    min_std, min_mean, max_black_fraction :
        Quality-gate thresholds.  Patches that fail are still tracked
        w/ (is_valid=False) so the grid can be reconstructed, but they
        are excluded from preprocessing as these patches are deemed to have no seafloor image.

    """

    def __init__(
        self,
        patch_size: int = 256,
        overlap: int = 32,
        min_std: float = 3.0,
        min_mean: float = 10.0,
        max_black_fraction: float = 0.25,
    ):
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.min_std = min_std
        self.min_mean = min_mean
        self.max_black_fraction = max_black_fraction


    def load_mosaic(self, filepath: Path) -> np.ndarray:
        """Load a mosaic from disk.  Can use TIF, PNG, JPG.

        For TIF files greather than 2 GB, I got a failsafe to "tifffile."
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

    def extract_patches(self, mosaic: np.ndarray) -> Tuple[List[np.ndarray], List[PatchInfo]]:
        """
        Divide mosaic into overlapping patches.

        Returns patches (that pass quality gate) in list array and metadata for all patches for restructuring grid in future.
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

    def reassemble(self, patch_outputs: List[np.ndarray], infos: List[PatchInfo], mosaic_shape: Tuple[int, int], dtype=np.float32) -> np.ndarray:
        """
        Recombine patches into a full-mosaic map.
       
        mosaic_shape is just the dimensions of original mosaic loaded before patching.
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
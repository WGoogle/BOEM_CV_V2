"""
training.sliding_window — Full-Mosaic Sliding-Window Inference
==============================================================
nnU-Net-style sliding window with **Gaussian importance weighting** for
seamless full-mosaic prediction. Replaces the patcher's uniform-average
``MosaicPatcher.reassemble`` for inference: edges of each window
contribute almost nothing to the stitched output, eliminating the
visible seams that uniform averaging produces at patch boundaries.

What this gives you over the existing per-patch inference path:

  - **No seam artifacts.** Each window's contribution to the global map
    is multiplied by a 2-D Gaussian peaked at the window centre and
    fading to ~0 at the edges, so adjacent windows blend smoothly.
  - **Mirror-pad-aware.** If the model was trained with the mirror-pad
    "extra context" trick (``TRAINING["mirror_pad"]`` > 0), the same
    padding is applied to every inference window so the encoder sees
    the same context distribution it learned on.
  - **Engineered-channel aware.** Mode-specific channel construction
    runs on every window via the existing
    :func:`training.dataset._prepare_image`, so engineered mode picks
    up the correct Sobel / LCR channels at inference.
  - **TTA built in.** Optional 4-way dihedral flip averaging matches
    the trainer's :meth:`Trainer._tta_probs` exactly.

Used by ``3_inference.py --mode mosaic`` and any future API that
needs full-mosaic prediction.
"""
from __future__ import annotations

import logging
from typing import Iterator

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import get_normalization_stats, _prepare_image

logger = logging.getLogger(__name__)


# ── Importance map ──────────────────────────────────────────────────────

def gaussian_importance_map(
    patch_size, sigma_scale = 0.125
):
    """2-D Gaussian importance map peaked at patch centre.

    Standard nnU-Net trick: the contribution of each sliding window to
    the stitched output is multiplied by this map so the centre of each
    window dominates and the edges (which see less context) contribute
    almost nothing. Eliminates the seam artifacts that uniform averaging
    produces at window boundaries.

    Parameters
    ----------
    patch_size : int
        Side length in pixels.
    sigma_scale : float
        σ as a fraction of the patch size. The nnU-Net default is 1/8
        which puts the FWHM at ~0.6·patch_size — strong centre weighting
        without zeroing the edges entirely.

    Returns
    -------
    np.ndarray
        ``(patch_size, patch_size)`` float32 with values in (0, 1].
    """
    sigma = patch_size * sigma_scale
    coords = np.arange(patch_size, dtype=np.float32) - (patch_size - 1) / 2.0
    g1d = np.exp(-0.5 * (coords / sigma) ** 2)
    g2d = np.outer(g1d, g1d)
    g2d /= g2d.max()
    # Floor at a small ε so the divide-by-weight at the end never explodes
    # in pathological cases (e.g. a window where literally only the corner
    # contributes). Cheaper and more stable than special-casing zeros.
    g2d = np.clip(g2d, 1e-3, 1.0)
    return g2d.astype(np.float32)


# ── Window grid (matches MosaicPatcher._grid_positions) ─────────────────

def _grid_positions(length, patch_size, stride):
    """Window start coordinates that fully cover [0, length).

    Last position snaps to ``length - patch_size`` so the trailing edge
    is always covered (consistent with ``MosaicPatcher._grid_positions``).
    """
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    if positions[-1] + patch_size < length:
        positions.append(length - patch_size)
    return sorted(set(positions))


# ── Per-window preprocessing ────────────────────────────────────────────

def _to_normalized_tensor(
    bgr_window, input_mode
):
    """BGR window → mode-specific channels → normalised tensor."""
    image = _prepare_image(bgr_window, input_mode)               # (H, W, 3) uint8
    image = image.astype(np.float32) / 255.0
    norm_mean, norm_std = get_normalization_stats(input_mode)
    mean = np.array(norm_mean, dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(norm_std,  dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()  # (3, H, W)


def _mirror_pad_tensor(t, pad):
    """Reflect-pad a (3, H, W) tensor by ``pad`` on every side."""
    if pad <= 0:
        return t
    return F.pad(t.unsqueeze(0), (pad, pad, pad, pad), mode="reflect").squeeze(0)


def _crop_center(t, h, w):
    """Centre-crop the last two dims of ``t`` to (h, w)."""
    H, W = t.shape[-2:]
    if (H, W) == (h, w):
        return t
    sh = (H - h) // 2
    sw = (W - w) // 2
    return t[..., sh:sh + h, sw:sw + w]


# ── TTA forward (matches Trainer._tta_probs) ────────────────────────────

@torch.no_grad()
def _forward(model, batch):
    return torch.sigmoid(model(batch))


@torch.no_grad()
def _tta_forward(model, batch):
    """4-way dihedral flip average. Each transform is its own inverse."""
    p  = _forward(model, batch)
    p += torch.flip(_forward(model, torch.flip(batch, dims=[-1])), dims=[-1])
    p += torch.flip(_forward(model, torch.flip(batch, dims=[-2])), dims=[-2])
    p += torch.flip(_forward(model, torch.flip(batch, dims=[-1, -2])), dims=[-1, -2])
    return p / 4.0


# ── Main entry point ────────────────────────────────────────────────────

@torch.no_grad()
def sliding_window_inference(
    model,
    mosaic_bgr,
    *,
    patch_size,
    overlap,
    input_mode = "rgb",
    device = "cpu",
    batch_size = 8,
    use_tta = True,
    mirror_pad = 0,
    use_amp = None,
):
    """Run a model over a full mosaic using overlapping windows.

    Each window is normalised, optionally mirror-padded, forwarded
    (optionally with 4-way TTA), un-padded back to ``patch_size``,
    multiplied by a Gaussian importance map, and accumulated into the
    full-mosaic probability buffer. The accumulator is divided by the
    accumulated importance weights at the end so the result is a
    proper weighted average.

    Parameters
    ----------
    model : nn.Module
        Already loaded, on ``device``, in ``eval()`` mode. Must accept a
        ``(B, 3, H, W)`` tensor and return ``(B, 1, H, W)`` logits.
    mosaic_bgr : np.ndarray
        ``(H, W, 3)`` uint8 BGR mosaic — already preprocessed by Step 1
        (the model was trained on preprocessed patches).
    patch_size, overlap : int
        Same values used at training time
        (``config.PATCHING["patch_size"]`` / ``["overlap"]``).
    input_mode : {"rgb", "grayscale", "engineered"}
        Channel construction mode. Must match
        ``config.MODEL["input_mode"]``.
    device : str
        ``"cuda"``, ``"mps"``, or ``"cpu"``.
    batch_size : int
        Number of windows forwarded per batch. Lower if you OOM.
    use_tta : bool
        Apply 4-way dihedral flip TTA per window. Worth ~1 Dice point.
    mirror_pad : int
        If > 0, every window is reflect-padded by this many pixels on
        each side before forwarding, then the model output is centre-
        cropped back to ``patch_size``. Must match
        ``config.TRAINING["mirror_pad"]`` if the model was trained with
        the extra-context trick.
    use_amp : bool | None
        Override AMP. ``None`` enables AMP automatically on CUDA only.

    Returns
    -------
    np.ndarray
        ``(H, W)`` float32 probability map in [0, 1] over the full mosaic.
    """
    if mosaic_bgr.ndim != 3 or mosaic_bgr.shape[2] != 3:
        raise ValueError(
            f"mosaic_bgr must be (H, W, 3) BGR uint8; got shape {mosaic_bgr.shape}"
        )

    H, W = mosaic_bgr.shape[:2]
    stride = max(1, patch_size - overlap)

    # When the mosaic is smaller than one patch, pad it up so the model
    # still gets the expected input size — the trailing edge is reflect-
    # padded and we crop the prediction back at the end.
    pad_h = max(0, patch_size - H)
    pad_w = max(0, patch_size - W)
    if pad_h or pad_w:
        mosaic_bgr = cv2.copyMakeBorder(
            mosaic_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101,
        )

    Hp, Wp = mosaic_bgr.shape[:2]
    y_starts = _grid_positions(Hp, patch_size, stride)
    x_starts = _grid_positions(Wp, patch_size, stride)
    n_windows = len(y_starts) * len(x_starts)

    importance = gaussian_importance_map(patch_size)             # (ps, ps) float32
    importance_t = torch.from_numpy(importance).to(device)        # for tensor mul

    accum  = torch.zeros((Hp, Wp), dtype=torch.float32, device=device)
    weight = torch.zeros((Hp, Wp), dtype=torch.float32, device=device)

    if use_amp is None:
        use_amp = (device == "cuda")
    amp_ctx = (
        torch.autocast(device_type="cuda")
        if use_amp
        else _nullcontext()
    )

    logger.info(
        "  Sliding-window: %d windows  (grid %dx%d, ps=%d, overlap=%d, "
        "tta=%s, mirror_pad=%d)",
        n_windows, len(y_starts), len(x_starts), patch_size, overlap,
        use_tta, mirror_pad,
    )

    # Stream windows in batches so we never blow up VRAM
    coords_iter = (
        (y, x)
        for y in y_starts
        for x in x_starts
    )

    with amp_ctx:
        for batch_coords, batch_tensor in _batched_windows(
            coords_iter, mosaic_bgr, patch_size, mirror_pad,
            input_mode, batch_size, device,
        ):
            probs = _tta_forward(model, batch_tensor) if use_tta else _forward(model, batch_tensor)
            # probs: (B, 1, H, W) — H/W include mirror padding
            if mirror_pad > 0:
                probs = _crop_center(probs, patch_size, patch_size)
            probs = probs.squeeze(1)                              # (B, ps, ps)

            # Gaussian-weight and accumulate
            weighted = probs * importance_t                      # (B, ps, ps)
            for (y, x), wp in zip(batch_coords, weighted):
                accum [y:y + patch_size, x:x + patch_size] += wp
                weight[y:y + patch_size, x:x + patch_size] += importance_t

    # Normalise by accumulated importance and crop back to original size
    weight.clamp_(min=1e-6)
    full = (accum / weight).clamp(0.0, 1.0).cpu().numpy()
    if pad_h or pad_w:
        full = full[:H, :W]
    return full.astype(np.float32)


# ── Helpers ─────────────────────────────────────────────────────────────

class _nullcontext:
    """Local nullcontext that doesn't pull in contextlib for one call site."""
    def __enter__(self): return None
    def __exit__(self, *a): return False


def _batched_windows(
    coords_iter,
    mosaic_bgr,
    patch_size,
    mirror_pad,
    input_mode,
    batch_size,
    device,
):
    """Yield (coords, batch_tensor) pairs of up to ``batch_size`` windows."""
    coords_buf: list[tuple[int, int]] = []
    tensors_buf: list[torch.Tensor] = []

    for (y, x) in coords_iter:
        window = mosaic_bgr[y:y + patch_size, x:x + patch_size]
        t = _to_normalized_tensor(window, input_mode)
        if mirror_pad > 0:
            t = _mirror_pad_tensor(t, mirror_pad)
        coords_buf.append((y, x))
        tensors_buf.append(t)

        if len(tensors_buf) == batch_size:
            batch = torch.stack(tensors_buf, dim=0).to(device, non_blocking=True)
            yield coords_buf, batch
            coords_buf, tensors_buf = [], []

    if tensors_buf:
        batch = torch.stack(tensors_buf, dim=0).to(device, non_blocking=True)
        yield coords_buf, batch

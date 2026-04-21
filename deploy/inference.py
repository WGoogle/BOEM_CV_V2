"""
inference.py — Self-contained nodule segmentation inference
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

ENGINEERED_MEAN = (0.4121, 0.0428, 0.0357)
ENGINEERED_STD  = (0.0685, 0.0820, 0.1501)

_LCR_BG_SIGMA = 30.0 

def get_normalization_stats(input_mode, checkpoint_dir=None):
    if input_mode != "engineered":
        return IMAGENET_MEAN, IMAGENET_STD

    if checkpoint_dir is not None:
        cache = Path(checkpoint_dir) / "engineered_norm_stats.json"
        if cache.exists():
            with open(cache) as f:
                data = json.load(f)
            return tuple(data["mean"]), tuple(data["std"])

    return ENGINEERED_MEAN, ENGINEERED_STD

def compute_engineered_channels(
    bgr, lcr_sigma = _LCR_BG_SIGMA
):
    """Build [L, Sobel mag, LCR] from a preprocessed BGR uint8 patch."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"expected BGR (H,W,3) uint8, got shape {bgr.shape}")

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]                                          # uint8

    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    sobel = cv2.magnitude(gx, gy)
    sobel_u8 = np.clip(sobel, 0.0, 255.0).astype(np.uint8)

    L_f = L.astype(np.float32)
    bg = cv2.GaussianBlur(L_f, (0, 0), sigmaX=lcr_sigma, sigmaY=lcr_sigma)
    lcr = (bg - L_f) / (bg + 1e-6)
    lcr_u8 = np.clip(lcr * 255.0, 0.0, 255.0).astype(np.uint8)

    return np.stack([L, sobel_u8, lcr_u8], axis=-1)

_VALID_MODES = ("rgb", "engineered")

def prepare_image(bgr, input_mode):
    if input_mode not in _VALID_MODES:
        raise ValueError(
            f"input_mode must be one of {_VALID_MODES}; got {input_mode!r}"
        )
    if input_mode == "rgb":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return compute_engineered_channels(bgr)

def load_model_config(config_path):
    # Read the model_config.json sidecar shipped with the checkpoint
    with open(config_path, "r") as f:
        return json.load(f)

def build_model(arch, encoder_name, in_channels = 3, classes = 1):
    """Instantiate an SMP model with raw-logit output."""
    return smp.create_model(
        arch=arch,
        encoder_name=encoder_name,
        encoder_weights=None,       
        in_channels=in_channels,
        classes=classes,
        activation=None,            
    )

def load_model(checkpoint_path, model_config, device):
    model = build_model(
        arch         = model_config["architecture"],
        encoder_name = model_config["encoder_name"],
        in_channels  = int(model_config.get("in_channels", 3)),
        classes      = int(model_config.get("classes", 1)),
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    best_threshold = ckpt.get("best_threshold", None)
    return model, best_threshold

def load_mosaic(filepath):
    filepath = Path(filepath)
    image = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)

    if image is None:
        try:
            import tifffile
        except ImportError as e:
            raise RuntimeError(
                f"OpenCV failed to load {filepath.name} and `tifffile` "
                "is not installed. Install it with `pip install tifffile`."
            ) from e
        image = tifffile.imread(str(filepath))
        if image.ndim == 3 and image.shape[2] >= 3:
            image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.dtype != np.uint8:
        if image.max() > 255:
            image = (image / image.max() * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    return image

def gaussian_importance_map(patch_size, sigma_scale = 0.125):
    sigma = patch_size * sigma_scale
    coords = np.arange(patch_size, dtype=np.float32) - (patch_size - 1) / 2.0
    g1d = np.exp(-0.5 * (coords / sigma) ** 2)
    g2d = np.outer(g1d, g1d)
    g2d /= g2d.max()
    g2d = np.clip(g2d, 1e-3, 1.0)
    return g2d.astype(np.float32)

def _grid_positions(length, patch_size, stride):
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    if positions[-1] + patch_size < length:
        positions.append(length - patch_size)
    return sorted(set(positions))

def _to_normalized_tensor(bgr_window, input_mode, norm_stats=None):
    image = prepare_image(bgr_window, input_mode).astype(np.float32) / 255.0
    if norm_stats is None:
        norm_stats = get_normalization_stats(input_mode)
    norm_mean, norm_std = norm_stats
    mean = np.array(norm_mean, dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(norm_std,  dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()

@torch.no_grad()
def _forward(model, batch):
    return torch.sigmoid(model(batch))

@torch.no_grad()
def _tta_forward(model, batch):
    """4-way dihedral flip averaging. Each flip is its own inverse."""
    p  = _forward(model, batch)
    p += torch.flip(_forward(model, torch.flip(batch, dims=[-1])), dims=[-1])
    p += torch.flip(_forward(model, torch.flip(batch, dims=[-2])), dims=[-2])
    p += torch.flip(_forward(model, torch.flip(batch, dims=[-1, -2])), dims=[-1, -2])
    return p / 4.0

def _batched_windows(coords_iter, mosaic_bgr, patch_size, input_mode, batch_size,
    device, norm_stats=None):

    coords_buf: list[tuple[int, int]] = []
    tensors_buf: list[torch.Tensor] = []
    for (y, x) in coords_iter:
        window = mosaic_bgr[y:y + patch_size, x:x + patch_size]
        t = _to_normalized_tensor(window, input_mode, norm_stats=norm_stats)
        coords_buf.append((y, x))
        tensors_buf.append(t)
        if len(tensors_buf) == batch_size:
            yield coords_buf, torch.stack(tensors_buf, dim=0).to(device, non_blocking=True)
            coords_buf, tensors_buf = [], []
    if tensors_buf:
        yield coords_buf, torch.stack(tensors_buf, dim=0).to(device, non_blocking=True)

@torch.no_grad()
def sliding_window_inference(model, mosaic_bgr, *, patch_size, overlap,
    input_mode = "engineered", device = "cpu", batch_size = 8, use_tta = True,
    use_amp = None, progress = True, norm_stats = None):

    if mosaic_bgr.ndim != 3 or mosaic_bgr.shape[2] != 3:
        raise ValueError(
            f"mosaic_bgr must be (H, W, 3) BGR uint8; got {mosaic_bgr.shape}"
        )

    H, W = mosaic_bgr.shape[:2]
    stride = max(1, patch_size - overlap)

    pad_h = max(0, patch_size - H)
    pad_w = max(0, patch_size - W)
    if pad_h or pad_w:
        mosaic_bgr = cv2.copyMakeBorder(
            mosaic_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
        )

    Hp, Wp = mosaic_bgr.shape[:2]
    y_starts = _grid_positions(Hp, patch_size, stride)
    x_starts = _grid_positions(Wp, patch_size, stride)
    n_windows = len(y_starts) * len(x_starts)

    importance = gaussian_importance_map(patch_size)
    importance_t = torch.from_numpy(importance).to(device)

    accum  = torch.zeros((Hp, Wp), dtype=torch.float32, device=device)
    weight = torch.zeros((Hp, Wp), dtype=torch.float32, device=device)

    if use_amp is None:
        use_amp = (device == "cuda")
    amp_ctx = torch.autocast(device_type="cuda") if use_amp else _NullCtx()

    if progress:
        print(
            f"  Sliding-window: {n_windows} windows  "
            f"(grid {len(y_starts)}x{len(x_starts)}, ps={patch_size}, "
            f"overlap={overlap}, tta={use_tta})"
        )

    coords_iter = ((y, x) for y in y_starts for x in x_starts)

    done = 0
    with amp_ctx:
        for batch_coords, batch_tensor in _batched_windows(
            coords_iter, mosaic_bgr, patch_size,
            input_mode, batch_size, device, norm_stats=norm_stats,
        ):
            probs = _tta_forward(model, batch_tensor) if use_tta else _forward(model, batch_tensor)
            probs = probs.squeeze(1)        # (B, ps, ps)

            weighted = probs * importance_t
            for (y, x), wp in zip(batch_coords, weighted):
                accum [y:y + patch_size, x:x + patch_size] += wp
                weight[y:y + patch_size, x:x + patch_size] += importance_t

            done += len(batch_coords)
            if progress and (done % max(1, batch_size * 10) == 0 or done == n_windows):
                print(f"    {done}/{n_windows} windows")

    weight.clamp_(min=1e-6)
    full = (accum / weight).clamp(0.0, 1.0).cpu().numpy()
    if pad_h or pad_w:
        full = full[:H, :W]
    return full.astype(np.float32)

class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *a): return False
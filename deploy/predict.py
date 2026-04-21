"""
predict.py — End-to-end nodule segmentation on raw seafloor mosaics.

One-click pipeline:
    raw mosaic  ->  preprocess  ->  model inference  ->  binary mask
                                                    ->  coverage %
                                                    ->  nodule density (per m^2)

Default:  `python predict.py`  scans ./input/, processes every mosaic found,
and writes outputs to ./predictions/.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from inference import (
    get_normalization_stats,
    load_model,
    load_model_config,
    sliding_window_inference,
)
from preprocess import preprocess_mosaic
from metrics import compute_metrics, format_metrics_report, seafloor_mask_from_raw

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _pick_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _gather_inputs(input_path):
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in _IMG_EXTS
        )
    raise FileNotFoundError(f"Input not found: {input_path}")


def _save_overlay(out_path, base_bgr, binary_mask, alpha=0.35):
    # Cyan tint on nodule pixels so you can eyeball-check against the seafloor.
    color = np.zeros_like(base_bgr)
    color[:, :, 0] = 255  # B
    color[:, :, 1] = 255  # G
    bool_mask = binary_mask > 0
    overlay = base_bgr.copy()
    overlay[bool_mask] = (
        (1.0 - alpha) * base_bgr[bool_mask] + alpha * color[bool_mask]
    ).astype(np.uint8)
    cv2.imwrite(str(out_path), overlay)


def _save_outline_overlay(out_path, base_bgr, binary_mask, thickness=1):
    overlay = base_bgr.copy()
    contours, _ = cv2.findContours(
        (binary_mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), thickness)
    cv2.imwrite(str(out_path), overlay)


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="BOEM nodule segmentation — raw mosaic in, metrics out.",
    )
    p.add_argument(
        "input", nargs="?", type=Path, default=here / "input",
        help="Raw mosaic file or directory of mosaics. Default: ./input/",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Checkpoint .pt. Default: ./checkpoints/checkpoint_best.pt",
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="model_config.json sidecar. Default: ./model_config.json",
    )
    p.add_argument(
        "--out", type=Path, default=here / "predictions",
        help="Output directory. Default: ./predictions",
    )
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Override binary threshold. Resolution order: flag -> checkpoint "
             "best_threshold -> config threshold -> 0.5.",
    )
    p.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
        help="Compute device. Default: auto-detect.",
    )
    p.add_argument(
        "--skip-preprocess", action="store_true",
        help="Treat inputs as ALREADY preprocessed. Skips the Step-1 filter chain.",
    )
    p.add_argument(
        "--save-preprocessed", action="store_true",
        help="Also write the intermediate preprocessed mosaic to predictions/.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    here = Path(__file__).resolve().parent

    config_path = args.config or (here / "model_config.json")
    if not config_path.exists():
        print(f"ERROR: model_config.json not found at {config_path}", file=sys.stderr)
        return 1
    cfg = load_model_config(config_path)

    ckpt_path = args.checkpoint or (here / "checkpoints" / "checkpoint_best.pt")
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}", file=sys.stderr)
        print("       Place your trained .pt file there or pass --checkpoint.", file=sys.stderr)
        return 1

    try:
        inputs = _gather_inputs(args.input)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if not inputs:
        print(f"ERROR: no images found in {args.input}", file=sys.stderr)
        print("       Drop a .tif/.png mosaic into ./input/ and re-run.", file=sys.stderr)
        return 1

    device = _pick_device(args.device)
    print(f"Device: {device}")

    model, ckpt_threshold = load_model(ckpt_path, cfg, device)
    print(f"Loaded checkpoint: {ckpt_path.name}")
    if ckpt_threshold is not None:
        print(f"  checkpoint best_threshold = {ckpt_threshold:.3f}")

    if args.threshold is not None:
        threshold = float(args.threshold)
    elif ckpt_threshold is not None:
        threshold = float(ckpt_threshold)
    else:
        threshold = float(cfg.get("threshold", 0.5))
    print(f"  using threshold = {threshold:.3f}")

    patch_size = int(cfg["patch_size"])
    overlap    = int(cfg["overlap"])
    input_mode = str(cfg.get("input_mode", "engineered"))

    norm_stats = get_normalization_stats(input_mode, checkpoint_dir=ckpt_path.parent)
    if input_mode == "engineered":
        print(f"  norm stats: mean={norm_stats[0]} std={norm_stats[1]}")

    args.out.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for i, mosaic_path in enumerate(inputs, 1):
        name = mosaic_path.stem
        print(f"\n[{i}/{len(inputs)}] {mosaic_path.name}")

        # 1) Preprocess (or skip if the caller already did it)
        if args.skip_preprocess:
            from inference import load_mosaic
            preprocessed_bgr = load_mosaic(mosaic_path)
            raw_bgr = preprocessed_bgr
            # No reliable m/px when skipping the preprocess loader; try TIFF tags.
            from preprocessing.geo_resolution import extract_meters_per_pixel
            from preprocess import FALLBACK_METERS_PER_PIXEL
            mpp = extract_meters_per_pixel(mosaic_path, fallback=FALLBACK_METERS_PER_PIXEL)
            print(f"  (skip-preprocess) shape={preprocessed_bgr.shape}")
        else:
            print("  preprocessing...")
            preprocessed_bgr, raw_bgr, mpp = preprocess_mosaic(mosaic_path)
            print(f"  preprocessed shape={preprocessed_bgr.shape}")

        if args.save_preprocessed:
            cv2.imwrite(str(args.out / f"{name}_preprocessed.png"), preprocessed_bgr)

        # 2) Model inference
        prob = sliding_window_inference(
            model, preprocessed_bgr,
            patch_size=patch_size,
            overlap=overlap,
            input_mode=input_mode,
            device=device,
            norm_stats=norm_stats,
        )

        # 3) Binarize
        binary = (prob > threshold).astype(np.uint8) * 255

        # 4) Metrics — restrict to real seafloor pixels (exclude black AUV border)
        seafloor = seafloor_mask_from_raw(raw_bgr)
        metrics = compute_metrics(
            binary, meters_per_pixel=mpp, seafloor_mask=seafloor,
        )

        # 5) Outputs
        cv2.imwrite(
            str(args.out / f"{name}_raw.jpg"), raw_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        _save_overlay(args.out / f"{name}_overlay.png", raw_bgr, binary)
        _save_outline_overlay(args.out / f"{name}_outline.png", raw_bgr, binary)

        metrics_path = args.out / f"{name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {"mosaic": mosaic_path.name, "threshold": threshold, **metrics},
                f, indent=2,
            )

        print(format_metrics_report(metrics, name))
        summary_rows.append({"mosaic": mosaic_path.name, "threshold": threshold, **metrics})

    # Batch summary
    summary_path = args.out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nDone. {len(inputs)} mosaic(s) processed.")
    print(f"Outputs : {args.out.resolve()}")
    print(f"Summary : {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

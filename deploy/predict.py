"""
predict.py — End-to-end nodule segmentation on raw seafloor mosaics.
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from inference import (
    get_normalization_stats,
    load_model,
    load_mosaic,
    load_model_config,
    sliding_window_inference,
)
from geo_resolution import compute_corner_coords, extract_geo_metadata
from metrics import compute_metrics, format_metrics_report, seafloor_mask_from_raw

_IMG_EXTS = {".tif", ".tiff"}
FALLBACK_METERS_PER_PIXEL = 0.005


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
        print("       Drop a .tif/.tiff mosaic into ./input/ and re-run.", file=sys.stderr)
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

        # 1) Load raw mosaic
        raw_bgr = load_mosaic(mosaic_path)
        geo = extract_geo_metadata(mosaic_path, fallback_mpp=FALLBACK_METERS_PER_PIXEL)
        mpp = geo["meters_per_pixel"]
        H, W = raw_bgr.shape[:2]
        geo["corners"] = compute_corner_coords(geo, height_px=H, width_px=W)
        print(f"  shape={raw_bgr.shape}")

        if geo.get("latitude") is not None and geo.get("longitude") is not None:
            print(
                f"  geo (top-left): lat={geo['latitude']:.6f}, lon={geo['longitude']:.6f}, "
                f"crs={geo['crs_type']}, mpp_source={geo['mpp_source']}"
            )
            for corner_name in ("top_left", "top_right", "bottom_left", "bottom_right"):
                c = geo["corners"][corner_name]
                if c["latitude"] is not None:
                    print(f"    {corner_name:<12}: lat={c['latitude']:.6f}, lon={c['longitude']:.6f}")

        # 2) Model inference
        prob = sliding_window_inference(
            model, raw_bgr,
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

        metrics_path = args.out / f"{name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "mosaic": mosaic_path.name,
                    "threshold": threshold,
                    "geo": geo,
                    **metrics,
                },
                f, indent=2,
            )

        report = format_metrics_report(metrics, name)
        txt_path = args.out / f"{name}_metrics.txt"
        with open(txt_path, "w") as f:
            f.write(f"Mosaic   : {mosaic_path.name}\n")
            f.write(f"Threshold: {threshold:.3f}\n")
            f.write("Geo      :\n")
            f.write(f"  meters_per_pixel : {geo['meters_per_pixel']:.6f} ({geo['mpp_source']})\n")
            f.write(f"  crs_type         : {geo.get('crs_type') or 'n/a'}\n")
            f.write("  corners (lat, lon):\n")
            for corner_name in ("top_left", "top_right", "bottom_left", "bottom_right"):
                c = geo["corners"][corner_name]
                if c["latitude"] is not None:
                    f.write(f"    {corner_name:<12} : {c['latitude']:.6f}, {c['longitude']:.6f}\n")
                else:
                    f.write(f"    {corner_name:<12} : n/a\n")
            f.write(report + "\n")

        print(report)
        summary_rows.append({
            "mosaic": mosaic_path.name,
            "threshold": threshold,
            "geo": geo,
            **metrics,
        })

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

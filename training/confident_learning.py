"""
training.confident_learning — Label-Quality Auditor
Confident-Learning-inspired label audit for proxy-labelled segmentation masks 
(Inspired by Northcutt et al., "Confident Learning: Estimating Uncertainty in Dataset Labels, JAIR 2021")

Use Audit Pipeline.txt file to see how to use this. 
Essentially, allows us to rank proxy-labelled patches by model/proxy disagreement, 
then export the worst offenders to the annotation inbox for manual review and correction.
Breaks bottleneck of having to manually annotate every single patch, rather just the worst ones where the model disagrees with the proxy.
"""
from __future__ import annotations
import csv
import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .dataset import NoduleSegmentationDataset, get_val_augmentations

logger = logging.getLogger(__name__)

@dataclass
class PatchAuditScore:
    patch_id: str
    image_path: str
    mask_path: str
    # Aggregate statistics
    mean_prob: float              
    proxy_coverage: float   
    pred_coverage: float      
    # Confident-learning disagreement scores
    confident_fp_frac: float      
    confident_fn_frac: float      
    dice_disagreement: float  
    combined_score: float        
    missed_pixels: int = 0     
    sum_prob_where_y0: float = 0.0
    count_y0:          float = 0.0
    sum_prob_where_y1: float = 0.0
    count_y1:          float = 0.0

class ConfidentLabelAuditor:
    def __init__(self, model, device = "cuda", *, low_thresh = 0.15, high_thresh = 0.85, w_fp = 0.6, w_fn = 0.3, w_dd = 0.1, adaptive_thresholds = False, batch_size = 16,
        num_workers = 4, input_mode = "rgb"):
        
        self.model = model.to(device).eval()
        self.device = device
        self.low_thresh  = low_thresh
        self.high_thresh = high_thresh
        self.w_fp = w_fp
        self.w_fn = w_fn
        self.w_dd = w_dd
        self.adaptive_thresholds = adaptive_thresholds
        self.batch_size  = batch_size
        self.num_workers = num_workers
        # Must match the input_mode (we use engineered mode, and while we initalize with rgb, dont worry it got changed to engineered downstream)
        self.input_mode = input_mode

    def score(self, records, corrected_masks_dir = None, save_pred_masks_dir = None):
        audit_records = self._filter_uncorrected(records, corrected_masks_dir)
        if not audit_records:
            logger.warning("  No proxy-only patches to audit (all corrected).")
            return []

        logger.info(
            f"  Auditing {len(audit_records)} proxy-only patches "
            f"(skipped {len(records) - len(audit_records)} already corrected)"
        )

        scores = self._single_pass(audit_records, save_pred_masks_dir=save_pred_masks_dir)

        if self.adaptive_thresholds:
            low, high = self._compute_adaptive_thresholds(scores)
            logger.info(
                f"  Adaptive thresholds: low={low:.4f}, high={high:.4f} "
                f"(overriding fixed {self.low_thresh}/{self.high_thresh})"
            )
            self.low_thresh, self.high_thresh = low, high
            scores = self._single_pass(audit_records, save_pred_masks_dir=save_pred_masks_dir)

        scores.sort(key=lambda s: s.confident_fp_frac, reverse=True)
        return scores

    @torch.no_grad()
    def _single_pass(self, records, *, save_pred_masks_dir=None):
        """Forward every patch once and compute disagreement stats."""
        dataset = NoduleSegmentationDataset(
            records,
            transform=get_val_augmentations(input_mode=self.input_mode),
            corrected_masks_dir=None,  # we want the proxy masks, not corrected
            input_mode=self.input_mode,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device == "cuda"),
        )

        if save_pred_masks_dir is not None:
            save_pred_masks_dir = Path(save_pred_masks_dir)
            save_pred_masks_dir.mkdir(parents=True, exist_ok=True)

        scores: list[PatchAuditScore] = []
        idx = 0
        use_amp = (self.device == "cuda")

        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device, enabled=use_amp):
                logits = self.model(images)
            probs = torch.sigmoid(logits.float())  

            # Save predicted masks if requested
            if save_pred_masks_dir is not None:
                pred_hard = (probs.squeeze(1) > 0.5).cpu().numpy().astype(np.uint8) * 255
                for i in range(pred_hard.shape[0]):
                    rec = records[idx + i]
                    pid = rec.get("patch_id", Path(rec["image_path"]).stem)
                    cv2.imwrite(str(save_pred_masks_dir / f"{pid}.png"), pred_hard[i])

            batch_scores = self._score_batch(probs, masks, records[idx : idx + images.size(0)])
            scores.extend(batch_scores)
            idx += images.size(0)

        return scores

    def _score_batch(self, probs, targets, batch_records):
        p = probs.squeeze(1)                 
        y = targets.squeeze(1)       

        n_pixels = p.shape[-1] * p.shape[-2]

        # Confident disagreement masks
        confident_fp = (y > 0.5) & (p < self.low_thresh)  
        confident_fn = (y < 0.5) & (p > self.high_thresh) 

        # Dice (hard prediction at 0.5)
        pred_hard = (p > 0.5).float()
        y_bin = (y > 0.5).float()
        missed_pixels = (pred_hard != y_bin).float().sum(dim=(1, 2)) 
        intersection = (pred_hard * y).sum(dim=(1, 2))
        denom        = pred_hard.sum(dim=(1, 2)) + y.sum(dim=(1, 2))
        dice         = (2.0 * intersection + 1e-6) / (denom + 1e-6)
        dice_disagreement = 1.0 - dice

        mean_prob       = p.mean(dim=(1, 2))
        proxy_coverage  = y.mean(dim=(1, 2))
        pred_coverage   = pred_hard.mean(dim=(1, 2))
        cf_fp_frac      = confident_fp.float().mean(dim=(1, 2))
        cf_fn_frac      = confident_fn.float().mean(dim=(1, 2))

        min_coverage = 0.001  
        both_empty = (proxy_coverage < min_coverage) & (pred_coverage < min_coverage)
        dice_disagreement = torch.where(both_empty, torch.zeros_like(dice_disagreement), dice_disagreement)

        # Class-conditional sum/count for adaptive threshold pass
        y_bool = y > 0.5
        sum_p_y1 = (p * y_bool).sum(dim=(1, 2))
        cnt_y1   = y_bool.sum(dim=(1, 2)).float()
        sum_p_y0 = (p * (~y_bool)).sum(dim=(1, 2))
        cnt_y0   = (~y_bool).sum(dim=(1, 2)).float()

        combined = (
            self.w_fp * cf_fp_frac
            + self.w_fn * cf_fn_frac
            + self.w_dd * dice_disagreement
        )

        out: list[PatchAuditScore] = []
        for i, rec in enumerate(batch_records):
            out.append(PatchAuditScore(
                patch_id          = rec.get("patch_id", Path(rec["image_path"]).stem),
                image_path        = rec["image_path"],
                mask_path         = rec["mask_path"],
                mean_prob         = float(mean_prob[i]),
                proxy_coverage    = float(proxy_coverage[i]),
                pred_coverage     = float(pred_coverage[i]),
                confident_fp_frac = float(cf_fp_frac[i]),
                confident_fn_frac = float(cf_fn_frac[i]),
                dice_disagreement = float(dice_disagreement[i]),
                combined_score    = float(combined[i]),
                missed_pixels     = int(missed_pixels[i].item()),
                sum_prob_where_y0 = float(sum_p_y0[i]),
                count_y0          = float(cnt_y0[i]),
                sum_prob_where_y1 = float(sum_p_y1[i]),
                count_y1          = float(cnt_y1[i]),
            ))
        return out

    def _compute_adaptive_thresholds(self, scores):
  
        sum0 = sum(s.sum_prob_where_y0 for s in scores)
        cnt0 = sum(s.count_y0          for s in scores)
        sum1 = sum(s.sum_prob_where_y1 for s in scores)
        cnt1 = sum(s.count_y1          for s in scores)

        t0 = (sum0 / cnt0) if cnt0 > 0 else self.low_thresh
        t1 = (sum1 / cnt1) if cnt1 > 0 else self.high_thresh
        low  = max(1e-3, min(t0, 0.5))
        high = min(1 - 1e-3, max(t1, 0.5))
        return float(low), float(high)

    @staticmethod
    def _filter_uncorrected(
        records,
        corrected_masks_dir,
    ):
        # Drop records whose patch_id already has a corrected mask
        if corrected_masks_dir is None:
            return list(records)
        cdir = Path(corrected_masks_dir)
        if not cdir.exists():
            return list(records)
        corrected_ids = {p.stem for p in cdir.glob("*.png")}
        return [r for r in records if r.get("patch_id", "") not in corrected_ids]

def export_audit_queue(scores, output_dir, top_k = 200, save_visualizations = True):

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    queued = scores if top_k is None else scores[:top_k]

    # CSV
    csv_path = out / "audit_queue.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank", "patch_id", "missed_pixels",
                "combined_score",
                "confident_fp_frac", "confident_fn_frac",
                "dice_disagreement", "proxy_coverage",
                "pred_coverage", "mean_prob",
                "image_path", "mask_path",
            ],
        )
        writer.writeheader()
        for rank, s in enumerate(queued, start=1):
            writer.writerow({
                "rank":              rank,
                "patch_id":          s.patch_id,
                "missed_pixels":     s.missed_pixels,
                "combined_score":    round(s.combined_score, 6),
                "confident_fp_frac": round(s.confident_fp_frac, 6),
                "confident_fn_frac": round(s.confident_fn_frac, 6),
                "dice_disagreement": round(s.dice_disagreement, 6),
                "proxy_coverage":    round(s.proxy_coverage, 6),
                "pred_coverage":     round(s.pred_coverage, 6),
                "mean_prob":         round(s.mean_prob, 6),
                "image_path":        s.image_path,
                "mask_path":         s.mask_path,
            })

    # JSON mirror
    json_path = out / "audit_queue.json"
    with open(json_path, "w") as f:
        json.dump([asdict(s) for s in queued], f, indent=2)

    if save_visualizations:
        _render_audit_visualizations(queued, out / "visualizations")

    logger.info(f"  Audit queue written: {csv_path}  ({len(queued)} patches)")
    return csv_path

def export_dice_audit_queue(metrics, output_dir, top_k = 200, save_visualizations = True, pred_masks_dir = None):

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    queued = metrics if top_k is None else metrics[:top_k]

    csv_path = out / "audit_queue.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank", "patch_id", "dice", "asd", "nsd",
                "image_path", "mask_path",
            ],
        )
        writer.writeheader()
        for rank, m in enumerate(queued, start=1):
            writer.writerow({
                "rank":       rank,
                "patch_id":   m["patch_id"],
                "dice":       round(m["dice"], 6),
                "asd":        None if m.get("asd") is None else round(m["asd"], 4),
                "nsd":        None if m.get("nsd") is None else round(m["nsd"], 6),
                "image_path": m["image_path"],
                "mask_path":  m["mask_path"],
            })

    json_path = out / "audit_queue.json"
    with open(json_path, "w") as f:
        json.dump(queued, f, indent=2)

    if save_visualizations:
        _render_dice_audit_visualizations(queued, out / "visualizations", pred_masks_dir=pred_masks_dir)

    logger.info(f"  Audit queue written: {csv_path}  ({len(queued)} patches)")
    return csv_path

def _render_dice_audit_visualizations(metrics, out_dir, *, pred_masks_dir=None):
    # Two-panel viz per patch: proxy contour overlay (left) | model prediction contour overlay (right).
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(pred_masks_dir) if pred_masks_dir is not None else None

    for m in metrics:
        image = cv2.imread(m["image_path"], cv2.IMREAD_COLOR)
        mask  = cv2.imread(m["mask_path"],  cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            logger.warning(f"  skipping viz for {m['patch_id']}: missing files")
            continue

        h, w = image.shape[:2]

        # Left panel — proxy label as red contour on the raw image
        proxy_panel = image.copy()
        proxy_bin = (mask > 127).astype(np.uint8)
        proxy_contours, _ = cv2.findContours(proxy_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(proxy_panel, proxy_contours, -1, (0, 0, 255), 1)

        # Right panel — model prediction as cyan contour on the raw image
        pred_panel = image.copy()
        if pred_dir is not None:
            pred_path = pred_dir / f"{m['patch_id']}.png"
            pred_img = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE) if pred_path.exists() else None
            if pred_img is not None:
                pred_bin = (pred_img > 127).astype(np.uint8)
                pred_contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(pred_panel, pred_contours, -1, (255, 255, 0), 1)

        panel = np.concatenate([proxy_panel, pred_panel], axis=1)

        label = f"{m['patch_id']}  dice={m['dice']:.3f}  proxy=red  pred=cyan"
        cv2.putText(panel, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(str(out_dir / f"{m['patch_id']}.png"), panel)


def _render_audit_visualizations(scores, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in scores:
        image = cv2.imread(s.image_path, cv2.IMREAD_COLOR)
        mask  = cv2.imread(s.mask_path,  cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            logger.warning(f"  skipping viz for {s.patch_id}: missing files")
            continue

        mask_bin = (mask > 127).astype(np.uint8)

        # Proxy overlay (red contour on image)
        overlay = image.copy()
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)

        # Proxy mask tinted
        tinted = np.zeros_like(image)
        tinted[mask_bin > 0] = (0, 0, 255)
        blended = cv2.addWeighted(image, 0.6, tinted, 0.4, 0)

        # Side-by-side panel
        h, w = image.shape[:2]
        panel = np.zeros((h, w * 3, 3), dtype=np.uint8)
        panel[:, :w]       = image
        panel[:, w:2*w]    = overlay
        panel[:, 2*w:3*w]  = blended

        label = (
            f"{s.patch_id}  score={s.combined_score:.3f}  "
            f"fp={s.confident_fp_frac:.3f}  fn={s.confident_fn_frac:.3f}  "
            f"dice_dis={s.dice_disagreement:.3f}"
        )
        cv2.putText(panel, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(str(out_dir / f"{s.patch_id}.png"), panel)

def load_model_from_checkpoint(checkpoint_path, model_cfg, device, patch_size = None):
    from .model import build_model  # local import to avoid circular

    model = build_model(model_cfg, patch_size=patch_size)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    logger.info(
        f"  Loaded checkpoint {Path(checkpoint_path).name} "
        f"(epoch {ckpt.get('epoch', '?')}, val_dice {ckpt.get('val_dice', float('nan')):.4f})"
    )
    return model.to(device)
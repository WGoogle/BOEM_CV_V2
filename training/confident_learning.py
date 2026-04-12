"""
training.confident_learning — Label-Quality Auditor
======================================================
Confident-Learning-inspired label audit for proxy-labelled segmentation
masks (inspired by Northcutt et al., *Confident Learning: Estimating
Uncertainty in Dataset Labels*, JAIR 2021).

Purpose
-------
Your proxy labels (top-hat × LCR pipeline from Step 1) are noisy. Manual
correction is expensive. Instead of correcting patches uniformly, this
module ranks proxy-labelled patches by *where the model disagrees most*
with the proxy label, so your annotation budget goes to the patches
that will actually move the needle.

Workflow
--------
1. Train a first-pass model on the current proxy + manual labels
   (``python 2_train.py``).
2. Run ``python 5_audit_labels.py`` to score every proxy-labelled patch
   against the trained model. Patches already in
   ``CORRECTED_MASKS_DIR`` are skipped — they're treated as ground
   truth.
3. The auditor writes the top-K disagreements into
   ``ANNOTATION_INBOX`` as ``(image, proxy_overlay, model_overlay,
   disagreement_heatmap)`` quads, plus an ``audit_queue.csv`` that
   drives the manual correction queue.
4. Correct that batch with your existing annotation tool; corrected
   masks land in ``CORRECTED_MASKS_DIR``; the dataset loader
   automatically prefers them at the next training run
   (``training/dataset.py`` already does this via
   ``corrected_masks_dir``).
5. Re-train, re-audit, repeat. This is the active-learning loop
   Northcutt et al. formalise.

Scoring
-------
For each patch, with model probability map ``p`` ∈ [0,1] and proxy
mask ``y`` ∈ {0,1}::

    confident_fp_frac = mean( y == 1  AND  p < low_thresh )
        → fraction of proxy-positive pixels the model confidently calls
          background. High value ⇒ proxy FP contamination (sediment /
          divots labelled as nodules).

    confident_fn_frac = mean( y == 0  AND  p > high_thresh )
        → fraction of proxy-negative pixels the model confidently calls
          nodule. High value ⇒ proxy FN omissions (missed nodules).

    dice_disagreement = 1 − Dice( y,  p > 0.5 )
        → overall region mismatch.

    combined_score    = w_fp · confident_fp_frac
                      + w_fn · confident_fn_frac
                      + w_dd · dice_disagreement

Default weights are **FP-biased** (``w_fp=0.6, w_fn=0.3, w_dd=0.1``) to
match the precision-biased training loss — auditing effort goes to
proxy FP contamination first, because that's the failure mode the
training loss and the downstream density metric care about.

Thresholds
----------
``low_thresh`` and ``high_thresh`` default to 0.15 and 0.85 (fixed).
For a more rigorous cleanlab-style calibration, pass
``adaptive_thresholds=True``; the auditor then runs a first pass to
compute per-class mean probabilities (Northcutt's self-confidence
thresholds) and rescores. Two-pass is slower but matches the JAIR 2021
formulation more closely.
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


# ── Per-patch scoring record ────────────────────────────────────────────

@dataclass
class PatchAuditScore:
    """All per-patch audit statistics for one proxy-labelled patch."""

    patch_id: str
    image_path: str
    mask_path: str
    # Aggregate statistics
    mean_prob: float              # model's mean predicted probability over patch
    proxy_coverage: float         # fraction of pixels labelled positive in proxy
    pred_coverage: float          # fraction of pixels model predicts positive (>0.5)
    # Confident-learning disagreement scores
    confident_fp_frac: float      # proxy positive ∧ model confidently negative
    confident_fn_frac: float      # proxy negative ∧ model confidently positive
    dice_disagreement: float      # 1 − Dice(proxy, pred)
    combined_score: float         # weighted rank signal
    # For downstream class-conditional threshold refinement
    sum_prob_where_y0: float = 0.0
    count_y0:          float = 0.0
    sum_prob_where_y1: float = 0.0
    count_y1:          float = 0.0


# ── Auditor ─────────────────────────────────────────────────────────────

class ConfidentLabelAuditor:
    """Score proxy-labelled patches by model/proxy disagreement.

    Parameters
    ----------
    model : nn.Module
        Trained segmentation model returning raw logits of shape
        ``(B, 1, H, W)``. Will be set to ``eval()`` internally.
    device : str
        ``"cuda"``, ``"mps"``, or ``"cpu"``.
    low_thresh : float
        Probability below which the model is considered "confidently
        negative". Default 0.15.
    high_thresh : float
        Probability above which the model is considered "confidently
        positive". Default 0.85.
    w_fp, w_fn, w_dd : float
        Weights for confident_fp_frac, confident_fn_frac, and
        dice_disagreement in the combined rank signal. Defaults are
        FP-biased (0.6 / 0.3 / 0.1) to match the training loss.
    adaptive_thresholds : bool
        If True, run a first pass to compute per-class mean predicted
        probabilities (cleanlab-style self-confidence thresholds), then
        rescore in a second pass using those as ``low_thresh`` /
        ``high_thresh``. Defaults to False for speed.
    batch_size, num_workers : int
        DataLoader settings for the forward pass.
    """

    def __init__(
        self,
        model,
        device = "cuda",
        *,
        low_thresh = 0.15,
        high_thresh = 0.85,
        w_fp = 0.6,
        w_fn = 0.3,
        w_dd = 0.1,
        adaptive_thresholds = False,
        batch_size = 16,
        num_workers = 4,
        input_mode = "rgb",
    ):
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
        # Must match the input_mode the loaded checkpoint was trained with,
        # otherwise the model sees statistically-foreign channels and the
        # disagreement ranking is garbage.
        self.input_mode = input_mode

    # ── Main entry point ────────────────────────────────────────────

    def score(
        self,
        records,
        corrected_masks_dir = None,
    ):
        """Score all proxy-labelled patches in ``records``.

        Records whose ``patch_id`` already has a corrected mask are
        **excluded** — they're treated as ground truth.

        Returns a list of :class:`PatchAuditScore` sorted in descending
        order of ``combined_score`` (worst disagreement first).
        """
        audit_records = self._filter_uncorrected(records, corrected_masks_dir)
        if not audit_records:
            logger.warning("  No proxy-only patches to audit (all corrected).")
            return []

        logger.info(
            f"  Auditing {len(audit_records)} proxy-only patches "
            f"(skipped {len(records) - len(audit_records)} already corrected)"
        )

        scores = self._single_pass(audit_records)

        if self.adaptive_thresholds:
            low, high = self._compute_adaptive_thresholds(scores)
            logger.info(
                f"  Adaptive thresholds: low={low:.4f}, high={high:.4f} "
                f"(overriding fixed {self.low_thresh}/{self.high_thresh})"
            )
            self.low_thresh, self.high_thresh = low, high
            scores = self._single_pass(audit_records)  # rescore with adapted thresholds

        scores.sort(key=lambda s: s.combined_score, reverse=True)
        return scores

    # ── Single forward pass over the dataset ────────────────────────

    @torch.no_grad()
    def _single_pass(self, records):
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

        scores: list[PatchAuditScore] = []
        idx = 0
        use_amp = (self.device == "cuda")

        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device, enabled=use_amp):
                logits = self.model(images)
            probs = torch.sigmoid(logits.float())  # upcast out of AMP for stats

            batch_scores = self._score_batch(probs, masks, records[idx : idx + images.size(0)])
            scores.extend(batch_scores)
            idx += images.size(0)

        return scores

    def _score_batch(
        self,
        probs,
        targets,
        batch_records,
    ):
        """Compute per-patch audit scores for one batch."""
        # Shapes: probs (B,1,H,W), targets (B,1,H,W)
        p = probs.squeeze(1)                 # (B, H, W)
        y = targets.squeeze(1)               # (B, H, W)  binary float

        n_pixels = p.shape[-1] * p.shape[-2]

        # Confident disagreement masks
        confident_fp = (y > 0.5) & (p < self.low_thresh)   # proxy pos, model strong neg
        confident_fn = (y < 0.5) & (p > self.high_thresh)  # proxy neg, model strong pos

        # Dice (hard prediction at 0.5)
        pred_hard = (p > 0.5).float()
        intersection = (pred_hard * y).sum(dim=(1, 2))
        denom        = pred_hard.sum(dim=(1, 2)) + y.sum(dim=(1, 2))
        dice         = (2.0 * intersection + 1e-6) / (denom + 1e-6)
        dice_disagreement = 1.0 - dice

        # Per-patch stats
        mean_prob       = p.mean(dim=(1, 2))
        proxy_coverage  = y.mean(dim=(1, 2))
        pred_coverage   = pred_hard.mean(dim=(1, 2))
        cf_fp_frac      = confident_fp.float().mean(dim=(1, 2))
        cf_fn_frac      = confident_fn.float().mean(dim=(1, 2))

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
                sum_prob_where_y0 = float(sum_p_y0[i]),
                count_y0          = float(cnt_y0[i]),
                sum_prob_where_y1 = float(sum_p_y1[i]),
                count_y1          = float(cnt_y1[i]),
            ))
        return out

    def _compute_adaptive_thresholds(
        self,
        scores,
    ):
        """Northcutt-style self-confidence thresholds.

        t_0 = mean model-probability over all pixels labelled 0 in the proxy.
        t_1 = mean model-probability over all pixels labelled 1 in the proxy.

        These are class-conditional expected probabilities. We use them
        as the low/high confident-disagreement thresholds: a pixel with
        ``y=1`` but ``p < t_1`` is below the *expected* confidence for
        that class, i.e. the model thinks the proxy label is wrong.
        """
        sum0 = sum(s.sum_prob_where_y0 for s in scores)
        cnt0 = sum(s.count_y0          for s in scores)
        sum1 = sum(s.sum_prob_where_y1 for s in scores)
        cnt1 = sum(s.count_y1          for s in scores)

        t0 = (sum0 / cnt0) if cnt0 > 0 else self.low_thresh
        t1 = (sum1 / cnt1) if cnt1 > 0 else self.high_thresh
        # t0 is the mean prob on proxy-negative pixels → use as low_thresh
        # (model says "confidently negative" if p < t0).
        # t1 is the mean prob on proxy-positive pixels → use as high_thresh
        # (model says "confidently positive" if p > t1). For FP detection we
        # want p < t1 on proxy-positive pixels, so t1 is the upper gate.
        # Guard pathological values.
        low  = max(1e-3, min(t0, 0.5))
        high = min(1 - 1e-3, max(t1, 0.5))
        return float(low), float(high)

    # ── Already-corrected patch filter ──────────────────────────────

    @staticmethod
    def _filter_uncorrected(
        records,
        corrected_masks_dir,
    ):
        """Drop records whose patch_id already has a corrected mask."""
        if corrected_masks_dir is None:
            return list(records)
        cdir = Path(corrected_masks_dir)
        if not cdir.exists():
            return list(records)
        corrected_ids = {p.stem for p in cdir.glob("*.png")}
        return [r for r in records if r.get("patch_id", "") not in corrected_ids]


# ── Export to annotation inbox ──────────────────────────────────────────

def export_audit_queue(
    scores,
    output_dir,
    top_k = 200,
    save_visualizations = True,
):
    """Write the top-K disagreements to the annotation inbox.

    Produces:
      - ``audit_queue.csv``  — one row per queued patch, sorted by
        combined_score (worst first), with all audit stats.
      - ``audit_queue.json`` — same data in JSON for programmatic use.
      - ``visualizations/<patch_id>/`` (if ``save_visualizations``) —
        image, proxy overlay, disagreement heatmap, and a side-by-side
        panel for the annotator to eyeball.

    Returns the CSV path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    queued = scores if top_k is None else scores[:top_k]

    # CSV
    csv_path = out / "audit_queue.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank", "patch_id", "combined_score",
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


def _render_audit_visualizations(
    scores,
    out_dir,
):
    """Save a side-by-side (image | proxy overlay | disagreement) panel
    for each queued patch. This is the annotator's at-a-glance view.

    Note: this renders from saved proxy masks only — no model
    probabilities are re-run. The disagreement heatmap shows the
    proxy-only outline, and the annotator refers to the CSV to see
    whether the failure mode is confident_fp (proxy over-labels) or
    confident_fn (proxy misses).
    """
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

        # Label text
        label = (
            f"{s.patch_id}  score={s.combined_score:.3f}  "
            f"fp={s.confident_fp_frac:.3f}  fn={s.confident_fn_frac:.3f}  "
            f"dice_dis={s.dice_disagreement:.3f}"
        )
        cv2.putText(panel, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(str(out_dir / f"{s.patch_id}.png"), panel)


# ── Convenience loader for checkpoints ──────────────────────────────────

def load_model_from_checkpoint(
    checkpoint_path,
    model_cfg,
    device,
    patch_size = None,
):
    """Rebuild a model from a config preset and load its weights.

    Separate from ``training.model.build_model`` so callers can use the
    auditor without importing config directly.
    """
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

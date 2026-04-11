"""
training.trainer — Training Loop Engine
========================================
Self-contained training loop with:
  - Mixed-precision training (torch.amp GradScaler) for GPU memory efficiency
  - Gradient clipping (unscale + clip_grad_norm_ 1.0) for stability with the
    strong region-level loss term
  - OneCycleLR scheduler (Smith 2018) stepped per-batch
  - EMA weights (Polyak averaging, decay 0.9999) used for validation + best
    checkpoint. ~1 Dice point for free.
  - Validation-time threshold sweep — probe 0.1..0.9 in 0.05 steps and pick
    the Dice-maximising threshold. Stored in the checkpoint alongside weights.
  - Test-time augmentation (TTA) at validation: 4-way dihedral flips
    (identity, hflip, vflip, rot180) averaged.
  - **Micro-averaged** IoU / Dice: intersection / sum_pred / sum_target are
    accumulated across the full epoch and reduced once. Zero-foreground
    batches no longer inflate the metric via macro-averaging.
  - Early stopping on validation Dice at the best threshold
  - Checkpoint saving (best + last) with full state for resume

The Trainer owns the optimisation loop.  All logging, manifest updates,
and progress reporting happen in 2_train.py so this module stays
focused on the numerical work.
"""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ── Metric primitives (micro-accumulation) ──────────────────────────────

_SMOOTH = 1e-6


def _binary_stats(
    preds: torch.Tensor, targets: torch.Tensor
) -> tuple[float, float, float]:
    """Return (intersection, sum_pred, sum_target) summed over the whole batch.

    Micro-accumulation: callers add these across every batch in an epoch and
    compute a single Dice / IoU at the end. A batch with zero foreground
    contributes zero intersection and zero sum_target — it no longer produces
    an artificial Dice ≈ 1 that would inflate a macro-average.
    """
    intersection = (preds * targets).sum().item()
    sum_pred     = preds.sum().item()
    sum_target   = targets.sum().item()
    return intersection, sum_pred, sum_target


def _dice_from_stats(inter: float, sp: float, st: float) -> float:
    return (2.0 * inter + _SMOOTH) / (sp + st + _SMOOTH)


def _iou_from_stats(inter: float, sp: float, st: float) -> float:
    return (inter + _SMOOTH) / (sp + st - inter + _SMOOTH)


# ── Epoch history record ─────────────────────────────────────────────────

@dataclass
class EpochLog:
    """Metrics for a single training epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    train_iou: float
    train_dice: float
    val_iou: float
    val_dice: float
    lr: float
    best_threshold: float = 0.5


@dataclass
class TrainingResult:
    """Final outcome returned by Trainer.fit()."""
    best_epoch: int
    best_val_dice: float
    best_threshold: float
    best_checkpoint_path: str
    last_checkpoint_path: str
    epochs_run: int
    history: list[EpochLog] = field(default_factory=list)


# ── Trainer ──────────────────────────────────────────────────────────────

class Trainer:
    """Manages the full training loop for a segmentation model.

    Parameters
    ----------
    model : nn.Module
        Segmentation model (returns raw logits).
    criterion : nn.Module
        Loss function operating on logits.
    train_cfg : dict
        From config.TRAINING.  Keys: learning_rate, weight_decay,
        num_epochs, early_stopping_patience, ema_decay (optional),
        onecycle_pct_start (optional).
    checkpoint_dir : Path
        Where to save best/last checkpoints.
    device : str
        "cuda", "mps", or "cpu".
    steps_per_epoch : int
        Number of training batches per epoch.  Required for OneCycleLR.
    """

    # Sweep thresholds 0.1..0.9 in 0.05 steps (17 values)
    _THRESHOLD_SWEEP = np.round(np.arange(0.10, 0.901, 0.05), 4)

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_cfg: dict,
        checkpoint_dir: Path,
        device: str = "cuda",
        steps_per_epoch: int | None = None,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.num_epochs = train_cfg["num_epochs"]
        self.patience   = train_cfg.get("early_stopping_patience", 15)
        self.max_lr     = train_cfg["learning_rate"]

        # Optimiser
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.max_lr,
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )

        # OneCycleLR — stepped per-batch, requires total_steps = epochs * batches
        if steps_per_epoch is None:
            raise ValueError(
                "Trainer requires `steps_per_epoch` for OneCycleLR. "
                "Pass steps_per_epoch=len(train_loader) from 2_train.py."
            )
        self.steps_per_epoch = steps_per_epoch
        total_steps = self.num_epochs * steps_per_epoch
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.max_lr,
            total_steps=total_steps,
            pct_start=train_cfg.get("onecycle_pct_start", 0.1),
            anneal_strategy="cos",
            div_factor=train_cfg.get("onecycle_div_factor", 25.0),
            final_div_factor=train_cfg.get("onecycle_final_div_factor", 1e4),
        )

        # Mixed-precision — only on CUDA (MPS/CPU autocast is brittle / slow here)
        self.use_amp = (device == "cuda")
        self.scaler  = GradScaler("cuda", enabled=self.use_amp)

        # EMA weights via AveragedModel with Polyak decay
        ema_decay = train_cfg.get("ema_decay", 0.9999)
        self.ema_decay = ema_decay

        def _ema_avg(avg_param, cur_param, num_avg):
            # Standard exponential moving average with fixed decay
            return ema_decay * avg_param + (1.0 - ema_decay) * cur_param

        # use_buffers=True so BN/LN buffers track the EMA as well. SegFormer
        # and ConvNeXt use LayerNorm so buffers are small — negligible cost.
        self.ema_model = AveragedModel(
            self.model, avg_fn=_ema_avg, use_buffers=True,
        ).to(device)

        # Best-threshold state, updated every val epoch and persisted
        self.best_threshold = 0.5

    # ── AMP context helper ──────────────────────────────────────────

    def _amp_ctx(self):
        """Return an autocast context on CUDA; nullcontext elsewhere."""
        if self.use_amp:
            return torch.autocast(device_type="cuda")
        return contextlib.nullcontext()

    # ── Single epoch ─────────────────────────────────────────────────

    def _train_one_epoch(self, loader: DataLoader) -> tuple[float, float, float]:
        """Run one training epoch.  Returns (loss, iou, dice).

        Dice / IoU are micro-averaged across the whole epoch.
        """
        self.model.train()
        running_loss = 0.0
        n_batches = 0
        tot_inter = tot_sp = tot_st = 0.0

        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with self._amp_ctx():
                logits = self.model(images)
                loss   = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()

            # Gradient clipping — unscale first, then clip norm at 1.0
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # OneCycleLR steps per-batch
            self.scheduler.step()

            # EMA update after each optimiser step
            self.ema_model.update_parameters(self.model)

            # Micro-stats for train metrics (threshold 0.5 is fine here —
            # threshold sweep is only applied at validation)
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                inter, sp, st = _binary_stats(preds, masks)
                tot_inter += inter
                tot_sp    += sp
                tot_st    += st

            running_loss += loss.item()
            n_batches += 1

        return (
            running_loss / max(n_batches, 1),
            _iou_from_stats(tot_inter, tot_sp, tot_st),
            _dice_from_stats(tot_inter, tot_sp, tot_st),
        )

    # ── TTA forward ──────────────────────────────────────────────────

    def _tta_probs(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        """Average sigmoid probabilities over 4-way dihedral flips.

        Identity + hflip + vflip + rot180. All four transforms are their own
        inverse (each is an involution), so no extra bookkeeping is needed —
        we just un-flip the prediction with the same operation.
        """
        with self._amp_ctx():
            p  = torch.sigmoid(model(images))
            # hflip
            xh = torch.flip(images, dims=[-1])
            p += torch.flip(torch.sigmoid(model(xh)), dims=[-1])
            # vflip
            xv = torch.flip(images, dims=[-2])
            p += torch.flip(torch.sigmoid(model(xv)), dims=[-2])
            # rot180 = hflip ∘ vflip
            xr = torch.flip(images, dims=[-1, -2])
            p += torch.flip(torch.sigmoid(model(xr)), dims=[-1, -2])
        return p / 4.0

    # ── Validation with TTA + threshold sweep ───────────────────────

    @torch.no_grad()
    def _validate(
        self, loader: DataLoader, *, sweep_thresholds: bool = True,
    ) -> tuple[float, float, float]:
        """Run validation with TTA (EMA weights) + optional threshold sweep.

        Returns (loss, iou, dice) at the selected threshold. When
        ``sweep_thresholds`` is True, scans 0.1..0.9 in 0.05 steps and picks
        the threshold that maximises micro-Dice. The winning threshold is
        stored in ``self.best_threshold`` and persisted in the next checkpoint.
        Otherwise uses the current ``self.best_threshold`` (for test eval).
        """
        self.ema_model.eval()
        self.model.eval()

        # Loss is computed on the identity forward of the EMA model. TTA
        # averaging is only used for metric probabilities.
        running_loss = 0.0
        n_batches = 0

        n_thr = len(self._THRESHOLD_SWEEP)
        tot_inter = np.zeros(n_thr, dtype=np.float64)
        tot_sp    = np.zeros(n_thr, dtype=np.float64)
        tot_st    = 0.0

        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device, non_blocking=True)

            # Loss from a single EMA forward (cheap, matches training signal)
            with self._amp_ctx():
                logits = self.ema_model(images)
                loss   = self.criterion(logits, masks)
            running_loss += loss.item()
            n_batches += 1

            # TTA-averaged probabilities for metric computation
            probs = self._tta_probs(self.ema_model, images)

            st_batch = masks.sum().item()
            tot_st  += st_batch

            for i, t in enumerate(self._THRESHOLD_SWEEP):
                preds = (probs > float(t)).float()
                inter = (preds * masks).sum().item()
                sp    = preds.sum().item()
                tot_inter[i] += inter
                tot_sp[i]    += sp

        # Compute Dice per threshold, pick best
        dices = np.array([
            _dice_from_stats(tot_inter[i], tot_sp[i], tot_st) for i in range(n_thr)
        ])

        if sweep_thresholds:
            best_i = int(np.argmax(dices))
            self.best_threshold = float(self._THRESHOLD_SWEEP[best_i])
        else:
            # Find nearest sweep index to current best_threshold
            best_i = int(np.argmin(np.abs(self._THRESHOLD_SWEEP - self.best_threshold)))

        val_dice = float(dices[best_i])
        val_iou  = _iou_from_stats(tot_inter[best_i], tot_sp[best_i], tot_st)
        val_loss = running_loss / max(n_batches, 1)
        return val_loss, val_iou, val_dice

    # ── Checkpointing ────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_dice: float, tag: str) -> Path:
        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     self.model.state_dict(),
            "ema_state_dict":       self.ema_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict":    self.scaler.state_dict(),
            "val_dice":             val_dice,
            "best_threshold":       self.best_threshold,
        }, path)
        return path

    def load_checkpoint(self, path: Path) -> int:
        """Resume from a checkpoint.  Returns the epoch to start from."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "ema_state_dict" in ckpt:
            self.ema_model.load_state_dict(ckpt["ema_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "best_threshold" in ckpt:
            self.best_threshold = float(ckpt["best_threshold"])
        logger.info(
            f"  Resumed from {path.name}  (epoch {ckpt['epoch']}, "
            f"val_dice {ckpt['val_dice']:.4f}, "
            f"thr {self.best_threshold:.2f})"
        )
        return ckpt["epoch"] + 1

    # ── Main training loop ───────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        start_epoch: int = 0,
        epoch_callback=None,
    ) -> TrainingResult:
        """Train the model for up to ``num_epochs`` with early stopping.

        Parameters
        ----------
        train_loader, val_loader : DataLoader
            Training and validation data loaders.
        start_epoch : int
            Epoch index to resume from (0-based).
        epoch_callback : callable | None
            Called after each epoch with (epoch_log: EpochLog).
            Used by 2_train.py for logging / progress bars.
        """
        best_dice  = 0.0
        best_epoch = 0
        no_improve = 0
        history: list[EpochLog] = []

        best_ckpt_path = ""
        last_ckpt_path = ""

        for epoch in range(start_epoch, self.num_epochs):
            # Train (scheduler steps per-batch inside)
            train_loss, train_iou, train_dice = self._train_one_epoch(train_loader)

            # Validate on EMA model with TTA + threshold sweep
            val_loss, val_iou, val_dice = self._validate(
                val_loader, sweep_thresholds=True,
            )

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Build log entry
            log = EpochLog(
                epoch=epoch,
                train_loss=train_loss, val_loss=val_loss,
                train_iou=train_iou, train_dice=train_dice,
                val_iou=val_iou, val_dice=val_dice,
                lr=current_lr,
                best_threshold=self.best_threshold,
            )
            history.append(log)

            # Checkpoint: always save "last"
            last_ckpt_path = str(self._save_checkpoint(epoch, val_dice, "last"))

            # Checkpoint: save "best" if improved
            if val_dice > best_dice:
                best_dice  = val_dice
                best_epoch = epoch
                no_improve = 0
                best_ckpt_path = str(self._save_checkpoint(epoch, val_dice, "best"))
            else:
                no_improve += 1

            # Callback for external logging
            if epoch_callback:
                epoch_callback(log)

            # Early stopping
            if no_improve >= self.patience:
                logger.info(
                    f"  Early stopping at epoch {epoch} "
                    f"(no improvement for {self.patience} epochs)"
                )
                break

        return TrainingResult(
            best_epoch=best_epoch,
            best_val_dice=best_dice,
            best_threshold=self.best_threshold,
            best_checkpoint_path=best_ckpt_path,
            last_checkpoint_path=last_ckpt_path,
            epochs_run=len(history),
            history=history,
        )

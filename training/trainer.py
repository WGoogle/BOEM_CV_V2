"""
The Trainer owns the optimisation loop. 
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
import torch
from torch.amp import GradScaler

logger = logging.getLogger(__name__)

# Metrics 
def _compute_metrics(logits, targets, threshold):
    # Compute IoU and Dice from raw logits + binary targets
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        preds_flat   = preds.view(preds.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (preds_flat * targets_flat).sum(dim=1)
        union        = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
        sum_preds    = preds_flat.sum(dim=1)
        sum_targets  = targets_flat.sum(dim=1)

        smooth = 1e-6
        iou  = (intersection + smooth) / (sum_preds + sum_targets - intersection + smooth)
        dice = (2 * intersection + smooth) / (union + smooth)

    return {"iou": iou.mean().item(), "dice": dice.mean().item()}

@dataclass
class EpochLog:
    # Metrics for a single training epoch
    epoch: int
    train_loss: float
    val_loss: float
    train_iou: float
    train_dice: float
    val_iou: float
    val_dice: float
    lr: float

@dataclass
class TrainingResult:
    # Final outcome returned by Trainer.fit().
    best_epoch: int
    best_val_dice: float
    best_checkpoint_path: str
    last_checkpoint_path: str
    epochs_run: int
    history: list[EpochLog] = field(default_factory=list)


# Trainer 
class Trainer:
    # Manages the full training loop for a segmentation model.

    def __init__(self, model, criterion, train_cfg, checkpoint_dir, device, **_kw):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimiser
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )

        # Scheduler: reduce LR when val Dice plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",                                
            factor=train_cfg.get("scheduler_factor", 0.5),
            patience=train_cfg.get("scheduler_patience", 7),
            min_lr=1e-7,
        )

        self.num_epochs = train_cfg["num_epochs"]
        self.patience   = train_cfg.get("early_stopping_patience", 15)

        # Mixed-precision
        self.use_amp = (device == "cuda")
        self.scaler  = GradScaler("cuda", enabled=self.use_amp)

    def _train_one_epoch(self, loader):
        """Run one training epoch.  Returns (loss, iou, dice)."""
        self.model.train()
        running_loss = 0.0
        running_iou  = 0.0
        running_dice = 0.0
        n_batches = 0

        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device, enabled=self.use_amp):
                logits = self.model(images)
                loss   = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            metrics = _compute_metrics(logits, masks, 0.5)
            running_loss += loss.item()
            running_iou  += metrics["iou"]
            running_dice += metrics["dice"]
            n_batches += 1

        return (
            running_loss / n_batches,
            running_iou  / n_batches,
            running_dice / n_batches,
        )

    @torch.no_grad()
    def _validate(self, loader):
        # Run validation.  Returns (loss, iou, dice)
        self.model.eval()
        running_loss = 0.0
        running_iou  = 0.0
        running_dice = 0.0
        n_batches = 0

        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device, enabled=self.use_amp):
                logits = self.model(images)
                loss   = self.criterion(logits, masks)

            metrics = _compute_metrics(logits, masks, 0.5)
            running_loss += loss.item()
            running_iou  += metrics["iou"]
            running_dice += metrics["dice"]
            n_batches += 1

        return (
            running_loss / n_batches,
            running_iou  / n_batches,
            running_dice / n_batches,
        )

    # Threshold optimisation 
    @torch.no_grad()
    def find_best_threshold(self, loader, low, high, steps):
        # Sweep thresholds on a loader and return (best_threshold, best_dice).
        self.model.eval()
        all_probs = []
        all_targets = []

        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device, enabled=self.use_amp):
                logits = self.model(images)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_targets.append(masks.cpu())

        probs = torch.cat(all_probs, dim=0)     
        targets = torch.cat(all_targets, dim=0)  
        best_thr = 0.5
        best_dice = 0.0
        smooth = 1e-6

        for thr in torch.linspace(low, high, steps):
            preds = (probs > thr.item()).float()
            p = preds.view(preds.size(0), -1)
            t = targets.view(targets.size(0), -1)
            intersection = (p * t).sum(dim=1)
            union = p.sum(dim=1) + t.sum(dim=1)
            dice = ((2 * intersection + smooth) / (union + smooth)).mean().item()
            if dice > best_dice:
                best_dice = dice
                best_thr = round(thr.item(), 3)

        return best_thr, best_dice

    # Checkpointing 
    def _save_checkpoint(self, epoch, val_dice, tag, best_threshold):
        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        data = {
            "epoch":            epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict":    self.scaler.state_dict(),
            "val_dice":         val_dice,
        }
        if best_threshold is not None:
            data["best_threshold"] = best_threshold
        torch.save(data, path)
        return path

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        logger.info(
            f"  Resumed from {path.name}  (epoch {ckpt['epoch']}, "
            f"val_dice {ckpt['val_dice']:.4f})"
        )
        return ckpt["epoch"] + 1

    # Main training loop
    def fit(self, train_loader, val_loader, *, start_epoch, epoch_callback):
        # Train the model for up to ``num_epochs`` with early stopping.

        best_dice  = 0.0
        best_epoch = 0
        no_improve = 0
        history: list[EpochLog] = []
        best_ckpt_path = ""
        last_ckpt_path = ""

        for epoch in range(start_epoch, self.num_epochs):
            # Train
            train_loss, train_iou, train_dice = self._train_one_epoch(train_loader)

            # Validate
            val_loss, val_iou, val_dice = self._validate(val_loader)

            # Scheduler step (tracks val Dice)
            self.scheduler.step(val_dice)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Build log entry
            log = EpochLog(
                epoch=epoch,
                train_loss=train_loss, val_loss=val_loss,
                train_iou=train_iou, train_dice=train_dice,
                val_iou=val_iou, val_dice=val_dice,
                lr=current_lr,
            )
            history.append(log)

            # Checkpoint: always save "last"
            last_ckpt_path = str(self._save_checkpoint(epoch, val_dice, "last", None))

            # Checkpoint: save "best" if improved
            if val_dice > best_dice:
                best_dice  = val_dice
                best_epoch = epoch
                no_improve = 0
                best_ckpt_path = str(self._save_checkpoint(epoch, val_dice, "best", None))
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
            best_checkpoint_path=best_ckpt_path,
            last_checkpoint_path=last_ckpt_path,
            epochs_run=len(history),
            history=history,
        )

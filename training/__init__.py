"""
training — Polymetallic Nodule Training Module
================================================
Public API re-exported here so callers can do::

    from training import (
        NoduleSegmentationDataset, build_model, CombinedLoss,
        split_dataset, Trainer,
    )
"""

from .dataset import NoduleSegmentationDataset, get_train_augmentations, get_val_augmentations
from .model import build_model, CombinedLoss, DiceLoss, FocalTverskyLoss
from .splits import split_dataset, save_split_info
from .trainer import Trainer, TrainingResult, EpochLog
from .confident_learning import (
    ConfidentLabelAuditor,
    PatchAuditScore,
    export_audit_queue,
    load_model_from_checkpoint,
)

__all__ = [
    "NoduleSegmentationDataset",
    "get_train_augmentations",
    "get_val_augmentations",
    "build_model",
    "CombinedLoss",
    "DiceLoss",
    "FocalTverskyLoss",
    "split_dataset",
    "save_split_info",
    "Trainer",
    "TrainingResult",
    "EpochLog",
    "ConfidentLabelAuditor",
    "PatchAuditScore",
    "export_audit_queue",
    "load_model_from_checkpoint",
]

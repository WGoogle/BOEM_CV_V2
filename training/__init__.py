"""
training — Polymetallic Nodule Training
"""
from .dataset import (
    NoduleSegmentationDataset,
    CopyPasteAugmentation,
    get_train_augmentations,
    get_val_augmentations,
    get_normalization_stats,
)
from .model import build_model, CombinedLoss
from .splits import split_dataset, save_split_info
from .trainer import Trainer, TrainingResult, EpochLog
from .confident_learning import (
    ConfidentLabelAuditor,
    PatchAuditScore,
    export_audit_queue,
    export_dice_audit_queue,
    load_model_from_checkpoint,
)

__all__ = [
    "NoduleSegmentationDataset",
    "CopyPasteAugmentation",
    "get_train_augmentations",
    "get_val_augmentations",
    "get_normalization_stats",
    "build_model",
    "CombinedLoss",
    "split_dataset",
    "save_split_info",
    "Trainer",
    "TrainingResult",
    "EpochLog",
    "ConfidentLabelAuditor",
    "PatchAuditScore",
    "export_audit_queue",
    "export_dice_audit_queue",
    "load_model_from_checkpoint",
]

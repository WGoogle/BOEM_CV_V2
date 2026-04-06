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
from .model import build_model, CombinedLoss, DiceLoss
from .splits import split_dataset, save_split_info
from .trainer import Trainer, TrainingResult, EpochLog

__all__ = [
    "NoduleSegmentationDataset",
    "get_train_augmentations",
    "get_val_augmentations",
    "build_model",
    "CombinedLoss",
    "DiceLoss",
    "split_dataset",
    "save_split_info",
    "Trainer",
    "TrainingResult",
    "EpochLog",
]

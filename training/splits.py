"""
training.splits — Stratified Dataset Splitting
================================================
Splits patch records into train / val / test sets, stratified by
nodule density so each split has a representative mix of sparse
and dense patches.

CoralNet-inspired design:
  - Image-level splitting is not needed here because patches are
    already spatially independent tiles.
  - Stratification bins by nodule coverage % so the model sees
    balanced difficulty in every split.
  - Split indices are saved to JSON for exact reproducibility.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def _coverage_bin(record: dict, n_bins: int = 5) -> int:
    """Assign a patch to a density stratum (0 = empty, n_bins-1 = dense).

    Uses ``label_stats.coverage_pct`` written by Step 1.  Falls back to
    nodule count if coverage is unavailable.
    """
    stats = record.get("label_stats", {})
    coverage = stats.get("coverage_pct", 0.0)

    # Bin edges: [0, 1, 3, 8, 20, 100] — empirically chosen for BOEM data
    edges = [0, 1, 3, 8, 20, 100]
    for i, upper in enumerate(edges[1:]):
        if coverage <= upper:
            return min(i, n_bins - 1)
    return n_bins - 1


def split_dataset(
    records: list[dict],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split patch records into train / val / test with stratification.

    Parameters
    ----------
    records : list[dict]
        Patch records (from patch_manifest.json).
    train_frac, val_frac, test_frac : float
        Must sum to 1.0.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_records, val_records, test_records : list[dict]
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, (
        f"Splits must sum to 1.0, got {train_frac + val_frac + test_frac}"
    )

    n = len(records)
    if n < 10:
        raise ValueError(f"Need at least 10 patches to split, got {n}")

    # Stratification labels
    strata = np.array([_coverage_bin(r) for r in records])

    # Merge rare strata (fewer than 2 members) into the nearest populated bin
    # to satisfy sklearn's stratification requirement.
    unique, counts = np.unique(strata, return_counts=True)
    rare = set(unique[counts < 2])
    if rare:
        safe = unique[counts >= 2]
        for i in range(len(strata)):
            if strata[i] in rare:
                strata[i] = safe[np.argmin(np.abs(safe - strata[i]))]

    indices = np.arange(n)

    # First split: train vs (val + test)
    holdout_frac = val_frac + test_frac
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=holdout_frac,
        random_state=seed,
        stratify=strata,
    )

    # Second split: val vs test (within holdout)
    holdout_strata = strata[holdout_idx]
    # Merge rare strata again for the smaller holdout set
    unique_h, counts_h = np.unique(holdout_strata, return_counts=True)
    rare_h = set(unique_h[counts_h < 2])
    if rare_h:
        safe_h = unique_h[counts_h >= 2]
        if len(safe_h) == 0:
            safe_h = unique_h  # fallback: no stratification possible
        for i in range(len(holdout_strata)):
            if holdout_strata[i] in rare_h:
                holdout_strata[i] = safe_h[np.argmin(np.abs(safe_h - holdout_strata[i]))]

    relative_test = test_frac / holdout_frac
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=relative_test,
        random_state=seed,
        stratify=holdout_strata,
    )

    train_records = [records[i] for i in train_idx]
    val_records   = [records[i] for i in val_idx]
    test_records  = [records[i] for i in test_idx]

    logger.info(
        f"  Split: {len(train_records)} train / "
        f"{len(val_records)} val / {len(test_records)} test"
    )

    return train_records, val_records, test_records


def compute_sampler_weights(
    records: list[dict],
    dense_multiplier: float = 5.0,
) -> np.ndarray:
    """Per-record weights for a :class:`torch.utils.data.WeightedRandomSampler`.

    Addresses the train-loop-level class imbalance that the stratified
    split does not solve: even after splitting, the vast majority of
    train patches have zero or near-zero nodule coverage, so uniform
    shuffling buries the dense-bin examples the network actually needs
    to learn the foreground signal.

    Strategy: assign each record a weight that grows linearly from 1.0
    (empty patches, bin 0) to ``dense_multiplier`` (dense patches,
    top bin). With ``dense_multiplier=5.0`` a dense patch is drawn ~5×
    more often per epoch than an empty one. The sampler then draws
    ``len(records)`` examples per epoch with replacement, so the
    effective epoch still sees roughly the same number of steps but
    with a foreground-enriched mix.

    Parameters
    ----------
    records : list[dict]
        Training patch records (passed straight from ``split_dataset``).
    dense_multiplier : float
        Weight assigned to the densest coverage bin. Values in the
        range 3.0–5.0 are the sweet spot per MODEL_IMPROVEMENTS.md §3;
        higher values over-fit the dense patches.

    Returns
    -------
    np.ndarray
        float64 weight per record, shape ``(len(records),)``.
    """
    if dense_multiplier < 1.0:
        raise ValueError(
            f"dense_multiplier must be ≥ 1.0 (got {dense_multiplier})"
        )

    n_bins = 5
    bins = np.array([_coverage_bin(r, n_bins=n_bins) for r in records])
    # Linear ramp from 1.0 (bin 0) to dense_multiplier (top bin).
    weights = 1.0 + (bins / (n_bins - 1)) * (dense_multiplier - 1.0)
    return weights.astype(np.float64)


def save_split_info(
    train_records: list[dict],
    val_records: list[dict],
    test_records: list[dict],
    output_path: Path,
    seed: int,
) -> None:
    """Persist split composition to JSON for reproducibility."""
    info = {
        "seed": seed,
        "counts": {
            "train": len(train_records),
            "val":   len(val_records),
            "test":  len(test_records),
        },
        "train_ids": [r["patch_id"] for r in train_records],
        "val_ids":   [r["patch_id"] for r in val_records],
        "test_ids":  [r["patch_id"] for r in test_records],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"  Split info saved → {output_path}")

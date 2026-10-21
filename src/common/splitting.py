"""Reproducible stratified train/validation/test split helpers."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_split_codes(
    labels: np.ndarray,
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> np.ndarray:
    """Return an int8 split code per source row: 0=train, 1=validation, 2=test.

    The validation fraction is applied to the train+validation remainder so the
    requested ratios describe the complete dataset.
    """
    ratios = np.asarray([train_ratio, validation_ratio, test_ratio], dtype=float)
    if np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Split ratios must be positive and sum to 1.")

    labels = np.asarray(labels)
    row_ids = np.arange(labels.size, dtype=np.int64)
    train_validation, test = train_test_split(
        row_ids,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels,
    )
    relative_validation_ratio = validation_ratio / (train_ratio + validation_ratio)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation_ratio,
        random_state=seed,
        stratify=labels[train_validation],
    )

    codes = np.empty(labels.size, dtype=np.int8)
    codes[train] = 0
    codes[validation] = 1
    codes[test] = 2
    return codes

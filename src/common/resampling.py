"""Training-only class imbalance strategies for the Stage 1 data matrices."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.utils import check_random_state

from src.common.config import DATASET_CONFIG, SEED


def resample_training_data(X, y, *, dataset: str, strategy: str, seed: int = SEED):
    """Apply a named strategy to training data and return X, y, count metadata.

    Validation and test data must never be passed to this function. ``class_weight``
    returns the unchanged matrix; callers should use balanced class weights when
    constructing the estimator.
    """
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset}")
    if strategy not in DATASET_CONFIG[dataset]["resampling_strategies"]:
        raise ValueError(f"Strategy {strategy!r} is not enabled for {dataset}.")
    labels = np.asarray(y).reshape(-1).astype(np.int8, copy=False)
    if X.shape[0] != labels.size:
        raise ValueError("X and y row counts do not match.")
    if not np.isin(labels, (0, 1)).all() or np.unique(labels).size != 2:
        raise ValueError("Training data must contain both binary classes.")

    before = _counts(labels)
    if strategy == "class_weight":
        return X, labels, {"before": before, "after": before, "method": strategy}

    # Keep all fraud rows and randomly cap legitimate rows. Sampling occurs only
    # on the already-created train split, before synthetic positives are added.
    rng = check_random_state(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    cap = int(DATASET_CONFIG[dataset]["cap_majority_per_minority"] * positive.size)
    if negative.size > cap:
        negative = rng.choice(negative, size=cap, replace=False)
    keep = np.concatenate((positive, negative))
    rng.shuffle(keep)
    X_capped = X[keep]
    y_capped = labels[keep]

    if strategy == "cap_smote_enn":
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import EditedNearestNeighbours

        # SMOTEENN resamples from the original capped training slice so its
        # internal SMOTE and ENN steps form one coherent strategy.
        sampler = SMOTEENN(
            sampling_strategy=0.5,
            random_state=seed,
            smote=SMOTE(sampling_strategy=0.5, random_state=seed),
            enn=EditedNearestNeighbours(n_neighbors=3, n_jobs=-1),
            n_jobs=-1,
        )
        X_resampled, y_resampled = sampler.fit_resample(X_capped, y_capped)
    else:
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE(sampling_strategy=0.5, random_state=seed)
        X_resampled, y_resampled = sampler.fit_resample(X_capped, y_capped)
    if sparse.issparse(X_resampled):
        X_resampled = X_resampled.tocsr().astype(np.float32, copy=False)
    y_resampled = np.asarray(y_resampled, dtype=np.int8)
    return X_resampled, y_resampled, {
        "before": before,
        "after_cap": _counts(y_capped),
        "after": _counts(y_resampled),
        "cap_majority_per_minority": int(DATASET_CONFIG[dataset]["cap_majority_per_minority"]),
        "method": strategy,
        "seed": int(seed),
    }


def _counts(y: np.ndarray) -> dict[str, int]:
    return {"negative": int(np.count_nonzero(y == 0)), "positive": int(np.count_nonzero(y == 1))}

"""Fraud-classification metrics and validation-only threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict[str, float]:
    """Compute threshold and ranking metrics from binary labels and scores."""
    y_true = np.asarray(y_true).reshape(-1)
    y_proba = np.asarray(y_proba, dtype=np.float64).reshape(-1)
    if y_true.shape != y_proba.shape:
        raise ValueError("y_true and y_proba must have the same number of rows.")
    if y_true.size == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    if not np.isfinite(y_proba).all():
        raise ValueError("Predicted probabilities contain non-finite values.")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must contain only binary labels 0 and 1.")

    y_pred = (y_proba >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def pick_threshold(y_true, y_proba, *, fpr_ceiling: float = 0.01) -> float:
    """Choose a validation threshold maximizing recall under the FPR ceiling.

    If no threshold meets the ceiling, return the next representable float
    above the maximum score, which predicts no positives. Never call this on test.
    """
    if not 0.0 <= fpr_ceiling <= 1.0:
        raise ValueError("fpr_ceiling must be between 0 and 1.")
    y_true = np.asarray(y_true).reshape(-1)
    scores = np.asarray(y_proba, dtype=np.float64).reshape(-1)
    if y_true.shape != scores.shape or y_true.size == 0:
        raise ValueError("y_true and y_proba must be nonempty arrays of equal length.")
    if not np.isfinite(scores).all() or not np.isin(y_true, (0, 1)).all():
        raise ValueError("Scores must be finite and labels must be binary.")

    positives = int(np.count_nonzero(y_true == 1))
    negatives = int(np.count_nonzero(y_true == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("Threshold selection requires both classes in validation.")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = y_true[order]
    true_positives = np.cumsum(sorted_labels == 1)
    false_positives = np.cumsum(sorted_labels == 0)
    # Consider only the last score in each tie group so predictions use >=.
    last_in_tie = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    candidate_idx = np.flatnonzero(last_in_tie)
    eligible = candidate_idx[
        false_positives[candidate_idx] / negatives <= fpr_ceiling
    ]
    if eligible.size == 0:
        return float(np.nextafter(scores.max(), np.inf))
    best_idx = eligible[np.argmax(true_positives[eligible] / positives)]
    return float(sorted_scores[best_idx])

import numpy as np
import pytest

from src.common.evaluate import compute_metrics, pick_threshold
from src.common.pipeline_utils import stratified_split


def test_compute_metrics_returns_required_stage_one_metrics():
    metrics = compute_metrics([0, 0, 1, 1], [0.05, 0.20, 0.80, 0.95])
    assert {"precision", "recall", "f1", "roc_auc", "pr_auc", "fpr"} <= metrics.keys()
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["fpr"] == pytest.approx(0.0)


def test_compute_metrics_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="non-finite"):
        compute_metrics([0, 1], [0.1, np.nan])
    with pytest.raises(ValueError, match="binary"):
        compute_metrics([0, 2], [0.1, 0.9])


def test_pick_threshold_respects_validation_fpr_ceiling():
    threshold = pick_threshold([0, 0, 0, 1, 1], [0.05, 0.10, 0.40, 0.60, 0.90], fpr_ceiling=0.0)
    metrics = compute_metrics([0, 0, 0, 1, 1], [0.05, 0.10, 0.40, 0.60, 0.90], threshold)
    assert metrics["fpr"] <= 0.0


def test_stratified_split_is_deterministic_and_disjoint():
    labels = np.array([0] * 80 + [1] * 20)
    first = stratified_split(labels, seed=42)
    second = stratified_split(labels, seed=42)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert not (set(first[0]) & set(first[1]))
    assert not (set(first[0]) & set(first[2]))
    assert not (set(first[1]) & set(first[2]))
    assert [len(part) for part in first] == [70, 15, 15]


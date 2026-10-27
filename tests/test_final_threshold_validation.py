"""Tests for validation-only threshold audit behavior."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.validate_final_thresholds import audit_dataset


class FakeModel:
    def predict_proba(self, matrix):
        return np.array([[0.01, 0.99], [0.05, 0.95], [0.8, 0.2], [0.9, 0.1]])


def test_audit_recomputes_validation_metrics_without_opening_test_files(tmp_path, monkeypatch):
    from scipy import sparse
    import scripts.validate_final_thresholds as validator

    dataset = "paysim"
    model_dir = tmp_path / "models" / dataset / "stage1"
    processed_dir = tmp_path / "data" / "processed" / dataset
    model_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    import joblib
    joblib.dump(FakeModel(), model_dir / "best_model.joblib")
    sparse.save_npz(processed_dir / "X_validation.npz", sparse.csr_matrix(np.ones((4, 1))))
    np.save(processed_dir / "y_validation.npy", np.array([1, 0, 1, 0]))
    from src.common.evaluate import compute_metrics
    expected = compute_metrics(np.array([1, 0, 1, 0]), np.array([0.99, 0.95, 0.2, 0.1]), 0.99)
    (model_dir / "metadata.json").write_text(json.dumps({
        "model": "lightgbm",
        "threshold": 0.99,
        "threshold_rule": "validation threshold",
        "validation_metrics": expected,
    }), encoding="utf-8")

    real_load = validator.np.load
    loaded_names = []

    def recording_load(path, *args, **kwargs):
        loaded_names.append(str(path))
        assert "test" not in Path(path).name.lower()
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(validator.np, "load", recording_load)
    result = audit_dataset(tmp_path, dataset)
    assert result["passed"]
    assert result["test_split_accessed"] is False
    assert {Path(path).name for path in loaded_names} == {
        "X_validation.npz", "y_validation.npy"
    }


def test_audit_fails_if_saved_metrics_disagree(tmp_path):
    from scipy import sparse
    import joblib

    dataset = "paysim"
    model_dir = tmp_path / "models" / dataset / "stage1"
    processed_dir = tmp_path / "data" / "processed" / dataset
    model_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    joblib.dump(FakeModel(), model_dir / "best_model.joblib")
    sparse.save_npz(processed_dir / "X_validation.npz", sparse.csr_matrix(np.ones((4, 1))))
    np.save(processed_dir / "y_validation.npy", np.array([1, 0, 1, 0]))
    (model_dir / "metadata.json").write_text(json.dumps({
        "model": "lightgbm", "threshold": 0.9,
        "validation_metrics": {"threshold": 0.5},
    }), encoding="utf-8")

    result = audit_dataset(tmp_path, dataset)
    assert not result["passed"]
    assert result["differences"]["threshold"]

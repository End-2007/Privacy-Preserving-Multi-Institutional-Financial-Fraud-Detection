from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.inference import DatasetArtifacts, predict_transaction, transform_transaction


class DummyPreprocessor:
    feature_names_in_ = np.array(["amount", "type"])

    def transform(self, frame):
        assert list(frame.columns) == ["amount", "type"]
        return frame[["amount"]].to_numpy(dtype=float)


class DummyModel:
    def __init__(self, probability=0.8):
        self.probability = probability

    def predict_proba(self, transformed):
        assert transformed.shape == (1, 1)
        return np.array([[1.0 - self.probability, self.probability]])


def make_artifacts(probability=0.8):
    return DatasetArtifacts(
        dataset="paysim",
        model=DummyModel(probability),
        preprocessor=DummyPreprocessor(),
        metadata={"dataset": "paysim", "threshold": 0.5, "model": "dummy"},
        feature_metadata={"one_hot_features": ["type"]},
    )


def test_predict_transaction_returns_thresholded_result():
    result = predict_transaction(
        "paysim",
        {"amount": 100.0, "type": "TRANSFER", "ignored": "allowed"},
        artifacts=make_artifacts(),
    )

    assert result.dataset == "paysim"
    assert result.fraud_probability == pytest.approx(0.8)
    assert result.predicted_label == 1
    assert result.threshold == pytest.approx(0.5)
    assert result.model_name == "dummy"


def test_probability_below_threshold_is_not_flagged():
    result = predict_transaction(
        "paysim",
        pd.Series({"amount": 100.0, "type": "PAYMENT"}),
        artifacts=make_artifacts(probability=0.2),
    )

    assert result.predicted_label == 0


def test_missing_feature_is_rejected():
    with pytest.raises(ValueError, match="Missing required feature columns"):
        transform_transaction({"amount": 100.0}, make_artifacts())


def test_multiple_rows_are_rejected():
    frame = pd.DataFrame([{"amount": 1.0, "type": "PAYMENT"}, {"amount": 2.0, "type": "TRANSFER"}])
    with pytest.raises(ValueError, match="Exactly one transaction"):
        transform_transaction(frame, make_artifacts())


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="Unsupported dataset"):
        predict_transaction("unknown", {"amount": 1.0}, artifacts=make_artifacts())



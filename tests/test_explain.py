from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common import explain as explain_module
from src.common.inference import DatasetArtifacts


class DummyPreprocessor:
    feature_names_in_ = np.array(["amount", "type", "card_1"], dtype=object)

    def transform(self, frame):
        assert list(frame.columns) == ["amount", "type", "card_1"]
        return np.array([[100.0, 1.0, 2.0]], dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.array(
            ["numeric__amount", "onehot__type_TRANSFER", "frequency__card_1"],
            dtype=object,
        )


class DummyBooster:
    def predict(self, transformed, raw_score=False):
        assert transformed.shape == (1, 3)
        if raw_score:
            return np.array([-0.05])
        return np.array([0.82])


class DummyModel:
    def __init__(self):
        self.booster_ = DummyBooster()


class DummyTreeExplainer:
    expected_value = 0.2

    def __init__(self, model, **kwargs):
        assert isinstance(model, DummyBooster)
        assert kwargs == {
            "feature_perturbation": "tree_path_dependent",
            "model_output": "raw",
        }

    def shap_values(self, transformed, **kwargs):
        assert transformed.shape == (1, 3)
        return np.array([[0.1, -0.4, 0.05]])


def make_artifacts(dataset: str) -> DatasetArtifacts:
    return DatasetArtifacts(
        dataset=dataset,
        model=DummyModel(),
        preprocessor=DummyPreprocessor(),
        metadata={
            "dataset": dataset,
            "threshold": 0.5,
            "model": "LightGBM",
        },
        feature_metadata={
            "numeric_features": ["amount"],
            "one_hot_features": ["type"],
            "frequency_encoded_features": ["card_1"],
            "missing_indicator_features": [],
        },
    )


def use_dummy_explainer(monkeypatch):
    monkeypatch.setattr(
        explain_module.shap,
        "TreeExplainer",
        DummyTreeExplainer,
    )


@pytest.mark.parametrize("dataset", ["paysim", "ieee_cis"])
def test_explanation_contract_rank_probability_and_additivity(dataset, monkeypatch):
    artifacts = make_artifacts(dataset)
    use_dummy_explainer(monkeypatch)

    result = explain_module.explain_transaction(
        dataset,
        {
            "amount": 100.0,
            "type": "TRANSFER",
            "card_1": 2,
        },
        artifacts=artifacts,
        max_features=2,
    )

    assert result.dataset == dataset
    assert result.fraud_probability == pytest.approx(0.82)
    assert result.base_value == pytest.approx(0.2)
    assert result.output_value == pytest.approx(-0.05)
    assert result.plain_language_reasons == ()

    assert len(result.contributions) == 2
    first, second = result.contributions
    assert first.feature_name == "onehot__type_TRANSFER"
    assert first.source_feature == "type"
    assert first.value == "TRANSFER"
    assert first.shap_value == pytest.approx(-0.4)
    assert first.direction == "decreases_risk"
    assert first.rank == 1
    assert second.feature_name == "numeric__amount"
    assert second.source_feature == "amount"
    assert second.value == 100.0
    assert second.shap_value == pytest.approx(0.1)
    assert second.direction == "increases_risk"
    assert second.rank == 2


def test_full_contributions_add_to_raw_model_output(monkeypatch):
    use_dummy_explainer(monkeypatch)
    result = explain_module.explain_transaction(
        "paysim",
        {"amount": 100.0, "type": "TRANSFER", "card_1": 2},
        artifacts=make_artifacts("paysim"),
        max_features=3,
    )

    explained_output = result.base_value + sum(
        contribution.shap_value for contribution in result.contributions
    )
    assert explained_output == pytest.approx(
        result.output_value,
        rel=explain_module.ADDITIVITY_RTOL,
        abs=explain_module.ADDITIVITY_ATOL,
    )


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="Unsupported dataset"):
        explain_module.explain_transaction(
            "unknown",
            {"amount": 100.0, "type": "TRANSFER", "card_1": 2},
            artifacts=make_artifacts("paysim"),
        )


@pytest.mark.parametrize("max_features", [0, -1])
def test_non_positive_max_features_is_rejected(max_features):
    with pytest.raises(ValueError, match="at least 1"):
        explain_module.explain_transaction(
            "paysim",
            {"amount": 100.0, "type": "TRANSFER", "card_1": 2},
            artifacts=make_artifacts("paysim"),
            max_features=max_features,
        )


def test_non_integer_max_features_is_rejected():
    with pytest.raises(TypeError, match="positive integer"):
        explain_module.explain_transaction(
            "paysim",
            {"amount": 100.0, "type": "TRANSFER", "card_1": 2},
            artifacts=make_artifacts("paysim"),
            max_features=1.5,
        )


def test_one_row_series_is_supported(monkeypatch):
    use_dummy_explainer(monkeypatch)
    result = explain_module.explain_transaction(
        "paysim",
        pd.Series({"amount": 100.0, "type": "TRANSFER", "card_1": 2}),
        artifacts=make_artifacts("paysim"),
    )
    assert result.fraud_probability == pytest.approx(0.82)
    assert len(result.contributions) == 3


def test_multi_row_dataframe_is_rejected(monkeypatch):
    use_dummy_explainer(monkeypatch)
    frame = pd.DataFrame(
        [
            {"amount": 100.0, "type": "TRANSFER", "card_1": 2},
            {"amount": 200.0, "type": "PAYMENT", "card_1": 3},
        ]
    )
    with pytest.raises(ValueError, match="Exactly one transaction"):
        explain_module.explain_transaction(
            "paysim",
            frame,
            artifacts=make_artifacts("paysim"),
        )


def test_missing_indicator_maps_to_original_source():
    artifacts = make_artifacts("ieee_cis")
    artifacts.feature_metadata["missing_indicator_features"] = ["D1"]

    source = explain_module._source_feature_name(
        "missing__missingindicator_D1",
        artifacts,
        ["D1"],
    )

    assert source == "D1"


def test_overlapping_one_hot_source_names_use_longest_match():
    artifacts = make_artifacts("ieee_cis")
    artifacts.feature_metadata["one_hot_features"] = ["id_3", "id_30"]

    source = explain_module._source_feature_name(
        "categorical__id_30_Windows 10",
        artifacts,
        ["id_3", "id_30"],
    )

    assert source == "id_30"


def test_unmapped_feature_keeps_transformed_name_and_has_no_value(monkeypatch):
    artifacts = make_artifacts("ieee_cis")
    artifacts.preprocessor.get_feature_names_out = lambda input_features=None: np.array(
        [
            "unmapped__feature",
            "onehot__type_TRANSFER",
            "frequency__card_1",
        ],
        dtype=object,
    )
    use_dummy_explainer(monkeypatch)

    result = explain_module.explain_transaction(
        "ieee_cis",
        {"amount": 100.0, "type": "TRANSFER", "card_1": 2},
        artifacts=artifacts,
        max_features=3,
    )

    item = next(
        contribution
        for contribution in result.contributions
        if contribution.feature_name == "unmapped__feature"
    )
    assert item.source_feature == "unmapped__feature"
    assert item.value is None


def test_tree_shap_additivity_mismatch_is_rejected(monkeypatch):
    class IncorrectRawScoreBooster(DummyBooster):
        def predict(self, transformed, raw_score=False):
            if raw_score:
                return np.array([1.0])
            return super().predict(transformed, raw_score=False)

    artifacts = make_artifacts("paysim")
    artifacts.model.booster_ = IncorrectRawScoreBooster()
    use_dummy_explainer(monkeypatch)

    with pytest.raises(ValueError, match="do not add up"):
        explain_module.explain_transaction(
            "paysim",
            {"amount": 100.0, "type": "TRANSFER", "card_1": 2},
            artifacts=artifacts,
        )

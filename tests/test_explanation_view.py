from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.components import explanation_view
from src.common.explain import (
    ExplanationResult,
    FeatureContribution,
    _plain_language_reasons,
)


def make_result(dataset: str = "paysim") -> ExplanationResult:
    # Reverse tuple order intentionally; the component should render by rank.
    contributions = (
        FeatureContribution(
            feature_name="categorical__type_TRANSFER",
            source_feature="type",
            value="TRANSFER",
            shap_value=-0.4,
            direction="decreases_risk",
            rank=2,
        ),
        FeatureContribution(
            feature_name="numeric__C1",
            source_feature="C1",
            value=12.0,
            shap_value=0.9,
            direction="increases_risk",
            rank=1,
        ),
    )
    ranked = tuple(sorted(contributions, key=lambda item: item.rank))
    return ExplanationResult(
        dataset=dataset,
        fraud_probability=0.87,
        base_value=-1.2,
        output_value=1.1,
        contributions=contributions,
        plain_language_reasons=_plain_language_reasons(ranked),
    )


@pytest.fixture
def ui_calls(monkeypatch):
    calls = {
        "caption": [],
        "info": [],
        "markdown": [],
        "metric": [],
        "pyplot": [],
        "table": [],
        "write": [],
    }

    for name in calls:
        def recorder(*args, _name=name, **kwargs):
            calls[_name].append((args, kwargs))

        monkeypatch.setattr(explanation_view.st, name, recorder)
    return calls


@pytest.mark.parametrize("dataset", ["paysim", "ieee_cis"])
def test_render_explanation_shows_score_chart_reasons_and_limitations(
    dataset,
    ui_calls,
):
    result = make_result(dataset)

    explanation_view.render_explanation(result)

    metric_args, metric_kwargs = ui_calls["metric"][0]
    assert metric_kwargs["label"] == "Fraud model score (uncalibrated)"
    assert metric_kwargs["value"] == "0.8700"
    assert "confirmed fraud finding" in ui_calls["info"][0][0][0]
    assert "do not prove causation" in ui_calls["info"][0][0][0]
    assert "anonymized" in ui_calls["info"][0][0][0]
    assert any(dataset in args[0] for args, _ in ui_calls["caption"])

    assert len(ui_calls["pyplot"]) == 1
    figure = ui_calls["pyplot"][0][0][0]
    axis = figure.axes[0]
    assert axis.get_xlabel() == "SHAP contribution (raw model-output scale)"
    assert [patch.get_width() for patch in axis.patches] == pytest.approx([0.9, -0.4])
    assert len(ui_calls["table"]) == 1
    table = ui_calls["table"][0][0][0]
    assert isinstance(table, pd.DataFrame)
    assert list(table["Feature"]) == ["C1", "type"]
    assert list(table["Direction"]) == ["Raises model score", "Lowers model score"]
    assert len(ui_calls["write"]) == 2
    assert "C1" in ui_calls["write"][0][0][0]
    assert "TRANSFER" in ui_calls["write"][1][0][0]


def test_missing_value_is_displayed_as_unavailable(ui_calls):
    contribution = FeatureContribution(
        feature_name="unmapped__feature",
        source_feature="unmapped__feature",
        value=None,
        shap_value=0.3,
        direction="increases_risk",
        rank=1,
    )
    result = ExplanationResult(
        dataset="ieee_cis",
        fraud_probability=0.2,
        base_value=-0.5,
        output_value=-0.2,
        contributions=(contribution,),
        plain_language_reasons=(
            "Model feature 'unmapped__feature' (observed input value unavailable) "
            "contributed to a higher model fraud-risk score (SHAP value +0.3).",
        ),
    )

    explanation_view.render_explanation(result)

    table = ui_calls["table"][0][0][0]
    assert table.loc[0, "Input value"] == "Unavailable"
    assert "unmapped__feature" in ui_calls["write"][0][0][0]


def test_empty_explanation_shows_helpful_states_without_chart(ui_calls):
    result = ExplanationResult(
        dataset="paysim",
        fraud_probability=0.1,
        base_value=0.0,
        output_value=0.0,
        contributions=(),
        plain_language_reasons=(),
    )

    explanation_view.render_explanation(result)

    assert ui_calls["pyplot"] == []
    assert ui_calls["table"] == []
    messages = [args[0] for args, _ in ui_calls["info"]]
    assert any("No feature contributions" in message for message in messages)
    assert any("No plain-language reasons" in message for message in messages)


@pytest.mark.parametrize("probability", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_probability_is_rejected(probability, ui_calls):
    result = make_result()
    invalid = ExplanationResult(
        dataset=result.dataset,
        fraud_probability=probability,
        base_value=result.base_value,
        output_value=result.output_value,
        contributions=result.contributions,
        plain_language_reasons=result.plain_language_reasons,
    )

    with pytest.raises(ValueError, match="fraud_probability"):
        explanation_view.render_explanation(invalid)


def test_non_explanation_result_is_rejected():
    with pytest.raises(TypeError, match="ExplanationResult"):
        explanation_view.render_explanation({"fraud_probability": 0.5})

from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from app import streamlit_app
from app.streamlit_app import load_metrics, metrics_table


def test_metrics_table_contains_both_datasets_and_required_metrics():
    table = metrics_table()

    assert set(table["Dataset"]) == {"PaySim", "IEEE-CIS"}
    assert {
        "Precision",
        "Recall",
        "F1",
        "PR-AUC",
        "ROC-AUC",
        "FPR",
    }.issubset(table.columns)
    assert table["Threshold"].notna().all()


def test_load_metrics_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unsupported dataset"):
        load_metrics("unknown")

def test_shap_integration_consumes_agreed_result_contract(monkeypatch):
    contributions = (
        SimpleNamespace(source_feature="amount", shap_value=0.42),
        SimpleNamespace(source_feature="type_TRANSFER", shap_value=-0.08),
    )
    explanation = SimpleNamespace(
        plain_language_reasons=("The amount is unusually high for this transaction type.",),
        contributions=contributions,
    )
    module = SimpleNamespace(
        explain_transaction=lambda *args, **kwargs: explanation,
    )
    monkeypatch.setitem(sys.modules, "src.common.explain", module)
    monkeypatch.setattr(streamlit_app.st, "subheader", lambda *args, **kwargs: None)
    rendered = []
    monkeypatch.setattr(streamlit_app.st, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "text", rendered.append)
    monkeypatch.setattr(
        streamlit_app.st,
        "bar_chart",
        lambda *args, **kwargs: pytest.fail("SHAP should use the stable text renderer"),
    )

    streamlit_app._render_explanation("paysim", pd.Series({"amount": 100}), SimpleNamespace())

    assert len(rendered) == 1
    assert "amount" in rendered[0]
    assert "+0.4200" in rendered[0]
    assert "-0.0800" in rendered[0]
    assert "Positive contributions" in rendered[0]


def test_explanation_failure_is_reported_without_crashing(monkeypatch):
    def fail_explanation(*args, **kwargs):
        raise RuntimeError("SHAP backend unavailable")

    module = SimpleNamespace(explain_transaction=fail_explanation)
    monkeypatch.setitem(sys.modules, "src.common.explain", module)
    warnings = []
    monkeypatch.setattr(streamlit_app.st, "warning", warnings.append)

    streamlit_app._render_explanation("paysim", pd.Series({"amount": 100}), SimpleNamespace())

    assert warnings == ["Explanation could not be generated: SHAP backend unavailable"]


def test_signed_bars_skip_malformed_and_non_finite_values():
    contributions = (
        SimpleNamespace(source_feature="good", shap_value=0.5),
        SimpleNamespace(source_feature="nan", shap_value=float("nan")),
        SimpleNamespace(source_feature="missing value"),
    )

    rendered = streamlit_app._signed_contribution_bars(contributions)

    assert "good" in rendered
    assert "nan" not in rendered
    assert "+0.5000" in rendered

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
    monkeypatch.setattr(streamlit_app.st, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "bar_chart", lambda *args, **kwargs: None)

    streamlit_app._render_explanation("paysim", pd.Series({"amount": 100}), SimpleNamespace())

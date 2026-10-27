from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from app import streamlit_app
from app.streamlit_app import load_metrics, metrics_table, validate_transaction_input


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


def test_paysim_input_validation_rejects_negative_amount_and_missing_type():
    transaction = pd.Series({"amount": -10, "oldbalanceOrg": 0, "newbalanceOrig": 0,
                             "oldbalanceDest": 0, "newbalanceDest": 0, "type": ""})

    errors = validate_transaction_input("paysim", transaction)

    assert "type is required." in errors
    assert "amount cannot be negative." in errors


def test_ieee_input_validation_accepts_complete_nonnegative_input():
    transaction = pd.Series({"TransactionAmt": 25.0, "card1": 1234,
                             "ProductCD": "W", "card4": "visa",
                             "P_emaildomain": "example.com", "R_emaildomain": "example.com"})

    assert validate_transaction_input("ieee_cis", transaction) == []


def test_input_validation_rejects_unknown_dataset():
    assert validate_transaction_input("unknown", pd.Series()) == ["Unsupported dataset: unknown"]


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


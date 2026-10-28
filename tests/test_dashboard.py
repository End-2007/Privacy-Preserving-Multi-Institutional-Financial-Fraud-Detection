from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from app import streamlit_app
from app.streamlit_app import (
    load_metrics,
    metrics_table,
    validate_transaction_input,
    validation_comparison_table,
)


def test_metrics_table_contains_both_datasets_and_required_metrics():
    table = metrics_table()

    assert set(table["Dataset"]) == {"PaySim", "IEEE-CIS"}
    assert {
        "Precision",
        "Recall",
        "F1",
        "Average Precision (AP)",
        "ROC-AUC",
        "FPR",
    }.issubset(table.columns)
    assert table["Validation-selected threshold"].notna().all()


def test_validation_comparison_table_has_four_families_per_dataset():
    table = validation_comparison_table()

    assert len(table) == 8
    assert table.groupby("Dataset")["Model"].nunique().to_dict() == {
        "IEEE-CIS": 4,
        "PaySim": 4,
    }
    assert {
        "Best validation strategy",
        "Validation Average Precision (AP)",
        "Validation ROC-AUC",
    }.issubset(table.columns)
    assert table["Validation Average Precision (AP)"].between(0, 1).all()
    assert table["Validation ROC-AUC"].between(0, 1).all()


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


def test_ieee_input_validation_allows_naturally_missing_categorical_fields():
    transaction = pd.Series({
        "TransactionAmt": 25.0,
        "card1": 1234,
        "ProductCD": "W",
        "card4": pd.NA,
        "P_emaildomain": pd.NA,
        "R_emaildomain": pd.NA,
    })

    assert validate_transaction_input("ieee_cis", transaction) == []


@pytest.mark.parametrize("product_code", ["", None, float("nan"), pd.NA])
def test_ieee_input_validation_still_requires_product_code(product_code):
    transaction = pd.Series({
        "TransactionAmt": 25.0,
        "card1": 1234,
        "ProductCD": product_code,
        "card4": "visa",
        "P_emaildomain": "example.com",
        "R_emaildomain": "example.com",
    })

    assert validate_transaction_input("ieee_cis", transaction) == ["ProductCD is required."]


@pytest.mark.parametrize("value", [None, float("nan"), pd.NA])
def test_missing_ieee_text_defaults_render_as_blank(value):
    assert streamlit_app._text_input_default(value) == ""


def test_input_validation_rejects_unknown_dataset():
    assert validate_transaction_input("unknown", pd.Series()) == ["Unsupported dataset: unknown"]


def test_shap_integration_consumes_agreed_result_contract(monkeypatch):
    contributions = (
        SimpleNamespace(
            feature_name="numeric__amount",
            source_feature="amount",
            value=100,
            shap_value=0.42,
            direction="increases_risk",
        ),
        SimpleNamespace(
            feature_name="categorical__type_TRANSFER",
            source_feature="type_TRANSFER",
            value=1,
            shap_value=-0.08,
            direction="decreases_risk",
        ),
    )
    explanation = SimpleNamespace(
        base_value=-0.2,
        plain_language_reasons=("The amount is unusually high for this transaction type.",),
        contributions=contributions,
    )
    calls = []
    module = SimpleNamespace(
        explain_transaction=lambda *args, **kwargs: calls.append((args, kwargs)) or explanation,
    )
    monkeypatch.setitem(sys.modules, "src.common.explain", module)
    monkeypatch.setattr(streamlit_app.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "write", lambda *args, **kwargs: None)
    rendered = []
    monkeypatch.setattr(streamlit_app.st, "caption", rendered.append)
    monkeypatch.setattr(
        streamlit_app.st,
        "markdown",
        lambda body, **kwargs: rendered.append(body),
    )
    monkeypatch.setattr(streamlit_app.st, "dataframe", lambda *args, **kwargs: None)
    transaction = pd.Series({"amount": 100})
    artifacts = SimpleNamespace()

    result = streamlit_app._load_explanation("paysim", transaction, artifacts)
    streamlit_app._show_explanation(result)

    assert result is explanation
    assert calls == [
        (("paysim", transaction), {"artifacts": artifacts, "max_features": 10})
    ]
    chart = next(value for value in rendered if isinstance(value, str) and "fg-shap-card" in value)
    assert "amount" in chart
    assert "+0.4200" in chart
    assert "-0.0800" in chart
    assert "Negative contributions" in chart


def test_explanation_failure_is_reported_without_crashing(monkeypatch):
    def fail_explanation(*args, **kwargs):
        raise RuntimeError("SHAP backend unavailable")

    module = SimpleNamespace(explain_transaction=fail_explanation)
    monkeypatch.setitem(sys.modules, "src.common.explain", module)
    warnings = []
    monkeypatch.setattr(streamlit_app.st, "warning", warnings.append)

    result = streamlit_app._load_explanation_or_report(
        "paysim", pd.Series({"amount": 100}), SimpleNamespace()
    )

    assert result is None
    assert warnings == [
        "SHAP explanation could not be generated: SHAP backend unavailable"
    ]


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

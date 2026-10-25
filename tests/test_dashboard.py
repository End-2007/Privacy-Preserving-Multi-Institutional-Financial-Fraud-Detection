from __future__ import annotations

import pandas as pd
import pytest

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


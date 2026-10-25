from __future__ import annotations

import pandas as pd
import pytest

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


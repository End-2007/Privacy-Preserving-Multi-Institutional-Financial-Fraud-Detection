"""Streamlit dashboard for the FedGuard Micro Project."""

from __future__ import annotations

import importlib
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from src.common.config import REPORTS_DIR
from src.common.inference import (
    DatasetArtifacts,
    load_dataset_artifacts,
    predict_transaction,
)


DATASET_LABELS = {
    "paysim": "PaySim",
    "ieee_cis": "IEEE-CIS",
}
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}
STRATEGY_LABELS = {
    "class_weight": "Class weights",
    "cap_smote": "Capped SMOTE",
    "cap_smote_enn": "Capped SMOTE-ENN",
}
EXCLUDED_COLUMNS = {"isFraud", "row_id", "TransactionID"}
USER_INPUT_FIELDS = {
    "paysim": {
        "numeric_nonnegative": (
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ),
        "required": ("type",),
    },
    "ieee_cis": {
        "numeric_nonnegative": ("TransactionAmt", "card1"),
        # Email domains and card network are naturally missing in IEEE-CIS.
        "required": ("ProductCD",),
    },
}


@st.cache_resource(show_spinner=False)
def cached_artifacts(dataset: str) -> DatasetArtifacts:
    return load_dataset_artifacts(dataset)


@st.cache_data(show_spinner=False)
def load_sample_transactions(dataset: str, limit: int = 50) -> pd.DataFrame:
    """Load reproducible validation examples, never the held-out test split."""
    if dataset not in DATASET_LABELS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    path = Path("data") / "processed" / dataset / "validation.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Validation samples were not found: {path}")
    frame = pd.read_parquet(path)
    frame = frame.drop(columns=list(EXCLUDED_COLUMNS.intersection(frame.columns)))
    return frame.head(limit).reset_index(drop=True)


def load_metrics(dataset: str) -> dict:
    if dataset not in DATASET_LABELS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    path = REPORTS_DIR / f"{dataset}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Metrics report was not found: {path}")
    import json

    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def metrics_table() -> pd.DataFrame:
    rows = []
    for dataset in DATASET_LABELS:
        report = load_metrics(dataset)
        metrics = report.get("test_metrics", {})
        rows.append(
            {
                "Dataset": DATASET_LABELS[dataset],
                "Model": report.get("model", "unknown"),
                "Validation-selected threshold": report.get("threshold", metrics.get("threshold")),
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "F1": metrics.get("f1"),
                "Average Precision (AP)": metrics.get("pr_auc"),
                "ROC-AUC": metrics.get("roc_auc"),
                "FPR": metrics.get("fpr"),
            }
        )
    return pd.DataFrame(rows)


def validation_comparison_table() -> pd.DataFrame:
    """Return the best validation-ranking result per model family and dataset.

    Average Precision and ROC-AUC do not depend on the candidate's 0.5
    classification threshold. Final thresholded metrics are shown separately.
    """
    rows = []
    model_order = tuple(MODEL_LABELS)
    for dataset, dataset_label in DATASET_LABELS.items():
        comparison = load_metrics(dataset).get("comparison", [])
        best_by_model = {}
        for candidate in comparison:
            model = candidate.get("model")
            validation = candidate.get("validation", {})
            if model not in MODEL_LABELS:
                continue
            try:
                average_precision = float(validation["pr_auc"])
                roc_auc = float(validation["roc_auc"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(average_precision) or not math.isfinite(roc_auc):
                continue
            current = best_by_model.get(model)
            if current is None or average_precision > current["average_precision"]:
                best_by_model[model] = {
                    "average_precision": average_precision,
                    "roc_auc": roc_auc,
                    "strategy": candidate.get("strategy", "unknown"),
                }

        missing_models = [model for model in model_order if model not in best_by_model]
        if missing_models:
            raise ValueError(
                f"Validation comparison is incomplete for {dataset}: {', '.join(missing_models)}"
            )

        for model in model_order:
            result = best_by_model[model]
            rows.append(
                {
                    "Dataset": dataset_label,
                    "Model": MODEL_LABELS[model],
                    "Best validation strategy": STRATEGY_LABELS.get(
                        result["strategy"], str(result["strategy"])
                    ),
                    "Validation Average Precision (AP)": result["average_precision"],
                    "Validation ROC-AUC": result["roc_auc"],
                }
            )
    return pd.DataFrame(rows)


def _risk_label(score: float, threshold: float) -> str:
    if score >= threshold:
        return "High"
    if score >= threshold * 0.5:
        return "Medium"
    return "Low"


def _render_explanation(dataset: str, transaction: pd.Series, artifacts: DatasetArtifacts) -> None:
    try:
        module = importlib.import_module("src.common.explain")
        explain_transaction = getattr(module, "explain_transaction")
    except (ImportError, AttributeError):
        st.info(
            "SHAP explanation module is not available yet. "
            "The risk score above is still available."
        )
        return

    try:
        explanation = explain_transaction(
            dataset,
            transaction,
            artifacts=artifacts,
            max_features=8,
        )
    except Exception as exc:
        st.warning(f"Explanation could not be generated: {exc}")
        return

    reasons = getattr(explanation, "plain_language_reasons", ())
    if reasons:
        st.subheader("Explanation of this prediction")
        for reason in reasons:
            st.write(f"- {reason}")

    contributions = getattr(explanation, "contributions", ())
    if contributions:
        st.subheader("Top feature contributions")
        chart_text = _signed_contribution_bars(contributions)
        if chart_text:
            # Use Streamlit's plain text renderer here: chart components rely on
            # lazily loaded browser bundles, which failed in the local demo and
            # took the page connection down. The exact values remain visible in
            # the labels, while the signed bars preserve direction and scale.
            st.text(chart_text)
        else:
            st.info("No finite SHAP contributions were available to display.")


def _signed_contribution_bars(contributions, *, bar_width: int = 20) -> str:
    """Format local SHAP values as dependency-light, signed text bars."""
    rows = []
    for item in contributions:
        try:
            feature = str(item.source_feature)
            value = float(item.shap_value)
        except (AttributeError, TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        rows.append((feature, value))

    if not rows:
        return ""

    largest = max(abs(value) for _, value in rows) or 1.0
    lines = ["Negative contributions  ←       0       →  Positive contributions"]
    for feature, value in rows:
        width = max(1, int(round(abs(value) / largest * bar_width)))
        bar = "█" * min(width, bar_width)
        left = bar.rjust(bar_width) if value < 0 else " " * bar_width
        right = " " * bar_width if value < 0 else bar.ljust(bar_width)
        lines.append(f"{feature[:32]:32} {left}|{right} {value:+.4f}")
    return "\n".join(lines)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number


def _text_input_default(value: object) -> str:
    """Render missing source values as blank instead of the text ``<NA>``."""
    try:
        if value is None or pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def validate_transaction_input(dataset: str, transaction: pd.Series) -> list[str]:
    """Return user-facing validation messages before model inference."""
    if dataset not in USER_INPUT_FIELDS:
        return [f"Unsupported dataset: {dataset}"]

    errors: list[str] = []
    rules = USER_INPUT_FIELDS[dataset]
    for field in rules["required"]:
        value = transaction.get(field)
        if not _text_input_default(value).strip():
            errors.append(f"{field} is required.")

    for field in rules["numeric_nonnegative"]:
        value = transaction.get(field)
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors.append(f"{field} must be a number.")
            continue
        if not math.isfinite(number):
            errors.append(f"{field} must be finite.")
        elif number < 0:
            errors.append(f"{field} cannot be negative.")
    return errors


def _derive_paysim_features(transaction: pd.Series) -> pd.Series:
    """Keep engineered PaySim fields consistent with user-entered balances."""
    amount = _safe_float(transaction.get("amount"))
    old_orig = _safe_float(transaction.get("oldbalanceOrg"))
    new_orig = _safe_float(transaction.get("newbalanceOrig"))
    old_dest = _safe_float(transaction.get("oldbalanceDest"))
    new_dest = _safe_float(transaction.get("newbalanceDest"))
    transaction["orig_balance_delta"] = new_orig - old_orig
    transaction["dest_balance_delta"] = new_dest - old_dest
    transaction["error_balance_orig"] = new_orig + amount - old_orig
    transaction["error_balance_dest"] = old_dest + amount - new_dest
    transaction["orig_zero_after"] = int(new_orig == 0)
    transaction["log_amount"] = math.log1p(max(amount, 0.0))
    return transaction


def _score_transaction_tab(dataset: str, transaction: pd.Series) -> None:
    validation_errors = validate_transaction_input(dataset, transaction)
    if validation_errors:
        st.error("Please correct the following input(s):")
        for error in validation_errors:
            st.write(f"- {error}")
        return

    try:
        artifacts = cached_artifacts(dataset)
        result = predict_transaction(dataset, transaction, artifacts=artifacts)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Transaction details")
        preview = transaction.head(30).rename("value").to_frame()
        st.dataframe(preview, width="stretch")
        if len(transaction) > 30:
            st.caption(f"Showing 30 of {len(transaction)} model input fields.")
    with right:
        st.subheader("Risk result")
        st.metric("Fraud model score (uncalibrated)", f"{result.fraud_probability:.4f}")
        st.metric("Validation-selected threshold", f"{result.threshold:.4f}")
        st.metric("Interface category (heuristic)", _risk_label(result.fraud_probability, result.threshold))
        st.caption("Heuristic presentation category based on the validation threshold; not a calibrated risk band.")
        st.write(
            "Predicted class: "
            + ("Flagged for review" if result.predicted_label else "Not flagged")
        )
        _render_explanation(dataset, transaction, artifacts)


def _render_paysim_tab() -> None:
    try:
        samples = load_sample_transactions("paysim", 50)
    except Exception as exc:
        st.error(f"PaySim setup failed: {exc}")
        return
    selected = st.selectbox(
        "Reference transaction",
        options=list(samples.index),
        format_func=lambda index: f"Example {index + 1}",
        key="paysim_reference",
    )
    transaction = samples.loc[selected].copy()
    st.caption("Enter the transaction details below. Engineered balance features are updated automatically.")
    with st.form("paysim_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            transaction["type"] = st.selectbox(
                "Transaction type",
                ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
                index=["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"].index(str(transaction.get("type", "PAYMENT")))
                if str(transaction.get("type", "PAYMENT")) in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"] else 3,
            )
            transaction["amount"] = st.number_input("Amount", min_value=0.0, value=_safe_float(transaction.get("amount")), step=100.0)
        with col2:
            transaction["oldbalanceOrg"] = st.number_input("Sender balance before", min_value=0.0, value=_safe_float(transaction.get("oldbalanceOrg")), step=100.0)
            transaction["newbalanceOrig"] = st.number_input("Sender balance after", min_value=0.0, value=_safe_float(transaction.get("newbalanceOrig")), step=100.0)
        with col3:
            transaction["oldbalanceDest"] = st.number_input("Receiver balance before", min_value=0.0, value=_safe_float(transaction.get("oldbalanceDest")), step=100.0)
            transaction["newbalanceDest"] = st.number_input("Receiver balance after", min_value=0.0, value=_safe_float(transaction.get("newbalanceDest")), step=100.0)
        submitted = st.form_submit_button("Check PaySim risk")
    if submitted:
        _score_transaction_tab("paysim", _derive_paysim_features(transaction))


def _render_ieee_tab() -> None:
    try:
        samples = load_sample_transactions("ieee_cis", 50)
    except Exception as exc:
        st.error(f"IEEE-CIS setup failed: {exc}")
        return
    selected = st.selectbox(
        "Reference transaction",
        options=list(samples.index),
        format_func=lambda index: f"Example {index + 1}",
        key="ieee_reference",
    )
    transaction = samples.loc[selected].copy()
    st.caption("Enter the main card/e-commerce details. Less visible model fields remain from the selected reference example.")
    st.caption("Leave optional fields blank when the reference transaction has no value; missing values are passed through the saved preprocessing.")
    with st.form("ieee_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            transaction["TransactionAmt"] = st.number_input("Transaction amount", min_value=0.0, value=_safe_float(transaction.get("TransactionAmt")), step=10.0)
            transaction["ProductCD"] = st.text_input("Product code", value=_text_input_default(transaction.get("ProductCD", "")))
        with col2:
            transaction["card1"] = st.number_input("Card identifier", min_value=0.0, value=_safe_float(transaction.get("card1")), step=1.0)
            card4_value = st.text_input("Card network", value=_text_input_default(transaction.get("card4", "")))
            transaction["card4"] = card4_value.strip() or pd.NA
        with col3:
            purchaser_domain = st.text_input("Purchaser email domain", value=_text_input_default(transaction.get("P_emaildomain", "")))
            recipient_domain = st.text_input("Recipient email domain", value=_text_input_default(transaction.get("R_emaildomain", "")))
            transaction["P_emaildomain"] = purchaser_domain.strip() or pd.NA
            transaction["R_emaildomain"] = recipient_domain.strip() or pd.NA
        submitted = st.form_submit_button("Check IEEE-CIS risk")
    if submitted:
        _score_transaction_tab("ieee_cis", transaction)


def main() -> None:
    st.set_page_config(page_title="FedGuard", layout="wide")
    st.title("FedGuard")
    st.caption(
        "Micro Project demo: centralized fraud-risk scoring with explainable outputs. "
        "A prediction is a risk signal, not proof of confirmed fraud."
    )

    paysim_tab, ieee_tab = st.tabs(["PaySim Risk Check", "IEEE-CIS Risk Check"])
    with paysim_tab:
        _render_paysim_tab()
    with ieee_tab:
        _render_ieee_tab()

    st.divider()
    st.subheader("Validation-only model-family comparison")
    st.caption(
        "Each family is represented by its strategy with the highest validation Average Precision. "
        "These ranking metrics are not held-out test results."
    )
    try:
        validation_table = validation_comparison_table()
        st.dataframe(
            validation_table.style.format(
                {
                    "Validation Average Precision (AP)": "{:.4f}",
                    "Validation ROC-AUC": "{:.4f}",
                }
            ),
            width="stretch",
        )
    except Exception as exc:
        st.warning(f"Validation comparison could not be loaded: {exc}")

    st.subheader("Selected-model held-out test results")
    try:
        table = metrics_table()
        st.dataframe(
            table.style.format(
                {
                    "Validation-selected threshold": "{:.4f}",
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1": "{:.3f}",
                    "Average Precision (AP)": "{:.3f}",
                    "ROC-AUC": "{:.3f}",
                    "FPR": "{:.3%}",
                }
            ),
            width="stretch",
        )
    except Exception as exc:
        st.warning(f"Metrics could not be loaded: {exc}")

    st.caption(
        "This table reports the final LightGBM test evaluation for each dataset. "
        "Model and threshold selection used validation data; the test split was evaluated once."
    )
    st.caption(
        "The dashboard uses validation samples for demonstration. "
        "The held-out test split is not used for interactive selection or tuning."
    )


if __name__ == "__main__":
    main()

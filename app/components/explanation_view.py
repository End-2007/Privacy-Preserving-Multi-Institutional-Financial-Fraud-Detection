"""Standalone Streamlit component for transaction-level SHAP explanations."""

from __future__ import annotations

import math
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import streamlit as st

from src.common.explain import ExplanationResult, FeatureContribution


EXPLANATION_DISCLAIMER = (
    "This is a model risk estimate, not a confirmed fraud finding. SHAP values "
    "describe how features contributed to this model's score for this one "
    "transaction; they do not prove causation. Some IEEE-CIS feature names are "
    "anonymized, so this panel preserves their names without assigning "
    "undocumented business meanings."
)


_DIRECTION_LABELS = {
    "increases_risk": "Raises model score",
    "decreases_risk": "Lowers model score",
    "no_effect": "No directional effect",
}
_DIRECTION_COLORS = {
    "increases_risk": "#d55e00",
    "decreases_risk": "#0072b2",
    "no_effect": "#777777",
}


def _display_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return "Unavailable"

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return "Unavailable"

    rendered = str(value)
    if len(rendered) > 80:
        rendered = f"{rendered[:77]}..."
    return rendered


def _chart_label(contribution: FeatureContribution) -> str:
    if contribution.source_feature == contribution.feature_name and "__" in contribution.feature_name:
        return f"Model feature: {contribution.feature_name}"
    if contribution.source_feature == contribution.feature_name:
        return contribution.feature_name
    return f"{contribution.source_feature} ({contribution.feature_name})"


def _contribution_table(
    contributions: tuple[FeatureContribution, ...],
) -> pd.DataFrame:
    rows = []
    for contribution in sorted(contributions, key=lambda item: item.rank):
        rows.append(
            {
                "Feature": contribution.source_feature,
                "Model feature": contribution.feature_name,
                "Input value": _display_value(contribution.value),
                "SHAP value": contribution.shap_value,
                "Direction": _DIRECTION_LABELS.get(
                    contribution.direction,
                    "Unknown",
                ),
                "Rank": contribution.rank,
            }
        )
    return pd.DataFrame(rows)


def _render_contribution_chart(
    contributions: tuple[FeatureContribution, ...],
) -> None:
    ranked = sorted(contributions, key=lambda item: item.rank)
    labels = [_chart_label(item) for item in ranked]
    values = [float(item.shap_value) for item in ranked]
    colors = [_DIRECTION_COLORS.get(item.direction, "#777777") for item in ranked]

    figure_height = max(3.0, 0.48 * len(ranked) + 1.2)
    figure = Figure(figsize=(10, figure_height))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.barh(labels, values, color=colors)
    axis.axvline(0.0, color="#555555", linewidth=0.9)
    axis.invert_yaxis()
    axis.set_xlabel("SHAP contribution (raw model-output scale)")
    axis.set_title("Feature contributions for this transaction")
    axis.grid(axis="x", linestyle=":", alpha=0.35)
    figure.tight_layout()
    st.pyplot(figure, width="stretch", clear_figure=True)


def render_explanation(result: ExplanationResult) -> None:
    """Render a SHAP chart, contribution details, reasons, and limitations.

    The component displays the uncalibrated model score from ``ExplanationResult`` as a
    model estimate. It does not infer a prediction label or threshold; those
    belong to the shared inference result consumed by the app shell.
    """
    if not isinstance(result, ExplanationResult):
        raise TypeError("result must be an ExplanationResult")

    try:
        fraud_probability = float(result.fraud_probability)
    except (TypeError, ValueError) as exc:
        raise ValueError("fraud_probability must be a finite number from 0 to 1") from exc
    if not math.isfinite(fraud_probability) or not 0.0 <= fraud_probability <= 1.0:
        raise ValueError("fraud_probability must be a finite number from 0 to 1")

    st.markdown("### Transaction explanation")
    st.caption(f"Dataset: {result.dataset}")
    st.metric(
        label="Fraud model score (uncalibrated)",
        value=f"{fraud_probability:.4f}",
    )
    st.info(EXPLANATION_DISCLAIMER)

    st.markdown("#### SHAP feature contributions")
    if result.contributions:
        _render_contribution_chart(result.contributions)
        st.caption(
            "Positive values move the model output toward higher estimated risk; "
            "negative values move it lower. SHAP values are on the model's raw "
            "output scale, not probability points."
        )
        st.table(_contribution_table(result.contributions))
    else:
        st.info("No feature contributions are available for this explanation.")

    st.markdown("#### Plain-language reasons")
    if result.plain_language_reasons:
        for reason in result.plain_language_reasons:
            # st.write displays the reason as text instead of interpreting raw
            # input values as Markdown or HTML.
            st.write(reason)
    else:
        st.info("No plain-language reasons are available for this explanation.")

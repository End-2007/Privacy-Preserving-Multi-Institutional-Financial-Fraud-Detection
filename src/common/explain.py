"""TreeSHAP explanations for single FedGuard transactions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from scipy import sparse

from src.common.config import DATASET_CONFIG, ROOT
from src.common.inference import (
    DatasetArtifacts,
    load_dataset_artifacts,
    predict_transaction,
    transform_transaction,
)


SUPPORTED_DATASETS = tuple(sorted(DATASET_CONFIG))

# SHAP contributions and LightGBM's raw score should agree within this
# floating-point tolerance for one transformed transaction.
ADDITIVITY_RTOL = 1e-5
ADDITIVITY_ATOL = 1e-5


@dataclass(frozen=True)
class FeatureContribution:
    """One transformed model feature and its source transaction field."""

    feature_name: str
    source_feature: str
    value: object
    shap_value: float
    direction: str
    rank: int


@dataclass(frozen=True)
class ExplanationResult:
    """Prediction and TreeSHAP explanation for one transaction."""

    dataset: str
    fraud_probability: float
    base_value: float
    output_value: float
    contributions: tuple[FeatureContribution, ...]
    plain_language_reasons: tuple[str, ...]


def _normalise_dataset(dataset: str) -> str:
    name = str(dataset).strip().lower()
    if name not in DATASET_CONFIG:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset {dataset!r}; choose one of: {supported}")
    return name


def _as_single_row(
    transaction: Mapping[str, Any] | pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(transaction, pd.DataFrame):
        frame = transaction.copy()
    elif isinstance(transaction, pd.Series):
        frame = transaction.to_frame().T
    elif isinstance(transaction, Mapping):
        frame = pd.DataFrame([dict(transaction)])
    else:
        raise TypeError(
            "transaction must be a mapping, pandas Series, or one-row DataFrame"
        )

    if len(frame) != 1:
        raise ValueError("Exactly one transaction is required for an explanation")
    return frame


def _input_feature_names(artifacts: DatasetArtifacts) -> list[str]:
    names = getattr(artifacts.preprocessor, "feature_names_in_", None)
    if names is not None:
        result = [str(name) for name in names]
        if result:
            return result

    metadata = artifacts.feature_metadata
    groups = (
        metadata.get("numeric_features", []),
        metadata.get("frequency_encoded_features", []),
        metadata.get("one_hot_features", []),
    )
    result = list(dict.fromkeys(str(name) for group in groups for name in group))
    if not result:
        raise ValueError(
            f"No source feature names found for {artifacts.dataset}"
        )
    return result


def _transformed_feature_names(
    artifacts: DatasetArtifacts,
    feature_count: int,
) -> list[str]:
    source_names = _input_feature_names(artifacts)
    getter = getattr(artifacts.preprocessor, "get_feature_names_out", None)

    if not callable(getter):
        if len(source_names) == feature_count:
            return source_names
        raise ValueError(
            "The fitted preprocessor does not expose transformed feature names, "
            "and its input feature count does not match the transformed matrix"
        )

    try:
        names = getter()
    except (TypeError, ValueError):
        names = getter(np.asarray(source_names, dtype=object))

    result = [str(name) for name in np.asarray(names, dtype=object).reshape(-1)]
    if len(result) != feature_count:
        raise ValueError(
            "Preprocessor feature-name count does not match the transformed "
            f"matrix width ({len(result)} names for {feature_count} columns)"
        )
    return result


def _source_feature_name(
    transformed_name: str,
    artifacts: DatasetArtifacts,
    source_names: list[str],
) -> str:
    # Keep the full transformed name in FeatureContribution.feature_name.
    unprefixed = transformed_name.rsplit("__", 1)[-1]

    if unprefixed in source_names:
        return unprefixed

    missing_sources = {
        str(name)
        for name in artifacts.feature_metadata.get("missing_indicator_features", [])
    }
    indicator_prefix = "missingindicator_"
    if unprefixed.startswith(indicator_prefix):
        candidate = unprefixed[len(indicator_prefix):]
        if candidate in missing_sources:
            return candidate

    one_hot_sources = {
        str(name)
        for name in artifacts.feature_metadata.get("one_hot_features", [])
    }
    matches = [
        name for name in one_hot_sources if unprefixed.startswith(f"{name}_")
    ]
    if matches:
        return max(matches, key=len)

    # The handoff says to preserve the transformed name when mapping is unclear.
    return transformed_name


def _positive_class_values(values: Any, feature_count: int) -> np.ndarray:
    # Older SHAP versions can return one array per class; select fraud/class 1.
    if isinstance(values, (list, tuple)):
        if not values:
            raise ValueError("SHAP returned no contribution arrays")
        values = values[1] if len(values) > 1 else values[0]

    array = np.asarray(values, dtype=float)

    if array.ndim == 3:
        # Current multi-output shape: samples x features x outputs.
        if array.shape[0] == 1 and array.shape[1] == feature_count:
            output_index = 1 if array.shape[2] > 1 else 0
            array = array[0, :, output_index]
        # Also accept outputs x samples x features.
        elif array.shape[1] == 1 and array.shape[2] == feature_count:
            output_index = 1 if array.shape[0] > 1 else 0
            array = array[output_index, 0, :]
        else:
            raise ValueError(f"Unexpected SHAP output shape: {array.shape}")
    elif array.ndim == 2:
        if array.shape == (1, feature_count):
            array = array[0]
        elif array.shape == (2, feature_count):
            array = array[1]
        else:
            raise ValueError(f"Unexpected SHAP output shape: {array.shape}")

    if array.ndim != 1 or array.size != feature_count:
        raise ValueError(f"Unexpected SHAP output shape: {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("SHAP returned a non-finite contribution")

    return array


def _positive_class_base_value(value: Any) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError("SHAP returned no base value")
    result = float(array[1] if array.size > 1 else array[0])
    if not np.isfinite(result):
        raise ValueError("SHAP returned a non-finite base value")
    return result


def _format_reason_value(value: Any) -> str | None:
    """Format an observed input value without inventing a missing value."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None

    rendered = repr(value) if isinstance(value, str) else str(value)
    if len(rendered) > 80:
        rendered = f"{rendered[:77]}..."
    return rendered


def _reason_for_contribution(contribution: FeatureContribution) -> str:
    """Describe a local model contribution using only observed evidence."""
    if contribution.source_feature == contribution.feature_name and "__" in contribution.feature_name:
        feature_label = f"Model feature {contribution.feature_name!r}"
    else:
        feature_label = f"Feature {contribution.source_feature!r}"

    value = _format_reason_value(contribution.value)
    value_text = (
        f" (observed value {value})"
        if value is not None
        else " (observed input value unavailable)"
    )

    effect_text = {
        "increases_risk": "contributed to a higher model fraud-risk score",
        "decreases_risk": "contributed to a lower model fraud-risk score",
        "no_effect": "had no measurable directional effect on the model score",
    }.get(contribution.direction)
    if effect_text is None:
        raise ValueError(
            f"Unsupported contribution direction: {contribution.direction!r}"
        )

    return (
        f"{feature_label}{value_text} {effect_text} "
        f"(SHAP value {contribution.shap_value:+.4g})."
    )


def _plain_language_reasons(
    contributions: tuple[FeatureContribution, ...],
) -> tuple[str, ...]:
    """Create rank-ordered audit text without assigning semantics to features."""
    ranked = sorted(contributions, key=lambda contribution: contribution.rank)
    return tuple(_reason_for_contribution(item) for item in ranked)


def explain_transaction(
    dataset: str,
    transaction: Mapping[str, Any] | pd.Series | pd.DataFrame,
    *,
    artifacts: DatasetArtifacts | None = None,
    max_features: int = 8,
    root: Path = ROOT,
) -> ExplanationResult:
    """Explain one prediction with TreeSHAP on the selected tree model.

    ``base_value`` and ``output_value`` use TreeSHAP's raw model-output scale.
    ``fraud_probability`` is the separate probability returned by inference.
    The TreeSHAP additive output is checked against LightGBM's raw score with
    ``rtol=1e-5`` and ``atol=1e-5``.
    """
    name = _normalise_dataset(dataset)
    if isinstance(max_features, bool) or not isinstance(max_features, (int, np.integer)):
        raise TypeError("max_features must be a positive integer")
    if max_features < 1:
        raise ValueError("max_features must be at least 1")

    bundle = artifacts or load_dataset_artifacts(name, root=root)
    if bundle.dataset != name:
        raise ValueError(f"Artifacts are for {bundle.dataset!r}, not {name!r}")

    frame = _as_single_row(transaction)

    # Use the shared inference implementation for the displayed probability.
    prediction = predict_transaction(
        name,
        frame,
        artifacts=bundle,
        root=root,
    )

    transformed = transform_transaction(frame, bundle)
    if len(transformed.shape) != 2 or transformed.shape[0] != 1:
        raise ValueError(
            "The preprocessor must return exactly one transformed transaction"
        )

    feature_count = int(transformed.shape[1])
    feature_names = _transformed_feature_names(bundle, feature_count)
    source_names = _input_feature_names(bundle)

    # The input is one row, so densifying a sparse matrix is small and lets
    # TreeExplainer receive a conventional NumPy matrix.
    shap_input = (
        transformed.toarray()
        if sparse.issparse(transformed)
        else np.asarray(transformed)
    )

    tree_model = getattr(bundle.model, "booster_", None)
    if tree_model is None:
        tree_model = bundle.model

    explainer = shap.TreeExplainer(
        tree_model,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )
    shap_values = _positive_class_values(
        explainer.shap_values(shap_input),
        feature_count,
    )
    base_value = _positive_class_base_value(explainer.expected_value)
    output_value = float(base_value + shap_values.sum())

    if not np.isfinite(output_value):
        raise ValueError("SHAP produced a non-finite output value")

    # The selected production models are LightGBM. Compare TreeSHAP's additive
    # output to the same booster in raw-score mode, while leaving fixture or
    # alternate tree estimators usable when they do not expose a Booster API.
    if tree_model is not bundle.model and callable(getattr(tree_model, "predict", None)):
        raw_output = np.asarray(
            tree_model.predict(shap_input, raw_score=True),
            dtype=float,
        ).reshape(-1)
        if raw_output.size != 1 or not np.isfinite(raw_output[0]):
            raise ValueError("The tree model returned an invalid raw output")
        if not np.isclose(
            output_value,
            float(raw_output[0]),
            rtol=ADDITIVITY_RTOL,
            atol=ADDITIVITY_ATOL,
        ):
            raise ValueError(
                "TreeSHAP contributions do not add up to the model raw output "
                f"within rtol={ADDITIVITY_RTOL} and atol={ADDITIVITY_ATOL}"
            )

    raw_values = frame.iloc[0].to_dict()
    ranked_indices = sorted(
        range(feature_count),
        key=lambda index: (-abs(float(shap_values[index])), index),
    )

    contributions = []
    for rank, index in enumerate(ranked_indices[:max_features], start=1):
        feature_name = feature_names[index]
        source_feature = _source_feature_name(
            feature_name,
            bundle,
            source_names,
        )
        shap_value = float(shap_values[index])

        direction = (
            "increases_risk"
            if shap_value > 0
            else "decreases_risk"
            if shap_value < 0
            else "no_effect"
        )
        value = raw_values.get(source_feature)
        if isinstance(value, np.generic):
            value = value.item()

        contributions.append(
            FeatureContribution(
                feature_name=feature_name,
                source_feature=source_feature,
                value=value,
                shap_value=shap_value,
                direction=direction,
                rank=rank,
            )
        )

    return ExplanationResult(
        dataset=name,
        fraud_probability=prediction.fraud_probability,
        base_value=base_value,
        output_value=output_value,
        contributions=tuple(contributions),
        plain_language_reasons=_plain_language_reasons(tuple(contributions)),
    )

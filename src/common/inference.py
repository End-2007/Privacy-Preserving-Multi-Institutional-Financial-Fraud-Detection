"""Shared single-transaction inference for the Micro Project dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from src.common.config import DATASET_CONFIG, ROOT


SUPPORTED_DATASETS = tuple(sorted(DATASET_CONFIG))


@dataclass(frozen=True)
class PredictionResult:
    """Stable result contract consumed by Streamlit and the SHAP layer."""

    dataset: str
    fraud_probability: float
    predicted_label: int
    threshold: float
    model_name: str


@dataclass
class DatasetArtifacts:
    """Loaded model bundle for one supported dataset."""

    dataset: str
    model: Any
    preprocessor: Any
    metadata: dict[str, Any]
    feature_metadata: dict[str, Any]


def _normalise_dataset(dataset: str) -> str:
    name = str(dataset).strip().lower()
    if name not in DATASET_CONFIG:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset {dataset!r}; choose one of: {supported}")
    return name


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required metadata file was not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _repair_legacy_sklearn_state(value: Any, seen: set[int] | None = None) -> None:
    """Repair a known 1.3 -> 1.9 SimpleImputer private-state rename.

    The transfer bundle contains preprocessors written by scikit-learn 1.3
    while the selected model bundle was written by 1.9. Current sklearn can
    load the old object but expects ``_fill_dtype`` during transform; 1.3
    stored the equivalent state as ``_fit_dtype``.
    """

    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))

    if value.__class__.__name__ == "SimpleImputer":
        if hasattr(value, "_fit_dtype") and not hasattr(value, "_fill_dtype"):
            value._fill_dtype = value._fit_dtype

    if isinstance(value, dict):
        for child in value.values():
            _repair_legacy_sklearn_state(child, seen)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _repair_legacy_sklearn_state(child, seen)
    elif hasattr(value, "__dict__"):
        for child in vars(value).values():
            _repair_legacy_sklearn_state(child, seen)


def load_dataset_artifacts(dataset: str, *, root: Path = ROOT) -> DatasetArtifacts:
    """Load the selected model, preprocessor, and metadata from the local bundle."""

    name = _normalise_dataset(dataset)
    model_dir = root / "models" / name
    stage_dir = model_dir / "stage1"
    model_path = stage_dir / "best_model.joblib"
    metadata_path = stage_dir / "metadata.json"
    preprocessor_path = model_dir / "preprocessor.joblib"
    feature_metadata_path = root / "data" / "processed" / name / "feature_metadata.json"

    required = (model_path, preprocessor_path, metadata_path, feature_metadata_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inference artifacts:\n" + "\n".join(missing))

    metadata = _read_json(metadata_path)
    feature_metadata = _read_json(feature_metadata_path)
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    _repair_legacy_sklearn_state(preprocessor)

    if metadata.get("dataset") != name:
        raise ValueError(
            f"Model metadata identifies dataset {metadata.get('dataset')!r}, expected {name!r}"
        )
    if "threshold" not in metadata:
        raise ValueError(f"Model metadata has no threshold: {metadata_path}")

    return DatasetArtifacts(
        dataset=name,
        model=model,
        preprocessor=preprocessor,
        metadata=metadata,
        feature_metadata=feature_metadata,
    )


def _as_single_row(transaction: Mapping[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(transaction, pd.DataFrame):
        frame = transaction.copy()
    elif isinstance(transaction, pd.Series):
        frame = transaction.to_frame().T
    elif isinstance(transaction, Mapping):
        frame = pd.DataFrame([dict(transaction)])
    else:
        raise TypeError("transaction must be a mapping, pandas Series, or one-row DataFrame")

    if len(frame) != 1:
        raise ValueError("Exactly one transaction is required for single-transaction inference")
    return frame


def _feature_columns(artifacts: DatasetArtifacts) -> list[str]:
    names = getattr(artifacts.preprocessor, "feature_names_in_", None)
    if names is not None:
        return [str(name) for name in names]

    metadata = artifacts.feature_metadata
    groups = (
        metadata.get("numeric_features", []),
        metadata.get("frequency_encoded_features", []),
        metadata.get("one_hot_features", []),
    )
    columns = [column for group in groups for column in group]
    if not columns:
        raise ValueError(f"No input feature columns found for {artifacts.dataset}")
    return columns


def transform_transaction(transaction: Mapping[str, Any] | pd.Series | pd.DataFrame, artifacts: DatasetArtifacts) -> pd.DataFrame:
    """Validate and transform one raw transaction using the fitted preprocessor."""

    frame = _as_single_row(transaction)
    feature_columns = _feature_columns(artifacts)
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing required feature columns for {artifacts.dataset}: {', '.join(missing[:10])}"
        )

    features = frame[feature_columns].copy()
    categorical = set(artifacts.feature_metadata.get("frequency_encoded_features", []))
    categorical.update(artifacts.feature_metadata.get("one_hot_features", []))
    for column in categorical.intersection(features.columns):
        features[column] = features[column].astype(object).where(features[column].notna(), np.nan)
    transformed = artifacts.preprocessor.transform(features)
    if sparse.issparse(transformed):
        return transformed.tocsr().astype(np.float32)
    return np.asarray(transformed, dtype=np.float32)


def predict_transaction(
    dataset: str,
    transaction: Mapping[str, Any] | pd.Series | pd.DataFrame,
    *,
    artifacts: DatasetArtifacts | None = None,
    root: Path = ROOT,
) -> PredictionResult:
    """Return a thresholded fraud prediction for one raw transaction."""

    name = _normalise_dataset(dataset)
    bundle = artifacts or load_dataset_artifacts(name, root=root)
    if bundle.dataset != name:
        raise ValueError(f"Artifacts are for {bundle.dataset!r}, not {name!r}")

    transformed = transform_transaction(transaction, bundle)
    # The transferred LightGBM sklearn wrapper can crash inside its
    # predict_proba adapter when loaded across supported dependency versions,
    # while the underlying Booster remains compatible and gives the same
    # positive-class probability. Keep the generic predict_proba path for all
    # other estimators and for the test doubles used by this module.
    booster = getattr(bundle.model, "booster_", None)
    if booster is not None:
        probability = float(np.asarray(booster.predict(transformed)).reshape(-1)[0])
    else:
        probabilities = np.asarray(bundle.model.predict_proba(transformed))
        if probabilities.ndim != 2 or probabilities.shape[0] != 1 or probabilities.shape[1] < 2:
            raise ValueError("The model must return a two-class predict_proba result")
        probability = float(probabilities[0, 1])

    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Model returned invalid fraud probability: {probability}")

    threshold = float(bundle.metadata["threshold"])
    return PredictionResult(
        dataset=name,
        fraud_probability=probability,
        predicted_label=int(probability >= threshold),
        threshold=threshold,
        model_name=str(bundle.metadata.get("model", type(bundle.model).__name__)),
    )



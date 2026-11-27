"""Load the reviewed feature lists used for Model A and Model B runs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.common.config import DATASET_CONFIG, ROOT

FEATURE_SET_NAMES = ("full", "auth")

FEATURE_LIST_FILES = {
    ("paysim", "full"): ROOT / "configs" / "features" / "PaySim_full.txt",
    ("paysim", "auth"): ROOT / "configs" / "features" / "PaySim_auth.txt",
    ("ieee_cis", "full"): ROOT / "configs" / "features" / "IEEECIS_full.txt",
    ("ieee_cis", "auth"): ROOT / "configs" / "features" / "IEEECIS_auth.txt",
}

# These engineered columns are included only when every source dependency is
# included by the reviewed list. This keeps post-transaction PaySim signals out
# of the authorization-time configuration.
DERIVED_FEATURE_DEPENDENCIES = {
    "paysim": (
        ("hour", frozenset({"step"})),
        ("log_amount", frozenset({"amount"})),
        ("orig_balance_delta", frozenset({"oldbalanceOrg", "newbalanceOrig"})),
        ("dest_balance_delta", frozenset({"oldbalanceDest", "newbalanceDest"})),
        ("error_balance_orig", frozenset({"amount", "oldbalanceOrg", "newbalanceOrig"})),
        ("error_balance_dest", frozenset({"amount", "oldbalanceDest", "newbalanceDest"})),
        ("orig_zero_after", frozenset({"newbalanceOrig"})),
    ),
    "ieee_cis": (
        ("transaction_hour", frozenset({"TransactionDT"})),
        ("transaction_day", frozenset({"TransactionDT"})),
        ("log_amt", frozenset({"TransactionAmt"})),
    ),
}

FORBIDDEN_PREDICTORS = frozenset(
    {"isFraud", "isFlaggedFraud", "row_id", "TransactionID", "nameOrig", "nameDest"}
)


def feature_set_paths(dataset: str, feature_set: str | None) -> tuple[Path, Path]:
    """Return the processed-matrix and model-artifact folders for a run."""
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if feature_set is None:
        return DATASET_CONFIG[dataset]["processed_dir"], DATASET_CONFIG[dataset]["model_dir"]
    if feature_set not in FEATURE_SET_NAMES:
        raise ValueError(
            f"Unsupported feature set {feature_set!r}; choose one of {FEATURE_SET_NAMES}."
        )
    return (
        DATASET_CONFIG[dataset]["processed_dir"] / "feature_sets" / feature_set,
        DATASET_CONFIG[dataset]["model_dir"] / "feature_sets" / feature_set,
    )


def feature_list_path(dataset: str, feature_set: str) -> Path:
    """Return the reviewed source-feature list for one dataset/configuration."""
    key = (dataset, feature_set)
    try:
        path = FEATURE_LIST_FILES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset/feature-set pair: {key}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Feature list was not found: {path}")
    return path


def load_feature_list(dataset: str, feature_set: str) -> list[str]:
    """Read a newline-delimited feature list, rejecting unsafe or ambiguous entries."""
    path = feature_list_path(dataset, feature_set)
    features = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not features:
        raise ValueError(f"Feature list is empty: {path}")
    seen = set()
    duplicates = set()
    for name in features:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    duplicates = sorted(duplicates)
    if duplicates:
        raise ValueError(f"Duplicate feature names in {path}: {', '.join(duplicates)}")
    forbidden = sorted(FORBIDDEN_PREDICTORS.intersection(features))
    if forbidden:
        raise ValueError(
            f"Feature list {path} contains target/identifier/leakage columns: "
            + ", ".join(forbidden)
        )
    return features


def resolve_feature_columns(
    dataset: str,
    feature_set: str,
    available_columns: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Resolve listed source fields plus safe, available engineered derivatives.

    Returns ``(model_columns, derived_columns)`` in deterministic order. A
    missing configured field is an error; the list is never silently weakened.
    """
    configured = load_feature_list(dataset, feature_set)
    if dataset not in DERIVED_FEATURE_DEPENDENCIES:
        raise ValueError(f"Unsupported dataset: {dataset}")
    available = set(available_columns)
    missing = [column for column in configured if column not in available]
    if missing:
        preview = ", ".join(missing[:12])
        raise ValueError(
            f"Prepared {dataset} data is missing {len(missing)} configured "
            f"{feature_set} feature(s): {preview}. Re-run scripts/prepare_stage1.py "
            "with the current source schema."
        )

    configured_set = set(configured)
    derived = [
        name
        for name, dependencies in DERIVED_FEATURE_DEPENDENCIES[dataset]
        if dependencies.issubset(configured_set)
    ]
    missing_derived = [name for name in derived if name not in available]
    if missing_derived:
        raise ValueError(
            f"Prepared {dataset} data is missing engineered feature(s): "
            + ", ".join(missing_derived)
            + ". Re-run scripts/prepare_stage1.py."
        )
    resolved = configured + derived
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"Resolved {dataset}/{feature_set} feature list contains duplicates.")
    return resolved, derived

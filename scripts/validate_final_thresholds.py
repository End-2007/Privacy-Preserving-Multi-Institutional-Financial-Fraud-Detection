"""Recompute Stage 1 final-model metrics on validation splits only.

This audit deliberately opens no test split. It checks the saved LightGBM
threshold and the validation snapshot in each dataset's stage1 metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse

from src.common.config import FPR_CEILING
from src.common.evaluate import compute_metrics
from src.common.pipeline_utils import save_json


DATASETS = ("paysim", "ieee_cis")
METRIC_KEYS = (
    "threshold", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr",
    "tn", "fp", "fn", "tp",
)


def _load_matrix(path: Path):
    matrix = sparse.load_npz(path)
    return matrix


def audit_dataset(artifact_root: Path, dataset: str) -> dict:
    """Audit one saved final model against its validation-only snapshot."""
    model_dir = artifact_root / "models" / dataset / "stage1"
    processed_dir = artifact_root / "data" / "processed" / dataset
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_path = model_dir / "best_model.joblib"
    model = joblib.load(model_path)

    # Explicit validation filenames: this function never reads X_test/y_test.
    x_validation = _load_matrix(processed_dir / "X_validation.npz")
    y_validation = np.load(processed_dir / "y_validation.npy", allow_pickle=False)
    probabilities = model.predict_proba(x_validation)[:, 1]
    threshold = float(metadata["threshold"])
    measured = compute_metrics(y_validation, probabilities, threshold)
    expected = metadata.get("validation_metrics")
    if not isinstance(expected, dict):
        raise ValueError(f"{dataset}: metadata has no validation_metrics snapshot")

    differences = {}
    for key in METRIC_KEYS:
        if key not in expected:
            differences[key] = "missing from saved validation snapshot"
            continue
        actual_value = measured[key]
        expected_value = expected[key]
        if key in {"tn", "fp", "fn", "tp"}:
            matches = int(actual_value) == int(expected_value)
        else:
            matches = bool(np.isclose(float(actual_value), float(expected_value), rtol=0, atol=1e-10))
        if not matches:
            differences[key] = {"recomputed": actual_value, "saved": expected_value}

    passed = not differences and measured["fpr"] <= FPR_CEILING + 1e-12
    return {
        "dataset": dataset,
        "model": metadata.get("model"),
        "model_path": str(Path("models") / dataset / "stage1" / "best_model.joblib"),
        "validation_features_path": str(Path("data") / "processed" / dataset / "X_validation.npz"),
        "validation_labels_path": str(Path("data") / "processed" / dataset / "y_validation.npy"),
        "test_split_accessed": False,
        "threshold_rule": metadata.get("threshold_rule"),
        "fpr_ceiling": FPR_CEILING,
        "validation_rows": int(y_validation.size),
        "validation_positive_rows": int(np.count_nonzero(y_validation == 1)),
        "metrics": measured,
        "matches_saved_validation_snapshot": not differences,
        "differences": differences,
        "passed": passed,
    }


def run_audit(artifact_root: Path) -> dict:
    results = [audit_dataset(artifact_root, dataset) for dataset in DATASETS]
    return {
        "audit": "final_threshold_validation_only",
        "test_data_opened": False,
        "passed": all(result["passed"] for result in results),
        "datasets": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root", type=Path, default=Path(__file__).resolve().parents[1],
        help="Root containing models/ and data/processed/ (defaults to this repo).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="JSON output path (defaults to reports/final_threshold_validation.json).",
    )
    args = parser.parse_args()
    report = run_audit(args.artifact_root.resolve())
    output = args.output or (Path(__file__).resolve().parents[1] / "reports" / "final_threshold_validation.json")
    save_json(output, report)
    for result in report["datasets"]:
        metrics = result["metrics"]
        print(
            f"{result['dataset']}: {'PASS' if result['passed'] else 'FAIL'}; "
            f"threshold={metrics['threshold']:.12g}, precision={metrics['precision']:.6f}, "
            f"recall={metrics['recall']:.6f}, F1={metrics['f1']:.6f}, "
            f"ROC-AUC={metrics['roc_auc']:.6f}, Average Precision={metrics['pr_auc']:.6f}, "
            f"FPR={metrics['fpr']:.8f}"
        )
        if result["differences"]:
            print(json.dumps(result["differences"], indent=2))
    print(f"Report: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

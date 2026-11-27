"""Search LightGBM configurations for IEEE-CIS full and auth feature sets.

For each feature set, this runs the Stage 1 LightGBM tuning budget under each
IEEE-CIS strategy, selects the configuration on validation AP, then evaluates
the selected model once on the held-out test split.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import ParameterSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import DATASET_CONFIG, FPR_CEILING, SEED, TUNING_BUDGETS  # noqa: E402
from src.common.evaluate import compute_metrics, pick_threshold  # noqa: E402
from src.common.feature_sets import feature_list_path, feature_set_paths  # noqa: E402
from src.common.models import build_estimator, parameter_space  # noqa: E402
from src.common.pipeline_utils import plot_pr_curves, save_artifacts, save_json  # noqa: E402
from src.common.resampling import resample_training_data  # noqa: E402


def _package_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm")
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _load_split(feature_set: str, split: str):
    processed_dir, _ = feature_set_paths("ieee_cis", feature_set)
    matrix_path = processed_dir / f"X_{split}.npz"
    labels_path = processed_dir / f"y_{split}.npy"
    if not matrix_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(
            f"Missing prepared IEEE-CIS {feature_set}/{split} matrix or labels: "
            f"{matrix_path} / {labels_path}. Run scripts/prepare_feature_sets.py first."
        )
    X = sparse.load_npz(matrix_path).tocsr().astype(np.float32, copy=False)
    y = np.load(labels_path, allow_pickle=False).astype(np.int8, copy=False)
    if X.shape[0] != y.size:
        raise ValueError(
            f"IEEE-CIS {feature_set}/{split} row mismatch: {X.shape[0]} != {y.size}"
        )
    return X, y


def train_feature_set(feature_set: str, candidate_params: list[dict]) -> dict:
    processed_dir, model_dir = feature_set_paths("ieee_cis", feature_set)
    X_train, y_train = _load_split(feature_set, "train")
    X_validation, y_validation = _load_split(feature_set, "validation")
    strategies = tuple(DATASET_CONFIG["ieee_cis"]["resampling_strategies"])
    total_fits = len(strategies) * len(candidate_params)
    fit_number = 0
    candidate_results = []
    comparison = []
    curves = {}
    selected = None
    search_started = time.perf_counter()
    print(
        f"Starting IEEE-CIS/{feature_set} LightGBM: {total_fits} fits across "
        f"{strategies}; {X_train.shape[0]:,} training rows, {X_train.shape[1]:,} "
        "features, CPU threads=all available.",
        flush=True,
    )

    for strategy in strategies:
        print(f"Preparing training strategy={strategy}", flush=True)
        strategy_started = time.perf_counter()
        X_fit, y_fit, resampling = resample_training_data(
            X_train,
            y_train,
            dataset="ieee_cis",
            strategy=strategy,
            seed=SEED,
        )
        class_ratio = float(np.count_nonzero(y_fit == 0) / np.count_nonzero(y_fit == 1))
        print(
            f"Prepared {strategy} in {(time.perf_counter() - strategy_started) / 60:.1f} min; "
            f"rows={X_fit.shape[0]:,}, class counts={resampling['after']}",
            flush=True,
        )
        best_for_strategy = None

        for candidate_index, params in enumerate(candidate_params, start=1):
            fit_number += 1
            estimator = build_estimator(
                "lightgbm",
                params,
                strategy=strategy,
                class_ratio=class_ratio,
                seed=SEED,
            )
            fit_started = time.perf_counter()
            print(
                f"[{fit_number}/{total_fits}] fitting {strategy} candidate "
                f"{candidate_index}/{len(candidate_params)}",
                flush=True,
            )
            estimator.fit(X_fit, y_fit)
            validation_scores = estimator.predict_proba(X_validation)[:, 1]
            validation_metrics = compute_metrics(y_validation, validation_scores)
            fit_seconds = time.perf_counter() - fit_started
            row = {
                "model": "lightgbm",
                "strategy": strategy,
                "candidate": candidate_index,
                "params": params,
                "validation": validation_metrics,
                "resampling": resampling,
                "compute_device": "cpu",
                "fit_seconds": float(fit_seconds),
            }
            candidate_results.append(row)
            print(
                f"[{fit_number}/{total_fits}] LightGBM/{strategy} candidate "
                f"{candidate_index}/{len(candidate_params)}: validation AP="
                f"{validation_metrics['pr_auc']:.6f}, fit={fit_seconds / 60:.2f} min",
                flush=True,
            )

            if (
                best_for_strategy is None
                or validation_metrics["pr_auc"] > best_for_strategy["validation"]["pr_auc"]
            ):
                best_for_strategy = {
                    **row,
                    "estimator": estimator,
                    "validation_scores": validation_scores,
                }
            else:
                del estimator
            del validation_scores

        comparison.append(
            {key: value for key, value in best_for_strategy.items()
             if key not in ("estimator", "validation_scores")}
        )
        precision, recall, _ = precision_recall_curve(
            y_validation, best_for_strategy["validation_scores"]
        )
        curves[strategy] = (
            precision,
            recall,
            best_for_strategy["validation"]["pr_auc"],
        )
        if (
            selected is None
            or best_for_strategy["validation"]["pr_auc"] > selected["validation"]["pr_auc"]
        ):
            selected = best_for_strategy
        del X_fit, y_fit, best_for_strategy

    search_seconds = time.perf_counter() - search_started
    if selected is None:
        raise RuntimeError(f"No LightGBM candidate completed for IEEE-CIS/{feature_set}.")

    threshold = pick_threshold(
        y_validation,
        selected["validation_scores"],
        fpr_ceiling=FPR_CEILING,
    )
    validation_metrics = compute_metrics(
        y_validation, selected["validation_scores"], threshold
    )

    # Test data is first loaded after the model and threshold are fixed on validation.
    X_test, y_test = _load_split(feature_set, "test")
    test_scores = selected["estimator"].predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_scores, threshold)

    feature_metadata_path = processed_dir / "feature_metadata.json"
    metadata = {
        "dataset": "ieee_cis",
        "feature_set": feature_set,
        "feature_list": str(feature_list_path("ieee_cis", feature_set).relative_to(ROOT)),
        "model": "lightgbm",
        "strategy": selected["strategy"],
        "seed": SEED,
        "params": selected["params"],
        "hyperparameter_search": True,
        "tuning_budget": len(candidate_params),
        "strategies": list(strategies),
        "search_candidate_fits": total_fits,
        "search_seconds": float(search_seconds),
        "training_rows": int(X_train.shape[0]),
        "training_features_after_preprocessing": int(X_train.shape[1]),
        "fit_seconds": float(selected["fit_seconds"]),
        "threshold_rule": f"maximize recall subject to FPR <= {FPR_CEILING}",
        "threshold": threshold,
        "validation_metric": "average_precision",
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "resampling_counts": selected["resampling"],
        "preprocessor": str((model_dir / "preprocessor.joblib").relative_to(ROOT)),
        "feature_metadata": str(feature_metadata_path.relative_to(ROOT)),
        "compute_device": "cpu",
        "compute_plan": "LightGBM CPU using all available logical threads",
        "package_versions": _package_versions(),
        "evaluated_test_once": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    artifact_dir = model_dir / "stage1"
    save_artifacts(selected["estimator"], metadata, artifact_dir)

    report = {
        **metadata,
        "comparison": comparison,
        "candidate_search": candidate_results,
        "artifact": str((artifact_dir / "best_model.joblib").relative_to(ROOT)),
    }
    report_path = ROOT / "reports" / f"ieee_cis_{feature_set}_metrics.json"
    save_json(report_path, report)

    curve_path = ROOT / "reports" / "figures" / f"ieee_cis_{feature_set}_lightgbm_pr.png"
    plot_pr_curves(
        {
            f"IEEE-CIS {feature_set} LightGBM/{strategy}": curve
            for strategy, curve in curves.items()
        },
        curve_path,
    )
    print(
        f"Completed IEEE-CIS/{feature_set}: selected {selected['strategy']} "
        f"candidate {selected['candidate']}, validation AP={validation_metrics['pr_auc']:.6f}, "
        f"test AP={test_metrics['pr_auc']:.6f}, test FPR={test_metrics['fpr']:.6f}, "
        f"search={search_seconds / 60:.1f} min. Report: {report_path}",
        flush=True,
    )
    return report


def main() -> None:
    candidate_params = list(
        ParameterSampler(
            parameter_space("lightgbm"),
            n_iter=TUNING_BUDGETS["lightgbm"],
            random_state=SEED,
        )
    )
    for feature_set in ("full", "auth"):
        train_feature_set(feature_set, candidate_params)


if __name__ == "__main__":
    main()

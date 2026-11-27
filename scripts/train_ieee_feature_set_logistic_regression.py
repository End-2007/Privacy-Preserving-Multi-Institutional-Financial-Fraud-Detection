"""Tune IEEE-CIS Logistic Regression on the full and auth feature sets.

For each feature set, evaluate the configured Logistic Regression budget under
each IEEE-CIS training strategy. Select by validation average precision, choose
the decision threshold on validation, and evaluate the selected estimator once
on the held-out test split.
"""

from __future__ import annotations

import sys
import shutil
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import DATASET_CONFIG, FPR_CEILING, SEED, TUNING_BUDGETS  # noqa: E402
from src.common.evaluate import compute_metrics, pick_threshold  # noqa: E402
from src.common.feature_sets import feature_list_path, feature_set_paths  # noqa: E402
from src.common.models import build_estimator, parameter_space  # noqa: E402
from src.common.pipeline_utils import plot_pr_curves, save_artifacts, save_json  # noqa: E402
from src.common.resampling import resample_training_data  # noqa: E402


def _package_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn")
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


def _fit_candidate(
    candidate_index: int,
    params: dict,
    strategy: str,
    class_ratio: float,
    resampling: dict,
    X_fit,
    y_fit,
    X_validation,
    y_validation,
) -> tuple[dict, object, np.ndarray]:
    """Fit one independent candidate in a worker process."""
    estimator = build_estimator(
        "logistic_regression",
        params,
        strategy=strategy,
        class_ratio=class_ratio,
        seed=SEED,
    )
    fit_started = time.perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as fit_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(X_fit, y_fit)
    convergence_warnings = [
        str(item.message)
        for item in fit_warnings
        if issubclass(item.category, ConvergenceWarning)
    ]
    with threadpool_limits(limits=1):
        validation_scores = estimator.predict_proba(X_validation)[:, 1]
    validation_metrics = compute_metrics(y_validation, validation_scores)
    fit_seconds = time.perf_counter() - fit_started
    row = {
        "model": "logistic_regression",
        "strategy": strategy,
        "candidate": candidate_index,
        "params": params,
        "validation": validation_metrics,
        "resampling": resampling,
        "compute_device": "cpu",
        "fit_seconds": float(fit_seconds),
        "iterations": np.asarray(estimator.n_iter_).astype(int).tolist(),
        "convergence_warnings": convergence_warnings,
    }
    return row, estimator, validation_scores


def train_feature_set(feature_set: str, candidate_params: list[dict]) -> dict:
    processed_dir, model_dir = feature_set_paths("ieee_cis", feature_set)
    X_train_raw, y_train = _load_split(feature_set, "train")
    X_validation_raw, y_validation = _load_split(feature_set, "validation")

    # Fit the sparse-compatible scaler on training rows only, matching Stage 1.
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train_raw).tocsr().astype(np.float32, copy=False)
    X_validation = scaler.transform(X_validation_raw).tocsr().astype(np.float32, copy=False)
    del X_validation_raw

    strategies = tuple(DATASET_CONFIG["ieee_cis"]["resampling_strategies"])
    total_fits = len(strategies) * len(candidate_params)
    fit_number = 0
    candidate_results = []
    comparison = []
    curves = {}
    selected = None
    search_started = time.perf_counter()
    checkpoint_root = (
        ROOT / ".venv" / ".cache" / f"ieee_cis_{feature_set}_logistic_regression"
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Starting IEEE-CIS/{feature_set} Logistic Regression: {total_fits} fits across "
        f"{strategies}; {X_train.shape[0]:,} training rows, {X_train.shape[1]:,} "
        "features; CPU/liblinear.",
        flush=True,
    )

    for strategy in strategies:
        print(f"Preparing training strategy={strategy}", flush=True)
        strategy_started = time.perf_counter()
        X_fit_raw, y_fit, resampling = resample_training_data(
            X_train_raw,
            y_train,
            dataset="ieee_cis",
            strategy=strategy,
            seed=SEED,
        )
        # Resample raw training rows; transform the result with the scaler
        # fitted above on the original training split only.
        X_fit = scaler.transform(X_fit_raw).tocsr().astype(np.float32, copy=False)
        del X_fit_raw
        class_ratio = float(np.count_nonzero(y_fit == 0) / np.count_nonzero(y_fit == 1))
        print(
            f"Prepared {strategy} in {(time.perf_counter() - strategy_started) / 60:.2f} min; "
            f"rows={X_fit.shape[0]:,}, class counts={resampling['after']}",
            flush=True,
        )
        strategy_checkpoint = checkpoint_root / strategy
        strategy_checkpoint.mkdir(parents=True, exist_ok=True)
        cached_rows = {}
        missing_candidates = []
        for candidate_index, params in enumerate(candidate_params, start=1):
            row_path = strategy_checkpoint / f"candidate_{candidate_index}.json"
            model_path = strategy_checkpoint / f"candidate_{candidate_index}.joblib"
            scores_path = strategy_checkpoint / f"candidate_{candidate_index}_validation.npy"
            if row_path.is_file() and model_path.is_file() and scores_path.is_file():
                import json

                row = json.loads(row_path.read_text(encoding="utf-8"))
                cached_rows[candidate_index] = row
                print(
                    f"Resuming cached {strategy} candidate {candidate_index}: "
                    f"validation AP={row['validation']['pr_auc']:.6f}",
                    flush=True,
                )
            else:
                missing_candidates.append((candidate_index, params))

        fit_number += len(cached_rows)
        for candidate_index, _ in missing_candidates:
            print(
                f"Queueing {strategy} candidate "
                f"{candidate_index}/{len(candidate_params)} with liblinear",
                flush=True,
            )

        if missing_candidates:
            worker_count = min(len(missing_candidates), len(candidate_params))
            print(
                f"Running {len(missing_candidates)} candidate(s) in parallel with "
                f"{worker_count} CPU worker processes.",
                flush=True,
            )
            fitted_candidates = Parallel(
                n_jobs=worker_count,
                backend="loky",
                verbose=10,
                max_nbytes="16M",
                return_as="generator_unordered",
            )(
                delayed(_fit_candidate)(
                    candidate_index,
                    params,
                    strategy,
                    class_ratio,
                    resampling,
                    X_fit,
                    y_fit,
                    X_validation,
                    y_validation,
                )
                for candidate_index, params in missing_candidates
            )
            completed_in_strategy = 0
            for row, estimator, validation_scores in fitted_candidates:
                candidate_index = row["candidate"]
                model_path = strategy_checkpoint / f"candidate_{candidate_index}.joblib"
                scores_path = strategy_checkpoint / f"candidate_{candidate_index}_validation.npy"
                row_path = strategy_checkpoint / f"candidate_{candidate_index}.json"
                joblib.dump(estimator, model_path, compress=3)
                np.save(scores_path, validation_scores)
                save_json(row_path, row)
                cached_rows[candidate_index] = row
                completed_in_strategy += 1
                print(
                    f"[{fit_number + completed_in_strategy}/{total_fits}] LogisticRegression/"
                    f"{strategy} candidate {candidate_index}/{len(candidate_params)}: "
                    f"validation AP={row['validation']['pr_auc']:.6f}, "
                    f"fit={row['fit_seconds'] / 60:.2f} min, iterations={row['iterations']}, "
                    f"convergence_warnings={len(row['convergence_warnings'])}",
                    flush=True,
                )
            fit_number += completed_in_strategy

        strategy_rows = [cached_rows[i] for i in range(1, len(candidate_params) + 1)]
        candidate_results.extend(strategy_rows)
        best_row = max(strategy_rows, key=lambda row: row["validation"]["pr_auc"])
        best_index = best_row["candidate"]
        best_model_path = strategy_checkpoint / f"candidate_{best_index}.joblib"
        best_scores_path = strategy_checkpoint / f"candidate_{best_index}_validation.npy"
        best_for_strategy = {
            **best_row,
            "estimator": joblib.load(best_model_path),
            "validation_scores": np.load(best_scores_path, allow_pickle=False),
        }

        comparison.append(best_row)
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
        raise RuntimeError(f"No Logistic Regression candidate completed for IEEE-CIS/{feature_set}.")

    threshold = pick_threshold(
        y_validation,
        selected["validation_scores"],
        fpr_ceiling=FPR_CEILING,
    )
    validation_metrics = compute_metrics(
        y_validation, selected["validation_scores"], threshold
    )

    # Load the held-out test split only after validation selection is complete.
    X_test_raw, y_test = _load_split(feature_set, "test")
    X_test = scaler.transform(X_test_raw).tocsr().astype(np.float32, copy=False)
    test_scores = selected["estimator"].predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_scores, threshold)

    artifact_dir = model_dir / "logistic_regression" / "stage1"
    scaler_path = artifact_dir / "lr_scaler.joblib"
    metadata = {
        "dataset": "ieee_cis",
        "feature_set": feature_set,
        "feature_list": str(feature_list_path("ieee_cis", feature_set).relative_to(ROOT)),
        "model": "logistic_regression",
        "selected_candidate": selected["candidate"],
        "strategy": selected["strategy"],
        "seed": SEED,
        "params": selected["params"],
        "hyperparameter_search": True,
        "tuning_budget_per_strategy": len(candidate_params),
        "strategies": list(strategies),
        "search_candidate_fits": total_fits,
        "search_seconds": float(search_seconds),
        "training_rows": int(X_train.shape[0]),
        "training_features_after_preprocessing": int(X_train.shape[1]),
        "fit_seconds": float(selected["fit_seconds"]),
        "iterations": selected["iterations"],
        "convergence_warnings": selected["convergence_warnings"],
        "scaler": "StandardScaler(with_mean=False), fitted on training rows only",
        "lr_scaler": str(scaler_path.relative_to(ROOT)),
        "threshold_rule": f"maximize recall subject to FPR <= {FPR_CEILING}",
        "threshold": threshold,
        "validation_metric": "average_precision",
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "resampling_counts": selected["resampling"],
        "feature_metadata": str((processed_dir / "feature_metadata.json").relative_to(ROOT)),
        "compute_device": "cpu",
        "compute_plan": "Logistic Regression with liblinear; three parallel CPU candidate workers",
        "package_versions": _package_versions(),
        "evaluated_test_once": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    save_artifacts(selected["estimator"], metadata, artifact_dir)
    joblib.dump(scaler, scaler_path, compress=3)
    report = {
        **metadata,
        "comparison": comparison,
        "candidate_search": candidate_results,
        "artifact": str((artifact_dir / "best_model.joblib").relative_to(ROOT)),
    }
    report_path = ROOT / "reports" / f"ieee_cis_{feature_set}_logistic_regression_metrics.json"
    save_json(report_path, report)

    curve_path = ROOT / "reports" / "figures" / f"ieee_cis_{feature_set}_logistic_regression_pr.png"
    plot_pr_curves(
        {
            f"IEEE-CIS {feature_set} Logistic Regression/{strategy}": curve
            for strategy, curve in curves.items()
        },
        curve_path,
    )
    print(
        f"Completed IEEE-CIS/{feature_set}: selected {selected['strategy']} "
        f"candidate {selected['candidate']}, validation AP={validation_metrics['pr_auc']:.6f}, "
        f"test AP={test_metrics['pr_auc']:.6f}, test FPR={test_metrics['fpr']:.6f}, "
        f"search={search_seconds / 60:.2f} min. Report: {report_path}",
        flush=True,
    )
    # Only remove this run's checkpoint directory after its final report and
    # model artifacts have been written successfully.
    if checkpoint_root.resolve().is_relative_to((ROOT / ".venv" / ".cache").resolve()):
        shutil.rmtree(checkpoint_root, ignore_errors=True)
    return report


def main() -> None:
    candidate_params = list(
        ParameterSampler(
            parameter_space("logistic_regression"),
            n_iter=TUNING_BUDGETS["logistic_regression"],
            random_state=SEED,
        )
    )
    for feature_set in ("full", "auth"):
        train_feature_set(feature_set, candidate_params)


if __name__ == "__main__":
    main()

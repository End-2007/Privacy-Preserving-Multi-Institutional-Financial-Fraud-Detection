"""Explicit Stage 1 training entry point for PaySim and IEEE-CIS.

This file only defines the pipeline when imported. Training begins only when
the user runs this script from the command line.
"""

from __future__ import annotations

import argparse
import json
import joblib
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import DATASET_CONFIG, FPR_CEILING, SEED, TUNING_BUDGETS  # noqa: E402
from src.common.evaluate import compute_metrics, pick_threshold  # noqa: E402
from src.common.models import build_estimator, parameter_space  # noqa: E402
from src.common.pipeline_utils import plot_pr_curves, save_artifacts, save_json  # noqa: E402
from src.common.resampling import resample_training_data  # noqa: E402
from src.common.tracking import finish_run, init_run, log_metrics  # noqa: E402

MODEL_NAMES = tuple(TUNING_BUDGETS)


def load_split(dataset: str, split: str, *, logistic: bool = False):
    data_dir = DATASET_CONFIG[dataset]["processed_dir"]
    matrix_name = f"X_lr_{split}.npz" if logistic else f"X_{split}.npz"
    matrix_path = data_dir / matrix_name
    label_path = data_dir / f"y_{split}.npy"
    if not matrix_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"Missing prepared {dataset} {split} matrix or labels; run scripts/prepare_stage1.py first.")
    X = sparse.load_npz(matrix_path).tocsr().astype(np.float32, copy=False)
    y = np.load(label_path, allow_pickle=False).astype(np.int8, copy=False)
    if X.shape[0] != y.size:
        raise ValueError(f"Matrix/label row mismatch for {dataset} {split}: {X.shape[0]} != {y.size}")
    return X, y


def package_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "xgboost", "lightgbm")
    found = {}
    for package in packages:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = "not-installed"
    return found


def preferred_xgboost_device() -> str:
    """Use CUDA when the installed XGBoost build includes it; otherwise use CPU."""
    import xgboost

    return "cuda" if xgboost.build_info().get("USE_CUDA", False) else "cpu"


def run_dataset(dataset: str, *, strategies: tuple[str, ...] | None = None, track: bool = True) -> dict:
    """Run the predeclared search, select on validation, and touch test once."""
    cfg = DATASET_CONFIG[dataset]
    enabled_strategies = strategies or tuple(cfg["resampling_strategies"])
    if not enabled_strategies:
        raise ValueError("At least one resampling strategy must be selected.")
    invalid = set(enabled_strategies) - set(cfg["resampling_strategies"])
    if invalid:
        raise ValueError(f"Strategies not enabled for {dataset}: {sorted(invalid)}")

    # Test remains unopened until the single validation-selected configuration is fixed.
    y_validation = np.load(cfg["processed_dir"] / "y_validation.npy", allow_pickle=False).astype(np.int8, copy=False)
    result_rows = []
    curves = {}
    experiment_config = {
        "dataset": dataset,
        "seed": SEED,
        "validation_metric": "pr_auc",
        "threshold_rule": f"maximize recall subject to FPR <= {FPR_CEILING}",
        "strategies": list(enabled_strategies),
        "tuning_budgets": TUNING_BUDGETS,
        "package_versions": package_versions(),
        "compute_plan": {
            "xgboost": preferred_xgboost_device(),
            "random_forest": "cpu, all logical threads",
            "lightgbm": "cpu, all logical threads (installed build has no GPU entry point)",
            "logistic_regression": "cpu",
        },
    }
    run = init_run(
        project="fedguard-stage1",
        name=f"{dataset}-stage1-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        config=experiment_config,
        enabled=track,
    )
    xgb_device = preferred_xgboost_device()
    total_fits = sum(TUNING_BUDGETS[name] for name in MODEL_NAMES) * len(enabled_strategies)
    fit_number = 0
    print(
        f"Starting {dataset}: {total_fits} candidate fits across {enabled_strategies}; "
        f"XGBoost device={xgb_device}; CPU estimators use all available threads.",
        flush=True,
    )
    try:
        X_train_tree, y_train = load_split(dataset, "train")
        X_validation_tree, y_validation_check = load_split(dataset, "validation")
        if not np.array_equal(y_validation, y_validation_check):
            raise ValueError(f"Validation labels changed while loading {dataset} matrices.")
        # Refit the LR-only scale on the already-prepared train matrix using
        # the active environment. This avoids loading a scaler pickled under a
        # different scikit-learn version and preserves the train-only boundary.
        lr_scaler = StandardScaler(with_mean=False)
        X_train_lr = lr_scaler.fit_transform(X_train_tree).tocsr().astype(np.float32, copy=False)
        X_validation_lr = lr_scaler.transform(X_validation_tree).tocsr().astype(np.float32, copy=False)
        sparse.save_npz(cfg["processed_dir"] / "X_lr_train.npz", X_train_lr, compressed=True)
        sparse.save_npz(cfg["processed_dir"] / "X_lr_validation.npz", X_validation_lr, compressed=True)
        joblib.dump(lr_scaler, cfg["model_dir"] / "lr_scaler.joblib", compress=3)

        for strategy in enabled_strategies:
            print(f"Preparing training strategy={strategy} once for all models", flush=True)
            resample_started = time.perf_counter()
            X_fit_tree, y_fit, resampling = resample_training_data(
                X_train_tree,
                y_train,
                dataset=dataset,
                strategy=strategy,
                seed=SEED,
            )
            if strategy == "class_weight":
                X_fit_lr = X_train_lr
            else:
                X_fit_lr = lr_scaler.transform(X_fit_tree).tocsr().astype(np.float32, copy=False)
            print(
                f"Prepared {strategy} in {(time.perf_counter() - resample_started) / 60:.1f} min; "
                f"class counts={resampling['after']}",
                flush=True,
            )
            ratio = float(np.count_nonzero(y_fit == 0) / np.count_nonzero(y_fit == 1))
            for model_name in MODEL_NAMES:
                logistic = model_name == "logistic_regression"
                X_fit = X_fit_lr if logistic else X_fit_tree
                X_validation = X_validation_lr if logistic else X_validation_tree
                ratio = float(np.count_nonzero(y_fit == 0) / np.count_nonzero(y_fit == 1))
                candidates = list(
                    ParameterSampler(
                        parameter_space(model_name),
                        n_iter=TUNING_BUDGETS[model_name],
                        random_state=SEED,
                    )
                )
                best_for_pair = None
                for candidate_index, params in enumerate(candidates, start=1):
                    fit_number += 1
                    fit_started = time.perf_counter()
                    print(
                        f"[{fit_number}/{total_fits}] fitting {model_name}/{strategy} "
                        f"candidate {candidate_index}/{len(candidates)}",
                        flush=True,
                    )
                    model = build_estimator(
                        model_name,
                        params,
                        strategy=strategy,
                        class_ratio=ratio,
                        seed=SEED,
                        xgb_device=xgb_device,
                    )
                    try:
                        model.fit(X_fit, y_fit)
                    except Exception as error:
                        if model_name != "xgboost" or xgb_device != "cuda":
                            raise
                        from xgboost.core import XGBoostError

                        if not isinstance(error, XGBoostError):
                            raise
                        warnings.warn(
                            f"XGBoost CUDA fit failed ({error}); switching remaining XGBoost candidates to CPU.",
                            RuntimeWarning,
                        )
                        xgb_device = "cpu"
                        model = build_estimator(
                            model_name,
                            params,
                            strategy=strategy,
                            class_ratio=ratio,
                            seed=SEED,
                            xgb_device=xgb_device,
                        )
                        model.fit(X_fit, y_fit)
                    val_scores = model.predict_proba(X_validation)[:, 1]
                    val_metrics = compute_metrics(y_validation, val_scores)
                    fit_seconds = time.perf_counter() - fit_started
                    print(
                        f"[{fit_number}/{total_fits}] {model_name}/{strategy} candidate "
                        f"{candidate_index}/{len(candidates)}: validation PR-AUC="
                        f"{val_metrics['pr_auc']:.6f}, elapsed={fit_seconds / 60:.1f} min, "
                        f"device={xgb_device if model_name == 'xgboost' else 'cpu'}",
                        flush=True,
                    )
                    row = {
                        "model": model_name,
                        "strategy": strategy,
                        "candidate": candidate_index,
                        "params": params,
                        "validation": val_metrics,
                        "resampling": resampling,
                        "compute_device": xgb_device if model_name == "xgboost" else "cpu",
                    }
                    log_metrics(
                        {
                            "validation/pr_auc": val_metrics["pr_auc"],
                            "validation/roc_auc": val_metrics["roc_auc"],
                            "candidate": candidate_index,
                            "model": model_name,
                            "strategy": strategy,
                            "compute_device": row["compute_device"],
                        }
                    )
                    if best_for_pair is None or val_metrics["pr_auc"] > best_for_pair["validation"]["pr_auc"]:
                        best_for_pair = {**row, "estimator": model, "validation_scores": val_scores}
                    del model, val_scores

                result_rows.append({key: value for key, value in best_for_pair.items() if key not in ("estimator", "validation_scores")})
                p, r, _ = precision_recall_curve(y_validation, best_for_pair["validation_scores"])
                curves[f"{model_name}/{strategy}"] = (p, r, best_for_pair["validation"]["pr_auc"])
                if result_rows[-1]["validation"]["pr_auc"] > (
                    max((entry["validation"]["pr_auc"] for entry in result_rows[:-1]), default=-1.0)
                ):
                    selected = best_for_pair
                del X_fit, best_for_pair
            del X_fit_tree, X_fit_lr, y_fit
        del X_train_tree, X_train_lr, X_validation_tree, X_validation_lr, y_train, y_validation_check

        threshold = pick_threshold(
            y_validation,
            selected["validation_scores"],
            fpr_ceiling=FPR_CEILING,
        )
        selected_validation = compute_metrics(y_validation, selected["validation_scores"], threshold)
        X_test_tree, y_test = load_split(dataset, "test")
        if selected["model"] == "logistic_regression":
            X_test = lr_scaler.transform(X_test_tree).tocsr().astype(np.float32, copy=False)
            sparse.save_npz(cfg["processed_dir"] / "X_lr_test.npz", X_test, compressed=True)
        else:
            X_test = X_test_tree
        test_scores = selected["estimator"].predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, test_scores, threshold)

        metadata = {
            **experiment_config,
            "model": selected["model"],
            "strategy": selected["strategy"],
            "compute_device": selected["compute_device"],
            "params": selected["params"],
            "threshold": threshold,
            "validation_metrics": selected_validation,
            "test_metrics": test_metrics,
            "resampling_counts": selected["resampling"],
            "preprocessor": str((cfg["model_dir"] / "preprocessor.joblib").relative_to(ROOT)),
            "lr_scaler": str((cfg["model_dir"] / "lr_scaler.joblib").relative_to(ROOT)),
            "evaluated_test_once": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        save_artifacts(selected["estimator"], metadata, cfg["model_dir"] / "stage1")
        report = {
            **metadata,
            "comparison": result_rows,
        }
        report_path = ROOT / "reports" / f"{dataset}_metrics.json"
        save_json(report_path, report)
        plot_pr_curves(curves, ROOT / "reports" / "figures" / f"{dataset}_pr_curves.png")
        log_metrics({f"test/{key}": value for key, value in test_metrics.items()})
        return report
    finally:
        finish_run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_CONFIG), required=True)
    parser.add_argument("--strategies", nargs="+", help="Defaults to that dataset's required Stage 1 strategies.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable local W&B offline tracking.")
    args = parser.parse_args()
    result = run_dataset(args.dataset, strategies=tuple(args.strategies) if args.strategies else None, track=not args.no_wandb)
    print(json.dumps({"dataset": args.dataset, "selected_model": result["model"], "selected_strategy": result["strategy"], "metrics_file": str(ROOT / "reports" / f"{args.dataset}_metrics.json")}, indent=2))


if __name__ == "__main__":
    main()

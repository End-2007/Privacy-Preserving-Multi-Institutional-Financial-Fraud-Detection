"""Finalize reproducible Model A/B results for all model families.

This script reuses saved validation-selected IEEE-CIS models and PaySim
LightGBM models. PaySim's other model families are refit using their already
recorded best validation candidates because the earlier family search did not
save those estimators or per-family test predictions. It never selects a
model or threshold using test results.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import DATASET_CONFIG, FPR_CEILING, SEED  # noqa: E402
from src.common.evaluate import compute_metrics, pick_threshold  # noqa: E402
from src.common.feature_sets import feature_list_path, feature_set_paths  # noqa: E402
from src.common.models import build_estimator  # noqa: E402
from src.common.resampling import resample_training_data  # noqa: E402

DATASETS = ("paysim", "ieee_cis")
FEATURE_SETS = ("full", "auth")
MODELS = ("logistic_regression", "random_forest", "xgboost", "lightgbm")
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}
RESULTS_DIR = ROOT / "reports" / "feature_set_results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
IMPORTANCE_DIR = RESULTS_DIR / "feature_importance"
FIGURES_DIR = RESULTS_DIR / "figures"
CONFIG_DIR = ROOT / "configs" / "experiments" / "feature_set_comparison"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _package_versions() -> dict[str, str]:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "xgboost", "lightgbm")
    result = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _relative_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def _report_for(dataset: str, feature_set: str, model: str) -> dict:
    if dataset == "paysim":
        return _read_json(ROOT / "reports" / f"paysim_{feature_set}_metrics.json")
    filename = (
        f"ieee_cis_{feature_set}_metrics.json"
        if model == "lightgbm"
        else f"ieee_cis_{feature_set}_{model}_metrics.json"
    )
    return _read_json(ROOT / "reports" / filename)


def _best_paysim_family_candidate(report: dict, model: str) -> dict:
    rows = [row for row in report.get("comparison", []) if row.get("model") == model]
    if not rows:
        raise ValueError(f"No PaySim validation-search rows for {model}.")
    return max(rows, key=lambda row: float(row["validation"]["pr_auc"]))


def _load_split(data_dir: Path, split: str, *, logistic: bool = False):
    suffix = "lr_" if logistic else ""
    X = sparse.load_npz(data_dir / f"X_{suffix}{split}.npz").tocsr().astype(np.float32, copy=False)
    y = np.load(data_dir / f"y_{split}.npy", allow_pickle=False).astype(np.int8, copy=False)
    if X.shape[0] != len(y):
        raise ValueError(f"{split} matrix and label rows differ: {X.shape[0]} != {len(y)}")
    return X, y


def _row_ids(dataset: str, split: str, expected_labels: np.ndarray) -> np.ndarray:
    base_dir = DATASET_CONFIG[dataset]["processed_dir"]
    frame = pd.read_parquet(base_dir / f"{split}.parquet")
    if dataset == "paysim":
        id_col = "row_id"
        target_col = "isFraud"
    else:
        id_col = "TransactionID"
        target_col = "isFraud"
    if id_col not in frame or target_col not in frame:
        raise ValueError(f"{dataset}/{split} parquet lacks {id_col} or {target_col}.")
    labels = frame[target_col].to_numpy(dtype=np.int8, copy=False)
    if not np.array_equal(labels, expected_labels):
        raise ValueError(f"Prepared {dataset}/{split} parquet labels are out of alignment with feature-set arrays.")
    return frame[id_col].to_numpy(copy=True)


def _load_family_artifact(dataset: str, feature_set: str, model: str, report: dict):
    model_root = feature_set_paths(dataset, feature_set)[1]
    if dataset == "paysim" and model != "lightgbm":
        candidate = _best_paysim_family_candidate(report, model)
        artifact_dir = model_root / model / "stage1"
        metadata_path = artifact_dir / "metadata.json"
        model_path = artifact_dir / "best_model.joblib"
        if metadata_path.is_file() and model_path.is_file():
            metadata = _read_json(metadata_path)
            if metadata.get("dataset") == dataset and metadata.get("feature_set") == feature_set and metadata.get("model") == model:
                return joblib.load(model_path), metadata, candidate
        return None, None, candidate

    if dataset == "paysim":
        artifact_dir = model_root / "stage1"
    elif model == "lightgbm":
        artifact_dir = model_root / "stage1"
    else:
        artifact_dir = model_root / model / "stage1"
    metadata_path = artifact_dir / "metadata.json"
    model_path = artifact_dir / "best_model.joblib"
    if not metadata_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"Saved selected {dataset}/{feature_set}/{model} model is missing under {artifact_dir}.")
    metadata = _read_json(metadata_path)
    if metadata.get("dataset") != dataset or metadata.get("feature_set") != feature_set or metadata.get("model") != model:
        raise ValueError(f"Saved artifact metadata does not match {dataset}/{feature_set}/{model}: {metadata_path}")
    if report.get("threshold") is not None and not math.isclose(
        float(metadata["threshold"]), float(report["threshold"]), rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError(f"Saved threshold differs from its result report for {dataset}/{feature_set}/{model}.")
    return joblib.load(model_path), metadata, None


def _fit_paysim_family(
    model_name: str,
    candidate: dict,
    *,
    dataset: str,
    model_dir: Path,
    X_train_tree,
    y_train: np.ndarray,
    lr_scaler,
):
    strategy = candidate["strategy"]
    X_fit_tree, y_fit, resampling = resample_training_data(
        X_train_tree, y_train, dataset=dataset, strategy=strategy, seed=SEED
    )
    X_fit = lr_scaler.transform(X_fit_tree).tocsr().astype(np.float32, copy=False) if model_name == "logistic_regression" else X_fit_tree
    class_ratio = float(np.count_nonzero(y_fit == 0) / np.count_nonzero(y_fit == 1))
    device = candidate.get("compute_device", "cpu")
    estimator = build_estimator(
        model_name,
        candidate["params"],
        strategy=strategy,
        class_ratio=class_ratio,
        seed=SEED,
        xgb_device=device,
    )
    started = time.perf_counter()
    try:
        estimator.fit(X_fit, y_fit)
    except Exception as exc:
        if model_name != "xgboost" or device != "cuda":
            raise
        from xgboost.core import XGBoostError

        if not isinstance(exc, XGBoostError):
            raise
        warnings.warn(f"CUDA fit failed for PaySim {model_name}; retrying the same candidate on CPU: {exc}")
        device = "cpu"
        estimator = build_estimator(
            model_name,
            candidate["params"],
            strategy=strategy,
            class_ratio=class_ratio,
            seed=SEED,
            xgb_device=device,
        )
        estimator.fit(X_fit, y_fit)
    fit_seconds = time.perf_counter() - started
    return estimator, resampling, device, fit_seconds


def _source_feature(transformed_name: str) -> str:
    group, separator, name = transformed_name.partition("__")
    if not separator:
        return transformed_name
    if group == "categorical":
        # One-hot names are source-column plus category, e.g. type_TRANSFER.
        for source in ("ProductCD", "DeviceType", "P_emaildomain", "R_emaildomain", "card4", "card6", "type"):
            if name.startswith(source + "_"):
                return source
        return name.split("_", 1)[0]
    return name


def _feature_importance(model, model_name: str, feature_names: list[str]) -> pd.DataFrame:
    if model_name == "logistic_regression":
        signed = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
        values = np.abs(signed)
        method = "absolute coefficient on train-scaled features"
    else:
        values = np.asarray(model.feature_importances_, dtype=np.float64).reshape(-1)
        signed = np.full(values.shape, np.nan, dtype=np.float64)
        method = f"native {model_name} feature_importances_"
    if len(values) != len(feature_names):
        raise ValueError(f"Importance count {len(values)} does not match transformed feature count {len(feature_names)}.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Invalid feature importance values for {model_name}.")
    total = float(values.sum())
    normalized = values / total if total else np.zeros_like(values)
    frame = pd.DataFrame({
        "transformed_feature": feature_names,
        "source_feature": [_source_feature(name) for name in feature_names],
        "importance": values,
        "normalized_importance": normalized,
        "signed_coefficient": signed,
        "importance_method": method,
    })
    return frame.sort_values("normalized_importance", ascending=False, kind="mergesort").reset_index(drop=True).assign(
        rank=lambda data: np.arange(1, len(data) + 1)
    )


def _write_prediction_file(path: Path, records: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("row_id", "true_label", "predicted_probability", "predicted_label", "split"))
        for split, ids, labels, scores, threshold in records:
            predicted = (scores >= threshold).astype(np.int8)
            for row_id, label, score, pred in zip(ids, labels, scores, predicted, strict=True):
                writer.writerow((row_id, int(label), f"{float(score):.10g}", int(pred), split))


def _plot_confusion(path: Path, title: str, validation: dict, test: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, name, metrics in zip(axes, ("Validation", "Held-out test"), (validation, test), strict=True):
        matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
        ConfusionMatrixDisplay(matrix, display_labels=("Legitimate", "Fraud")).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format=",d"
        )
        ax.set_title(f"{name}\nFPR={metrics['fpr']:.3%}  Recall={metrics['recall']:.3%}")
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_pr(path: Path, title: str, validation_labels, validation_scores, test_labels, test_scores) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for split, labels, scores in (
        ("Validation", validation_labels, validation_scores),
        ("Held-out test", test_labels, test_scores),
    ):
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = compute_metrics(labels, scores)["pr_auc"]
        ax.plot(recall, precision, label=f"{split} AP={ap:.4f}", linewidth=2)
    ax.set(xlabel="Recall", ylabel="Precision", title=title, xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.22)
    ax.legend(loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _validate_shared_splits() -> None:
    for dataset in DATASETS:
        reference = None
        reference_labels = {}
        for feature_set in FEATURE_SETS:
            data_dir, _ = feature_set_paths(dataset, feature_set)
            manifest = _read_json(data_dir / "split_manifest.json")
            current = manifest["splits"]
            if reference is None:
                reference = current
            else:
                for split in ("train", "validation", "test"):
                    if current[split]["rows"] != reference[split]["rows"] or current[split]["fraud_rows"] != reference[split]["fraud_rows"]:
                        raise ValueError(f"{dataset} full/auth splits differ in {split} row counts or labels.")
            for split in ("train", "validation", "test"):
                _, y = _load_split(data_dir, split)
                if y.size != current[split]["rows"] or int(y.sum()) != current[split]["fraud_rows"]:
                    raise ValueError(f"{dataset}/{feature_set}/{split} arrays do not match split manifest.")
                if feature_set == FEATURE_SETS[0]:
                    reference_labels[split] = y.copy()
                elif not np.array_equal(reference_labels[split], y):
                    raise ValueError(f"{dataset} full/auth row labels are not aligned in {split}.")
        print(f"Verified shared {dataset} full/auth split sizes and class counts.", flush=True)


def _run_configuration(dataset: str, feature_set: str, model_name: str) -> dict:
    started = time.perf_counter()
    data_dir, model_root = feature_set_paths(dataset, feature_set)
    report = _report_for(dataset, feature_set, model_name)
    existing_model, existing_meta, candidate = _load_family_artifact(dataset, feature_set, model_name, report)
    logistic = model_name == "logistic_regression"
    X_train, y_train = _load_split(data_dir, "train", logistic=logistic)
    X_validation, y_validation = _load_split(data_dir, "validation", logistic=logistic)
    X_test, y_test = _load_split(data_dir, "test", logistic=logistic)
    X_train_tree = None
    lr_scaler = None
    fit_seconds = 0.0

    if dataset == "paysim" and model_name != "lightgbm" and existing_model is None:
        # Model-family validation search already selected candidate params.
        # Refit only those params on train, then threshold on validation.
        X_train_tree, _ = _load_split(data_dir, "train", logistic=False)
        model_dir = model_root
        lr_scaler_path = model_dir / "lr_scaler.joblib"
        lr_scaler = joblib.load(lr_scaler_path)
        existing_model, resampling, device, fit_seconds = _fit_paysim_family(
            model_name,
            candidate,
            dataset=dataset,
            model_dir=model_dir,
            X_train_tree=X_train_tree,
            y_train=y_train,
            lr_scaler=lr_scaler,
        )
        params = candidate["params"]
        strategy = candidate["strategy"]
        validation_scores = existing_model.predict_proba(X_validation)[:, 1]
        threshold = pick_threshold(y_validation, validation_scores, fpr_ceiling=FPR_CEILING)
        artifact_dir = model_root / model_name / "stage1"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(existing_model, artifact_dir / "best_model.joblib", compress=3)
        if model_name == "logistic_regression":
            joblib.dump(lr_scaler, artifact_dir / "lr_scaler.joblib", compress=3)
    else:
        params = (existing_meta or {}).get("params", report.get("params", {}))
        strategy = (existing_meta or {}).get("strategy", report.get("strategy"))
        device = (existing_meta or {}).get("compute_device", report.get("compute_device", "cpu"))
        if dataset == "paysim":
            artifact_dir = model_root / "stage1" if model_name == "lightgbm" else model_root / model_name / "stage1"
        elif model_name == "lightgbm":
            artifact_dir = model_root / "stage1"
        else:
            artifact_dir = _relative_path(report["artifact"]).parent
        if logistic:
            scaler_path = artifact_dir / "lr_scaler.joblib"
            if not scaler_path.is_file():
                scaler_path = model_root / "lr_scaler.joblib"
            lr_scaler = joblib.load(scaler_path)
        validation_scores = existing_model.predict_proba(X_validation)[:, 1]
        threshold = float((existing_meta or {}).get("threshold", report["threshold"]))
        expected_validation = (existing_meta or {}).get("validation_metrics", report.get("validation_metrics", {}))
        if expected_validation:
            expected_ap = float(expected_validation["pr_auc"])
            observed_ap = compute_metrics(y_validation, validation_scores)["pr_auc"]
            if not math.isclose(observed_ap, expected_ap, rel_tol=2e-3, abs_tol=2e-5):
                raise ValueError(
                    f"Saved validation AP does not reproduce for {dataset}/{feature_set}/{model_name}: "
                    f"{observed_ap:.8f} != {expected_ap:.8f}"
                )
        resampling = (existing_meta or {}).get("resampling_counts", report.get("resampling_counts", {}))

    validation_metrics = compute_metrics(y_validation, validation_scores, threshold)
    if dataset == "paysim" and model_name == "lightgbm":
        # The stored selected model is already the validation-selected LightGBM.
        historical = report.get("validation_metrics", {})
        if historical and not math.isclose(validation_metrics["pr_auc"], historical["pr_auc"], rel_tol=2e-3, abs_tol=2e-5):
            raise ValueError(f"PaySim {feature_set} LightGBM validation AP failed reproducibility check.")

    # Score the held-out test partition only after model, strategy and threshold are fixed.
    test_scores = existing_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_scores, threshold)
    historical_test = (
        {} if dataset == "paysim" and model_name != "lightgbm"
        else (existing_meta or report).get("test_metrics", {})
    )
    if historical_test and not math.isclose(
        test_metrics["pr_auc"], float(historical_test["pr_auc"]), rel_tol=2e-3, abs_tol=2e-5
    ):
        raise ValueError(
            f"Held-out test AP did not reproduce for {dataset}/{feature_set}/{model_name}: "
            f"{test_metrics['pr_auc']:.8f} != {float(historical_test['pr_auc']):.8f}"
        )

    base_feature_meta = _read_json(data_dir / "feature_metadata.json")
    feature_names = [str(name) for name in base_feature_meta["transformed_feature_names"]]
    importance = _feature_importance(existing_model, model_name, feature_names)
    importance_path = IMPORTANCE_DIR / f"{dataset}_{feature_set}_{model_name}.csv"
    importance.to_csv(importance_path, index=False, float_format="%.10g")

    prediction_path = PREDICTIONS_DIR / f"{dataset}_{feature_set}_{model_name}_predictions.csv.gz"
    prediction_records = []
    for split, labels, scores in (("validation", y_validation, validation_scores), ("test", y_test, test_scores)):
        ids = _row_ids(dataset, split, labels)
        prediction_records.append((split, ids, labels, scores, threshold))
    _write_prediction_file(prediction_path, prediction_records)

    metrics_path = METRICS_DIR / f"{dataset}_{feature_set}_{model_name}_metrics.json"
    matrix_path = FIGURES_DIR / f"{dataset}_{feature_set}_{model_name}_confusion.png"
    pr_path = FIGURES_DIR / f"{dataset}_{feature_set}_{model_name}_pr.png"
    _plot_confusion(matrix_path, f"{dataset.upper()} | {feature_set.upper()} | {MODEL_LABELS[model_name]}", validation_metrics, test_metrics)
    _plot_pr(pr_path, f"{dataset.upper()} | {feature_set.upper()} | {MODEL_LABELS[model_name]}", y_validation, validation_scores, y_test, test_scores)

    feature_list = feature_list_path(dataset, feature_set)
    config = {
        "dataset": dataset,
        "model": model_name,
        "feature_set": feature_set,
        "feature_list": str(feature_list.relative_to(ROOT)),
        "split_manifest": str((data_dir / "split_manifest.json").relative_to(ROOT)),
        "feature_metadata": str((data_dir / "feature_metadata.json").relative_to(ROOT)),
        "preprocessor": str((model_root / "preprocessor.joblib").relative_to(ROOT)),
        "lr_scaler": (
            str((artifact_dir / "lr_scaler.joblib").relative_to(ROOT))
            if logistic and (artifact_dir / "lr_scaler.joblib").is_file()
            else str((model_root / "lr_scaler.joblib").relative_to(ROOT))
            if logistic and (model_root / "lr_scaler.joblib").is_file()
            else None
        ),
        "seed": SEED,
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15, "stratified": True},
        "selection_metric": "validation Average Precision (AP)",
        "model_candidate_selection": "highest validation AP among the recorded candidates for this model family",
        "threshold_rule": f"maximize validation recall subject to validation FPR <= {FPR_CEILING}",
        "threshold_selected_on": "validation",
        "final_metrics_split": "test",
        "strategy": strategy,
        "params": params,
        "search_candidate": candidate.get("candidate") if candidate else (existing_meta or {}).get("selected_candidate"),
        "search_validation_ap": float(candidate["validation"]["pr_auc"]) if candidate else (existing_meta or {}).get("validation_metrics", {}).get("pr_auc"),
        "compute_device": device,
        "train_only_resampling": True,
        "preprocessing_fit_split": "train",
        "package_versions": _package_versions(),
        "source_search_report": str((ROOT / "reports" / f"{dataset}_{feature_set}_metrics.json").relative_to(ROOT)) if dataset == "paysim" else str((ROOT / "reports" / f"ieee_cis_{feature_set}_{model_name}_metrics.json").relative_to(ROOT)) if model_name != "lightgbm" else str((ROOT / "reports" / f"ieee_cis_{feature_set}_metrics.json").relative_to(ROOT)),
        "artifact": str((artifact_dir / "best_model.joblib").relative_to(ROOT)),
        "prediction_file": str(prediction_path.relative_to(ROOT)),
        "feature_importance_file": str(importance_path.relative_to(ROOT)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    config_path = CONFIG_DIR / f"{dataset}_{feature_set}_{model_name}.json"
    _write_json(config_path, config)
    result = {
        **config,
        "threshold": float(threshold),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "resampling_counts": resampling,
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_validation)),
        "test_rows": int(len(y_test)),
        "fit_seconds": float(fit_seconds),
        "search_validation_ap": float(candidate["validation"]["pr_auc"]) if candidate else (existing_meta or {}).get("validation_metrics", {}).get("pr_auc"),
        "test_scored_once_for_this_export": True,
        "metrics_file": str(metrics_path.relative_to(ROOT)),
        "confusion_figure": str(matrix_path.relative_to(ROOT)),
        "pr_figure": str(pr_path.relative_to(ROOT)),
    }
    _write_json(metrics_path, result)
    if dataset == "paysim" and model_name != "lightgbm":
        artifact_metadata = {
            **config,
            "finalized_export": True,
            "threshold": float(threshold),
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "resampling_counts": resampling,
            "evaluated_test_once": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(artifact_dir / "metadata.json", artifact_metadata)
    print(
        f"Completed {dataset}/{feature_set}/{model_name}: val AP={validation_metrics['pr_auc']:.6f}, "
        f"test AP={test_metrics['pr_auc']:.6f}, test FPR={test_metrics['fpr']:.3%}; "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return result


def _write_comparison_outputs(results: list[dict]) -> None:
    frame = pd.DataFrame([
        {
            "Dataset": row["dataset"],
            "Model": MODEL_LABELS[row["model"]],
            "Model key": row["model"],
            "Feature set": "Model A (Full)" if row["feature_set"] == "full" else "Model B (Authorization-time)",
            "Strategy": row["strategy"],
            "Threshold": row["threshold"],
            "Validation AP": row["validation_metrics"]["pr_auc"],
            "Test AP": row["test_metrics"]["pr_auc"],
            "Test ROC-AUC": row["test_metrics"]["roc_auc"],
            "Test Precision": row["test_metrics"]["precision"],
            "Test Recall": row["test_metrics"]["recall"],
            "Test F1": row["test_metrics"]["f1"],
            "Test FPR": row["test_metrics"]["fpr"],
            "Test TN": row["test_metrics"]["tn"],
            "Test FP": row["test_metrics"]["fp"],
            "Test FN": row["test_metrics"]["fn"],
            "Test TP": row["test_metrics"]["tp"],
            "Threshold split": "Validation",
            "Final evaluation split": "Test",
        }
        for row in results
    ]).sort_values(["Dataset", "Model key", "Feature set"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_DIR / "all_configurations_metrics.csv", index=False, float_format="%.10g")

    pivot_rows = []
    narratives = []
    for dataset in DATASETS:
        narratives.append(f"## {dataset.replace('_', '-').upper()}\n")
        narratives.append("Held-out Average Precision (AP) compares the exact feature lists recorded in the run configs. It is a dataset-specific result and does not imply production performance.\n")
        subset = frame[frame["Dataset"] == dataset]
        for model in MODELS:
            pair = subset[subset["Model key"] == model].set_index("Feature set")
            full = pair.loc["Model A (Full)"]
            auth = pair.loc["Model B (Authorization-time)"]
            delta = float(auth["Test AP"] - full["Test AP"])
            pct = (delta / float(full["Test AP"]) * 100) if float(full["Test AP"]) else float("nan")
            pivot_rows.append({
                "Dataset": dataset,
                "Model": MODEL_LABELS[model],
                "Model A full test AP": full["Test AP"],
                "Model B auth test AP": auth["Test AP"],
                "Auth minus full AP": delta,
                "Relative AP change percent": pct,
                "Full test recall": full["Test Recall"],
                "Auth test recall": auth["Test Recall"],
                "Full test FPR": full["Test FPR"],
                "Auth test FPR": auth["Test FPR"],
            })
            direction = "higher" if delta > 0 else "lower" if delta < 0 else "the same"
            narratives.append(
                f"- **{MODEL_LABELS[model]}:** Model B test AP was {abs(delta):.4f} "
                f"({'+' if delta > 0 else ''}{delta:.4f}; {pct:+.1f}%) {direction} than Model A "
                f"({float(full['Test AP']):.4f} vs {float(auth['Test AP']):.4f}). Test recall was "
                f"{float(full['Test Recall']):.3%} for full and {float(auth['Test Recall']):.3%} for auth; "
                f"test FPR was {float(full['Test FPR']):.3%} and {float(auth['Test FPR']):.3%}, respectively."
            )
            importance_notes = []
            for feature_set, set_label in (("full", "full"), ("auth", "auth")):
                importance = pd.read_csv(IMPORTANCE_DIR / f"{dataset}_{feature_set}_{model}.csv").head(5)
                names = ", ".join(
                    f"{row.transformed_feature} ({row.normalized_importance:.1%})"
                    for row in importance.itertuples()
                )
                importance_notes.append(f"{set_label}: {names}")
            narratives.append(
                "  Top transformed features by each model's native importance measure — "
                f"{importance_notes[0]}; {importance_notes[1]}. Importance scales are model-specific and should not be compared across families."
            )
        narratives.append("")
    pair_frame = pd.DataFrame(pivot_rows)
    pair_frame.to_csv(RESULTS_DIR / "model_a_vs_model_b_summary.csv", index=False, float_format="%.10g")

    md = [
        "# Full and authorization-time model comparison",
        "",
        "This report compares all four model families for PaySim and IEEE-CIS under the reviewed full and authorization-time feature lists. Model selection and the operating threshold use validation data. The held-out test split is reported separately and is not used for selection.",
        "",
        "## Scope and interpretation",
        "",
        "PaySim is synthetic; its full feature set includes post-transaction balance information and its high scores should be interpreted as retrospective benchmark performance. IEEE-CIS includes anonymized and timing-uncertain fields in the full list; the authorization-time comparison follows the conservative classifications in the feature lists. The A/B differences describe these declared feature configurations and do not prove real-world availability for each field.",
        "",
        "The test predictions were exported after each model and validation threshold were fixed. Scores are model outputs and are not calibrated fraud probabilities.",
        "",
        "Feature-importance ranks are reported on transformed inputs. Logistic Regression uses absolute coefficients from train-scaled features; tree models use their native feature_importances_ measure. These scales differ across model families, so interpret importance within each family and feature set.",
        "",
        "## Per-configuration held-out results",
        "",
        "| Dataset | Model | Feature set | Strategy | Validation AP | Test AP | Test ROC-AUC | Precision | Recall | F1 | FPR | Threshold |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in frame.iterrows():
        md.append(
            f"| {row['Dataset']} | {row['Model']} | {row['Feature set']} | {row['Strategy']} | "
            f"{row['Validation AP']:.6f} | {row['Test AP']:.6f} | {row['Test ROC-AUC']:.6f} | "
            f"{row['Test Precision']:.4f} | {row['Test Recall']:.4f} | {row['Test F1']:.4f} | "
            f"{row['Test FPR']:.4%} | {row['Threshold']:.6f} |"
        )
    md.extend(["", "## Model B minus Model A observations", ""])
    md.extend(narratives)
    md.extend([
        "## Saved outputs",
        "",
        "- `all_configurations_metrics.csv`: all 16 model/dataset/feature-set results, including confusion counts.",
        "- `model_a_vs_model_b_summary.csv`: paired AP, recall, and FPR deltas.",
        "- `metrics/`: per-configuration JSON with thresholds, parameters, package versions, and output references.",
        "- `predictions/`: validation and test row predictions as compressed CSV files.",
        "- `feature_importance/`: complete transformed-feature importance rankings for each configuration.",
        "- `figures/`: validation/test confusion matrices and precision-recall curves for each configuration.",
    ])
    (RESULTS_DIR / "model_a_vs_model_b_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    _plot_importance_pairs(results)


def _plot_importance_pairs(results: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for dataset in DATASETS:
        for model in MODELS:
            fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
            for ax, feature_set, label in zip(axes, FEATURE_SETS, ("Model A | Full", "Model B | Authorization-time"), strict=True):
                path = IMPORTANCE_DIR / f"{dataset}_{feature_set}_{model}.csv"
                table = pd.read_csv(path).head(15).sort_values("normalized_importance")
                ax.barh(table["transformed_feature"], table["normalized_importance"], color="#4263eb" if feature_set == "full" else "#8b5cf6")
                ax.set_title(label)
                ax.set_xlabel("Normalized model importance")
                ax.grid(axis="x", alpha=0.2)
            fig.suptitle(f"{dataset.replace('_', '-').upper()} | {MODEL_LABELS[model]} | Top 15 transformed features")
            output = FIGURES_DIR / f"{dataset}_{model}_importance_full_vs_auth.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=170, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    for directory in (METRICS_DIR, PREDICTIONS_DIR, IMPORTANCE_DIR, FIGURES_DIR, CONFIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    _validate_shared_splits()
    results = []
    total = len(DATASETS) * len(FEATURE_SETS) * len(MODELS)
    number = 0
    for dataset in DATASETS:
        for feature_set in FEATURE_SETS:
            for model_name in MODELS:
                number += 1
                print(f"[{number}/{total}] Starting {dataset}/{feature_set}/{model_name}", flush=True)
                results.append(_run_configuration(dataset, feature_set, model_name))
    _write_comparison_outputs(results)
    print(f"Completed all {total} configurations. Outputs: {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()

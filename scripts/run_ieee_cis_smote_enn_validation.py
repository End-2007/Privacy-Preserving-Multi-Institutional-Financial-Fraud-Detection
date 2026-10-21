"""Validation-only IEEE-CIS SMOTE-ENN experiment.

Keeps the completed Stage 1 report and test split untouched. Prepares the
SMOTE-ENN training matrix once, then runs the existing 39-candidate search
budgets and saves each family's validation-best challenger model separately.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from scipy import sparse
from sklearn.model_selection import ParameterSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import DATASET_CONFIG, FPR_CEILING, SEED, TUNING_BUDGETS  # noqa: E402
from src.common.evaluate import compute_metrics, pick_threshold  # noqa: E402
from src.common.models import build_estimator, parameter_space  # noqa: E402


def counts(y: np.ndarray) -> dict[str, int]:
    return {
        "negative": int(np.count_nonzero(y == 0)),
        "positive": int(np.count_nonzero(y == 1)),
    }


def main() -> None:
    dataset = "ieee_cis"
    cfg = DATASET_CONFIG[dataset]
    data_dir = cfg["processed_dir"]
    out_dir = cfg["model_dir"] / "smote_enn_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "reports" / "ieee_cis_smote_enn_validation.json"

    started_utc = datetime.now(timezone.utc).isoformat()
    X_train = sparse.load_npz(data_dir / "X_train.npz").tocsr().astype(np.float32, copy=False)
    y_train = np.load(data_dir / "y_train.npy", allow_pickle=False).astype(np.int8, copy=False)
    X_validation = sparse.load_npz(data_dir / "X_validation.npz").tocsr().astype(np.float32, copy=False)
    y_validation = np.load(data_dir / "y_validation.npy", allow_pickle=False).astype(np.int8, copy=False)

    before = counts(y_train)
    positive_rows = np.flatnonzero(y_train == 1)
    negative_rows = np.flatnonzero(y_train == 0)
    cap = int(cfg["cap_majority_per_minority"] * positive_rows.size)
    rng = np.random.RandomState(SEED)
    if negative_rows.size > cap:
        negative_rows = rng.choice(negative_rows, size=cap, replace=False)
    selected_rows = np.concatenate((positive_rows, negative_rows))
    rng.shuffle(selected_rows)
    X_capped = X_train[selected_rows]
    y_capped = y_train[selected_rows]

    print(
        f"Preparing IEEE-CIS cap_smote_enn once: X={X_capped.shape}, "
        f"counts={counts(y_capped)}, features={X_capped.shape[1]}",
        flush=True,
    )
    resample_started = time.perf_counter()
    sampler = SMOTEENN(
        sampling_strategy=0.5,
        random_state=SEED,
        smote=SMOTE(sampling_strategy=0.5, random_state=SEED),
        enn=EditedNearestNeighbours(n_neighbors=3, n_jobs=-1),
        n_jobs=-1,
    )
    X_fit_tree, y_fit = sampler.fit_resample(X_capped, y_capped)
    if sparse.issparse(X_fit_tree):
        X_fit_tree = X_fit_tree.tocsr().astype(np.float32, copy=False)
    y_fit = np.asarray(y_fit, dtype=np.int8)
    resampling = {
        "method": "cap_smote_enn",
        "seed": SEED,
        "cap_majority_per_minority": cfg["cap_majority_per_minority"],
        "before": before,
        "after_cap": counts(y_capped),
        "after_smote_enn": counts(y_fit),
        "prepared_once_for_all_candidates": True,
    }
    print(
        f"Prepared cap_smote_enn in {(time.perf_counter() - resample_started) / 60:.1f} min; "
        f"X={X_fit_tree.shape}; counts={resampling['after_smote_enn']}",
        flush=True,
    )

    # Reuse the original train-fitted scaler so the LR challenger uses the
    # same feature scaling as the completed IEEE-CIS search.
    scaler = joblib.load(cfg["model_dir"] / "lr_scaler.joblib")
    X_fit_lr = scaler.transform(X_fit_tree).tocsr().astype(np.float32, copy=False)
    X_validation_lr = scaler.transform(X_validation).tocsr().astype(np.float32, copy=False)

    try:
        import xgboost

        xgb_device = "cuda" if xgboost.build_info().get("USE_CUDA", False) else "cpu"
    except Exception:
        xgb_device = "cpu"

    model_names = tuple(TUNING_BUDGETS)
    total_fits = sum(TUNING_BUDGETS[name] for name in model_names)
    experiment = {
        "dataset": dataset,
        "strategy": "cap_smote_enn",
        "seed": SEED,
        "validation_metric": "pr_auc",
        "test_loaded_or_evaluated": False,
        "started_utc": started_utc,
        "resampling": resampling,
        "compute_plan": {
            "cpu_estimators": "all logical threads where estimator supports n_jobs",
            "xgboost": xgb_device,
            "enn": "CPU all threads; GPU not applicable",
        },
        "tuning_budgets": TUNING_BUDGETS,
        "candidate_results": [],
        "family_best": [],
        "status": "running",
    }
    report_path.write_text(json.dumps(experiment, indent=2, sort_keys=True), encoding="utf-8")

    fit_number = 0
    for model_name in model_names:
        best_model = None
        best_candidate = None
        candidates = list(
            ParameterSampler(
                parameter_space(model_name),
                n_iter=TUNING_BUDGETS[model_name],
                random_state=SEED,
            )
        )
        X_model_fit = X_fit_lr if model_name == "logistic_regression" else X_fit_tree
        X_model_validation = X_validation_lr if model_name == "logistic_regression" else X_validation
        class_ratio = float(np.count_nonzero(y_fit == 0) / np.count_nonzero(y_fit == 1))

        for candidate_index, params in enumerate(candidates, start=1):
            fit_number += 1
            print(
                f"[{fit_number}/{total_fits}] fitting {model_name}/cap_smote_enn "
                f"candidate {candidate_index}/{len(candidates)}",
                flush=True,
            )
            fit_started = time.perf_counter()
            model = build_estimator(
                model_name,
                params,
                strategy="cap_smote_enn",
                class_ratio=class_ratio,
                seed=SEED,
                xgb_device=xgb_device,
            )
            model.fit(X_model_fit, y_fit)
            scores = model.predict_proba(X_model_validation)[:, 1]
            metrics = compute_metrics(y_validation, scores)
            row = {
                "model": model_name,
                "strategy": "cap_smote_enn",
                "candidate": candidate_index,
                "params": params,
                "compute_device": xgb_device if model_name == "xgboost" else "cpu",
                "elapsed_minutes": (time.perf_counter() - fit_started) / 60,
                "validation": metrics,
            }
            experiment["candidate_results"].append(row)
            print(
                f"[{fit_number}/{total_fits}] {model_name}/cap_smote_enn candidate "
                f"{candidate_index}/{len(candidates)}: validation PR-AUC={metrics['pr_auc']:.6f}, "
                f"elapsed={row['elapsed_minutes']:.1f} min, device={row['compute_device']}",
                flush=True,
            )
            if best_candidate is None or metrics["pr_auc"] > best_candidate["validation"]["pr_auc"]:
                best_candidate = row
                best_model = model
            else:
                del model
            experiment["completed_fits"] = fit_number
            experiment["status"] = "running"
            report_path.write_text(json.dumps(experiment, indent=2, sort_keys=True), encoding="utf-8")
            del scores

        threshold = pick_threshold(y_validation, best_model.predict_proba(X_model_validation)[:, 1], fpr_ceiling=FPR_CEILING)
        validation_scores = best_model.predict_proba(X_model_validation)[:, 1]
        validation_metrics = compute_metrics(y_validation, validation_scores, threshold)
        family_dir = out_dir / model_name
        family_dir.mkdir(parents=True, exist_ok=True)
        model_path = family_dir / "best_model.joblib"
        joblib.dump(best_model, model_path, compress=3)
        metadata = {
            "dataset": dataset,
            "model": model_name,
            "strategy": "cap_smote_enn",
            "params": best_candidate["params"],
            "compute_device": best_candidate["compute_device"],
            "seed": SEED,
            "threshold_rule": f"maximize recall subject to FPR <= {FPR_CEILING}",
            "threshold": threshold,
            "validation_metrics": validation_metrics,
            "validation_search_candidate": best_candidate["candidate"],
            "validation_search_pr_auc": best_candidate["validation"]["pr_auc"],
            "resampling": resampling,
            "evaluated_test": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        (family_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        family_result = {
            "model": model_name,
            "strategy": "cap_smote_enn",
            "candidate": best_candidate["candidate"],
            "params": best_candidate["params"],
            "validation_search_pr_auc": best_candidate["validation"]["pr_auc"],
            **validation_metrics,
            "model_path": str(model_path),
            "evaluated_test": False,
        }
        experiment["family_best"].append(family_result)
        experiment["completed_families"] = len(experiment["family_best"])
        report_path.write_text(json.dumps(experiment, indent=2, sort_keys=True), encoding="utf-8")
        print(
            f"Saved validation-best {model_name}: PR-AUC={validation_metrics['pr_auc']:.6f}; "
            f"path={model_path}",
            flush=True,
        )
        del best_model, X_model_fit, X_model_validation, validation_scores

    experiment["status"] = "completed"
    experiment["completed_utc"] = datetime.now(timezone.utc).isoformat()
    experiment["completed_fits"] = fit_number
    report_path.write_text(json.dumps(experiment, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Completed {fit_number}/{total_fits} validation-only candidates; report={report_path}", flush=True)


if __name__ == "__main__":
    main()

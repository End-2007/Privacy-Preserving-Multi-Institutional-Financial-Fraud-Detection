"""Refit and save the validation-best IEEE-CIS model for each model family.

Uses the completed candidate search in reports/ieee_cis_metrics.json. The test
split is deliberately not loaded or evaluated here.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import DATASET_CONFIG, FPR_CEILING, SEED  # noqa: E402
from src.common.evaluate import compute_metrics, pick_threshold  # noqa: E402
from src.common.models import build_estimator  # noqa: E402
from src.common.resampling import resample_training_data  # noqa: E402


def main() -> None:
    report_path = ROOT / "reports" / "ieee_cis_metrics.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cfg = DATASET_CONFIG["ieee_cis"]
    data_dir = cfg["processed_dir"]
    out_dir = cfg["model_dir"] / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = sparse.load_npz(data_dir / "X_train.npz").tocsr().astype(np.float32, copy=False)
    y_train = np.load(data_dir / "y_train.npy", allow_pickle=False).astype(np.int8, copy=False)
    X_validation_tree = sparse.load_npz(data_dir / "X_validation.npz").tocsr().astype(np.float32, copy=False)
    y_validation = np.load(data_dir / "y_validation.npy", allow_pickle=False).astype(np.int8, copy=False)
    lr_scaler = joblib.load(cfg["model_dir"] / "lr_scaler.joblib")
    X_validation_lr = lr_scaler.transform(X_validation_tree).tocsr().astype(np.float32, copy=False)

    families = sorted({row["model"] for row in report["comparison"]})
    if len(families) != 4:
        raise RuntimeError(f"Expected four model families in report; found {families}.")

    saved = []
    for family in families:
        candidate = max(
            (row for row in report["comparison"] if row["model"] == family),
            key=lambda row: row["validation"]["pr_auc"],
        )
        strategy = candidate["strategy"]
        print(
            f"Refitting {family} with validation-best strategy={strategy}, "
            f"candidate={candidate['candidate']}, params={candidate['params']}",
            flush=True,
        )
        X_fit_tree, y_fit, resampling = resample_training_data(
            X_train, y_train, dataset="ieee_cis", strategy=strategy, seed=SEED
        )
        if family == "logistic_regression":
            X_fit = lr_scaler.transform(X_fit_tree).tocsr().astype(np.float32, copy=False)
            X_validation = X_validation_lr
        else:
            X_fit = X_fit_tree
            X_validation = X_validation_tree

        class_ratio = float(np.count_nonzero(y_fit == 0) / np.count_nonzero(y_fit == 1))
        device = candidate["compute_device"]
        model = build_estimator(
            family,
            candidate["params"],
            strategy=strategy,
            class_ratio=class_ratio,
            seed=SEED,
            xgb_device=device,
        )
        model.fit(X_fit, y_fit)
        validation_scores = model.predict_proba(X_validation)[:, 1]
        threshold = pick_threshold(y_validation, validation_scores, fpr_ceiling=FPR_CEILING)
        validation_metrics = compute_metrics(y_validation, validation_scores, threshold)

        family_dir = out_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        model_path = family_dir / "best_model.joblib"
        joblib.dump(model, model_path, compress=3)
        metadata = {
            "dataset": "ieee_cis",
            "model": family,
            "strategy": strategy,
            "params": candidate["params"],
            "compute_device": device,
            "seed": SEED,
            "threshold_rule": f"maximize recall subject to FPR <= {FPR_CEILING}",
            "threshold": threshold,
            "validation_metrics": validation_metrics,
            "validation_search_candidate": candidate["candidate"],
            "validation_search_pr_auc": candidate["validation"]["pr_auc"],
            "resampling_counts": resampling,
            "preprocessor": str((cfg["model_dir"] / "preprocessor.joblib").relative_to(ROOT)),
            "lr_scaler": str((cfg["model_dir"] / "lr_scaler.joblib").relative_to(ROOT))
            if family == "logistic_regression"
            else None,
            "evaluated_test": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path = family_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        saved.append(
            {
                "model": family,
                "strategy": strategy,
                "validation_search_pr_auc": candidate["validation"]["pr_auc"],
                **validation_metrics,
                "model_path": str(model_path),
                "metadata_path": str(metadata_path),
                "evaluated_test": False,
            }
        )
        summary_path = out_dir / "comparison_summary.json"
        summary_path.write_text(json.dumps(saved, indent=2, sort_keys=True), encoding="utf-8")
        print(
            f"Saved {family}: validation PR-AUC={validation_metrics['pr_auc']:.6f}; "
            f"path={family_dir}",
            flush=True,
        )
        del model, X_fit, X_fit_tree, y_fit, validation_scores

    print(f"Saved comparison summary: {out_dir / 'comparison_summary.json'}", flush=True)


if __name__ == "__main__":
    main()

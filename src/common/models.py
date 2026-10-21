"""Stage 1 estimator factories and compact random-search spaces."""

from __future__ import annotations

from scipy.stats import loguniform

from src.common.config import SEED, TUNING_BUDGETS


def parameter_space(model_name: str) -> dict:
    if model_name == "logistic_regression":
        # SAGA was prohibitively slow on PaySim's multi-million row split in
        # the local timing run. Keep three C samples with the faster solver.
        return {"C": loguniform(1e-3, 1e2), "solver": ["liblinear"]}
    if model_name == "random_forest":
        return {
            "n_estimators": [100, 150, 200],
            "max_depth": [12, 16, 20],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", 0.5],
        }
    if model_name == "xgboost":
        return {
            "n_estimators": [200, 350, 500],
            "max_depth": [3, 5, 7],
            "learning_rate": loguniform(0.02, 0.2),
            "subsample": [0.75, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
            "min_child_weight": [1, 5, 10],
            "reg_lambda": loguniform(0.1, 10.0),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": [200, 350, 500],
            "num_leaves": [15, 31, 63],
            "max_depth": [-1, 8, 12],
            "learning_rate": loguniform(0.02, 0.2),
            "subsample": [0.75, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
            "min_child_samples": [20, 50, 100],
            "reg_lambda": loguniform(0.1, 10.0),
        }
    raise ValueError(f"Unknown model: {model_name}")


def build_estimator(
    model_name: str,
    params: dict,
    *,
    strategy: str,
    class_ratio: float,
    seed: int = SEED,
    xgb_device: str = "cpu",
):
    """Build one estimator; fit is intentionally left to the explicit runner."""
    weighted = strategy == "class_weight"
    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            **params,
            max_iter=2000,
            class_weight="balanced" if weighted else None,
            random_state=seed,
        )
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            **params,
            class_weight="balanced_subsample" if weighted else None,
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            device=xgb_device,
            scale_pos_weight=class_ratio if weighted else 1.0,
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            **params,
            objective="binary",
            class_weight="balanced" if weighted else None,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def tuning_budget(model_name: str) -> int:
    try:
        return TUNING_BUDGETS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc

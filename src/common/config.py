"""Shared, reproducible defaults for Stage 1 experiments."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

SEED = 42
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
FPR_CEILING = 0.01
TARGET_COLUMN = "isFraud"

# Fixed tuning budgets from the Stage 1 plan. These are configuration only;
# importing this module never constructs or fits an estimator.
TUNING_BUDGETS = {
    "logistic_regression": 3,
    "random_forest": 6,
    "xgboost": 15,
    "lightgbm": 15,
}

DATASET_CONFIG = {
    "paysim": {
        "processed_dir": PROCESSED_DIR / "paysim",
        "model_dir": MODELS_DIR / "paysim",
        "positive_label": 1,
        "cap_majority_per_minority": 20,
        "resampling_strategies": ("class_weight", "cap_smote", "cap_smote_enn"),
    },
    "ieee_cis": {
        "processed_dir": PROCESSED_DIR / "ieee_cis",
        "model_dir": MODELS_DIR / "ieee_cis",
        "positive_label": 1,
        "cap_majority_per_minority": 5,
        # SMOTE-ENN is optional for IEEE-CIS due its runtime on the wide matrix.
        "resampling_strategies": ("class_weight", "cap_smote"),
    },
}

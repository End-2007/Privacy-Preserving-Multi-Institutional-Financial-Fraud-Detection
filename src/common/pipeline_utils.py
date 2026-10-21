"""Artifact, figure, and split helpers shared by Stage 1 runners."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src.common.config import SEED
from src.common.evaluate import pick_threshold


def stratified_split(y, *, seed: int = SEED, train_size: float = 0.70, validation_size: float = 0.15):
    """Return train/validation/test row indices with stratified labels."""
    labels = np.asarray(y).reshape(-1)
    if labels.size == 0 or not np.isin(labels, (0, 1)).all():
        raise ValueError("y must be a nonempty vector of binary labels.")
    if train_size <= 0 or validation_size <= 0 or train_size + validation_size >= 1:
        raise ValueError("train_size and validation_size must be positive and sum to less than 1.")
    indices = np.arange(labels.size)
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=1.0 - train_size - validation_size,
        stratify=labels,
        random_state=seed,
    )
    relative_val_size = validation_size / (train_size + validation_size)
    train_idx, validation_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_size,
        stratify=labels[train_val_idx],
        random_state=seed,
    )
    return train_idx, validation_idx, test_idx


def choose_validation_threshold(y_validation, validation_scores, *, fpr_ceiling: float = 0.01) -> float:
    """Named wrapper to make the validation-only threshold boundary explicit."""
    return pick_threshold(y_validation, validation_scores, fpr_ceiling=fpr_ceiling)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def save_artifacts(model, metadata: dict, output_dir: str | Path, *, filename: str = "best_model.joblib") -> None:
    """Persist a fitted model and matching metadata in an output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / filename, compress=3)
    save_json(output_dir / "metadata.json", metadata)


def plot_pr_curves(curves: dict[str, tuple], output_path: str | Path) -> None:
    """Save precision-recall curves: mapping label -> (precision, recall, AP)."""
    mpl_config = Path(__file__).resolve().parents[2] / ".venv" / ".cache" / "matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, (precision, recall, average_precision) in curves.items():
        ax.plot(recall, precision, label=f"{label} (AP={average_precision:.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)

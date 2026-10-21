"""Optional Weights & Biases tracking with a safe offline default."""

from __future__ import annotations

import os
from pathlib import Path


def init_run(*, project: str, name: str, config: dict, enabled: bool = True):
    """Start an offline W&B run unless online sync was explicitly enabled."""
    if not enabled:
        return None
    root = Path(__file__).resolve().parents[2]
    wandb_dir = root / "wandb"
    cache_dir = root / ".venv" / ".cache" / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))
    os.environ.setdefault("WANDB_CACHE_DIR", str(cache_dir))
    import wandb

    mode = "online" if os.getenv("WANDB_MODE", "offline").lower() == "online" else "offline"
    return wandb.init(project=project, name=name, config=config, mode=mode, reinit="finish_previous")


def log_metrics(metrics: dict, *, step: int | None = None) -> None:
    import wandb

    if wandb.run is not None:
        wandb.log(metrics, step=step)


def finish_run() -> None:
    import wandb

    if wandb.run is not None:
        wandb.finish()

"""Sync locally stored offline W&B runs after account access is available."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--directory", type=Path, default=Path("wandb"))
    args = parser.parse_args()
    if not args.directory.is_dir():
        raise SystemExit(f"Offline run directory does not exist: {args.directory}")

    run_files = sorted(args.directory.rglob("*.wandb"))
    if not run_files:
        raise SystemExit(f"No W&B run files found under: {args.directory}")

    for run_file in run_files:
        subprocess.run(
            [
                "wandb",
                "sync",
                "--include-offline",
                "--no-include-online",
                "--project",
                args.project,
                str(run_file),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()


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
    if not args.directory.exists():
        raise SystemExit(f"Offline run directory does not exist: {args.directory}")
    subprocess.run(
        ["wandb", "sync", "--include-offline", "--project", args.project, str(args.directory)],
        check=True,
    )


if __name__ == "__main__":
    main()


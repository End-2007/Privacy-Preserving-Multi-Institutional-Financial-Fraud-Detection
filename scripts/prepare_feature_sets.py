"""Prepare train-only matrices for the reviewed full/auth feature sets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_stage1 import IEEE_HIGH_CARD, fit_and_transform  # noqa: E402
from src.common.config import DATASET_CONFIG  # noqa: E402
from src.common.feature_sets import FEATURE_SET_NAMES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_CONFIG),
        default=list(DATASET_CONFIG),
        help="Datasets to prepare (defaults to both).",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=FEATURE_SET_NAMES,
        default=list(FEATURE_SET_NAMES),
        help="Reviewed feature sets to prepare (defaults to full and auth).",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        source_dir = DATASET_CONFIG[dataset]["processed_dir"]
        required = [source_dir / f"{split}.parquet" for split in ("train", "validation", "test")]
        missing = [path for path in required if not path.is_file()]
        if missing:
            missing_paths = ", ".join(str(path.relative_to(ROOT)) for path in missing)
            raise FileNotFoundError(
                f"Prepared {dataset} split files are missing: {missing_paths}. "
                "Run scripts/prepare_stage1.py first."
            )

        categorical_high = sorted(IEEE_HIGH_CARD) if dataset == "ieee_cis" else []
        for feature_set in args.feature_sets:
            print(f"Preparing {dataset}/{feature_set} feature matrices", flush=True)
            fit_and_transform(
                dataset,
                categorical_high=categorical_high,
                feature_set=feature_set,
            )


if __name__ == "__main__":
    main()

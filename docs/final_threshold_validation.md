# Final threshold validation audit

This validation-only audit checks the selected Stage 1 LightGBM model on PaySim and IEEE-CIS.

## What it checks

- Loads each saved final model and its validation features and labels.
- Recomputes validation metrics at the threshold recorded in model metadata.
- Compares the recomputed metrics with the saved validation snapshot.
- Checks that the false-positive rate is at or below the configured 1% ceiling.
- Does not open or score either test split.

## Run it locally

From the repository root, after placing the local model artifacts and processed data in the standard `models/` and `data/processed/` directories. These files are excluded from Git.

```powershell
python -m scripts.validate_final_thresholds
```

The default artifact root is the repository root, and the default report destination is `reports/final_threshold_validation.json`. Use `--artifact-root PATH` for another artifact root or `--output PATH` to write the report elsewhere.

## Recorded validation snapshot

The JSON report records the latest audit output. Both dataset results matched their saved validation metric snapshots and met the configured FPR ceiling:

| Dataset | Validation rows | Threshold | Average Precision | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|
| PaySim | 954,394 | 0.0568012041 | 0.999986 | 1.000000 | 0.000022 |
| IEEE-CIS | 88,581 | 0.7675237348 | 0.758107 | 0.960109 | 0.009967 |

These are validation results, not final test results. PaySim is synthetic; its high scores do not establish real-world performance.

## Reproduction status

The audit was freshly rerun on 2026-10-28 against the local artifacts in the repository root. Both datasets matched their saved validation snapshots and passed the 1% FPR ceiling. The script opens only validation arrays; it does not load or score test data. The local environment may emit scikit-learn version warnings because the saved preprocessing objects were serialized with scikit-learn 1.3.0 while the current environment uses a newer version; the validation audit completed successfully, but matching the artifact-generation environment is preferable for strict reproducibility.

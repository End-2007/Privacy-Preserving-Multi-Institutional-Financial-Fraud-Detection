# Final threshold validation audit

This change adds a reproducible validation-only audit for the selected Stage 1 LightGBM model on PaySim and IEEE-CIS.

## What it checks

- Loads each saved final model and its validation features and labels.
- Recomputes validation metrics at the threshold recorded in model metadata.
- Compares the recomputed metrics with the saved validation snapshot.
- Checks that the false-positive rate is at or below the configured 1% ceiling.
- Does not open or score either test split.

## Run it locally

From the repository root, with the Stage 1 transfer bundle extracted to a directory containing `models/` and `data/processed/`:

```powershell
python -m scripts.validate_final_thresholds --artifact-root D:\MICRO\stage1_transfer
```

The default report destination is `reports/final_threshold_validation.json`. Use `--output PATH` to write it elsewhere. Keep the model/data bundle outside Git.

## Recorded validation snapshot

The committed JSON report records the prior audit output. Both dataset results matched their saved validation metric snapshots and met the configured FPR ceiling:

| Dataset | Validation rows | Threshold | Average Precision | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|
| PaySim | 954,394 | 0.0568012041 | 0.999986 | 1.000000 | 0.000022 |
| IEEE-CIS | 88,581 | 0.7675237348 | 0.758107 | 0.960109 | 0.009967 |

These are validation results, not final test results. PaySim is synthetic; its high scores do not establish real-world performance.

## Reproduction status

The report is retained from the earlier audit run. Re-run the command above against the local Stage 1 bundle before relying on it as a fresh reproduction. In the current Windows environment, the saved LightGBM predictor previously raised a native access-violation error, so a fresh run has not been confirmed here. The audit code and report do not replace the required local reproduction.

# PaySim Data Card

**Project:** FedGuard Micro, Stage 1  
**Prepared by:** Naman Dugar  
**Audit date:** 2026-10-21  
**Status:** Audit, label-aware EDA, stratified splits, train-fitted preprocessing, four-model comparison, and selected-model test evaluation complete

## Summary

PaySim is a synthetic mobile-money transaction dataset. In this repository it is used as the Stage 1 mobile-money institution dataset for developing and evaluating a centralized fraud baseline. It is not a sample of current bank transactions and should not be used to make decisions about real customers.

## Provenance and files

| Item | Detail |
| --- | --- |
| Source listed by the project | [Kaggle PaySim1](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| Local file | `data/raw/PS_20174392719_1491204439457_log.csv` |
| File size | 493,534,783 bytes |
| Format | CSV |
| Source-file handling | Raw file remains local and is excluded from Git |
| Audit output | `reports/data_audit_summary.json` |

## Audited contents

| Measure | Observed value |
| --- | ---: |
| Rows | 6,362,620 |
| Columns | 11 |
| Fraud label | `isFraud` |
| Non-fraud rows (`isFraud = 0`) | 6,354,407 |
| Fraud rows (`isFraud = 1`) | 8,213 |
| Fraud prevalence | 0.1291% |
| Missing values | 0 |
| Duplicate full rows detected | 0 |

The duplicate-row check hashes rows in chunks and checks the hashes across the full file. The audit found no duplicate PaySim rows.

### Label counts by transaction type

| Transaction type | Non-fraud | Fraud |
| --- | ---: | ---: |
| `PAYMENT` | 2,151,495 | 0 |
| `TRANSFER` | 528,812 | 4,097 |
| `CASH_OUT` | 2,233,384 | 4,116 |
| `DEBIT` | 41,432 | 0 |
| `CASH_IN` | 1,399,284 | 0 |

The audit confirms that all labeled fraud rows are `TRANSFER` or `CASH_OUT` in this local file. `isFlaggedFraud` is 1 for 16 rows and 0 for 6,362,604 rows; all 16 flagged rows have `isFraud = 1`. Exclude this pre-existing rule flag from model inputs because it directly discloses a subset of the target.

## Fields and processing notes

The source fields are `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, and `isFlaggedFraud`.

For Stage 1, `isFraud` is the target. The implementation derives balance-difference, zero-balance, hour, and log-amount features, treats `type` as categorical, and excludes account identifiers and `isFlaggedFraud`. The final 18 transformed feature names and order are recorded in local `data/processed/paysim/feature_metadata.json`; both that file and the fitted preprocessor in `models/paysim/preprocessor.joblib` are excluded from Git.

## Known limitations and risks

- PaySim is a simulator. Its balance rules can make fraud easier to identify than fraud in real payment systems, so performance should not be presented as expected production performance.
- Fraud is rare (0.1291%); report PR-AUC, recall, precision, and false-positive rate. Accuracy alone is not informative.
- The fraud labels occur only in two transaction types in this file. Models may rely heavily on those patterns; document feature importance and avoid claims that the result generalizes to other fraud types.
- `step` is the simulator's time index. Any time-based split or feature derived from it needs a documented rationale.
- No nulls or duplicate rows were observed in this audit. Additional checks for ranges, impossible values, and feature-label relationships belong in EDA.

## Stage 1 evaluation protocol

Use a stratified 70/15/15 train/validation/test split with seed 42, as specified in the active Stage 1 plan. Fit preprocessing and resampling on the training portion only. Select models and thresholds using validation data, then evaluate the selected final model on the untouched test portion once. Preserve the natural fraud rate in validation and test.

The selected Stage 1 model is LightGBM with class weights. Its validation PR-AUC was 0.999986; its single held-out test evaluation gave PR-AUC 0.997968, ROC-AUC 0.998691, precision 0.973080, recall 0.997565, F1 0.985170, and FPR 0.00003567 at threshold 0.0568012. See `reports/paysim_metrics.json` and `docs/HANDOVER.md` for the per-family validation comparison and interpretation.

## Audit method and reproducibility

The local CSV was scanned in 25,000-row chunks. Label counts, null counts, and duplicate-row hashes were accumulated across the whole file. The reusable command is:

```powershell
python scripts\audit_data.py
```

The complete machine-readable audit is in `reports/data_audit_summary.json`. Re-run the audit if the raw file changes.

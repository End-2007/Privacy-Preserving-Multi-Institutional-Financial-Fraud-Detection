# IEEE-CIS Fraud Detection Data Card

**Project:** FedGuard Micro, Stage 1  
**Prepared by:** Naman Dugar  
**Audit date:** 2026-10-21  
**Status:** Audit, identity join check, label-aware EDA, stratified splits, train-fitted preprocessing, four-model comparison, and selected-model test evaluation complete

## Summary

IEEE-CIS Fraud Detection is the card and e-commerce dataset used for the second simulated institution in FedGuard Stage 1. The labeled transaction and identity training files are joined for analysis and modeling. The competition test files are unlabeled for fraud evaluation and are not the Stage 1 holdout set.

## Provenance and files

| Item | Detail |
| --- | --- |
| Source listed by the project | [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) |
| Transaction training file | `data/raw/train_transaction.csv` |
| Identity training file | `data/raw/train_identity.csv` |
| Competition test files | `data/raw/test_transaction.csv`, `data/raw/test_identity.csv` |
| Format | CSV |
| Source-file handling | Raw files remain local and are excluded from Git |
| Audit output | `reports/data_audit_summary.json` |

## Audited contents

| Measure | Transaction training file | Identity training file |
| --- | ---: | ---: |
| Rows | 590,540 | 144,233 |
| Columns | 394 | 41 |
| Label | `isFraud` | None |
| Fraud rows | 20,663 | N/A |
| Fraud prevalence | 3.499% | N/A |
| Duplicate `TransactionID` values | 0 | 0 |

All 144,233 identity keys match a transaction key. Identity data is available for 24.42% of the 590,540 transactions; the remaining transactions receive no identity row in a left join. The joined table therefore has 434 columns: the 394 transaction columns plus 40 identity attributes, with the duplicate join key counted once.

## Missingness and Stage 1 column rule

The transaction file has missing values in 374 of 394 columns. The identity file has missing values in 38 of 41 columns. After a left join, 414 of 434 columns have at least one missing value.

The following columns exceed 90% missingness in the full labeled training file after joining:

| Column | Missing rows | Missing |
| --- | ---: | ---: |
| `dist2` | 552,913 | 93.63% |
| `D7` | 551,623 | 93.41% |
| `id_07` | 585,385 | 99.13% |
| `id_08` | 585,385 | 99.13% |
| `id_18` | 545,427 | 92.36% |
| `id_21` | 585,381 | 99.13% |
| `id_22` | 585,371 | 99.12% |
| `id_23` | 585,371 | 99.12% |
| `id_24` | 585,793 | 99.20% |
| `id_25` | 585,408 | 99.13% |
| `id_26` | 585,377 | 99.13% |
| `id_27` | 585,371 | 99.12% |

These are **full-file audit candidates**, not the final fitted feature filter. To prevent validation or test data from influencing preprocessing, create the 70/15/15 split first, calculate the 90% missingness rule on the training partition only, and save the resulting dropped-column list. The list from that training-only calculation may differ from this preliminary full-file list.

The fitted training-only rule dropped 12 columns: `dist2`, `D7`, `id_07`, `id_08`, `id_18`, `id_21`, `id_22`, `id_23`, `id_24`, `id_25`, `id_26`, and `id_27`. The exact transformed model feature order is recorded in local `data/processed/ieee_cis/feature_metadata.json` (1,208 features); this file and the fitted preprocessor are excluded from Git.

## Fields and processing notes

The transaction file contains `TransactionID`, `TransactionDT`, `TransactionAmt`, `ProductCD`, card and address fields, email domains, `C*`, `D*`, `M*`, and `V*` feature families, plus `isFraud`. The identity file contains `TransactionID`, `id_*` attributes, `DeviceType`, and `DeviceInfo`.

For Stage 1, use `isFraud` from `train_transaction.csv` as the target and left-join identity data on `TransactionID`. Keep a `has_identity` indicator to distinguish absent identity records from null values within an identity record. Exclude `TransactionID` from model predictors after joining. Fit imputers, encoders, frequency mappings, scalers, and feature selectors on the training partition only.

## Known limitations and risks

- The dataset has substantial and structured missingness. In particular, identity features apply to only about one quarter of transactions, and several become more than 90% missing after the join.
- Many features are anonymized or represented by feature families, which limits direct human interpretation. The SHAP layer preserves supplied names and does not assign undocumented business meaning; see docs/HANDOVER.md.
- The competition test files do not provide `isFraud`. Use a stratified internal split of the labeled training files for Stage 1 metrics; do not report competition-test metrics as labeled evaluation.
- The fraud prevalence (3.499%) is higher than PaySim's, so resampling ratios and operating thresholds should be dataset-specific.
- Row counts and missingness describe the local files at audit time. Re-run the audit if the files are replaced or modified.

## Stage 1 evaluation protocol

Merge only the labeled training files, then make a stratified 70/15/15 train/validation/test split with seed 42, as specified in the active Stage 1 plan. Fit all preprocessing and any resampling on the training portion only. Keep validation and test at their natural class distributions. Tune and choose the threshold on validation data, then evaluate the selected final model on the untouched test portion once.

The selected Stage 1 model is LightGBM with class weights. Its validation AP was 0.758107; its single held-out test evaluation gave AP 0.756757, ROC-AUC 0.959075, precision 0.710193, recall 0.687964, F1 0.698902, and FPR 0.010178 at threshold 0.7675237. The separate IEEE-CIS SMOTE-ENN trial was validation-only and did not improve AP for any model family. See `reports/ieee_cis_metrics.json`, `reports/ieee_cis_smote_enn_validation.json`, and `docs/HANDOVER.md`.

## Audit method and reproducibility

The transaction file was scanned in 25,000-row chunks for row counts, label counts, null counts, and duplicate join keys. The identity file was checked for null and duplicate keys, then matched against the transaction keys. Joined missingness was calculated for the left-joined training data. The reusable command is:

```powershell
python scripts\audit_data.py
```

The complete machine-readable audit is in `reports/data_audit_summary.json`. Re-run the audit if either raw training file changes.

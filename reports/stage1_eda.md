# Stage 1 Exploratory Data Analysis

**Prepared by:** Naman Dugar  
**Date:** 2026-10-21  
**Scope:** PaySim and labeled IEEE-CIS training data only. No models were trained.  
**Method:** Chunked scans; amount and balance quantiles below use deterministic class-stratified samples (seed 42), while row counts, category counts, and label rates use all rows.

## PaySim

- Rows and columns: 6,362,620 x 11.
- Fraud: 8,213 / 6,362,620 (0.1291%); no nulls or duplicate full rows in the source audit.
- `step` range: 1 to 743.
- All 8,213 fraud labels occur in `TRANSFER` and `CASH_OUT`.
- `isFlaggedFraud` is 1 for 16 rows, and all 16 are labeled fraud. Exclude this pre-existing rule flag from model inputs because it directly discloses a subset of the target.

### Fraud rate by transaction type (all rows)

| PaySim type | Rows | Fraud | Fraud rate |
| --- | ---: | ---: | ---: |
| `CASH_OUT` | 2,237,500 | 4,116 | 0.184% |
| `PAYMENT` | 2,151,495 | 0 | 0.000% |
| `CASH_IN` | 1,399,284 | 0 | 0.000% |
| `TRANSFER` | 532,909 | 4,097 | 0.769% |
| `DEBIT` | 41,432 | 0 | 0.000% |

### Balance and amount distributions

Values are approximate sample summaries, split by target. Engineered errors follow the active plan's formulas: `newbalanceOrig + amount - oldbalanceOrg` and `oldbalanceDest + amount - newbalanceDest`.

#### Amount

| Label | Sample n | Mean | Median | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 30,000 | 175,659.81 | 73,398.53 | 513,106.30 | 40,351,145.29 |
| 1 | 8,213 | 1,467,967.30 | 441,423.44 | 8,006,429.04 | 10,000,000.00 |

#### Origin balance error

| Label | Sample n | Mean | Median | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 30,000 | 198,933.16 | 67,202.05 | 711,651.97 | 40,351,145.29 |
| 1 | 8,213 | 10,692.33 | 0.0000 | 0.0000 | 10,000,000.00 |

#### Destination balance error

| Label | Sample n | Mean | Median | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 30,000 | 56,932.62 | 3,634.47 | 461,211.05 | 2,624,238.78 |
| 1 | 8,213 | 732,509.30 | 2,231.46 | 4,555,771.26 | 10,000,000.00 |

### Zero ending-balance rates (all rows)

| Label | `newbalanceOrig == 0` | `newbalanceDest == 0` |
| --- | ---: | ---: |
| Non-fraud | 56.677% | 38.325% |
| Fraud | 98.052% | 49.811% |

### PaySim interpretation

Transaction type and balance-related features are worth testing in the Stage 1 baselines, but the strong synthetic balance behavior and the fact that fraud occurs only in two transaction types limit real-world generalization. Keep account IDs and `isFlaggedFraud` out of model inputs under the active plan. Confirm the exact feature list and record it in run metadata.

## IEEE-CIS

- Labeled transaction rows: 590,540; fraud: 20,663 (3.4990%).
- `TransactionDT` range: 86,400 to 15,811,131 (dataset-relative time units).
- Identity rows: 144,233; left-join coverage: 24.42%; unmatched identity keys: 0.
- Full-file joined columns over 90% missing: `dist2`, `D7`, `id_07`, `id_08`, `id_18`, `id_21`, `id_22`, `id_23`, `id_24`, `id_25`, `id_26`, `id_27`. Recalculate this rule on the training split only before fitting or dropping columns.

### Most-missing joined columns

| Column | Missing rows | Missing |
| --- | ---: | ---: |
| `id_24` | 585,793 | 99.20% |
| `id_25` | 585,408 | 99.13% |
| `id_07` | 585,385 | 99.13% |
| `id_08` | 585,385 | 99.13% |
| `id_21` | 585,381 | 99.13% |
| `id_26` | 585,377 | 99.13% |
| `id_22` | 585,371 | 99.12% |
| `id_23` | 585,371 | 99.12% |
| `id_27` | 585,371 | 99.12% |
| `dist2` | 552,913 | 93.63% |
| `D7` | 551,623 | 93.41% |
| `id_18` | 545,427 | 92.36% |
| `D13` | 528,588 | 89.51% |
| `D14` | 528,353 | 89.47% |
| `D12` | 525,823 | 89.04% |

### Transaction amount

The summary uses deterministic stratified samples, not full-file quantiles.

| Label | Sample n | Mean | Median | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 30,000 | 133.3082 | 69.1725 | 424.9500 | 4,515.95 |
| 1 | 20,663 | 149.2448 | 75.0000 | 500.0000 | 5,191.00 |

### Categorical fraud rates (all rows; categories with at least 500 records)

| ProductCD | Rows | Fraud | Fraud rate |
| --- | ---: | ---: | ---: |
| `W` | 439,670 | 8,969 | 2.040% |
| `C` | 68,519 | 8,008 | 11.687% |
| `R` | 37,699 | 1,426 | 3.783% |
| `H` | 33,024 | 1,574 | 4.766% |
| `S` | 11,628 | 686 | 5.900% |

| card4 | Rows | Fraud | Fraud rate |
| --- | ---: | ---: | ---: |
| `visa` | 384,767 | 13,373 | 3.476% |
| `mastercard` | 189,217 | 6,496 | 3.433% |
| `american express` | 8,328 | 239 | 2.870% |
| `discover` | 6,651 | 514 | 7.728% |
| `<missing>` | 1,577 | 41 | 2.600% |

| card6 | Rows | Fraud | Fraud rate |
| --- | ---: | ---: | ---: |
| `debit` | 439,938 | 10,674 | 2.426% |
| `credit` | 148,986 | 9,950 | 6.678% |
| `<missing>` | 1,571 | 39 | 2.482% |

| P_emaildomain | Rows | Fraud | Fraud rate |
| --- | ---: | ---: | ---: |
| `gmail.com` | 228,355 | 9,943 | 4.354% |
| `yahoo.com` | 100,934 | 2,297 | 2.276% |
| `<missing>` | 94,456 | 2,790 | 2.954% |
| `hotmail.com` | 45,250 | 2,396 | 5.295% |
| `anonymous.com` | 36,998 | 859 | 2.322% |
| `aol.com` | 28,289 | 617 | 2.181% |
| `comcast.net` | 7,888 | 246 | 3.119% |
| `icloud.com` | 6,267 | 197 | 3.143% |
| `outlook.com` | 5,096 | 482 | 9.458% |
| `msn.com` | 4,092 | 90 | 2.199% |
| `att.net` | 4,033 | 30 | 0.744% |
| `live.com` | 3,041 | 84 | 2.762% |
| `sbcglobal.net` | 2,970 | 12 | 0.404% |
| `verizon.net` | 2,705 | 22 | 0.813% |
| `ymail.com` | 2,396 | 50 | 2.087% |
| `bellsouth.net` | 1,909 | 53 | 2.776% |
| `yahoo.com.mx` | 1,543 | 16 | 1.037% |
| `me.com` | 1,522 | 27 | 1.774% |
| `cox.net` | 1,393 | 29 | 2.082% |
| `optonline.net` | 1,011 | 17 | 1.682% |
| *Remaining 5 categories omitted* | | | |

| DeviceType | Rows | Fraud | Fraud rate |
| --- | ---: | ---: | ---: |
| `<missing>` | 449,730 | 9,452 | 2.102% |
| `desktop` | 85,165 | 5,554 | 6.521% |
| `mobile` | 55,645 | 5,657 | 10.166% |

### IEEE-CIS interpretation

The fraud rate is materially higher than PaySim's, and identity attributes cover roughly one quarter of transactions. Preserve a `has_identity` indicator so “no identity row” remains distinguishable from null values inside a matched identity row. The high-cardinality and sparse fields need train-only encoders and imputers. Use the internal stratified split for labeled metrics; the competition test files are unlabeled.

## Stage 1 actions from this EDA

1. Compare Logistic Regression, Random Forest, XGBoost, and LightGBM on each dataset. The updated sprint plan assigns 3, 6, 15, and 15 tuning sets respectively to LR, RF, XGBoost, and LightGBM.
2. Use the planned 70/15/15 stratified split with seed 42. Derive IEEE-CIS missingness filtering and all preprocessing from train only.
3. Keep validation and test at the natural class distribution; resample only train.
4. Select models using validation Average Precision (AP); choose the operating threshold on validation and report test metrics once for the final selected model. Report FPR with the other metrics.
5. Treat the distribution patterns as diagnostic findings, not evidence of production performance. SHAP and Streamlit remain post-Stage-1 work.

## Reproduction

Run `python scripts\eda_stage1.py` from the repository root. The script reads raw files without modifying them and writes this report to `reports/stage1_eda.md`.

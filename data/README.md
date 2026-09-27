# Data Directory

> [!NOTE]
> **Datasets are not committed to Git.** They are excluded via `.gitignore` due to their large size.

## Download Instructions

### 1. PaySim (Mobile-Money Transactions)
- **Source:** [Kaggle — PaySim1](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **File:** `PS_20174392719_1491204439457_log.csv` (~471 MB)
- **Rows:** ~6.3 million transactions
- **Columns:** 11 — `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`
- **Place in:** `data/raw/`

### 2. IEEE-CIS Fraud Detection (Card / E-commerce Transactions)
- **Source:** [Kaggle — IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data)
- **Files:**
  | File | Size | Role | Columns |
  |---|:---:|:---:|:---:|
  | `train_transaction.csv` | ~652 MB | <img src="https://img.shields.io/badge/Partition-Training_Tx-2979FF?style=flat-square" alt="Train Tx" /> | 394 |
  | `train_identity.csv` | ~25 MB | <img src="https://img.shields.io/badge/Partition-Training_ID-00B0FF?style=flat-square" alt="Train ID" /> | 41 |
  | `test_transaction.csv` | ~585 MB | <img src="https://img.shields.io/badge/Partition-Test_Tx-7C4DFF?style=flat-square" alt="Test Tx" /> | 393 |
  | `test_identity.csv` | ~25 MB | <img src="https://img.shields.io/badge/Partition-Test_ID-AA00FF?style=flat-square" alt="Test ID" /> | 41 |
  | `sample_submission.csv` | ~6 MB | <img src="https://img.shields.io/badge/Partition-Benchmark-757575?style=flat-square" alt="Benchmark" /> | 2 |
- **Place in:** `data/raw/`

## Directory Structure

```
data/
├── raw/              # Original downloaded CSVs (gitignored)
├── processed/        # Cleaned, encoded, SMOTE-resampled data (gitignored)
└── sample/           # Tiny sample CSVs for quick testing (committed)
```

## After Downloading

Your `data/raw/` folder should look like:

```
data/raw/
├── PS_20174392719_1491204439457_log.csv    # PaySim
├── train_transaction.csv                     # IEEE-CIS
├── train_identity.csv                        # IEEE-CIS
├── test_transaction.csv                      # IEEE-CIS
├── test_identity.csv                         # IEEE-CIS
└── sample_submission.csv                     # IEEE-CIS
```

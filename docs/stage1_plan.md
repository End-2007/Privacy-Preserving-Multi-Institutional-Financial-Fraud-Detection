# FedGuard Micro: Stage 1, 2-Day Solo Sprint and Handover Plan

Team: Naman Dugar, Anvi Tyagi, Riyanshu. Guide: Prof. Mohit Tiwari.

## 1. How the work is split in time

| Window | Who | What |
| --- | --- | --- |
| Day 1 and Day 2 | **Naman, alone** | Everything needed to produce working, saved results on both datasets, including the minimal shared code both pipelines depend on |
| Day 3 onward | **Anvi + Riyanshu** | Review, harden, test, document and integrate what Naman hands over |

Goal of the 2 days: at handover, both pipelines run end to end from raw data, all four models are trained per dataset, artifacts are saved, and a handover note tells the others exactly what to take over. Polish, tests and documentation are deliberately pushed to the handover.

## 2. Scope: must-have vs cut-if-late

Two days is tight, so decide the cut line now.

**Must-have (done by end of Day 2):**
- PaySim and IEEE-CIS cleaned, split (stratified 70/15/15, seed 42), saved as parquet
- Preprocessing fitted on train only
- Strategy A (class weights) and Strategy B (cap + SMOTE) for both datasets
- LR, RF, XGBoost and LightGBM per dataset, tuned on validation PR-AUC
- Threshold chosen on validation, one final test evaluation
- Best model + preprocessor + `metadata.json` + metrics JSON + PR curve per dataset
- Data cards, `decisions.md`, `HANDOVER.md`

**Cut first, hand over as open tasks if time runs out:**
1. Strategy C (SMOTE-ENN) on IEEE-CIS (slow on 400+ columns; later completed as a validation-only experiment, which did not improve PR-AUC)
2. IEEE-CIS temporal check
3. `feature_metadata.json` readable-vs-anonymised grouping (write the raw list at least)
4. W&B online sync (log offline if access is not ready)
5. Full tuning budget for RF (reduce to 6 sets)

## 3. Shared defaults (write into `docs/decisions.md` first)

| Setting | Value |
| --- | --- |
| Seed | 42 |
| Split | 70/15/15, stratified |
| Selection metric | Validation PR-AUC |
| Tuning | XGBoost 15 sets, LightGBM 15 sets, RF 6 sets, LR 3 sets (reduced for the 2-day window; raise at handover if time allows) |
| Primary-model rule for Stage 2 | Highest validation PR-AUC with FPR at most 1% |
| Threshold | Chosen on validation, applied once to test |
| IEEE-CIS split | Stratified primary; temporal check is optional |

## 4. Day 1: foundation and PaySim end to end

### Block 1.1: Setup and shared code (about 2 hours)

Because nobody else is writing it for 2 days, Naman writes the minimal shared layer so the pipelines and the handover use the same code. Keep these files in the current working tree; version control happens after both datasets finish training.

1. Folder structure, `requirements.txt`, `.gitignore` (`data/`, `models/`, `*.joblib`, `wandb/`), `docs/decisions.md`.
2. `src/common/config.py`: `SEED=42`, `SPLIT=(0.70, 0.15, 0.15)`, paths, `FPR_CEILING=0.01`.
3. `src/common/evaluate.py`: `compute_metrics(y_true, y_proba, threshold=0.5)` returning `precision, recall, f1, roc_auc, pr_auc, fpr`.
4. `src/common/tracking.py`: `init_run`, `log_metrics`, `finish_run`. W&B with an offline fallback.
5. `src/common/pipeline_utils.py`: `stratified_split`, `pick_threshold`, `save_artifacts`, `plot_pr_curves`, so both pipelines call the same functions.

```python
# src/common/evaluate.py
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)

def compute_metrics(y_true, y_proba, threshold=0.5):
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
    }
```

### Block 1.2: PaySim data card and EDA (about 1.5 hours)

Create `docs/data_card_paysim.md` and an exploratory notebook or report.

Answer in the EDA (verify the expected figures): about 6.36M rows, fraud around 0.13%, fraud only in `TRANSFER` and `CASH_OUT`, `isFlaggedFraud` almost always 0, balance behaviour on fraud rows, `amount` on log scale.

### Block 1.3: PaySim preprocessing and split (about 2 hours)

Implement the PaySim preprocessing and split in the shared working tree.

- `src/paysim/preprocess.py`: downcast dtypes, record nulls and duplicates, derive `dest_is_merchant` from `nameDest`, then drop `nameOrig`, `nameDest`, `isFlaggedFraud` (written reason: almost no positives, rule-based flag).
- Features, each documented in the docstring: `orig_balance_delta`, `dest_balance_delta`, `error_balance_orig`, `error_balance_dest`, `orig_zero_after`, `hour = step % 24`, `log_amount`.
- Stratified split via `pipeline_utils.stratified_split`, saved to `data/processed/paysim/`. Print fraud counts per split into the data card.
- Fit one-hot for `type` and a scaler (LR only) on train. Save `models/paysim/preprocessor.joblib`.

### Block 1.4: PaySim imbalance and baselines (about 3 hours)

Implement PaySim imbalance handling and training in the shared working tree.

- `src/paysim/resample.py`: Strategy A class weights; Strategy B cap majority at 1:20 then SMOTE to 1:2; Strategy C same cap then SMOTE-ENN (run it on PaySim, the capped set is small enough). Return class counts before and after.
- `src/paysim/train.py`: loop models x strategies, random search, validation PR-AUC, threshold on validation, one test evaluation via `compute_metrics`, log to W&B.
- Save best model, `metadata.json`, `reports/paysim_metrics.json`, `reports/figures/paysim_pr_curves.png`.

Run the full training in the background while you start Day 1 evening work below.

### Block 1.5: Day 1 closing (about 30 minutes)

- Confirm the same seed reproduces the same split and the same metrics.
- Confirm no resampled row exists outside train and test fraud rate is untouched.
- Save a working backup of the source and run notes. Start the IEEE-CIS data card so Day 2 starts clean. Do not commit or push during training.

**Day 1 exit criteria:** PaySim runs end to end, shared layer is written, and PaySim artifacts and run notes are saved locally.

## 5. Day 2: IEEE-CIS end to end, then handover package

### Block 2.1: IEEE-CIS merge, data card, EDA (about 1.5 hours)

Continue in the shared working tree.

- Left join `train_identity` onto `train_transaction` on `TransactionID`. Add `has_identity`.
- Save missingness per column to `reports/ieee_cis_missingness.csv`. Write `docs/data_card_ieee_cis.md`.
- Verify: about 590K rows, about 394 plus 41 columns before merge, fraud around 3.5%.
- Downcast to float32 and int32 immediately after loading to protect memory.

### Block 2.2: IEEE-CIS cleaning, encoding, split (about 2.5 hours)

Implement IEEE-CIS cleaning, encoding, and splitting in the shared working tree.

- Drop columns above 90% missing; save the dropped list.
- Split first (stratified, seed 42), then fit everything on train only.
- Numeric: median imputation plus missing-indicators for `D*` and `id_*`.
- Low-cardinality categoricals (`ProductCD`, `card4`, `card6`, `DeviceType`, `M*`): one-hot or ordinal.
- High-cardinality (`card1`, `card2`, `addr1`, `P_emaildomain`, `R_emaildomain`, `DeviceInfo`): frequency encoding fitted on train.
- Features: hour of day, day, `log_amt`.
- Save to `data/processed/ieee_cis/` and `models/ieee_cis/preprocessor.joblib`.

### Block 2.3: IEEE-CIS imbalance and baselines (about 3.5 hours)

Implement IEEE-CIS imbalance handling and training in the shared working tree.

- Strategy A (class weights) and B (cap 1:5 then SMOTE after encoding; document why SMOTE-NC is not used) are must-have.
- Strategy C (SMOTE-ENN): time a 50K-row sample first. If projected time is more than about 30 minutes, skip and list it as an open task in `HANDOVER.md`.
- Tuning budget as in section 3. Cap RF at 200 trees and depth 20 if memory is tight.
- Save best model, preprocessor, `metadata.json`, `reports/ieee_cis_metrics.json`, PR curves.
- Write `feature_metadata.json` with at least the raw feature list in model order.

### Block 2.4: Handover package (about 1.5 hours, do not skip)

Write `docs/HANDOVER.md` containing:

1. **Status table:** each deliverable done, partly done or not started.
2. **How to run:** exact commands to regenerate processed data and train each pipeline, with runtimes seen on your machine.
3. **Artifact contract:** the `metadata.json` fields (seed, split ratios, strategy, params, threshold, feature list and order, library versions, git commit, test metrics, resampling counts), identical for both datasets.
4. **Results snapshot:** the six metrics for each dataset's best model, with the chosen strategy and threshold; include the per-model comparison table.
5. **Known caveats:** PaySim scores are inflated by synthetic balance rules, anonymised IEEE-CIS features limit SHAP readability, anything cut under section 2.
6. **Open tasks list** (section 6 below) with suggested owners.

**Day 2 exit criteria:** both datasets have saved artifacts and metrics JSON, and `HANDOVER.md` is written. After the training work is complete, perform the Git version-control work as a separate final step.

## 6. Handover: what Anvi and Riyanshu take over (Day 3 onward)

| Task | Suggested owner |
| --- | --- |
| Review PaySim PRs | Anvi |
| Review IEEE-CIS PRs | Riyanshu |
| Unit tests for `compute_metrics` and `pipeline_utils` | Riyanshu |
| Branch protection, PR template, CONTRIBUTING, issue labels | Riyanshu |
| W&B project, invites, sync offline runs | Riyanshu |
| `docs/schema_alignment.md` | Anvi |
| `src/common/model_interface.py` (reads `metadata.json`) | Anvi |
| `reports/stage1_results.md` from the two metrics JSONs | Anvi |
| README and progress-report cleanup | Both, split by section |
| Optional follow-up: IEEE-CIS temporal check, readable-only feature comparison, or expanded RF tuning | Agree at handover meeting |

Naman reviews their PRs and stays available for pipeline questions.

## 7. Working rules during the solo sprint

1. Do not create branches, commits, or PRs while completing the two dataset pipelines and training runs. Keep a separate working backup and save run configurations and results as they are produced.
2. Once both training pipelines and their results are complete, add the source, documentation, and small reports to Git. Keep raw data, processed datasets, and model binaries out of Git.
3. Notebooks are for exploration only. Keep final logic in `src/`.
4. If metadata needs a Git reference, first commit the frozen source and reports, then add that source commit ID to the metadata in a follow-up commit. Do not retrain solely to add the reference.

## 8. Leakage and correctness checks (before each final run)

1. Split first, then fit anything.
2. Imputers, encoders, scalers and feature selectors are fitted on train only.
3. Resampling touches train only; val and test keep the natural fraud rate.
4. Tune and choose the threshold on validation; test is evaluated once per final model.
5. Never headline accuracy. Lead with PR-AUC, recall, FPR.
6. Every number in a report comes from a logged run with its seed, dataset identifiers, package versions, and run configuration. Add the source commit reference after training during the final version-control step.

## 9. Time risks and fallbacks

| Risk | Fallback |
| --- | --- |
| PaySim too heavy for RAM | Parquet, downcast dtypes, cap the majority class before any model fitting |
| SMOTE-ENN too slow on IEEE-CIS | Skip it, or run on top-100 features by train-only XGBoost importance, and document |
| RF too slow or too large | 6 tuning sets, 200 trees, depth 20, `joblib.dump(..., compress=3)` |
| W&B access not ready | `WANDB_MODE=offline`, sync at handover |
| Day 1 overruns | Move Strategy C on PaySim to the open-task list; protect Day 2 |
| Day 2 overruns | Cut in the order given in section 2; HANDOVER.md still gets written |

## 10. Definition of done at handover

- [ ] Both datasets regenerate from raw with one command each
- [ ] Resampling only on train; transformers fitted on train only; test untouched
- [ ] LR, RF, XGBoost and LightGBM trained per dataset with at least strategies A and B
- [ ] Metrics JSON per dataset from the shared `compute_metrics`
- [ ] Best model, preprocessor and `metadata.json` saved per dataset
- [ ] Data cards, `decisions.md`, `HANDOVER.md` written
- [ ] Git version control completed after both dataset training runs finish
- [ ] Open-task list handed to Anvi and Riyanshu

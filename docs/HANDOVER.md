# FedGuard Stage 1 Handover

**Prepared by:** Naman Dugar  
**Status:** Stage 1 training and evaluation complete; SHAP and Streamlit are later-stage work.  
**Git status:** Final commit and push are pending Naman's review. Local `main` has commit `b785e7e` ahead of `origin/main`; the W&B sync helper and final handoff-document updates are uncommitted.
**Recipients:** Anvi Tyagi and Riyanshu (Riya).

## Handover summary

Naman completed data audit, exploratory analysis, split/preprocessing, model tuning, imbalance-strategy comparisons, selected-model evaluation, and saved per-family comparison models for both PaySim and IEEE-CIS. Both datasets use the same seed (42), stratified 70/15/15 split, train-only preprocessing/resampling, validation PR-AUC model selection, and validation threshold selection subject to FPR at most 1%. The selected model for each dataset was evaluated once on its held-out test split.

Stage 1 did **not** include SHAP or Streamlit. Those are later-stage deliverables. The training, evaluation, W&B summaries, and transfer bundle are complete. Naman's remaining repository task is to review, commit, and push the final source and documentation changes.

## Deliverable status

| Deliverable | Status | Location / notes |
| --- | --- | --- |
| Raw data audit | Done | `scripts/audit_data.py`; `reports/data_audit_summary.json` |
| EDA | Done | `scripts/eda_stage1.py`; `reports/stage1_eda.md` |
| Data cards and decisions | Done | `docs/data_card_paysim.md`, `docs/data_card_ieee_cis.md`, `docs/decisions.md` |
| Shared training/preprocessing code | Done | `src/common/`, `scripts/` |
| PaySim preparation and model-ready splits | Done locally | `data/processed/paysim/` (ignored by Git) |
| IEEE-CIS preparation and model-ready splits | Done locally | `data/processed/ieee_cis/` (ignored by Git) |
| PaySim model search | Done | 117 fits: 39 candidates for each of class weights, capped SMOTE, and capped SMOTE-ENN; `reports/paysim_metrics.json` |
| IEEE-CIS model search | Done | 78 fits: 39 candidates for each of class weights and capped SMOTE; `reports/ieee_cis_metrics.json` |
| IEEE-CIS SMOTE-ENN experiment | Done; not selected | Separate 39-fit validation-only experiment; lower PR-AUC than class weights for all four families; `reports/ieee_cis_smote_enn_validation.json` |
| Selected models and preprocessing artifacts | Done locally | `models/paysim/stage1/`, `models/ieee_cis/stage1/` |
| Four-family comparison models | Done locally | `models/{paysim,ieee_cis}/comparison/`; these were selected/refit on train and checked on validation only, not test |
| PR curve figures | Done | `reports/figures/paysim_pr_curves.png`, `reports/figures/ieee_cis_pr_curves.png` |
| W&B comparison summaries | Done | Four per-model summary runs for each dataset in the `fedguard-stage1` project; see `docs/github_and_wandb.md` |
| SHAP explanations | Not started | Planned after Stage 1 |
| Streamlit app | Not started | Planned after Stage 1 |
| Final Git commit/push | Pending Naman | One earlier local commit is ahead of `origin/main`; the sync helper and final documentation updates still need review and commit |

## Results snapshot

### Selected models: one held-out test evaluation each

| Dataset | Selected model / strategy | Threshold | Validation PR-AUC | Test PR-AUC | Test ROC-AUC | Test precision | Test recall | Test F1 | Test FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PaySim | LightGBM / class weights | 0.0568012 | 0.999986 | 0.997968 | 0.998691 | 0.973080 | 0.997565 | 0.985170 | 0.000036 |
| IEEE-CIS | LightGBM / class weights | 0.7675237 | 0.758107 | 0.756757 | 0.959075 | 0.710193 | 0.687964 | 0.698902 | 0.010178 |

Thresholds maximize recall while constraining validation FPR to at most 1%. IEEE-CIS test FPR is slightly above 1%, as expected from threshold selection on validation rather than tuning on test. Do not adjust the threshold using the test split.

### Validation comparison across model families

These are validation results at each family's threshold selected under the same FPR rule. They are not test metrics. The accompanying `comparison_summary.json` files and family metadata are local under `models/` and excluded from Git.

| Dataset | Model | Selected strategy | Validation PR-AUC | Validation ROC-AUC | Precision | Recall | FPR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| PaySim | LightGBM | class weights | 0.999986 | 1.000000 | 0.983240 | 1.000000 | 0.000022 |
| PaySim | XGBoost | capped SMOTE | 0.999970 | 1.000000 | 0.968553 | 1.000000 | 0.000042 |
| PaySim | Random Forest | capped SMOTE | 0.999909 | 1.000000 | 0.895349 | 1.000000 | 0.000151 |
| PaySim | Logistic Regression | capped SMOTE | 0.636580 | 0.996018 | 0.106539 | 0.909903 | 0.009863 |
| IEEE-CIS | LightGBM | class weights | 0.758107 | 0.960109 | 0.712744 | 0.681935 | 0.009967 |
| IEEE-CIS | XGBoost | class weights | 0.729666 | 0.953336 | 0.704333 | 0.655484 | 0.009979 |
| IEEE-CIS | Random Forest | class weights | 0.635609 | 0.937811 | 0.671791 | 0.563871 | 0.009991 |
| IEEE-CIS | Logistic Regression | class weights | 0.469064 | 0.878558 | 0.584841 | 0.385806 | 0.009932 |

The IEEE-CIS capped SMOTE-ENN follow-up used the same 39-candidate budget and validation protocol but did not outperform class weights: best PR-AUC was 0.429363 (LR), 0.598264 (RF), 0.651665 (XGBoost), and 0.692170 (LightGBM). It ended with 46,123 negative and 36,160 positive training rows after cap + SMOTE-ENN. The test split was not loaded or evaluated in this experiment.

## Data and split summary

| Dataset | Train | Validation | Test | Fraud prevalence | Transformed feature count |
| --- | ---: | ---: | ---: | ---: | ---: |
| PaySim | 4,453,833 (5,749 fraud) | 954,394 (1,232 fraud) | 954,393 (1,232 fraud) | 0.1291% | 18 |
| IEEE-CIS | 413,378 (14,464 fraud) | 88,581 (3,100 fraud) | 88,581 (3,099 fraud) | 3.499% | 1,208 |

IEEE-CIS's train-only >90%-missingness rule dropped `dist2`, `D7`, `id_07`, `id_08`, `id_18`, `id_21`, `id_22`, `id_23`, `id_24`, `id_25`, `id_26`, and `id_27`. Feature order is in each local `data/processed/<dataset>/feature_metadata.json`.

## How to reproduce from raw data

Use Python 3.10 or later. Put the required raw files in `data/raw/` as described in `data/README.md`. Raw data, processed matrices, and trained models are excluded from Git. The current machine already has `.venv`; from PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\audit_data.py
python scripts\eda_stage1.py
python scripts\prepare_stage1.py
```

`prepare_stage1.py` prepares both datasets in one run. Then run either dataset independently:

```powershell
python scripts\train_stage1.py --dataset paysim
python scripts\train_stage1.py --dataset ieee_cis
```

The default strategy sets are defined in `src/common/config.py`: PaySim uses class weights, capped SMOTE, and capped SMOTE-ENN; IEEE-CIS uses class weights and capped SMOTE. Each strategy evaluates 3 Logistic Regression, 6 Random Forest, 15 XGBoost, and 15 LightGBM parameter candidates. XGBoost uses CUDA when the installed build supports it; RF and LightGBM use CPU; the installed LightGBM build has no GPU support. The scripts use seed 42.

After a completed search, the optional scripts below refit the validation-best candidate for each family and save family models/metadata. They do not load the test split:

```powershell
python scripts\save_paysim_comparison_models.py
python scripts\save_ieee_cis_comparison_models.py
```

The separate IEEE-CIS SMOTE-ENN validation experiment is reproducible with:

```powershell
python scripts\run_ieee_cis_smote_enn_validation.py
```

**Do not rerun training merely to inspect the saved results:** the normal training command performs a new fit and evaluates its selected model on the test split once. Existing results are in `reports/*_metrics.json`. The persisted training reports record candidate-level validation results and final selected-model test metrics. PaySim training took about 42 minutes in the recorded run; the IEEE-CIS report does not preserve a reliable end-to-end wall-clock runtime, so no estimate is claimed here.

## Artifact locations and Git handoff

| Purpose | PaySim | IEEE-CIS |
| --- | --- | --- |
| Final selected estimator and metadata | `models/paysim/stage1/` | `models/ieee_cis/stage1/` |
| Per-family estimator and metadata | `models/paysim/comparison/` | `models/ieee_cis/comparison/` |
| Main results | `reports/paysim_metrics.json` | `reports/ieee_cis_metrics.json` |
| PR curve | `reports/figures/paysim_pr_curves.png` | `reports/figures/ieee_cis_pr_curves.png` |
| Prepared matrices, splits, feature mapping | `data/processed/paysim/` | `data/processed/ieee_cis/` |

`.gitignore` excludes raw and processed data and everything under `models/`, including model binaries and their metadata. The transfer bundle is at `D:\MICRO\stage1_transfer`, outside this Git checkout. It includes the raw training inputs used, both datasets' processed splits and feature maps, all saved model artifacts, and the reference documents. It does not include the unused unlabeled IEEE-CIS competition test files. The bundle has a `MANIFEST.sha256` checksum list. It still needs to be zipped and shared separately; do not add it to Git.

The reports, code, requirements, and Stage 1 documentation are intended for Git. **GitHub will not contain the fitted models, transformers, splits, or feature metadata.**

Before the final commit, review `git status` and ensure it contains only intended source, docs, requirements, and compact reports. The current local `main` already contains commit `b785e7e`, which is one commit ahead of `origin/main`; pushing `main` will publish it too. The current model metadata predates the final source commit. Do not retrain solely to add a Git reference.

## Caveats and follow-up ownership

- PaySim is synthetic and its balance behavior makes fraud unusually separable. Its near-perfect score is not evidence of comparable real-world performance.
- IEEE-CIS has 3.499% fraud prevalence and many anonymized features. Its PR-AUC of 0.7568 is the selected model's test result; the PR-AUC need not exceed 0.9 to be useful. Stage 2 explanations should show both source feature names and encoded feature mapping where possible.
- IEEE-CIS SMOTE-ENN did not improve validation PR-AUC for any family; retain class weights as the selected approach unless a future, preplanned experiment produces better validation evidence.
- IEEE-CIS test FPR was 1.0178%, slightly over the 1% validation constraint. The threshold was chosen without test data; do not tune on test to force the test FPR under 1%.
- No temporal split/check was performed. A temporal robustness analysis is an optional follow-up, separate from the completed stratified Stage 1 comparison.
- The saved per-family comparison refits were assessed on validation only. Only the selected LightGBM for each dataset received the final test evaluation.
- Naman: review, commit, and push the remaining source/documentation changes; zip and share `D:\MICRO\stage1_transfer` separately.
- Anvi: schema alignment and the shared model interface remain follow-on tasks; the Stage 1 results snapshot is documented above.
- Riyanshu: validation tests and GitHub workflow/repository setup are present in the repository; GitHub labels and branch protection remain to verify only if the team requires them. W&B project setup and invitations were handled by Naman.
- Anvi and Riya/Riyanshu: agree who leads the Stage 2 SHAP implementation and who leads Streamlit integration; define those responsibilities at handover.

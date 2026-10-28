# FedGuard Micro Project Handover

**Prepared by:** Naman Dugar
**Current scope:** Completed centralized Micro Project using PaySim and IEEE-CIS
**Recipients:** Anvi Tyagi and Riyanshu
**Repository note:** Source, tests, reports, and documentation are versioned. Large datasets and fitted model artifacts remain in the separately shared local transfer bundle and are excluded from Git.

## Project status

The Micro Project combines Stage 1 training/evaluation with the later inference, SHAP, and Streamlit implementation. The current application is a local research demonstration using centralized, dataset-specific LightGBM models. It is not a federated-learning deployment, does not connect to real institutions, and does not block or approve transactions.

The merged project history in the checked repository includes Naman's threshold-validation and dashboard-fix work, Anvi's inference interface, Streamlit integration and demo hardening, and Riyanshu's SHAP core, plain-language reasons, and explanation-view work. This handover summarizes the current code and the mentor's Git and dashboard guidance.

## Deliverable status

| Deliverable | Status | Source / notes |
| --- | --- | --- |
| Data audit and EDA | Complete | scripts/audit_data.py, scripts/eda_stage1.py, reports/data_audit_summary.json, reports/stage1_eda.md |
| Dataset cards and training decisions | Complete | docs/data_card_paysim.md, docs/data_card_ieee_cis.md, docs/decisions.md |
| Train/validation/test preparation | Complete | Stratified 70/15/15 split, seed 42; preprocessing and resampling fitted/applied on train only |
| Four-family model comparison | Complete | Logistic Regression, Random Forest, XGBoost, and LightGBM; validation results only for comparison models |
| PaySim imbalance experiments | Complete | Class weights, capped SMOTE, capped SMOTE-ENN; 117 candidate fits |
| IEEE-CIS imbalance experiments | Complete | Class weights and capped SMOTE; 78 candidate fits. Separate 39-candidate SMOTE-ENN experiment was validation-only and did not improve Average Precision |
| Final selected-model evaluation | Complete | LightGBM with class weights per dataset; each selected model evaluated once on its held-out test split |
| Final threshold validation audit | Complete | reports/final_threshold_validation.json and docs/final_threshold_validation.md; checks validation snapshots without opening test data |
| W&B experiment summaries | Complete | FedGuard Stage 1 project; summaries use saved results and do not retrain |
| Shared inference interface | Complete | src/common/inference.py and tests/test_inference.py |
| SHAP explanation core and evidence-based reasons | Complete | src/common/explain.py and tests/test_explain.py |
| Streamlit demo and error handling | Complete | app/streamlit_app.py and tests/test_dashboard.py; uses local validation examples |
| Reusable explanation view | Implemented | app/components/explanation_view.py and tests/test_explanation_view.py; current app shell uses text-based signed bars rather than this standalone Matplotlib view |
| Public hosting | Not evidenced | Repository and reviewed demo evidence support local execution; do not claim a public deployment |
| Federated learning, differential privacy, secure aggregation | Future work | Not implemented in this Micro Project |

## Evaluation protocol

Both datasets are evaluated separately using a stratified 70/15/15 train/validation/test split with seed 42. Validation and test retain their natural class distributions. Imputation, encoding, feature filtering, scaling, and resampling are learned or applied using training data only.

Models and thresholds are selected using validation results. The operating threshold maximizes recall subject to validation FPR at most 1%. The selected model is then evaluated once on the held-out test split. Test outcomes must not be used to revise thresholds or choose models.

The saved reports call the scalar field pr_auc; for report and dashboard wording, present it consistently as Average Precision (AP). The model outputs have not been calibrated as real-world fraud probabilities; describe them as uncalibrated model scores.

## Selected models: held-out test results

| Dataset | Selected model / strategy | Validation-selected threshold | Validation AP | Test AP | Test ROC-AUC | Test precision | Test recall | Test F1 | Test FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PaySim | LightGBM / class weights | 0.0568012 | 0.999986 | 0.997968 | 0.998691 | 0.973080 | 0.997565 | 0.985170 | 0.003567% |
| IEEE-CIS | LightGBM / class weights | 0.7675237 | 0.758107 | 0.756757 | 0.959075 | 0.710193 | 0.687964 | 0.698902 | 1.017758% |

PaySim validation FPR at the selected threshold was 0.002203%. IEEE-CIS validation FPR was 0.996713%. IEEE-CIS test FPR was 1.017758%, slightly above the validation target; report it as observed and do not tune on test.

| Dataset | Test TP | Test FP | Test FN | Test TN |
| --- | ---: | ---: | ---: | ---: |
| PaySim | 1,229 | 34 | 3 | 953,127 |
| IEEE-CIS | 2,132 | 870 | 967 | 84,612 |

## Four-model validation comparison

These are validation-only family comparisons at the family thresholds selected under the same FPR rule. They are not test results. Family models and their detailed threshold metadata are local artifacts excluded from Git.

| Dataset | Model | Selected strategy | Validation AP | Validation ROC-AUC | Precision | Recall | FPR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| PaySim | LightGBM | Class weights | 0.999986 | 1.000000 | 0.983240 | 1.000000 | 0.0022% |
| PaySim | XGBoost | Capped SMOTE | 0.999970 | 1.000000 | 0.968553 | 1.000000 | 0.0042% |
| PaySim | Random Forest | Capped SMOTE | 0.999909 | 1.000000 | 0.895349 | 1.000000 | 0.0151% |
| PaySim | Logistic Regression | Capped SMOTE | 0.636580 | 0.996018 | 0.106539 | 0.909903 | 0.9863% |
| IEEE-CIS | LightGBM | Class weights | 0.758107 | 0.960109 | 0.712744 | 0.681935 | 0.9967% |
| IEEE-CIS | XGBoost | Class weights | 0.729666 | 0.953336 | 0.704333 | 0.655484 | 0.9979% |
| IEEE-CIS | Random Forest | Class weights | 0.635609 | 0.937811 | 0.671791 | 0.563871 | 0.9991% |
| IEEE-CIS | Logistic Regression | Class weights | 0.469064 | 0.878558 | 0.584841 | 0.385806 | 0.9932% |

The top-level comparison arrays in each dataset metrics JSON record candidate metrics at the default 0.5 threshold. Those entries are not the threshold-optimized family summary above; keep the two result types separate.

The separate IEEE-CIS SMOTE-ENN run was validation-only. Its best AP values were 0.429363 for Logistic Regression, 0.598264 for Random Forest, 0.651665 for XGBoost, and 0.692170 for LightGBM, below the corresponding class-weight results. The test split was not loaded for that experiment.

## Data and feature summary

| Dataset | Train rows / fraud | Validation rows / fraud | Test rows / fraud | Prevalence | Transformed features |
| --- | ---: | ---: | ---: | ---: | ---: |
| PaySim | 4,453,833 / 5,749 | 954,394 / 1,232 | 954,393 / 1,232 | 0.1291% | 18 |
| IEEE-CIS | 413,378 / 14,464 | 88,581 / 3,100 | 88,581 / 3,099 | 3.499% | 1,208 |

PaySim excludes account identifiers and the isFlaggedFraud rule field and includes engineered balance/time signals. IEEE-CIS left-joins labeled identity records to transactions, adds a has_identity indicator, applies a training-only missingness filter, and retains anonymized source names. The exact feature order and fitted transformers are in the local transfer bundle.

## Application behavior

The Streamlit application is app/streamlit_app.py. Its sidebar selects the dataset, scoring model, and local artifact folder. The transaction view accepts a one-row CSV or a small set of validation reference examples and supports editing selected fields. PaySim engineered balance fields are recalculated when relevant balances change. Other IEEE-CIS model inputs stay from the selected validation example. Separate Model results and Scope and limitations views keep validation comparisons distinct from held-out test metrics and explain the demo's limits.

The selected dataset-specific model and threshold are loaded from local artifacts. The dashboard presents the four-family validation ranking comparison separately from the selected models' held-out test results. The demo displays the output score, threshold decision, and local explanation when available. Validation examples are used for interaction; held-out test rows are not offered as demo inputs. Low, medium, and high labels in the interface are heuristic presentation categories based on the validation threshold; they are not calibrated risk bands.

The SHAP core ranks local feature effects and maps transformed fields to source fields where possible. It reports contribution direction and evidence-based statements, not causality or proof of fraud. The current app renders a signed HTML/CSS contribution chart and a detailed table. The reusable Matplotlib explanation view remains available as a separate component.

## Local artifacts and running the demo

A clean Git clone does not include raw/processed data or model artifacts. The separately shared transfer bundle is outside Git. The dashboard defaults to the repository root; either place the bundle's models and data/processed subfolders there, or set FEDGUARD_ARTIFACT_ROOT to the extracted bundle folder or enter that folder in the dashboard sidebar. Keep those large artifacts out of Git.

From PowerShell at the repository root:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python -m streamlit run app\streamlit_app.py

See docs/micro_demo_runbook.md for the demonstration sequence and common setup problems. The current evidence is local execution; a public hosted URL is not documented.

## Reports, figures, and tracking

- PaySim metrics: reports/paysim_metrics.json
- IEEE-CIS metrics: reports/ieee_cis_metrics.json
- IEEE-CIS SMOTE-ENN validation experiment: reports/ieee_cis_smote_enn_validation.json
- Final threshold audit: reports/final_threshold_validation.json
- PaySim PR curves: reports/figures/paysim_pr_curves.png
- IEEE-CIS PR curves: reports/figures/ieee_cis_pr_curves.png
- Dataset audit and EDA: reports/data_audit_summary.json and reports/stage1_eda.md
- W&B project: https://forge.coreweave.com/wandb/ndugar007-bharati-vidyapeeths-college-of-engineering/fedguard-stage1



## Team contributions in the merged project history

| Member | Contribution |
| --- | --- |
| Naman Dugar | Data audit/EDA, preparation, training and model comparison, threshold validation and results; dashboard error/rendering fixes. |
| Anvi Tyagi | Shared inference interface, Streamlit integration and demo hardening. |
| Riyanshu | SHAP explanation core, evidence-based plain-language reasons and reusable explanation-view component. |

The local fetched main history includes the related merged PRs (#5, #6, #7, #8, #9, #10, #11, and #12). Verify contribution wording with the team before using it as an authorship statement.

## Limitations and future work

- PaySim is synthetic. Near-perfect performance on this benchmark does not establish performance on real payment activity.
- PaySim balance-after-transaction fields make the analysis retrospective, not pre-transaction fraud prevention.
- IEEE-CIS has substantial missingness and many anonymized/coded attributes, which limits semantic explanations.
- No probability calibration study is documented. Scores are not calibrated real-world fraud probabilities.
- No temporal robustness experiment is reported.
- The app is a local research demonstration, not a production risk-control service.
- Federated learning, differential privacy, secure aggregation, real institution integration, and public production deployment remain future work.

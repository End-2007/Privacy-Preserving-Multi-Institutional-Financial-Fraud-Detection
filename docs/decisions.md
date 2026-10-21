# Stage 1 Decisions Log

This log records decisions for the current centralized training stage. SHAP explanations and the Streamlit dashboard are planned after Stage 1 and are outside its current exit criteria.

| ID | Decision | Rationale / evidence | Status |
| --- | --- | --- | --- |
| D1 | Stage 1 covers data preparation, imbalance handling, baseline model training, and evaluation for PaySim and IEEE-CIS. SHAP and Streamlit follow after this stage. | Naman's confirmed stage boundary. | Confirmed |
| D2 | Keep raw datasets and generated model/data artifacts out of Git. | Raw files are large; the repository's `.gitignore` already excludes raw/processed data and model artifacts. | Confirmed |
| D3 | Use stratified 70/15/15 train/validation/test splits with seed 42. Fit preprocessing and resampling on train only; select models and threshold on validation; evaluate the chosen final model on test once. | Active Stage 1 plan and leakage controls. | Applied |
| D4 | Use the labeled IEEE-CIS training transaction and identity files for Stage 1. Left-join identity onto transactions by `TransactionID`, add `has_identity`, and exclude `TransactionID` from predictors. | Local audit found 590,540 transaction rows, 144,233 matching identity rows, no duplicate keys, and 24.42% identity coverage. | Confirmed |
| D5 | Apply the IEEE-CIS greater-than-90%-missing feature rule based on the training partition only. Save the final dropped-column list. | Prevents validation/test rows from influencing feature selection. The final list is in `data/processed/ieee_cis/feature_metadata.json`. | Applied |
| D6 | Train and compare Logistic Regression, Random Forest, XGBoost, and LightGBM for each dataset. | Naman explicitly added LightGBM to the Stage 1 model list. The updated two-day plan uses 15 tuning sets each for XGBoost and LightGBM, six for Random Forest, and three for Logistic Regression. | Confirmed |
| D7 | Keep Git version control until both dataset training pipelines and their results are complete. Record run configuration, seed, input files, package versions, and metrics during training; add the source commit reference during end-of-stage version control. | Naman's explicit workflow preference. | Confirmed |
| D8 | Save each prepared split as Parquet, save model-ready CSR matrices and labels, and persist train-fitted preprocessors. Use the base matrix for tree models and a train-fitted scaled matrix for Logistic Regression. | Preparation completed by `scripts/prepare_stage1.py`; split manifests and feature metadata record the outputs. | Applied |
| D9 | Use a local `.venv` for the project dependencies; keep the existing Anaconda installation untouched. | The Anaconda environment had a scikit-learn/imbalanced-learn incompatibility and lacked LightGBM. All declared requirements now install in `.venv`; `pip check` reports no broken requirements. | Applied |
| D10 | Cap only training negatives before SMOTE: 1:20 for PaySim and 1:5 for IEEE-CIS; target 1:2 after SMOTE. Use SMOTE-ENN only for PaySim in the default run. | Stage 1 compute/runtime constraints and the high dimensional IEEE-CIS encoded matrix. Class-weight and capped-SMOTE comparisons remain available for both datasets. | Confirmed |
| D11 | Compare the four models and strategies by validation PR-AUC; choose the threshold on validation to maximize recall with FPR at most 1%; evaluate the selected model on test once. | PaySim completed 117 candidate fits; IEEE-CIS completed 78. Each dataset's saved report records the final validation selection and one test evaluation of the selected model. | Applied for both datasets |
| D12 | Use CUDA for XGBoost when the installed build exposes CUDA; run Random Forest and LightGBM on CPU with available threads. | The RTX 4060 is available, XGBoost reports `USE_CUDA=true`, and the installed LightGBM build has no GPU entry point. The runner falls back to CPU if an XGBoost CUDA fit fails. | Applied |
| D13 | Fit and persist the Logistic Regression scaler under the active project environment before training. | Preparation artifacts were made with scikit-learn 1.3, while `.venv` uses 1.9.1. The runner now refits the scaler using only prepared train features, updates the train/validation scaled matrices, and avoids unpickling the older scaler. | Applied for PaySim |
| D14 | PaySim model selection is LightGBM with class weights, threshold 0.0568012. | Validation PR-AUC 0.999986; one test evaluation gave PR-AUC 0.997968, ROC-AUC 0.998691, precision 0.973080, recall 0.997565, F1 0.985170, and FPR 0.00003567. Results and all 12 model/strategy comparisons are in `reports/paysim_metrics.json`. | Complete |
| D15 | IEEE-CIS model selection is LightGBM with class weights, threshold 0.7675237. | Validation PR-AUC 0.758107; one test evaluation gave PR-AUC 0.756757, ROC-AUC 0.959075, precision 0.710193, recall 0.687964, F1 0.698902, and FPR 0.010178. Results and all 8 model/strategy comparisons are in `reports/ieee_cis_metrics.json`. | Complete |
| D16 | Do not use capped SMOTE-ENN for the IEEE-CIS winner. | A separate 39-candidate validation-only run completed without loading/evaluating test. Its best PR-AUC by family was below that family's class-weighted result; LightGBM scored 0.692170 versus 0.758107. Full results: `reports/ieee_cis_smote_enn_validation.json`. | Complete |

## Audit evidence

The raw-file audit was run on 2026-10-21 using `scripts/audit_data.py`. The detailed counts and per-column missingness are stored in `reports/data_audit_summary.json`. Key label counts:

- PaySim: 6,362,620 rows, 8,213 fraud labels (0.1291%); no missing values; no duplicate full rows detected.
- IEEE-CIS labeled transactions: 590,540 rows, 20,663 fraud labels (3.499%).
- IEEE-CIS identity: 144,233 rows, all keys matched to transactions; no duplicate keys.

The 70/15/15 splits and train-only preprocessing were generated on 2026-10-21. See each dataset's `split_manifest.json` and `feature_metadata.json` for exact split counts, fitted feature lists, and matrix shapes.

## Training setup status

- The project-local environment is `.venv`; activate it with `\.venv\Scripts\Activate.ps1` in PowerShell.
- The declared dependencies are installed and `python -m pip check` reports no broken requirements.
- PaySim training completed on 2026-10-21: 117 candidate fits (39 candidates per strategy across class weights, capped SMOTE, and capped SMOTE-ENN). The recorded run took about 42 minutes from W&B run start to report creation.
- IEEE-CIS training completed on 2026-10-21: 78 candidate fits (39 candidates per strategy across class weights and capped SMOTE). The selected model was evaluated once on the held-out test split.
- IEEE-CIS SMOTE-ENN was also run as a separate 39-candidate validation-only experiment. It did not improve PR-AUC for any family; the test split was not loaded or evaluated for that experiment.
- Selected models, preprocessing artifacts, family comparison models, and their metadata are saved under `models/` locally. Raw and processed data and model binaries are excluded from Git; arrange a separate artifact transfer if teammates need the fitted binaries rather than rerunning from the raw data.
- SHAP and Streamlit remain post-Stage 1 work.
- The PaySim score is exceptionally high on this synthetic dataset; interpret it with the documented balance-rule caveat in `docs/data_card_paysim.md`.

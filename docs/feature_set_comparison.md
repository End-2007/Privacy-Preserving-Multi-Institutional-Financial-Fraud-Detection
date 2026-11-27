# Model A and Model B feature-set comparison

Anvi's reviewed lists define two configurations for each dataset:

- **Model A (`full`)** uses eligible source predictors after targets and technical identifiers are excluded.
- **Model B (`auth`)** uses source predictors classified as available at authorization time.

The four newline-delimited inputs live under `configs/features/`. `src/common/feature_sets.py` reads them by dataset and feature-set name. It fails on duplicate entries or source columns missing from the prepared data instead of silently training on a smaller list.

## Reproducible preparation

First prepare the raw datasets using the existing seed-42 stratified 70/15/15 split:

```powershell
python scripts/prepare_stage1.py
```

Then prepare the two feature configurations from those same saved split Parquet files:

```powershell
python scripts/prepare_feature_sets.py --datasets paysim ieee_cis --feature-sets full auth
```

The feature-set preparation does not create a new split. Imputation, encoding, missingness filtering, and scaling are fit on each configuration's training rows only; validation and test rows are transformed with those training-fitted objects. Features above 90% missing are removed using the training split and recorded in `feature_metadata.json`.

Engineered columns are added only when their source fields are present in the selected list. For PaySim, `step` enables `hour`, `amount` enables `log_amount`, and post-transaction balance features and their derivatives are available only to `full`. For IEEE-CIS, `TransactionDT` enables `transaction_hour` and `transaction_day`, while `TransactionAmt` enables `log_amt`. Identifier-derived `dest_is_merchant` and generated `has_identity` are not added to either configuration unless a reviewed feature list explicitly includes them.

## Model runs

Run each dataset/configuration separately. Model-family and strategy selection use validation Average Precision (AP), and the threshold is selected on validation by maximizing recall while keeping validation FPR at or below 1%.

```powershell
python scripts/train_stage1.py --dataset paysim --feature-set full
python scripts/train_stage1.py --dataset paysim --feature-set auth
python scripts/train_ieee_feature_set_lightgbm.py
python scripts/train_ieee_feature_set_xgboost.py
python scripts/train_ieee_feature_set_random_forest.py
python scripts/train_ieee_feature_set_logistic_regression.py
python scripts/finalize_feature_comparison.py
```

The PaySim commands search all four model families for each feature set. Each IEEE-CIS runner handles both full and auth sets for its model family. The finalization command uses the saved validation-selected IEEE-CIS models, the saved PaySim LightGBM models, and refits the validation-best PaySim candidate for each other family before producing per-configuration results. It does not use test metrics to change a model or threshold.

Processed matrices and preprocessing metadata are kept separately under `data/processed/<dataset>/feature_sets/<full|auth>/`. Models and scalers are saved under `models/<dataset>/feature_sets/<full|auth>/`; model binaries remain local and are excluded from Git. The separate Model A/Model B outputs are written under `reports/feature_set_results/` and run configurations under `configs/experiments/feature_set_comparison/`.

## Completed four-family comparison

The completed comparison includes Logistic Regression, Random Forest, XGBoost, and LightGBM for PaySim and IEEE-CIS under both feature sets: 16 configurations total. Each configuration has a validation-selected threshold, held-out test metrics, a validation/test prediction file with source row ID and uncalibrated score, and a model-specific feature-importance ranking.

- [Full Model A/Model B analysis and results](../reports/feature_set_results/model_a_vs_model_b_analysis.md)
- [`all_configurations_metrics.csv`](../reports/feature_set_results/all_configurations_metrics.csv): per-configuration validation and test scores, thresholds, and confusion counts.
- [`model_a_vs_model_b_summary.csv`](../reports/feature_set_results/model_a_vs_model_b_summary.csv): paired full/auth AP, recall, and FPR differences.
- `reports/feature_set_results/predictions/`: 16 compressed CSV files containing validation and test row predictions.
- `reports/feature_set_results/feature_importance/`: full transformed-feature rankings for every configuration.
- `reports/feature_set_results/figures/`: one validation/test confusion matrix and PR curve per configuration, plus paired full/auth feature-importance charts.
- `reports/feature_set_results/metrics/` and `configs/experiments/feature_set_comparison/`: per-configuration metrics and reproducibility records.

The validation FPR target is not a guarantee about test FPR. Some held-out test FPRs are slightly above 1%; these are reported as observed and were not used to adjust thresholds. PaySim is synthetic and Model A includes post-transaction balance-derived features, so its full-feature scores are retrospective benchmark results. IEEE-CIS feature timing is partly uncertain; the comparison follows Anvi's declared lists and does not establish real-world availability for every field.

## Interpretation limits

The full/auth lists encode reviewable availability assumptions. In particular, IEEE-CIS anonymized fields remain out of `auth` until their operational timing is documented. A difference in test metrics is a comparison between these defined feature configurations, not evidence that every selected field is available in a real authorization workflow.

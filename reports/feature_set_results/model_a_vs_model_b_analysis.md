# Full and authorization-time model comparison

This report compares all four model families for PaySim and IEEE-CIS under the reviewed full and authorization-time feature lists. Model selection and the operating threshold use validation data. The held-out test split is reported separately and is not used for selection.

## Scope and interpretation

PaySim is synthetic; its full feature set includes post-transaction balance information and its high scores should be interpreted as retrospective benchmark performance. IEEE-CIS includes anonymized and timing-uncertain fields in the full list; the authorization-time comparison follows the conservative classifications in the feature lists. The A/B differences describe these declared feature configurations and do not prove real-world availability for each field.

The test predictions were exported after each model and validation threshold were fixed. Scores are model outputs and are not calibrated fraud probabilities.

Feature-importance ranks are reported on transformed inputs. Logistic Regression uses absolute coefficients from train-scaled features; tree models use their native feature_importances_ measure. These scales differ across model families, so interpret importance within each family and feature set.

## Per-configuration held-out results

| Dataset | Model | Feature set | Strategy | Validation AP | Test AP | Test ROC-AUC | Precision | Recall | F1 | FPR | Threshold |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ieee_cis | LightGBM | Model A (Full) | class_weight | 0.759261 | 0.755636 | 0.960174 | 0.7154 | 0.6896 | 0.7023 | 0.9944% | 0.773661 |
| ieee_cis | LightGBM | Model B (Authorization-time) | cap_smote | 0.469469 | 0.463604 | 0.911509 | 0.5618 | 0.3830 | 0.4555 | 1.0833% | 0.677705 |
| ieee_cis | Logistic Regression | Model A (Full) | class_weight | 0.469068 | 0.465952 | 0.875318 | 0.5877 | 0.3837 | 0.4643 | 0.9756% | 0.906840 |
| ieee_cis | Logistic Regression | Model B (Authorization-time) | cap_smote | 0.143720 | 0.150716 | 0.756948 | 0.2833 | 0.1175 | 0.1661 | 1.0774% | 0.818309 |
| ieee_cis | Random Forest | Model A (Full) | class_weight | 0.632591 | 0.632362 | 0.938532 | 0.6731 | 0.5595 | 0.6111 | 0.9850% | 0.615724 |
| ieee_cis | Random Forest | Model B (Authorization-time) | class_weight | 0.538728 | 0.528824 | 0.913079 | 0.5792 | 0.4201 | 0.4870 | 1.1067% | 0.674994 |
| ieee_cis | XGBoost | Model A (Full) | class_weight | 0.729780 | 0.724262 | 0.952564 | 0.6905 | 0.6538 | 0.6716 | 1.0622% | 0.777486 |
| ieee_cis | XGBoost | Model B (Authorization-time) | class_weight | 0.412559 | 0.406359 | 0.906549 | 0.5150 | 0.3207 | 0.3953 | 1.0950% | 0.875097 |
| paysim | LightGBM | Model A (Full) | class_weight | 0.999984 | 0.997938 | 0.999055 | 0.9685 | 0.9976 | 0.9828 | 0.0042% | 0.002780 |
| paysim | LightGBM | Model B (Authorization-time) | class_weight | 0.948516 | 0.938744 | 0.999239 | 0.1241 | 0.9976 | 0.2207 | 0.9103% | 0.018200 |
| paysim | Logistic Regression | Model A (Full) | cap_smote | 0.660690 | 0.648362 | 0.996039 | 0.1063 | 0.9180 | 0.1906 | 0.9972% | 0.692493 |
| paysim | Logistic Regression | Model B (Authorization-time) | cap_smote | 0.265580 | 0.259702 | 0.977850 | 0.0789 | 0.6542 | 0.1409 | 0.9868% | 0.849563 |
| paysim | Random Forest | Model A (Full) | class_weight | 0.999941 | 0.997568 | 0.998782 | 0.9178 | 0.9976 | 0.9560 | 0.0115% | 0.029974 |
| paysim | Random Forest | Model B (Authorization-time) | cap_smote | 0.885745 | 0.866489 | 0.998027 | 0.1106 | 0.9667 | 0.1985 | 1.0049% | 0.403478 |
| paysim | XGBoost | Model A (Full) | class_weight | 0.999981 | 0.997789 | 0.999238 | 0.9670 | 0.9976 | 0.9820 | 0.0044% | 0.080336 |
| paysim | XGBoost | Model B (Authorization-time) | class_weight | 0.937873 | 0.931217 | 0.999527 | 0.1322 | 0.9951 | 0.2334 | 0.8441% | 0.141283 |

## Model B minus Model A observations

## PAYSIM

Held-out Average Precision (AP) compares the exact feature lists recorded in the run configs. It is a dataset-specific result and does not imply production performance.

- **Logistic Regression:** Model B test AP was 0.3887 (-0.3887; -59.9%) lower than Model A (0.6484 vs 0.2597). Test recall was 91.802% for full and 65.422% for auth; test FPR was 0.997% and 0.987%, respectively.
  Top transformed features by each model's native importance measure — full: categorical__type_CASH_IN (34.3%), categorical__type_PAYMENT (11.0%), categorical__type_CASH_OUT (7.8%), numeric__orig_balance_delta (5.9%), numeric__orig_zero_after (5.8%); auth: categorical__type_CASH_IN (59.1%), numeric__oldbalanceOrg (15.5%), categorical__type_CASH_OUT (10.6%), categorical__type_TRANSFER (6.5%), categorical__type_PAYMENT (5.7%). Importance scales are model-specific and should not be compared across families.
- **Random Forest:** Model B test AP was 0.1311 (-0.1311; -13.1%) lower than Model A (0.9976 vs 0.8665). Test recall was 99.756% for full and 96.672% for auth; test FPR was 0.012% and 1.005%, respectively.
  Top transformed features by each model's native importance measure — full: numeric__error_balance_orig (39.2%), numeric__orig_balance_delta (22.3%), numeric__newbalanceOrig (13.4%), numeric__orig_zero_after (9.7%), numeric__oldbalanceOrg (7.1%); auth: categorical__type_TRANSFER (25.1%), numeric__oldbalanceOrg (18.6%), numeric__amount (14.1%), numeric__log_amount (10.8%), numeric__oldbalanceDest (8.7%). Importance scales are model-specific and should not be compared across families.
- **XGBoost:** Model B test AP was 0.0666 (-0.0666; -6.7%) lower than Model A (0.9978 vs 0.9312). Test recall was 99.756% for full and 99.513% for auth; test FPR was 0.004% and 0.844%, respectively.
  Top transformed features by each model's native importance measure — full: numeric__error_balance_orig (51.1%), numeric__newbalanceOrig (40.5%), numeric__log_amount (3.0%), numeric__orig_balance_delta (1.4%), numeric__error_balance_dest (1.0%); auth: categorical__type_CASH_IN (57.0%), categorical__type_PAYMENT (21.7%), numeric__log_amount (7.3%), numeric__oldbalanceOrg (4.0%), numeric__amount (2.3%). Importance scales are model-specific and should not be compared across families.
- **LightGBM:** Model B test AP was 0.0592 (-0.0592; -5.9%) lower than Model A (0.9979 vs 0.9387). Test recall was 99.756% for full and 99.756% for auth; test FPR was 0.004% and 0.910%, respectively.
  Top transformed features by each model's native importance measure — full: numeric__newbalanceDest (13.3%), numeric__oldbalanceDest (12.7%), numeric__step (12.7%), numeric__error_balance_orig (11.0%), numeric__dest_balance_delta (9.3%); auth: numeric__oldbalanceOrg (21.1%), numeric__amount (19.8%), numeric__step (17.8%), numeric__oldbalanceDest (15.1%), numeric__hour (11.3%). Importance scales are model-specific and should not be compared across families.

## IEEE-CIS

Held-out Average Precision (AP) compares the exact feature lists recorded in the run configs. It is a dataset-specific result and does not imply production performance.

- **Logistic Regression:** Model B test AP was 0.3152 (-0.3152; -67.7%) lower than Model A (0.4660 vs 0.1507). Test recall was 38.367% for full and 11.746% for auth; test FPR was 0.976% and 1.077%, respectively.
  Top transformed features by each model's native importance measure — full: numeric__V332 (4.1%), numeric__C11 (3.3%), numeric__V179 (3.1%), numeric__V323 (2.8%), numeric__C14 (2.3%); auth: numeric__log_amt (3.8%), categorical__addr2_87.0 (3.1%), categorical__card6_debit (2.3%), categorical__ProductCD_W (2.2%), frequency__P_emaildomain (2.2%). Importance scales are model-specific and should not be compared across families.
- **Random Forest:** Model B test AP was 0.1035 (-0.1035; -16.4%) lower than Model A (0.6324 vs 0.5288). Test recall was 55.954% for full and 42.014% for auth; test FPR was 0.985% and 1.107%, respectively.
  Top transformed features by each model's native importance measure — full: numeric__V258 (8.9%), numeric__C14 (4.1%), numeric__V317 (3.5%), numeric__V264 (3.5%), numeric__C13 (3.4%); auth: numeric__TransactionDT (11.4%), frequency__card1 (9.3%), numeric__log_amt (7.9%), frequency__addr1 (7.7%), numeric__TransactionAmt (7.6%). Importance scales are model-specific and should not be compared across families.
- **XGBoost:** Model B test AP was 0.3179 (-0.3179; -43.9%) lower than Model A (0.7243 vs 0.4064). Test recall was 65.376% for full and 32.075% for auth; test FPR was 1.062% and 1.095%, respectively.
  Top transformed features by each model's native importance measure — full: numeric__V258 (12.8%), numeric__V222 (5.9%), numeric__V201 (4.8%), numeric__V90 (4.5%), numeric__V70 (3.8%); auth: categorical__addr2_87.0 (40.8%), categorical__card3_185.0 (5.8%), categorical__card6_credit (2.7%), frequency__R_emaildomain (2.4%), categorical__ProductCD_C (2.2%). Importance scales are model-specific and should not be compared across families.
- **LightGBM:** Model B test AP was 0.2920 (-0.2920; -38.6%) lower than Model A (0.7556 vs 0.4636). Test recall was 68.958% for full and 38.303% for auth; test FPR was 0.994% and 1.083%, respectively.
  Top transformed features by each model's native importance measure — full: frequency__card1 (6.1%), frequency__card2 (4.8%), numeric__TransactionAmt (4.5%), numeric__TransactionDT (4.2%), frequency__addr1 (3.9%); auth: numeric__TransactionDT (13.3%), frequency__card1 (12.7%), frequency__addr1 (11.3%), frequency__card2 (9.6%), numeric__TransactionAmt (8.8%). Importance scales are model-specific and should not be compared across families.

## Saved outputs

- `all_configurations_metrics.csv`: all 16 model/dataset/feature-set results, including confusion counts.
- `model_a_vs_model_b_summary.csv`: paired AP, recall, and FPR deltas.
- `metrics/`: per-configuration JSON with thresholds, parameters, package versions, and output references.
- `predictions/`: validation and test row predictions as compressed CSV files.
- `feature_importance/`: complete transformed-feature importance rankings for each configuration.
- `figures/`: validation/test confusion matrices and precision-recall curves for each configuration.

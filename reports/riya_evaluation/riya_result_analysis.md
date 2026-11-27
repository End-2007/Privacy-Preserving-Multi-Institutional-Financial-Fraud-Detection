# Riyanshu: held-out feature-set evaluation

This evaluation reads Naman's saved prediction files for the eight assigned LightGBM/XGBoost configurations. It does not train models, change thresholds, or score test matrices again. Thresholded metrics use the saved predicted labels; AP is recalculated from the saved test scores. Each result is checked against its per-run metrics JSON.

## Model A vs Model B

| Dataset | Model | A AP | B AP | A Precision | B Precision | A Recall | B Recall | A FPR | B FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| paysim | LightGBM | 0.9979 | 0.9387 | 0.9685 | 0.1241 | 0.9976 | 0.9976 | 0.004% | 0.910% |
| paysim | XGBoost | 0.9978 | 0.9312 | 0.9670 | 0.1322 | 0.9976 | 0.9951 | 0.004% | 0.844% |
| ieee_cis | LightGBM | 0.7556 | 0.4636 | 0.7154 | 0.5618 | 0.6896 | 0.3830 | 0.994% | 1.083% |
| ieee_cis | XGBoost | 0.7243 | 0.4064 | 0.6905 | 0.5150 | 0.6538 | 0.3207 | 1.062% | 1.095% |

## Observations by dataset

### paysim

- **LightGBM:** Model B AP changed by -5.9% (0.9979 to 0.9387); precision 0.9685 to 0.1241; recall 0.9976 to 0.9976; FPR 0.004% to 0.910%.
- **XGBoost:** Model B AP changed by -6.7% (0.9978 to 0.9312); precision 0.9670 to 0.1322; recall 0.9976 to 0.9951; FPR 0.004% to 0.844%.

### ieee_cis

- **LightGBM:** Model B AP changed by -38.6% (0.7556 to 0.4636); precision 0.7154 to 0.5618; recall 0.6896 to 0.3830; FPR 0.994% to 1.083%.
- **XGBoost:** Model B AP changed by -43.9% (0.7243 to 0.4064); precision 0.6905 to 0.5150; recall 0.6538 to 0.3207; FPR 1.062% to 1.095%.

## Interpretation and limitations

For PaySim, authorization-time features retain high AP and recall, but precision is much lower and false positives increase compared with the full set. The full set contains post-transaction balance information, so its stronger scores are retrospective and are not a fair authorization-time expectation.

For IEEE-CIS, restricting the feature set produces a larger reduction in AP and recall for both models. The anonymized feature groups remain excluded from Model B because their operational timing is not established; the comparison reflects the reviewed feature lists, not proof that every Model A field is available in practice.

The repository records a stratified 70/15/15 split with seed 42. This is the implemented team decision; it is not a future-time evaluation. Validation FPR was constrained to at most 1% when choosing thresholds, but that does not guarantee test FPR at or below 1%. Test values above 1% are reported as observed and must not be used to retune thresholds.

The eight confusion-matrix SVGs are in the confusion_matrices folder. The full metric rows, including confusion counts, are in riya_evaluation_metrics.csv. All eight rows passed comparison against Naman's saved per-run metrics.

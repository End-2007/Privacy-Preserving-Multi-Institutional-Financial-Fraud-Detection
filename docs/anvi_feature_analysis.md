# Anvi Feature Classification

This handoff defines the two feature configurations for the FedGuard
authorization-time comparison. Model A is the full eligible predictor
set after removing the target and technical identifiers. Model B keeps
only features classified as available at authorization time.

## Rules used

- `isFraud` is the target and is excluded from both models.
- `TransactionID`, `row_id`, and account identifiers are excluded as
  technical or entity identifiers.
- Post-transaction balances are excluded from Model B because they are
  not known when authorization is decided.
- Anonymized IEEE-CIS groups (`C*`, `D*`, `M*`, and `V*`) are retained
  in Model A but excluded from Model B until their operational timing
  is documented.
- IEEE-CIS identity and device fields are treated conservatively as
  uncertain because the supplied data does not prove when they become
  available.
- PaySim `isFlaggedFraud` is excluded from both models and marked high
  leakage risk, following the active Stage 1 decision for this
  rule-based screening output.

## Files

- `configs/features/PaySim_full.txt` and `PaySim_auth.txt`
- `configs/features/IEEECIS_full.txt` and `IEEECIS_auth.txt`
- `docs/feature_classification_paysim.csv`
- `docs/feature_classification_ieee_cis.csv`

The CSV files preserve the classification, model inclusion, leakage
risk, and rationale for every source column. These are reviewable
assumptions, not claims that masked feature semantics are known.

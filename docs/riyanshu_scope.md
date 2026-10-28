# Team contributions and Micro Project handoff

This note summarizes the contributions represented in the merged Micro Project history. It replaces the earlier Stage 1 handoff status, which predated the inference, SHAP, and Streamlit PRs.

## Contributions in the merged project

| Member | Work |
| --- | --- |
| Naman Dugar | Data audit and EDA, split/preprocessing, model training and strategy comparisons, selected-model thresholds and test evaluation, threshold validation (PR #8), and dashboard error/rendering fixes (PR #9). |
| Anvi Tyagi | Shared inference interface (PR #5), Streamlit application integration (PR #6), and demo hardening (PR #7). |
| Riyanshu | SHAP explanation core (PR #10), evidence-based plain-language reasons (PR #11), and reusable XAI view component (PR #12). |

The checked main history contains merges for these PRs. Confirm the contribution descriptions with the team before using them as formal authorship statements.

## Current implementation boundary

The current Micro Project is a local centralized demonstration using separate PaySim and IEEE-CIS models. It includes saved-model inference, validation-selected thresholds, local SHAP contributions/reasons, and the Streamlit app. It is not a federated system and does not include differential privacy or secure aggregation.

Large model/data artifacts remain in the separately shared transfer bundle and are excluded from Git. A fresh checkout needs the matching models and processed data placed in the paths described in docs/micro_demo_runbook.md.

## Remaining project direction

This final documentation PR aligns README.md, ARCHITECTURE.md, the runbook, and this handoff with the merged code. Federated learning, differential privacy, and secure aggregation remain future research stages. GitHub labels and branch protection are repository settings; this note does not assert that they have been configured.

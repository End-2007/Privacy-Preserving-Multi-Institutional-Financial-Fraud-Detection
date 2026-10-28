# FedGuard Micro Project Demo Runbook

This runbook covers the local centralized demo for PaySim and IEEE-CIS. It does not require federated-learning services or raw data once the Stage 1 artifacts and validation examples are available.

## Before the demo

1. Open a terminal at the repository root.
2. Obtain the separately shared Stage 1 transfer bundle. A fresh Git clone does not include the models, preprocessors, feature metadata, or processed validation examples.
3. Extract the separately shared bundle. Keep it outside Git. The app defaults to the repository root; either copy the bundle's models and data/processed folders into matching locations under the repository root, set `FEDGUARD_ARTIFACT_ROOT` to the extracted bundle folder, or enter that folder in the dashboard sidebar.
4. Create and activate an environment and install the declared dependencies:

       python -m venv .venv
       .\.venv\Scripts\Activate.ps1
       python -m pip install -r requirements.txt

5. Start the app:

       python -m streamlit run app\streamlit_app.py

The app expects model bundles under `models/<dataset>`, feature metadata, and validation reference data at `data/processed/<dataset>/validation.parquet` within the selected artifact root. Each dataset directory has a `stage1` selected-model bundle and `comparison/<model-family>` bundles for interactive model selection.

## Demonstration flow

1. Choose PaySim or IEEE-CIS in the sidebar and select a scoring model.
2. In **Transaction score + SHAP**, choose either a one-row CSV or a validation reference example. The CSV template is label-free; demo examples come from validation data.
3. For PaySim, edit transaction type, amount, and sender/receiver balances. The app recalculates the associated engineered balance features. For IEEE-CIS, edit the displayed card/e-commerce fields; other inputs remain from the selected validation example.
4. Score the transaction and show the uncalibrated model score, validation-selected threshold, threshold decision, heuristic interface category, and SHAP contribution view/table.
5. Open **Model results** to show the four-family validation comparison and the selected LightGBM held-out test metrics as separate result groups.
6. Open **Scope and limitations**. For IEEE-CIS coded fields, preserve source names and avoid inventing business meanings.

## Interpretation notes

- A model score is not a confirmed fraud finding and is not calibrated as a real-world fraud probability.
- The selected threshold was chosen on validation data. The held-out test data is not used for interactive examples, model choice, or threshold tuning.
- SHAP values describe how input features affected this model’s raw output for this one row. They do not establish causation.
- PaySim is synthetic and uses balance fields observed after transactions, so its example is retrospective analysis rather than pre-transaction prevention.
- The demo is for research review; it does not block payments or make customer decisions.

## Common issues

- **Missing model or preprocessing files:** obtain the transfer bundle and copy the models and processed-data directories into the expected local paths. Do not commit the bundle.
- **Missing validation examples:** confirm that data/processed/paysim/validation.parquet and data/processed/ieee_cis/validation.parquet exist.
- **Prediction error:** confirm the selected dataset’s model, preprocessor, metadata, and feature metadata all belong to the same bundle and that its Python environment matches the saved artifacts.
- **Explanation error:** the app reports a warning and leaves the risk output visible. Continue by explaining the score and threshold; do not present a missing explanation as a successful SHAP result.
- **Browser disconnect or frontend rendering failure:** confirm the Streamlit process is still running in the terminal. The current app uses a self-contained HTML/CSS signed-contribution chart rather than the older lazily loaded Plotly chart path.

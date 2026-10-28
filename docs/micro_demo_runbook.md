# FedGuard Micro Project Demo Runbook

This runbook covers the local centralized demo for PaySim and IEEE-CIS. It does not require federated-learning services or raw data once the Stage 1 artifacts and validation examples are available.

## Before the demo

1. Open a terminal at the repository root.
2. Obtain the separately shared Stage 1 transfer bundle. A fresh Git clone does not include the models, preprocessors, feature metadata, or processed validation examples.
3. Copy the bundle’s models and data/processed folders into matching locations under the repository root. Do not add these large artifacts to Git.
4. Create and activate an environment and install the declared dependencies:

       python -m venv .venv
       .\.venv\Scripts\Activate.ps1
       python -m pip install -r requirements.txt

5. Start the app:

       python -m streamlit run app\streamlit_app.py

The app expects selected model bundles under models/paysim and models/ieee_cis, processed feature metadata, and validation reference data at data/processed/<dataset>/validation.parquet.

## Demonstration flow

1. Open the PaySim Risk Check tab and choose a reference transaction. The reference comes from validation data.
2. Edit transaction type, amount, and sender/receiver balances. The app recalculates the associated engineered balance features.
3. Submit the form. Show the uncalibrated model score, the validation-selected threshold, the threshold decision, and the explanation output.
4. Open the IEEE-CIS Risk Check tab and choose a validation reference. Edit the displayed card/e-commerce fields; other inputs remain from that reference example.
5. Submit and show the score, threshold decision, and explanation.
6. Show the centralized performance summary. Explain that the selected LightGBM test metrics are reported separately and the interactive inputs use validation examples.
7. For a SHAP explanation, point out the source/model feature names, contribution direction, and evidence-based reasons. For IEEE-CIS coded fields, preserve the source names and avoid inventing their business meaning.

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
- **Browser disconnect or frontend rendering failure:** confirm the Streamlit process is still running in the terminal. The current app shell uses text-based signed contribution bars to avoid relying on the older lazily loaded browser chart path.

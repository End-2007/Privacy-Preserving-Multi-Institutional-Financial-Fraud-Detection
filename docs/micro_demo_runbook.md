# FedGuard Micro Project Demo Runbook

This runbook is for the centralized Micro Project demonstration. It does not require federated-learning services or access to raw datasets.

## Before the demo

1. Activate the project virtual environment.
2. Confirm that the local model transfer bundle has populated `models/paysim/`, `models/ieee_cis/`, and the corresponding `data/processed/` folders. These artifacts are intentionally excluded from Git.
3. Run the test suite:

   ```powershell
   python -m pytest -q
   ```

4. Start the dashboard:

   ```powershell
   python -m streamlit run app/streamlit_app.py
   ```

## Demonstration flow

1. Open **PaySim Risk Check** and choose a reference example.
2. Change the amount or account balances and select **Check PaySim risk**.
3. Show the fraud probability, decision threshold, predicted class, and risk category.
4. Open **IEEE-CIS Risk Check** and change the visible card/e-commerce fields.
5. Select **Check IEEE-CIS risk** and show the same risk outputs.
6. Show the centralized model-performance table below the tabs.
7. If the SHAP module is installed, show the plain-language reasons and feature-contribution chart. If it is not available, explain that the risk score remains functional and the dashboard displays a clear fallback message.

## Important interpretation note

The output is a model-generated risk signal, not proof that a transaction is fraudulent. The dashboard uses validation examples for demonstration and does not expose the held-out test split for interactive selection.

## Common issues

- **Missing artifacts:** obtain the separately shared stage transfer bundle and place its contents in the expected local folders. Do not commit the files.
- **Prediction failed:** confirm that the virtual environment is active and run the test suite before restarting Streamlit.
- **Explanation unavailable:** continue the demo with the probability and threshold output; SHAP is an optional follow-on module.

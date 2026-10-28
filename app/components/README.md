# Transaction explanation component

`render_explanation(result)` displays the SHAP explanation for one transaction.
It shows the uncalibrated model score, a signed contribution chart, a
feature detail table, available plain-language reasons, and an explanation
limitations notice.

The current app shell does not call this component; it renders compact text-based signed bars.
The example below documents a possible component integration.

## Example integration from an app shell

```python
import streamlit as st

from app.components.explanation_view import render_explanation
from src.common.explain import explain_transaction
from src.common.inference import predict_transaction

prediction = predict_transaction(dataset, transaction)
explanation = explain_transaction(dataset, transaction)

# Keep the thresholded label and threshold display tied to inference.
st.write(f"Predicted label: {prediction.predicted_label}")
st.write(f"Decision threshold: {prediction.threshold:.4f}")
render_explanation(explanation)
```

The component does not produce or infer a predicted label or decision
threshold. Those values belong to `PredictionResult` from the shared inference
module.

## Interpretation limits

- The displayed score is uncalibrated and is not a confirmed fraud finding.
- SHAP values show how features contributed to this model's score for this
  transaction; they do not establish causation.
- SHAP values are shown on the model's raw output scale, not as probability
  points.
- IEEE-CIS fields can be anonymized. The component preserves their supplied
  names and does not assign undocumented business meanings.
- An empty contribution list or reason list is displayed as an unavailable
  state rather than being filled with invented explanations.

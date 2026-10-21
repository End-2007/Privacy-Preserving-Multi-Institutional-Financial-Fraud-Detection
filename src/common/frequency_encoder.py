"""Train-only frequency encoding for high-cardinality categorical columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


MISSING_TOKEN = "__FEDGUARD_MISSING__"


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace categories with their relative frequency in the fit data.

    Unseen validation/test categories map to 0.0. Missing values are treated
    as an explicit category and receive their training-set frequency.
    """

    def fit(self, X, y=None):
        frame = self._frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.frequency_maps_ = {}
        for column in frame.columns:
            values = frame[column].astype("string").fillna(MISSING_TOKEN)
            self.frequency_maps_[column] = values.value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        check_is_fitted(self, "frequency_maps_")
        frame = self._frame(X)
        encoded = {}
        for column in self.feature_names_in_:
            values = frame[column].astype("string").fillna(MISSING_TOKEN)
            encoded[column] = values.map(self.frequency_maps_[column]).fillna(0.0).astype(np.float32)
        return pd.DataFrame(encoded, index=frame.index)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return np.asarray(self.feature_names_in_ if input_features is None else input_features, dtype=object)

    @staticmethod
    def _frame(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

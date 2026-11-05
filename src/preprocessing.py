"""Leak-free preprocessing (fitted inside every CV fold via Pipeline)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


class NumericCoercer(BaseEstimator, TransformerMixin):
    """Convert dirty string columns to numbers (bad values -> NaN).

    In the Telco data ``TotalCharges`` contains blank strings for brand-new
    customers, which makes pandas read the whole column as text.
    """

    def __init__(self, columns=("TotalCharges",)):
        self.columns = columns

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

    def transform(self, X):
        X = X.copy()
        for c in self.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        return X


def build_preprocessor() -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    column_transformer = ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # drops customerID
    )
    return Pipeline([
        ("coerce", NumericCoercer(columns=("TotalCharges",))),
        ("column_transformer", column_transformer),
    ])

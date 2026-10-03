"""Feature preprocessing as a fittable ColumnTransformer.

Two design choices, both for correctness:

1. SEX, EDUCATION, MARRIAGE are nominal categories encoded as small
   integers (1/2, 1-4, 1-3) in the raw data. Passing them to a scaler or
   linear model as numbers would imply an ordering (e.g. MARRIAGE=3 is
   "more" than MARRIAGE=1). One-hot encoding removes that false order.
2. Everything is wrapped in a ColumnTransformer so it is fit exactly
   once per training fold (see pipeline.py) and only ever `transform`s
   held-out data, rather than being fit separately on train and test.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_default.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), NUMERIC_COLUMNS),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )

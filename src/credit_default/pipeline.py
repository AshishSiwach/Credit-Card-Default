"""The single leakage-safe pipeline: preprocess -> classifier.

No resampling step here: the ~78/22 class balance is handled via
class_weight / scale_pos_weight on the estimator itself. Resampling
inside cross-validation is easy to get subtly wrong and adds a second
imbalance-handling technique without a demonstrated need.
"""
from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline

from credit_default.features import build_preprocessor


def build_pipeline(model: ClassifierMixin) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("clf", model),
        ]
    )

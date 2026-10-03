"""Metrics and threshold selection.

Choosing a model on accuracy or F1 hides a precision/recall trade-off:
a model can look best on those while catching far fewer defaulters than
another. This module makes the trade-off explicit and configurable: for
a default-risk model, missing an actual defaulter (false negative) is
assumed to be the costlier error, so the threshold rule here is recall
at a precision floor, not raw accuracy or F1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


class ThresholdInfeasibleError(ValueError):
    """No threshold achieves the requested precision floor."""


@dataclass(frozen=True)
class ThresholdResult:
    """Chosen threshold plus the out-of-fold precision/recall it achieved
    on training data (an honest estimate — not the test-set numbers).
    """

    threshold: float
    oof_precision: float
    oof_recall: float


def tune_threshold_recall_at_precision(
    estimator: BaseEstimator,
    X_train,
    y_train,
    min_precision: float,
    cv_folds: int = 5,
    random_state: int = 42,
) -> ThresholdResult:
    """Pick the threshold with the highest recall whose precision is at
    or above `min_precision`.

    Why out-of-fold predictions: scoring the training rows with a model
    fit on those same rows is optimistic, which would pick a threshold
    that looks safer than it is. Each training row is instead scored by
    a clone fit on the other folds, so the final model — and the test
    set — are never involved. The passed estimator is cloned, never
    fit or mutated here.

    Why the exact PR curve: it lists every distinct score as a candidate
    threshold, so there is no coarse grid to miss the best operating point.

    Raises ThresholdInfeasibleError rather than returning a default
    threshold when the floor cannot be met — falling back to 0.5 would
    silently ship a model that violates the stated business rule.
    """
    if not 0.0 < min_precision <= 1.0:
        raise ValueError(f"min_precision must be in (0, 1], got {min_precision}")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    y_proba = cross_val_predict(
        clone(estimator), X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_train, y_proba)
    # precision/recall have one more entry than thresholds (the
    # recall=0 endpoint); drop it so indices line up with thresholds.
    precision, recall = precision[:-1], recall[:-1]

    feasible = precision >= min_precision
    if not feasible.any():
        raise ThresholdInfeasibleError(
            f"No threshold reaches precision >= {min_precision:.2f} "
            f"(best out-of-fold precision is {precision.max():.3f}). "
            "Lower min_precision or improve the model."
        )

    # Recall is non-increasing in the threshold, so among feasible
    # thresholds the highest recall is at the lowest feasible one.
    best = int(np.argmax(np.where(feasible, recall, -1.0)))
    return ThresholdResult(
        threshold=float(thresholds[best]),
        oof_precision=float(precision[best]),
        oof_recall=float(recall[best]),
    )


def evaluate_at_threshold(fitted_estimator, X_test, y_test, threshold: float) -> dict:
    y_proba = fitted_estimator.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "average_precision": float(average_precision_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

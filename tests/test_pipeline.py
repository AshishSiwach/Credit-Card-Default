"""Tests for the core correctness properties: no leakage of test data
into preprocessing, nominal categories one-hot encoded, a threshold
that honours the precision floor, and reproducibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.utils.validation import check_is_fitted

from credit_default.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN, split
from credit_default.evaluate import (
    ThresholdInfeasibleError,
    evaluate_at_threshold,
    tune_threshold_recall_at_precision,
)
from credit_default.pipeline import build_pipeline


def test_split_is_stratified(toy_df):
    X_train, X_test, y_train, y_test = split(toy_df, test_size=0.2, random_state=42)
    assert abs(y_train.mean() - y_test.mean()) < 0.1


def test_scaler_fit_only_on_training_data(toy_df):
    """Leakage guard: a scaler fit on (or refit with) test data would
    carry test-set statistics into its own evaluation. Three checks:
      1. the fitted mean/scale equal the *training* statistics exactly;
      2. they differ from the full-data (train+test) statistics, so a
         scaler that had seen the test set would be caught;
      3. transforming test data afterwards leaves the fitted parameters
         unchanged (transform-only, no refit).
    """
    X_train, X_test, y_train, _ = split(toy_df, test_size=0.2, random_state=42)
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(X_train, y_train)

    scaler = pipe.named_steps["preprocess"].named_transformers_["scale"]
    train_mean = X_train[NUMERIC_COLUMNS].mean().to_numpy()
    train_std = X_train[NUMERIC_COLUMNS].std(ddof=0).to_numpy()
    full = pd.concat([X_train, X_test])[NUMERIC_COLUMNS]

    np.testing.assert_allclose(scaler.mean_, train_mean)
    np.testing.assert_allclose(scaler.scale_, train_std)
    assert not np.allclose(scaler.mean_, full.mean().to_numpy())

    mean_before, scale_before = scaler.mean_.copy(), scaler.scale_.copy()
    pipe.predict_proba(X_test)
    np.testing.assert_array_equal(scaler.mean_, mean_before)
    np.testing.assert_array_equal(scaler.scale_, scale_before)


def test_scaler_params_follow_whatever_training_data_it_is_given(toy_df):
    """Fit on two different halves: parameters must differ, i.e. the
    pipeline controls what the scaler sees."""
    X_train, _, y_train, _ = split(toy_df, test_size=0.2, random_state=42)
    half = len(X_train) // 2
    pipe_a = build_pipeline(LogisticRegression(max_iter=1000))
    pipe_a.fit(X_train.iloc[:half], y_train.iloc[:half])
    pipe_b = build_pipeline(LogisticRegression(max_iter=1000))
    pipe_b.fit(X_train.iloc[half:], y_train.iloc[half:])
    a = pipe_a.named_steps["preprocess"].named_transformers_["scale"]
    b = pipe_b.named_steps["preprocess"].named_transformers_["scale"]
    assert not np.allclose(a.mean_, b.mean_)


def test_categorical_columns_are_one_hot_not_ordinal(toy_df):
    """SEX/EDUCATION/MARRIAGE must not reach the model as raw integers
    (which would imply a false ordering). Assert the exact
    expanded feature set: the raw columns are gone, one indicator exists
    per category (binary SEX collapses to one), and the width matches.
    """
    X_train, _, y_train, _ = split(toy_df, test_size=0.2, random_state=42)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    names = set(pipe.named_steps["preprocess"].get_feature_names_out())

    categorical = {n for n in names if n.startswith("onehot__")}
    assert categorical == (
        {"onehot__SEX_2"}
        | {f"onehot__EDUCATION_{v}" for v in (1, 2, 3, 4)}
        | {f"onehot__MARRIAGE_{v}" for v in (1, 2, 3)}
    )
    for col in CATEGORICAL_COLUMNS:
        assert f"onehot__{col}" not in names and col not in names
    assert len(names) == len(NUMERIC_COLUMNS) + 8


@pytest.mark.filterwarnings("ignore:Found unknown categories")
def test_unseen_category_at_predict_time_does_not_crash(toy_df):
    """The real data has rare codes (EDUCATION 5/6, MARRIAGE 0); a code
    absent from a training fold must be ignored, not raise."""
    X_train, X_test, y_train, _ = split(toy_df, test_size=0.2, random_state=42)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    X_odd = X_test.copy()
    X_odd["EDUCATION"] = 6
    X_odd["MARRIAGE"] = 0
    proba = pipe.predict_proba(X_odd)
    assert np.isfinite(proba).all()


@pytest.mark.parametrize("floor", [0.6, 0.8])
def test_threshold_respects_precision_floor(signal_df, floor):
    """The returned threshold must achieve at least the configured
    precision on out-of-fold training predictions, and the result must
    be independently reproducible from those predictions.
    """
    X_train, _, y_train, _ = split(signal_df, test_size=0.2, random_state=42)
    pipe = build_pipeline(LogisticRegression(max_iter=1000))

    result = tune_threshold_recall_at_precision(
        pipe, X_train, y_train, min_precision=floor, cv_folds=5, random_state=42
    )
    assert result.oof_precision >= floor
    assert 0.0 < result.oof_recall <= 1.0

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(
        clone(pipe), X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    pred = (proba >= result.threshold).astype(int)
    assert precision_score(y_train, pred) >= floor
    assert recall_score(y_train, pred) == pytest.approx(result.oof_recall)


def test_higher_floor_never_increases_recall(signal_df):
    X_train, _, y_train, _ = split(signal_df, test_size=0.2, random_state=42)
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    lo = tune_threshold_recall_at_precision(pipe, X_train, y_train, min_precision=0.6)
    hi = tune_threshold_recall_at_precision(pipe, X_train, y_train, min_precision=0.8)
    assert hi.oof_recall <= lo.oof_recall


def test_infeasible_floor_raises_instead_of_falling_back(toy_df):
    """A constant-score model has precision == base rate (~0.22) at every
    threshold, so a 0.5 floor is impossible. The tuner must say so, not
    return a default threshold that silently breaks the floor.
    """
    X_train, _, y_train, _ = split(toy_df, test_size=0.2, random_state=42)
    pipe = build_pipeline(DummyClassifier(strategy="prior"))
    with pytest.raises(ThresholdInfeasibleError):
        tune_threshold_recall_at_precision(pipe, X_train, y_train, min_precision=0.5)


@pytest.mark.parametrize("bad_floor", [0.0, -0.1, 1.5])
def test_invalid_floor_raises(toy_df, bad_floor):
    X_train, _, y_train, _ = split(toy_df, test_size=0.2, random_state=42)
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    with pytest.raises(ValueError, match="min_precision"):
        tune_threshold_recall_at_precision(pipe, X_train, y_train, min_precision=bad_floor)


def test_threshold_tuning_does_not_fit_or_mutate_estimator(signal_df):
    X_train, _, y_train, _ = split(signal_df, test_size=0.2, random_state=42)
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    tune_threshold_recall_at_precision(pipe, X_train, y_train, min_precision=0.6)
    with pytest.raises(NotFittedError):
        check_is_fitted(pipe)


def test_threshold_tuning_is_deterministic(signal_df):
    X_train, _, y_train, _ = split(signal_df, test_size=0.2, random_state=42)
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    a = tune_threshold_recall_at_precision(pipe, X_train, y_train, 0.6, random_state=7)
    b = tune_threshold_recall_at_precision(pipe, X_train, y_train, 0.6, random_state=7)
    assert a == b


def test_evaluate_at_threshold_reports_expected_keys(signal_df):
    X_train, X_test, y_train, y_test = split(signal_df, test_size=0.2, random_state=42)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    metrics = evaluate_at_threshold(pipe, X_test, y_test, 0.5)
    assert {"roc_auc", "average_precision", "precision", "recall", "confusion_matrix"} <= set(metrics)
    assert sum(metrics["confusion_matrix"].values()) == len(y_test)


def test_reproducible_given_same_random_state(toy_df):
    X_train, X_test, y_train, y_test = split(toy_df, test_size=0.2, random_state=42)

    pipe_1 = build_pipeline(LogisticRegression(max_iter=1000))
    pipe_1.fit(X_train, y_train)
    proba_1 = pipe_1.predict_proba(X_test)[:, 1]

    pipe_2 = build_pipeline(LogisticRegression(max_iter=1000))
    pipe_2.fit(X_train, y_train)
    proba_2 = pipe_2.predict_proba(X_test)[:, 1]

    assert np.allclose(proba_1, proba_2)

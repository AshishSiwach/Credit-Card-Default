"""Tests for the SHAP explanation module."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pytest
import yaml
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from credit_default.data import TARGET_COLUMN
from credit_default.explain import main, shap_contributions, source_feature, summarise
from credit_default.pipeline import build_pipeline


@pytest.fixture
def fitted(signal_df_large):
    """XGBoost pipeline on data whose target depends on LIMIT_BAL."""
    X = signal_df_large.drop(columns=[TARGET_COLUMN])
    y = signal_df_large[TARGET_COLUMN]
    pipe = build_pipeline(
        XGBClassifier(n_estimators=40, max_depth=3, eval_metric="logloss", random_state=0)
    ).fit(X, y)
    return pipe, X


@pytest.mark.parametrize(
    "name, expected",
    [
        ("scale__LIMIT_BAL", "LIMIT_BAL"),
        ("scale__PAY_0", "PAY_0"),
        ("onehot__EDUCATION_2", "EDUCATION"),
        ("onehot__SEX_2", "SEX"),
        ("onehot__MARRIAGE_0", "MARRIAGE"),
    ],
)
def test_source_feature_maps_transformed_names_to_raw_columns(name, expected):
    assert source_feature(name) == expected


def test_contributions_sum_exactly_to_model_margin(fitted):
    """The defining property of SHAP: contributions + bias == raw score."""
    pipe, X = fitted
    shap_df, bias, max_error = shap_contributions(pipe, X)

    Z = pipe.named_steps["preprocess"].transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    margin = pipe.named_steps["clf"].predict(Z, output_margin=True)
    np.testing.assert_allclose(shap_df.sum(axis=1) + bias, margin, atol=1e-3)
    assert max_error < 1e-3


def test_one_hot_columns_are_grouped_into_source_features(fitted):
    pipe, X = fitted
    shap_df, _, _ = shap_contributions(pipe, X)
    assert set(shap_df.columns) == set(X.columns)  # 23 raw features, no dummies
    assert shap_df.index.equals(X.index)


def test_informative_feature_ranks_first_with_correct_direction(fitted):
    pipe, X = fitted
    shap_df, _, _ = shap_contributions(pipe, X)
    rows = summarise(shap_df, X)
    assert rows[0]["feature"] == "LIMIT_BAL"          # the only feature the target depends on
    assert rows[0]["direction_corr"] > 0.5            # higher limit -> higher score here
    assert sum(r["share_of_total"] for r in rows) == pytest.approx(1.0)
    assert [r["mean_abs_shap"] for r in rows] == sorted(
        (r["mean_abs_shap"] for r in rows), reverse=True
    )
    assert all(r["direction_corr"] is None for r in rows if r["feature"] in {"SEX", "EDUCATION", "MARRIAGE"})


def test_non_xgboost_model_is_rejected(signal_df_large):
    X = signal_df_large.drop(columns=[TARGET_COLUMN])
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, signal_df_large[TARGET_COLUMN])
    with pytest.raises(TypeError, match="XGBClassifier"):
        shap_contributions(pipe, X)


def _cfg(tmp_path, signal_df_large) -> dict:
    data_path = tmp_path / "toy.csv"
    signal_df_large.to_csv(data_path, index=False)
    return {
        "data_path": str(data_path),
        "output_dir": str(tmp_path / "out"),
        "random_state": 42,
        "test_size": 0.2,
        "cv_folds": 3,
        "cv_scoring": "average_precision",
        "min_precision": 0.5,
        "model": {"type": "xgboost", "params": {}},
        "explain": {"n_rows": 100},
    }


def test_main_writes_report_from_saved_model(tmp_path, signal_df_large, fitted):
    pipe, _ = fitted
    cfg = _cfg(tmp_path, signal_df_large)
    (tmp_path / "out").mkdir()
    joblib.dump(pipe, tmp_path / "out" / "model.joblib")
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    report = main(str(cfg_path))

    assert json.loads((tmp_path / "out" / "explainability.json").read_text()) == report
    assert report["n_rows"] == 100 and report["explained_on"] == "training split sample"
    assert report["max_additivity_error"] < 1e-3
    assert len(report["features"]) == 23


def test_main_requires_trained_model(tmp_path, signal_df_large):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg(tmp_path, signal_df_large)))
    with pytest.raises(FileNotFoundError, match="train"):
        main(str(cfg_path))


def test_main_requires_explain_section(tmp_path, signal_df_large):
    cfg = _cfg(tmp_path, signal_df_large)
    del cfg["explain"]
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="explain"):
        main(str(cfg_path))

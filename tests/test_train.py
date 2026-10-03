"""Tests for config validation and model construction in train.py."""
from __future__ import annotations

import pandas as pd
import pytest

from credit_default.train import build_model, validate_config


@pytest.fixture
def cfg() -> dict:
    return {
        "data_path": "x.csv",
        "output_dir": "out",
        "random_state": 1,
        "test_size": 0.2,
        "cv_folds": 3,
        "cv_scoring": "average_precision",
        "min_precision": 0.4,
        "model": {"type": "xgboost", "params": {}},
    }


def test_valid_config_passes(cfg):
    validate_config(cfg)


def test_missing_key_raises(cfg):
    del cfg["min_precision"]
    with pytest.raises(ValueError, match="min_precision"):
        validate_config(cfg)


def test_unknown_model_type_raises_readable_error(cfg):
    cfg["model"]["type"] = "random_forest_9000"
    with pytest.raises(ValueError, match="Unknown model type"):
        validate_config(cfg)


def test_baseline_section_is_validated_too(cfg):
    cfg["baseline"] = {"type": "nope", "params": {}}
    with pytest.raises(ValueError, match="Unknown baseline type"):
        validate_config(cfg)


def test_auto_scale_pos_weight_matches_training_ratio():
    y = pd.Series([0] * 75 + [1] * 25)
    model = build_model(
        {"type": "xgboost", "auto_scale_pos_weight": True, "params": {}}, y, 0
    )
    assert model.get_params()["scale_pos_weight"] == pytest.approx(3.0)


def test_auto_and_manual_scale_pos_weight_conflict():
    y = pd.Series([0, 0, 1])
    with pytest.raises(ValueError, match="not both"):
        build_model(
            {"type": "xgboost", "auto_scale_pos_weight": True,
             "params": {"scale_pos_weight": 2.0}},
            y, 0,
        )


def test_class_weight_balanced_rejected_for_xgboost():
    with pytest.raises(ValueError, match="not supported for xgboost"):
        build_model(
            {"type": "xgboost", "class_weight_balanced": True, "params": {}},
            pd.Series([0, 1]), 0,
        )


def test_class_weight_balanced_applied_to_sklearn_model():
    model = build_model(
        {"type": "logistic_regression", "class_weight_balanced": True, "params": {}},
        pd.Series([0, 1]), 0,
    )
    assert model.get_params()["class_weight"] == "balanced"


def test_auto_scale_pos_weight_rejected_for_non_xgboost():
    with pytest.raises(ValueError, match="only applies to xgboost"):
        build_model(
            {"type": "decision_tree", "auto_scale_pos_weight": True, "params": {}},
            pd.Series([0, 1]), 0,
        )

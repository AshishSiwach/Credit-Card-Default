"""Tests for the model-comparison CLI and its config validation."""
from __future__ import annotations

import json

import pytest
import yaml

from credit_default.compare import main
from credit_default.train import validate_config


@pytest.fixture
def cmp_cfg(tmp_path, signal_df_large) -> dict:
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
        "model": {"type": "logistic_regression", "params": {"max_iter": 500}},
        "comparison": [
            {"name": "lr", "type": "logistic_regression",
             "class_weight_balanced": True, "params": {"max_iter": 500}},
            {"name": "tree", "type": "decision_tree",
             "class_weight_balanced": True, "params": {"max_depth": 4}},
        ],
    }


def test_comparison_ranks_by_cv_only_and_labels_each_model(cmp_cfg, tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cmp_cfg))
    report = main(str(cfg_path))

    assert json.loads((tmp_path / "out" / "comparison.json").read_text()) == report
    assert "training-set cross-validation only" in report["ranked_by"]
    scores = [m["cv_mean_average_precision"] for m in report["models"]]
    assert scores == sorted(scores, reverse=True)
    assert {m["name"] for m in report["models"]} == {"lr", "tree"}
    # each row's metrics belong to the model named in that row
    by_name = {m["name"]: m for m in report["models"]}
    assert by_name["lr"]["model_type"] == "logistic_regression"
    assert by_name["tree"]["model_type"] == "decision_tree"
    assert "recall" in by_name["lr"]["at_default_threshold"]


def test_comparison_entry_needs_name(cmp_cfg):
    del cmp_cfg["comparison"][0]["name"]
    with pytest.raises(ValueError, match="needs a 'name'"):
        validate_config(cmp_cfg)


def test_comparison_entry_type_is_validated(cmp_cfg):
    cmp_cfg["comparison"][0]["type"] = "nope"
    with pytest.raises(ValueError, match="Unknown comparison"):
        validate_config(cmp_cfg)


def test_compare_without_section_raises(cmp_cfg, tmp_path):
    del cmp_cfg["comparison"]
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cmp_cfg))
    with pytest.raises(ValueError, match="comparison"):
        main(str(cfg_path))

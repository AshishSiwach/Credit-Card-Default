"""Tests for the hyperparameter-search step."""
from __future__ import annotations

import json

import pytest
import yaml
from scipy.stats._distn_infrastructure import rv_frozen

from credit_default.train import validate_config
from credit_default.tune import main, parse_search_space


@pytest.fixture
def tune_cfg(tmp_path, signal_df_large) -> dict:
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
        "model": {
            "type": "xgboost",
            "auto_scale_pos_weight": True,
            "params": {"n_estimators": 20, "max_depth": 3, "eval_metric": "logloss"},
        },
        "tuning": {
            "n_iter": 4,
            "search_space": {
                "max_depth": [2, 3],
                "n_estimators": [10, 20],
                "learning_rate": {"loguniform": [0.05, 0.3]},
            },
        },
    }


def test_parse_search_space_targets_classifier_step():
    parsed = parse_search_space(
        {"max_depth": [2, 3], "learning_rate": {"loguniform": [0.01, 0.2]}}
    )
    assert set(parsed) == {"clf__max_depth", "clf__learning_rate"}
    assert parsed["clf__max_depth"] == [2, 3]
    assert isinstance(parsed["clf__learning_rate"], rv_frozen)
    draws = parsed["clf__learning_rate"].rvs(200, random_state=0)
    assert 0.01 <= draws.min() and draws.max() <= 0.2


@pytest.mark.parametrize("bad", [5, "x", {"uniform": [0, 1]}, {"loguniform": [1], "extra": 1}])
def test_parse_search_space_rejects_unsupported_specs(bad):
    with pytest.raises(ValueError, match="search_space"):
        parse_search_space({"max_depth": bad})


def test_tuning_config_is_validated(tune_cfg):
    tune_cfg["tuning"]["n_iter"] = "many"
    with pytest.raises(ValueError, match="tuning"):
        validate_config(tune_cfg)


def test_tune_reports_comparison_with_current_params_and_leaves_config_alone(tune_cfg, tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(tune_cfg))
    before = cfg_path.read_text()

    report = main(str(cfg_path))

    assert json.loads((tmp_path / "out" / "tuning.json").read_text()) == report
    assert len(report["top_candidates"]) == 4
    means = [c["cv_mean_average_precision"] for c in report["top_candidates"]]
    assert means == sorted(means, reverse=True)
    assert report["best"] == report["top_candidates"][0]
    assert report["gain_over_current"] == pytest.approx(
        report["best"]["cv_mean_average_precision"] - report["current_cv_mean_average_precision"]
    )
    assert report["current_params"] == tune_cfg["model"]["params"]
    assert "unbiased" in report["note"]
    assert cfg_path.read_text() == before  # tuning reports; it never rewrites config


def test_tune_is_reproducible(tune_cfg, tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(tune_cfg))
    a, b = main(str(cfg_path)), main(str(cfg_path))
    assert a == b


def test_tune_without_section_raises(tune_cfg, tmp_path):
    del tune_cfg["tuning"]
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(tune_cfg))
    with pytest.raises(ValueError, match="tuning"):
        main(str(cfg_path))

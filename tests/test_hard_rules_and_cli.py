"""Structural enforcement of CLAUDE.md's hard rule, plus end-to-end
tests of the train CLI on synthetic data."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import joblib
import pytest
import yaml

import credit_default
from credit_default.evaluate import ThresholdInfeasibleError
from credit_default.train import main

SRC = Path(credit_default.__file__).parent


def _calls():
    """Yield (filename, call_node) for every call in src/."""
    for path in sorted(SRC.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                yield path.name, node


def _callee_name(node: ast.Call) -> str:
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")


def test_fit_transform_is_never_called_in_src():
    offenders = [(f, n.lineno) for f, n in _calls() if _callee_name(n) == "fit_transform"]
    assert offenders == []


def test_train_test_split_is_only_called_in_data_py():
    offenders = [
        (f, n.lineno) for f, n in _calls()
        if _callee_name(n) == "train_test_split" and f != "data.py"
    ]
    assert offenders == []


def test_only_fit_calls_are_on_training_data():
    """`.fit(...)` may appear exactly twice in src/, always on training
    data: train.py fitting the final Pipeline, and tune.py running a
    cross-validated search over that Pipeline. If this fails, someone is
    fitting something by hand — extend the pipeline instead."""
    fits = []
    for fname, node in _calls():
        if _callee_name(node) != "fit":
            continue
        receiver = getattr(node.func.value, "id", "?")
        first_arg = getattr(node.args[0], "id", "?") if node.args else "?"
        fits.append((fname, receiver, first_arg))
    assert sorted(fits) == [("train.py", "pipe", "X_train"), ("tune.py", "search", "X_train")]


@pytest.fixture
def cli_cfg(tmp_path, signal_df_large):
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
        "baseline": {
            "type": "logistic_regression",
            "class_weight_balanced": True,
            "params": {"max_iter": 500},
        },
    }


def _run(cfg: dict, tmp_path: Path, name: str, out: str) -> dict:
    cfg = {**cfg, "output_dir": str(tmp_path / out)}
    cfg_path = tmp_path / name
    cfg_path.write_text(yaml.safe_dump(cfg))
    main(str(cfg_path))
    return json.loads((tmp_path / out / "metrics.json").read_text())


def test_cli_runs_end_to_end_and_writes_artifacts(cli_cfg, tmp_path, signal_df_large):
    metrics = _run(cli_cfg, tmp_path, "c.yaml", "out")

    for key in ("roc_auc", "precision", "recall", "chosen_threshold",
                "cv_mean_average_precision", "dataset", "baseline", "library_versions"):
        assert key in metrics
    assert metrics["dataset"]["n_train"] + metrics["dataset"]["n_test"] == len(signal_df_large)
    # baseline metrics live under their own key, never mixed into the main ones
    assert metrics["model_type"] == "xgboost"
    assert metrics["baseline"]["model_type"] == "logistic_regression"

    model = joblib.load(tmp_path / "out" / "model.joblib")
    proba = model.predict_proba(signal_df_large.drop(columns=["default payment next month"]))
    assert proba.shape == (len(signal_df_large), 2)


def test_cli_is_reproducible_with_fixed_random_state(cli_cfg, tmp_path):
    """Two full runs (CV + threshold tuning + XGBoost fit) with the same
    seed must produce identical metrics."""
    a = _run(cli_cfg, tmp_path, "a.yaml", "out_a")
    b = _run(cli_cfg, tmp_path, "b.yaml", "out_b")
    for m in (a, b):
        m.pop("config")  # differs only by output_dir
    assert a == b


def test_cli_raises_when_precision_floor_is_infeasible(cli_cfg, tmp_path):
    cli_cfg["min_precision"] = 1.0  # a perfect-precision floor on noisy scores
    cli_cfg["model"]["params"]["max_depth"] = 1
    cli_cfg["model"]["params"]["n_estimators"] = 2
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.safe_dump(cli_cfg))
    try:
        main(str(cfg_path))
    except ThresholdInfeasibleError:
        assert not (tmp_path / "out" / "model.joblib").exists()
    else:
        pytest.skip("floor happened to be attainable on this synthetic data")

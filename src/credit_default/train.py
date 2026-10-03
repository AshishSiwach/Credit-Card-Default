"""CLI entry point: load config, split once, cross-validate on train
only, tune threshold on out-of-fold training predictions, fit the final
pipeline, evaluate once on the untouched test set, persist model and
metrics.

Usage:
    python -m credit_default.train --config config.yaml

Every model — the main one and the optional baseline — goes through the
single `run_model` function. There is no second copy of the evaluation
code, so one model's metrics cannot end up reported under another's
name.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from credit_default.data import load_raw, split
from credit_default.evaluate import (
    ThresholdInfeasibleError,
    evaluate_at_threshold,
    tune_threshold_recall_at_precision,
)
from credit_default.pipeline import build_pipeline

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "xgboost": XGBClassifier,
}

REQUIRED_TOP_LEVEL_KEYS = [
    "data_path",
    "output_dir",
    "random_state",
    "test_size",
    "cv_folds",
    "cv_scoring",
    "min_precision",
    "model",
]


def validate_config(cfg: dict) -> None:
    """Fail up front with a readable message instead of a KeyError deep
    inside a run."""
    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")
    if "tuning" in cfg:
        tuning = cfg["tuning"]
        if not isinstance(tuning.get("n_iter"), int) or not tuning.get("search_space"):
            raise ValueError("Config section 'tuning' needs integer 'n_iter' and a 'search_space'")
    sections = {name: cfg[name] for name in ("model", "baseline") if name in cfg}
    for i, entry in enumerate(cfg.get("comparison", [])):
        if "name" not in entry:
            raise ValueError(f"comparison entry {i} needs a 'name'")
        sections[f"comparison[{entry['name']}]"] = entry
    for name, section in sections.items():
        if "type" not in section or "params" not in section:
            raise ValueError(f"Config section '{name}' needs 'type' and 'params'")
        if section["type"] not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown {name} type {section['type']!r}; "
                f"choose one of {sorted(MODEL_REGISTRY)}"
            )


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def build_model(model_cfg: dict, y_train: pd.Series, random_state: int):
    """Instantiate the classifier named in config.

    `auto_scale_pos_weight` derives XGBoost's imbalance weight
    (negatives / positives) from the training labels instead of a
    hand-typed constant that would go stale if the split or data
    changed. It uses training labels only. `class_weight_balanced` is
    the equivalent for sklearn models; each option is rejected for model
    types that don't support it rather than being silently ignored.
    """
    model_type = model_cfg["type"]
    params = dict(model_cfg["params"])

    if model_cfg.get("class_weight_balanced", False):
        if model_type == "xgboost":
            raise ValueError(
                "class_weight_balanced is not supported for xgboost; "
                "use auto_scale_pos_weight or scale_pos_weight"
            )
        params["class_weight"] = "balanced"

    if model_cfg.get("auto_scale_pos_weight", False):
        if model_type != "xgboost":
            raise ValueError("auto_scale_pos_weight only applies to xgboost")
        if "scale_pos_weight" in params:
            raise ValueError(
                "Set either auto_scale_pos_weight or params.scale_pos_weight, not both"
            )
        n_pos = int((y_train == 1).sum())
        if n_pos == 0:
            raise ValueError("No positive examples in training labels")
        params["scale_pos_weight"] = float((y_train == 0).sum() / n_pos)

    return MODEL_REGISTRY[model_type](**params, random_state=random_state)


def run_model(
    model_cfg: dict,
    cfg: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[Pipeline, dict]:
    """Cross-validate, tune the threshold, fit once, evaluate once.

    The one code path for every model. Returns the fitted pipeline and
    its metrics. If the precision floor is unattainable the metrics
    carry `threshold_infeasible` (and no test metrics) and the pipeline
    is returned unfitted; the caller decides whether that is fatal.
    """
    random_state = cfg["random_state"]
    model = build_model(model_cfg, y_train, random_state)
    pipe = build_pipeline(model)

    # Preprocessing is refit per fold because it lives inside the Pipeline.
    cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(
        pipe, X_train, y_train, cv=cv, scoring=cfg["cv_scoring"], n_jobs=-1
    )
    metrics: dict = {
        "model_type": model_cfg["type"],
        f"cv_mean_{cfg['cv_scoring']}": float(cv_scores.mean()),
        f"cv_std_{cfg['cv_scoring']}": float(cv_scores.std()),
        "min_precision_constraint": cfg["min_precision"],
    }
    if "scale_pos_weight" in model.get_params() and model.get_params()["scale_pos_weight"]:
        metrics["scale_pos_weight_used"] = float(model.get_params()["scale_pos_weight"])

    # Threshold is tuned on out-of-fold training predictions; the
    # estimator is cloned inside, so `pipe` is fit exactly once, below.
    try:
        selection = tune_threshold_recall_at_precision(
            pipe,
            X_train,
            y_train,
            min_precision=cfg["min_precision"],
            cv_folds=cfg["cv_folds"],
            random_state=random_state,
        )
    except ThresholdInfeasibleError as exc:
        metrics["threshold_infeasible"] = str(exc)
        return pipe, metrics

    pipe.fit(X_train, y_train)
    metrics.update(evaluate_at_threshold(pipe, X_test, y_test, selection.threshold))
    metrics["chosen_threshold"] = selection.threshold
    metrics["threshold_oof_precision"] = selection.oof_precision
    metrics["threshold_oof_recall"] = selection.oof_recall
    return pipe, metrics


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    random_state = cfg["random_state"]

    df = load_raw(cfg["data_path"])
    X_train, X_test, y_train, y_test = split(
        df, test_size=cfg["test_size"], random_state=random_state
    )

    pipe, metrics = run_model(cfg["model"], cfg, X_train, X_test, y_train, y_test)
    if "threshold_infeasible" in metrics:
        raise ThresholdInfeasibleError(metrics["threshold_infeasible"])

    report = {
        "dataset": {
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "train_default_rate": float(y_train.mean()),
            "test_default_rate": float(y_test.mean()),
        },
        "random_state": random_state,
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "config": cfg,
        **metrics,
    }

    if "baseline" in cfg:
        _, baseline_metrics = run_model(cfg["baseline"], cfg, X_train, X_test, y_train, y_test)
        report["baseline"] = baseline_metrics

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "model.joblib")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)

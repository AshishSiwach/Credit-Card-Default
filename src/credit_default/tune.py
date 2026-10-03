"""Hyperparameter search for the main model, scored on cross-validated
average precision.

This is a separate, explicit step — not part of `train` — so that a
hyperparameter change is a reviewable edit to `config.yaml` rather than
something that happens invisibly at training time. The workflow is:

    python -m credit_default.tune --config config.yaml   # writes artifacts/tuning.json
    # inspect it, copy the chosen values into config.yaml `model.params`
    python -m credit_default.train --config config.yaml

Discipline:
- The search runs on the *training* split only. The test set is not
  touched, so the test-set numbers produced afterwards by `train` remain
  an unbiased estimate of the tuned model.
- The search and the currently-configured parameters are scored on the
  same CV folds, so the comparison between them is like for like.
- The best CV score from a search is optimistically biased (it is the
  maximum over many noisy estimates). `tuning.json` therefore reports
  the gain relative to the CV standard deviation, and the honest read of
  a gain of about one standard deviation or less is "no meaningful
  improvement".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score

from credit_default.data import load_raw, split
from credit_default.pipeline import build_pipeline
from credit_default.train import build_model, load_config

TOP_N = 5


def parse_search_space(space: dict) -> dict:
    """Turn the YAML search space into `RandomizedSearchCV` distributions
    on the pipeline's classifier step. A list is a discrete set; a
    `{loguniform: [low, high]}` mapping is a log-uniform range.
    """
    parsed = {}
    for name, spec in space.items():
        key = f"clf__{name}"
        if isinstance(spec, list):
            parsed[key] = spec
        elif isinstance(spec, dict) and set(spec) == {"loguniform"}:
            low, high = spec["loguniform"]
            parsed[key] = loguniform(low, high)
        else:
            raise ValueError(
                f"search_space[{name!r}] must be a list or {{loguniform: [low, high]}}, got {spec!r}"
            )
    return parsed


def _py(value):
    """Make numpy scalars JSON-serialisable."""
    return value.item() if isinstance(value, np.generic) else value


def main(config_path: str) -> dict:
    cfg = load_config(config_path)
    tuning = cfg.get("tuning")
    if not tuning:
        raise ValueError("Config has no 'tuning' section to run")

    random_state = cfg["random_state"]
    df = load_raw(cfg["data_path"])
    X_train, _, y_train, _ = split(df, cfg["test_size"], random_state)

    pipe = build_pipeline(build_model(cfg["model"], y_train, random_state))
    if "n_jobs" in pipe.named_steps["clf"].get_params():
        # Parallelism is across CV fits (n_jobs=-1 below); a threaded
        # estimator inside would oversubscribe the machine.
        pipe.set_params(clf__n_jobs=1)

    cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=random_state)
    scoring = cfg["cv_scoring"]

    current = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    # refit=False: this step only ranks parameter sets; `train` does the final fit.
    search = RandomizedSearchCV(
        pipe,
        param_distributions=parse_search_space(tuning["search_space"]),
        n_iter=tuning["n_iter"],
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=False,
    )
    search.fit(X_train, y_train)

    res = search.cv_results_
    order = np.argsort(res["rank_test_score"])
    top = [
        {
            "params": {k.removeprefix("clf__"): _py(v) for k, v in res["params"][i].items()},
            f"cv_mean_{scoring}": float(res["mean_test_score"][i]),
            f"cv_std_{scoring}": float(res["std_test_score"][i]),
        }
        for i in order[:TOP_N]
    ]
    best_mean = float(res["mean_test_score"][order[0]])
    gain = best_mean - float(current.mean())

    report = {
        "scoring": scoring,
        "cv_folds": cfg["cv_folds"],
        "n_iter": tuning["n_iter"],
        "search_space": tuning["search_space"],
        "current_params": cfg["model"]["params"],
        f"current_cv_mean_{scoring}": float(current.mean()),
        f"current_cv_std_{scoring}": float(current.std()),
        "best": top[0],
        "gain_over_current": gain,
        "gain_in_current_cv_stds": gain / float(current.std()) if current.std() > 0 else None,
        "top_candidates": top,
        "note": (
            "Best CV score is selected as the maximum over n_iter noisy estimates and is "
            "optimistically biased. The test set was not used; run `train` for an unbiased "
            "estimate of the tuned model."
        ),
    }

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tuning.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"current  CV {scoring}: {current.mean():.4f} +/- {current.std():.4f}")
    print(f"best     CV {scoring}: {best_mean:.4f}  (gain {gain:+.4f}, "
          f"{report['gain_in_current_cv_stds']:+.2f} current-CV stds)")
    print("best params:", top[0]["params"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)

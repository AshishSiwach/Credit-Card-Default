"""SHAP feature importance for the trained XGBoost pipeline.

Values are exact TreeSHAP contributions computed by XGBoost itself
(`pred_contribs=True`), so no extra dependency is needed. They are in
log-odds units: for each client, the contributions plus a bias term sum
exactly to the model's raw score, and `explainability.json` records the
largest additivity error as a self-check.

Choices worth knowing:
- Computed on a seeded sample of the *training* split, not the test set,
  so explanation work cannot leak test information into later decisions.
- Contributions of one-hot columns are summed back to their source
  feature (EDUCATION_1..6 -> EDUCATION). Otherwise a categorical's
  importance is split across dummies and looks smaller than it is.
- Importance is descriptive of what the model uses, not causal: a large
  SHAP value says the model relies on a feature, not that changing it
  would change a client's outcome. `direction_corr` is only a rank
  correlation between a numeric feature and its contribution — a
  one-number summary of a possibly non-monotone relationship.

Usage:
    python -m credit_default.explain --config config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from credit_default.data import CATEGORICAL_COLUMNS, load_raw, split
from credit_default.train import load_config

ADDITIVITY_TOLERANCE = 1e-3  # log-odds; float32 round-off is far below this


def source_feature(transformed_name: str) -> str:
    """Map a ColumnTransformer output name back to its raw column:
    'scale__LIMIT_BAL' -> 'LIMIT_BAL', 'onehot__EDUCATION_2' -> 'EDUCATION'."""
    _, _, rest = transformed_name.partition("__")
    for col in CATEGORICAL_COLUMNS:
        if rest == col or rest.startswith(col + "_"):
            return col
    return rest


def shap_contributions(
    pipe: Pipeline, X: pd.DataFrame
) -> tuple[pd.DataFrame, float, float]:
    """Return (per-source-feature SHAP matrix, bias, max additivity error).

    Rows align with `X`; columns are raw feature names, with one-hot
    columns already summed into their source feature.
    """
    clf = pipe.named_steps["clf"]
    if not isinstance(clf, XGBClassifier):
        raise TypeError(
            f"SHAP via XGBoost is only available for an XGBClassifier, got {type(clf).__name__}"
        )
    pre = pipe.named_steps["preprocess"]
    Z = pre.transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    names = list(pre.get_feature_names_out())

    booster = clf.get_booster()
    dmatrix = xgb.DMatrix(Z)
    contribs = booster.predict(dmatrix, pred_contribs=True)  # (n, n_features + 1), last = bias
    margin = booster.predict(dmatrix, output_margin=True)
    max_error = float(np.abs(contribs.sum(axis=1) - margin).max())
    if max_error > ADDITIVITY_TOLERANCE:
        raise RuntimeError(
            f"SHAP contributions do not sum to the model margin (max error {max_error:.2e})"
        )

    per_column = pd.DataFrame(contribs[:, :-1], columns=names, index=X.index)
    grouped = per_column.T.groupby([source_feature(n) for n in names]).sum().T
    return grouped, float(contribs[0, -1]), max_error


def summarise(shap_df: pd.DataFrame, X: pd.DataFrame) -> list[dict]:
    """One row per source feature, sorted by mean |SHAP|."""
    mean_abs = shap_df.abs().mean()
    total = float(mean_abs.sum())
    rows = []
    for feature in mean_abs.sort_values(ascending=False).index:
        direction = None
        if feature in X.columns and feature not in CATEGORICAL_COLUMNS:
            if X[feature].nunique() > 1 and shap_df[feature].nunique() > 1:
                direction = float(spearmanr(X[feature], shap_df[feature]).statistic)
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(mean_abs[feature]),
                "share_of_total": float(mean_abs[feature] / total),
                "direction_corr": direction,
            }
        )
    return rows


def main(config_path: str) -> dict:
    cfg = load_config(config_path)
    settings = cfg.get("explain")
    if not settings or "n_rows" not in settings:
        raise ValueError("Config needs an 'explain' section with 'n_rows'")

    model_path = Path(cfg["output_dir"]) / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found; run `python -m credit_default.train` first")
    pipe = joblib.load(model_path)

    df = load_raw(cfg["data_path"])
    X_train, _, _, _ = split(df, cfg["test_size"], cfg["random_state"])
    sample = X_train.sample(
        n=min(settings["n_rows"], len(X_train)), random_state=cfg["random_state"]
    )

    shap_df, bias, max_error = shap_contributions(pipe, sample)
    report = {
        "method": "TreeSHAP via XGBoost pred_contribs (log-odds units)",
        "explained_on": "training split sample",
        "n_rows": int(len(sample)),
        "random_state": cfg["random_state"],
        "bias": bias,
        "max_additivity_error": max_error,
        "features": summarise(shap_df, sample),
    }
    if settings.get("figures_dir"):
        # Imported here so the numeric report works without matplotlib.
        from credit_default.plots import save_figures

        report["figures"] = save_figures(
            shap_df, sample, report["features"], Path(settings["figures_dir"])
        )

    out_dir = Path(cfg["output_dir"])
    with open(out_dir / "explainability.json", "w") as f:
        json.dump(report, f, indent=2)
    for row in report["features"][:10]:
        print(f"{row['feature']:12} mean|SHAP|={row['mean_abs_shap']:.3f}  share={row['share_of_total']:.1%}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)

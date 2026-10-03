"""Figures for the SHAP explanation, written as PNGs for the README.

matplotlib is an optional dependency (`pip install -e ".[notebook]"`), so
it is imported lazily and a missing install produces a clear error
rather than breaking unrelated modules.

Design notes:
- Bars are coloured by feature group using the Okabe-Ito colour-blind-safe
  palette, so the chart lines up with the group table next to it.
- Every figure has a white background and dark text so it reads the same
  on GitHub's light and dark themes.
- The summary plot only shows numeric features: its colour encodes the
  feature's value percentile, which has no meaning for category codes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from credit_default.data import CATEGORICAL_COLUMNS
from credit_default.report import feature_group

# Okabe-Ito palette, one colour per feature group.
GROUP_COLOURS = {
    "Repayment status": "#0072B2",
    "Payment amounts": "#E69F00",
    "Bill amounts": "#009E73",
    "Credit limit": "#CC79A7",
    "Demographics": "#7F7F7F",
    "Other": "#000000",
}
IMPORTANCE_FILE = "shap_importance.png"
SUMMARY_FILE = "shap_summary.png"
TOP_BARS = 10
TOP_DOTS = 8


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed (CI, servers)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            'Plotting needs matplotlib: pip install -e ".[notebook]"'
        ) from exc
    return plt


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot_importance(features: list[dict], path: Path) -> None:
    """Horizontal bars of mean |SHAP| for the top features, coloured by group."""
    plt = _pyplot()
    top = features[:TOP_BARS][::-1]  # reversed so the largest bar is on top
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor="white")
    ax.barh(
        [r["feature"] for r in top],
        [r["mean_abs_shap"] for r in top],
        color=[GROUP_COLOURS[feature_group(r["feature"])] for r in top],
    )
    ax.set_xlabel("Mean |SHAP value| (log-odds) — average influence on the model's score")
    ax.set_title(f"Top {len(top)} features by SHAP importance", loc="left", fontweight="bold")
    present = list(dict.fromkeys(feature_group(r["feature"]) for r in features[:TOP_BARS]))
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOURS[g]) for g in present]
    ax.legend(handles, present, title="Feature group", loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def plot_summary(
    shap_df: pd.DataFrame, X: pd.DataFrame, features: list[dict], path: Path, seed: int = 0
) -> None:
    """Per-client SHAP values for the top numeric features, coloured by the
    client's value of that feature (percentile within the sample)."""
    plt = _pyplot()
    numeric = [r["feature"] for r in features if r["feature"] not in CATEGORICAL_COLUMNS]
    numeric = numeric[:TOP_DOTS][::-1]
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(8, 5.2), facecolor="white")
    scatter = None
    for y, feature in enumerate(numeric):
        scatter = ax.scatter(
            shap_df[feature],
            y + rng.uniform(-0.3, 0.3, len(shap_df)),
            c=X[feature].rank(pct=True),
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=4,
            alpha=0.55,
            linewidths=0,
        )
    ax.set_yticks(range(len(numeric)))
    ax.set_yticklabels(numeric)
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_xlabel("SHAP value (log-odds): right of zero pushes the score towards default")
    ax.set_title("How feature values move each client's score", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Feature value (percentile; red = high, blue = low)")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def save_figures(
    shap_df: pd.DataFrame, X: pd.DataFrame, features: list[dict], out_dir: Path
) -> list[dict]:
    """Write both figures and return their descriptors for the JSON artifact
    (`path` is relative to the repository root, where the README lives)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        (IMPORTANCE_FILE, "Bar chart of mean absolute SHAP value for the top features, coloured by feature group"),
        (SUMMARY_FILE, "Dot plot of per-client SHAP values for the top numeric features, coloured by feature value"),
    ]
    plot_importance(features, out_dir / IMPORTANCE_FILE)
    plot_summary(shap_df, X, features, out_dir / SUMMARY_FILE)
    return [
        {"alt": alt, "path": (out_dir / name).as_posix(), "sha256": sha256_of(out_dir / name)}
        for name, alt in specs
    ]

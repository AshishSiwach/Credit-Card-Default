"""Tests that README numbers come from the artifacts, never by hand."""
from __future__ import annotations

from pathlib import Path

import pytest

from credit_default.report import END, START, build, render_results, replace_block

METRICS = {
    "model_type": "xgboost",
    "dataset": {"n_train": 80, "n_test": 20, "test_default_rate": 0.25},
    "min_precision_constraint": 0.45,
    "cv_mean_average_precision": 0.5,
    "cv_std_average_precision": 0.01,
    "roc_auc": 0.7,
    "average_precision": 0.55,
    "chosen_threshold": 0.4,
    "threshold_oof_precision": 0.45,
    "threshold_oof_recall": 0.6,
    "precision": 0.46,
    "recall": 0.61,
    "confusion_matrix": {"true_negative": 10, "false_positive": 5, "false_negative": 2, "true_positive": 3},
}
SENS = {"floors": [
    {"min_precision": 0.35, "feasible": True, "test_precision": 0.4, "test_recall": 0.7,
     "flagged_share_of_test": 0.5, "false_positives_per_true_positive": 1.5},
    {"min_precision": 0.99, "feasible": False},
]}


def test_render_contains_exact_metric_values():
    text = render_results(METRICS, None, SENS)
    assert "0.460 / 0.610" in text
    assert "10 / 5 / 2 / 3" in text
    assert "| 0.35 | 0.400 | 0.700 | 50.0% | 1.50 |" in text
    assert "| 0.99 | infeasible" in text


def test_replace_block_only_touches_marked_region():
    readme = f"before\n{START}\nold\n{END}\nafter\n"
    out = replace_block(readme, "NEW")
    assert out == f"before\n{START}\nNEW\n{END}\nafter\n"


def test_replace_block_requires_markers():
    with pytest.raises(ValueError, match="markers"):
        replace_block("no markers here", "x")


def test_repo_readme_matches_artifacts_when_present():
    """If artifacts exist locally, the committed README must match them
    exactly. Skipped on a fresh clone (artifacts are gitignored)."""
    root = Path(__file__).resolve().parents[1]
    artifacts = root / "artifacts"
    if not (artifacts / "metrics.json").exists():
        pytest.skip("no artifacts; run training first")
    readme = root / "README.md"
    assert build(readme, artifacts) == readme.read_text(encoding="utf-8"), (
        "README results are stale; run `python -m credit_default.report`"
    )

"""Tests that README numbers come from the artifacts, never by hand."""
from __future__ import annotations

from pathlib import Path

import pytest

from credit_default.report import (
    END,
    START,
    build,
    explainability_observations,
    feature_group,
    group_shares,
    render_results,
    replace_block,
)

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


def _tuning(gain_in_stds: float) -> dict:
    return {
        "scoring": "average_precision", "cv_folds": 5, "n_iter": 60,
        "current_params": {"max_depth": 4, "eval_metric": "logloss"},
        "current_cv_mean_average_precision": 0.5600, "current_cv_std_average_precision": 0.0100,
        "best": {"params": {"max_depth": 5, "learning_rate": 0.0130356},
                 "cv_mean_average_precision": 0.5700, "cv_std_average_precision": 0.0090},
        "gain_over_current": 0.0100, "gain_in_current_cv_stds": gain_in_stds,
    }


EXPL = {
    "method": "TreeSHAP via XGBoost pred_contribs (log-odds units)",
    "explained_on": "training split sample", "n_rows": 5000,
    "features": [
        {"feature": "PAY_0", "mean_abs_shap": 0.538, "share_of_total": 0.276, "direction_corr": 0.32},
        {"feature": "EDUCATION", "mean_abs_shap": 0.01, "share_of_total": 0.005, "direction_corr": None},
    ],
}


def test_tuning_within_noise_says_parameters_were_kept():
    text = render_results(METRICS, None, None, tuning=_tuning(0.44))
    assert "0.5600 ± 0.0100" in text and "0.5700 ± 0.0090" in text
    assert "learning_rate=0.013" in text and "eval_metric" not in text
    assert "(+0.44 current-CV standard deviations)" in text
    assert "within CV noise" in text and "kept" in text
    assert "optimistically biased" in text


def test_tuning_clear_gain_is_flagged_for_consideration():
    text = render_results(METRICS, None, None, tuning=_tuning(2.5))
    assert "worth considering" in text and "within CV noise" not in text


def test_explainability_table_and_caveat():
    text = render_results(METRICS, None, None, explainability=EXPL)
    assert "| PAY_0 | 0.538 | 27.6% | +0.32 |" in text
    assert "| EDUCATION | 0.010 | 0.5% | n/a (categorical) |" in text
    assert "5,000-row sample of the training split" in text and "not causal" in text


def _feat(name, share, corr):
    return {"feature": name, "mean_abs_shap": share * 2, "share_of_total": share, "direction_corr": corr}


EXPL_RICH = {
    "method": "TreeSHAP", "explained_on": "training split sample", "n_rows": 100,
    "features": [
        _feat("PAY_0", 0.40, 0.3), _feat("LIMIT_BAL", 0.20, -0.9), _feat("PAY_AMT1", 0.15, -0.8),
        _feat("PAY_3", 0.10, 0.85), _feat("BILL_AMT1", 0.05, -0.2), _feat("AGE", 0.04, 0.1),
        _feat("EDUCATION", 0.03, None), _feat("SEX", 0.03, None),
    ],
}


@pytest.mark.parametrize(
    "feature, group",
    [("PAY_0", "Repayment status"), ("PAY_6", "Repayment status"), ("PAY_AMT3", "Payment amounts"),
     ("BILL_AMT2", "Bill amounts"), ("LIMIT_BAL", "Credit limit"), ("SEX", "Demographics"),
     ("AGE", "Demographics"), ("MYSTERY", "Other")],
)
def test_feature_group_assignment(feature, group):
    assert feature_group(feature) == group  # PAY_AMT must not fall into PAY_ (status)


def test_group_shares_partition_total_importance():
    shares = group_shares(EXPL_RICH["features"])
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["Repayment status"] == pytest.approx(0.50)
    assert shares["Demographics"] == pytest.approx(0.10)


def test_observations_are_derived_from_artifact_numbers():
    obs = "\n".join(explainability_observations(EXPL_RICH))
    assert "`PAY_0` is the single most influential feature (40.0%" in obs
    assert "top three features together account for 75.0%" in obs
    assert "repayment status features carry the most importance (50.0%)" in obs
    assert "followed by credit limit (20.0%)" in obs
    assert "`LIMIT_BAL`, `PAY_AMT1` push the score towards lower default risk" in obs
    assert "`PAY_3` push the score towards higher default risk" in obs
    assert "10.0% of total importance. That is a material share" in obs  # demographics at the threshold


def test_demographic_share_wording_follows_the_number():
    small = {**EXPL_RICH, "features": [_feat("PAY_0", 0.95, 0.3), _feat("AGE", 0.05, 0.1)]}
    assert "small but not zero" in "\n".join(explainability_observations(small))


def test_weak_direction_correlations_are_not_stated():
    expl = {**EXPL_RICH, "features": [_feat("PAY_0", 0.6, 0.3), _feat("BILL_AMT1", 0.4, -0.2)]}
    obs = "\n".join(explainability_observations(expl))
    assert "lower default risk" not in obs and "higher default risk" not in obs


def test_rendered_explainability_includes_group_table_and_observations():
    text = render_results(METRICS, None, None, explainability=EXPL_RICH)
    assert "| Repayment status | PAY_0, PAY_2–PAY_6 | 50.0% |" in text
    assert "| Demographics | SEX, EDUCATION, MARRIAGE, AGE | 10.0% |" in text
    assert "Observations (derived from the tables above):" in text
    assert "not causal" in text


def test_optional_sections_are_omitted_when_artifacts_absent():
    text = render_results(METRICS, None, None)
    assert "Hyperparameter search" not in text and "SHAP" not in text


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

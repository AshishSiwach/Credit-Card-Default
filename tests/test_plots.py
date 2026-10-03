"""Tests for the SHAP figures and their integrity checks in the README build."""
from __future__ import annotations

import json

import joblib
import pytest
import yaml
from xgboost import XGBClassifier

pytest.importorskip("matplotlib")

from credit_default.data import TARGET_COLUMN  # noqa: E402
from credit_default.explain import main, shap_contributions, summarise  # noqa: E402
from credit_default.pipeline import build_pipeline  # noqa: E402
from credit_default.plots import (  # noqa: E402
    GROUP_COLOURS,
    IMPORTANCE_FILE,
    SUMMARY_FILE,
    save_figures,
    sha256_of,
)
from credit_default.report import (  # noqa: E402
    FEATURE_GROUPS,
    build,
    render_results,
    verify_figures,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def explained(signal_df_large):
    X = signal_df_large.drop(columns=[TARGET_COLUMN])
    pipe = build_pipeline(
        XGBClassifier(n_estimators=30, max_depth=3, eval_metric="logloss", random_state=0)
    ).fit(X, signal_df_large[TARGET_COLUMN])
    shap_df, _, _ = shap_contributions(pipe, X)
    return shap_df, X, summarise(shap_df, X)


def test_every_group_has_a_colour():
    assert {name for name, _ in FEATURE_GROUPS} | {"Other"} <= set(GROUP_COLOURS)


def test_save_figures_writes_valid_pngs_with_matching_hashes(explained, tmp_path):
    shap_df, X, features = explained
    figs = save_figures(shap_df, X, features, tmp_path / "img")

    assert [f["path"].rsplit("/", 1)[-1] for f in figs] == [IMPORTANCE_FILE, SUMMARY_FILE]
    for fig in figs:
        data = (tmp_path / "img" / fig["path"].rsplit("/", 1)[-1]).read_bytes()
        assert data.startswith(PNG_MAGIC) and len(data) > 5_000
        assert fig["sha256"] == sha256_of(tmp_path / "img" / fig["path"].rsplit("/", 1)[-1])
        assert fig["alt"]  # accessibility: every image has alt text


def test_figures_are_reproducible_for_the_same_input(explained, tmp_path):
    shap_df, X, features = explained
    a = save_figures(shap_df, X, features, tmp_path / "a")
    b = save_figures(shap_df, X, features, tmp_path / "b")
    assert [f["sha256"] for f in a] == [f["sha256"] for f in b]


def _write_cfg(tmp_path, signal_df_large, figures_dir):
    data_path = tmp_path / "toy.csv"
    signal_df_large.to_csv(data_path, index=False)
    cfg = {
        "data_path": str(data_path), "output_dir": str(tmp_path / "out"), "random_state": 42,
        "test_size": 0.2, "cv_folds": 3, "cv_scoring": "average_precision", "min_precision": 0.5,
        "model": {"type": "xgboost", "params": {}},
        "explain": {"n_rows": 120, **({"figures_dir": str(figures_dir)} if figures_dir else {})},
    }
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_explain_main_records_figures_only_when_configured(tmp_path, signal_df_large, explained):
    shap_df, X, _ = explained
    pipe = build_pipeline(
        XGBClassifier(n_estimators=30, max_depth=3, eval_metric="logloss", random_state=0)
    ).fit(X, signal_df_large[TARGET_COLUMN])
    (tmp_path / "out").mkdir()
    joblib.dump(pipe, tmp_path / "out" / "model.joblib")

    without = main(str(_write_cfg(tmp_path, signal_df_large, None)))
    assert "figures" not in without

    with_figs = main(str(_write_cfg(tmp_path, signal_df_large, tmp_path / "figs")))
    assert len(with_figs["figures"]) == 2
    assert json.loads((tmp_path / "out" / "explainability.json").read_text()) == with_figs
    assert (tmp_path / "figs" / IMPORTANCE_FILE).exists()


def _expl_with_figures(tmp_path, explained):
    shap_df, X, features = explained
    readme_dir = tmp_path / "repo"
    figs = save_figures(shap_df, X, features, readme_dir / "docs" / "images")
    for f in figs:  # paths in the artifact are relative to the README directory
        f["path"] = "docs/images/" + f["path"].rsplit("/", 1)[-1]
    return {
        "method": "TreeSHAP", "explained_on": "training split sample", "n_rows": 600,
        "features": features, "figures": figs,
    }, readme_dir


METRICS = {
    "model_type": "xgboost", "dataset": {"n_train": 8, "n_test": 2, "test_default_rate": 0.2},
    "min_precision_constraint": 0.45, "cv_mean_average_precision": 0.5, "cv_std_average_precision": 0.01,
    "roc_auc": 0.7, "average_precision": 0.5, "chosen_threshold": 0.4, "threshold_oof_precision": 0.45,
    "threshold_oof_recall": 0.6, "precision": 0.46, "recall": 0.61,
    "confusion_matrix": {"true_negative": 1, "false_positive": 1, "false_negative": 0, "true_positive": 0},
}


def test_render_embeds_both_figures_in_order(tmp_path, explained):
    expl, _ = _expl_with_figures(tmp_path, explained)
    text = render_results(METRICS, None, None, explainability=expl)
    first = text.index("![Bar chart")
    second = text.index("![Dot plot")
    assert text.index("| Feature | Mean") > first          # bars sit above the importance table
    assert text.index("Importance by feature group") < second  # dots follow the group table
    assert "(docs/images/shap_importance.png)" in text and "(docs/images/shap_summary.png)" in text


def test_verify_figures_accepts_matching_files(tmp_path, explained):
    expl, readme_dir = _expl_with_figures(tmp_path, explained)
    verify_figures(expl, readme_dir)  # does not raise


def test_verify_figures_rejects_missing_file(tmp_path, explained):
    expl, readme_dir = _expl_with_figures(tmp_path, explained)
    (readme_dir / "docs" / "images" / IMPORTANCE_FILE).unlink()
    with pytest.raises(FileNotFoundError, match="explain"):
        verify_figures(expl, readme_dir)


def test_verify_figures_rejects_stale_or_edited_image(tmp_path, explained):
    expl, readme_dir = _expl_with_figures(tmp_path, explained)
    (readme_dir / "docs" / "images" / SUMMARY_FILE).write_bytes(PNG_MAGIC + b"not the real figure")
    with pytest.raises(ValueError, match="does not match"):
        verify_figures(expl, readme_dir)


def test_build_refuses_to_embed_a_stale_figure(tmp_path, explained):
    expl, readme_dir = _expl_with_figures(tmp_path, explained)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "metrics.json").write_text(json.dumps(METRICS))
    (artifacts / "explainability.json").write_text(json.dumps(expl))
    readme = readme_dir / "README.md"
    readme.write_text("x\n<!-- RESULTS:START (generated by `python -m credit_default.report` - do not edit) -->\n<!-- RESULTS:END -->\n")

    assert "![Bar chart" in build(readme, artifacts)  # consistent -> builds

    (readme_dir / "docs" / "images" / IMPORTANCE_FILE).write_bytes(PNG_MAGIC + b"stale")
    with pytest.raises(ValueError, match="does not match"):
        build(readme, artifacts)

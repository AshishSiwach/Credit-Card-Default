"""Compare the candidate models listed under `comparison:` in config.

Models can look very different depending on the operating point: a
balanced decision tree catches far more defaulters than an unweighted
XGBoost at the default 0.5 threshold, while XGBoost ranks better
overall. Each model is therefore shown at two operating points — the
default 0.5 threshold and the recall-at-precision-floor threshold
(what this repo ships) — so the difference is visible rather than
implied.

Selection discipline: models are ranked by cross-validated average
precision on the training set only. Test-set numbers are reported for
transparency but are never used to rank or choose, otherwise the test
set stops being an unbiased estimate.

Every model runs through `train.run_model` — the same single code path
as the main CLI.

Usage:
    python -m credit_default.compare --config config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from credit_default.data import load_raw, split
from credit_default.evaluate import evaluate_at_threshold
from credit_default.train import load_config, run_model

DEFAULT_THRESHOLD = 0.5


def main(config_path: str) -> dict:
    cfg = load_config(config_path)
    if not cfg.get("comparison"):
        raise ValueError("Config has no 'comparison' section to run")

    df = load_raw(cfg["data_path"])
    X_train, X_test, y_train, y_test = split(
        df, test_size=cfg["test_size"], random_state=cfg["random_state"]
    )

    cv_key = f"cv_mean_{cfg['cv_scoring']}"
    rows = []
    for entry in cfg["comparison"]:
        pipe, metrics = run_model(entry, cfg, X_train, X_test, y_train, y_test)
        row = {"name": entry["name"], **metrics}
        if "chosen_threshold" in metrics:  # pipeline was fitted
            at_default = evaluate_at_threshold(pipe, X_test, y_test, DEFAULT_THRESHOLD)
            row["at_default_threshold"] = {
                "threshold": DEFAULT_THRESHOLD,
                "precision": at_default["precision"],
                "recall": at_default["recall"],
            }
        rows.append(row)

    rows.sort(key=lambda r: r[cv_key], reverse=True)
    report = {
        "ranked_by": f"{cv_key} (training-set cross-validation only)",
        "min_precision_constraint": cfg["min_precision"],
        "models": rows,
    }

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(report, f, indent=2)

    _print_table(rows, cv_key)
    return report


def _print_table(rows: list[dict], cv_key: str) -> None:
    header = f"{'model':34}{'cv_AP':>8}{'@0.5 P':>9}{'@0.5 R':>9}{'@floor P':>10}{'@floor R':>10}"
    print(header)
    for r in rows:
        d = r.get("at_default_threshold", {})
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else "  n/a"
        print(
            f"{r['name']:34}{r[cv_key]:>8.3f}{fmt(d.get('precision')):>9}"
            f"{fmt(d.get('recall')):>9}{fmt(r.get('precision')):>10}{fmt(r.get('recall')):>10}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)

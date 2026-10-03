"""How the deployment decision moves as the precision floor changes.

The floor encodes a business assumption (a missed defaulter costs more
than an unnecessary review, down to a limit). Nobody should have to
take the 0.45 on faith, so this runs the main model at several floors
and shows what each one buys in recall and costs in review workload.

Usage:
    python -m credit_default.sensitivity --config config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from credit_default.data import load_raw, split
from credit_default.train import load_config, run_model


def main(config_path: str) -> dict:
    cfg = load_config(config_path)
    floors = cfg.get("sensitivity_floors")
    if not floors:
        raise ValueError("Config has no 'sensitivity_floors' to run")

    df = load_raw(cfg["data_path"])
    X_train, X_test, y_train, y_test = split(
        df, test_size=cfg["test_size"], random_state=cfg["random_state"]
    )

    rows = []
    for floor in floors:
        # Same single code path as training, only the floor differs.
        _, m = run_model(
            cfg["model"], {**cfg, "min_precision": floor}, X_train, X_test, y_train, y_test
        )
        if "threshold_infeasible" in m:
            rows.append({"min_precision": floor, "feasible": False})
            continue
        cm = m["confusion_matrix"]
        flagged = cm["true_positive"] + cm["false_positive"]
        rows.append(
            {
                "min_precision": floor,
                "feasible": True,
                "chosen_threshold": m["chosen_threshold"],
                "test_precision": m["precision"],
                "test_recall": m["recall"],
                "flagged_share_of_test": flagged / len(y_test),
                "false_positives_per_true_positive": cm["false_positive"] / cm["true_positive"],
            }
        )

    report = {"model_type": cfg["model"]["type"], "floors": rows}
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sensitivity.json", "w") as f:
        json.dump(report, f, indent=2)
    for r in rows:
        print(r)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)

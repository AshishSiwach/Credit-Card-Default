"""Static hygiene checks on the EDA notebook (it is never executed in
tests): it must be one coherent top-to-bottom run, error-free, and must
not bypass the package's leakage guards."""
from __future__ import annotations

from pathlib import Path

import nbformat

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda_and_experiments.ipynb"


def _nb():
    return nbformat.read(NB_PATH, as_version=4)


def _code_cells():
    return [c for c in _nb().cells if c.cell_type == "code"]


def test_notebook_was_run_top_to_bottom_in_order():
    counts = [c.execution_count for c in _code_cells()]
    assert counts == list(range(1, len(counts) + 1)), counts


def test_notebook_has_no_error_outputs():
    for cell in _code_cells():
        assert not [o for o in cell.outputs if o.output_type == "error"]


def test_notebook_does_not_bypass_package_leakage_guards():
    source = "\n".join(c.source for c in _code_cells())
    for forbidden in ("fit_transform", "train_test_split", ".fit("):
        assert forbidden not in source, f"notebook must not use {forbidden}"
    assert "from credit_default.data import" in source

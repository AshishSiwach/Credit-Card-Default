"""Tests for data loading/validation, including the UCI xls layout quirk
(a junk label row above the real header)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_default.data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    load_raw,
    validate_raw,
)


def _make_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {c: rng.integers(0, 1000, n) for c in NUMERIC_COLUMNS}
    data["SEX"] = rng.integers(1, 3, n)
    data["EDUCATION"] = rng.integers(0, 7, n)
    data["MARRIAGE"] = rng.integers(0, 4, n)
    data[TARGET_COLUMN] = np.tile([0, 0, 0, 1], n)[:n]
    df = pd.DataFrame(data)
    df.insert(0, "ID", np.arange(1, n + 1))
    return df


def _write_uci_layout(df: pd.DataFrame, path) -> None:
    """Mimic the UCI file: row 1 is X1..X23,Y labels, row 2 the real header."""
    label_row = [None] + [f"X{i}" for i in range(1, len(df.columns) - 1)] + ["Y"]
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([label_row]).to_excel(writer, header=False, index=False, startrow=0)
        df.to_excel(writer, index=False, startrow=1)


def test_load_raw_handles_uci_header_row_and_drops_id(tmp_path):
    df = _make_df()
    path = tmp_path / "uci.xlsx"
    _write_uci_layout(df, path)

    loaded = load_raw(path)

    assert "ID" not in loaded.columns
    assert loaded.shape == (len(df), len(df.columns) - 1)
    assert set(CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [TARGET_COLUMN]) == set(loaded.columns)


def test_load_raw_reads_csv(tmp_path):
    path = tmp_path / "d.csv"
    _make_df().to_csv(path, index=False)
    assert TARGET_COLUMN in load_raw(path).columns


def test_load_raw_accepts_dotted_target_alias(tmp_path):
    path = tmp_path / "d.csv"
    _make_df().rename(columns={TARGET_COLUMN: "default.payment.next.month"}).to_csv(
        path, index=False
    )
    assert TARGET_COLUMN in load_raw(path).columns


def test_missing_column_raises():
    with pytest.raises(ValueError, match="missing expected columns"):
        validate_raw(_make_df().drop(columns=["AGE"]))


def test_nan_raises():
    df = _make_df()
    df.loc[0, "LIMIT_BAL"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        validate_raw(df)


def test_non_binary_target_raises():
    df = _make_df()
    df.loc[0, TARGET_COLUMN] = 2
    with pytest.raises(ValueError, match="binary"):
        validate_raw(df)


def test_unexpected_category_code_raises():
    df = _make_df()
    df.loc[0, "EDUCATION"] = 9
    with pytest.raises(ValueError, match="EDUCATION"):
        validate_raw(df)


def test_duplicate_ids_raise():
    df = _make_df()
    df.loc[1, "ID"] = df.loc[0, "ID"]
    with pytest.raises(ValueError, match="duplicates"):
        validate_raw(df)

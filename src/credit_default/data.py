"""Data loading and the single source of truth for the train/test split.

Split once, here, before anything else touches the data. The classic
failure this guards against is a scaler (or any other fitted
transform) refit on the test set, which leaks test-set statistics into
its own evaluation. The structural defence is that nothing downstream
of `split()` may call `.fit()` or `.fit_transform()` on anything but
training data; features.py and pipeline.py enforce that, and a test
scans the source to keep it true.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "default payment next month"

CATEGORICAL_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE"]

NUMERIC_COLUMNS = (
    ["LIMIT_BAL", "AGE"]
    + [f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)


ID_COLUMN = "ID"

# Codes observed in the real UCI file. The documented codes are
# EDUCATION 1-4 and MARRIAGE 1-3; 0/5/6 and 0 are undocumented but
# present (EDUCATION: 345 rows, MARRIAGE: 54 rows). They are kept as
# their own one-hot categories rather than silently recoded — a data
# problem should stay visible — but anything outside these sets means
# the file is not the dataset this package was written for.
ALLOWED_CODES = {
    "SEX": {1, 2},
    "EDUCATION": {0, 1, 2, 3, 4, 5, 6},
    "MARRIAGE": {0, 1, 2, 3},
}

# Some redistributions of the dataset use this spelling for the target.
_TARGET_ALIASES = {"default.payment.next.month": TARGET_COLUMN}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
        if "LIMIT_BAL" not in df.columns:
            # The UCI xls has a junk label row (X1..X23, Y) above the
            # real header, so the real column names sit on row 2.
            df = pd.read_excel(path, header=1)
        return df
    return pd.read_csv(path)


def validate_raw(df: pd.DataFrame) -> None:
    """Fail loudly on anything that would otherwise surface later as a
    silent modelling problem. Raises ValueError; never repairs data.
    """
    expected = set(CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [TARGET_COLUMN])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Input data is missing expected columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Input data has no rows")

    null_counts = df[sorted(expected)].isna().sum()
    if null_counts.any():
        raise ValueError(
            f"Input data has missing values: {null_counts[null_counts > 0].to_dict()}"
        )
    if not set(df[TARGET_COLUMN].unique()) <= {0, 1}:
        raise ValueError(f"Target {TARGET_COLUMN!r} must be binary 0/1")
    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError(f"Target {TARGET_COLUMN!r} has only one class")

    for col, allowed in ALLOWED_CODES.items():
        unexpected = set(df[col].unique()) - allowed
        if unexpected:
            raise ValueError(
                f"{col} has unexpected codes {sorted(unexpected)}; allowed {sorted(allowed)}"
            )

    if ID_COLUMN in df.columns and not df[ID_COLUMN].is_unique:
        raise ValueError(f"{ID_COLUMN} column contains duplicates")


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the UCI credit-default file (csv or the UCI xls),
    validate it, and drop the ID column (an identifier, not a feature).

    No value transforms happen here — only structural handling of the
    file format and validation.
    """
    df = _read_table(Path(path)).rename(columns=_TARGET_ALIASES)
    validate_raw(df)
    return df.drop(columns=[ID_COLUMN], errors="ignore")


def split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split — default rate is ~22%, stratification keeps
    train/test base rates aligned.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

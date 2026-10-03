"""Shared synthetic datasets. Real data is never needed by the tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_default.data import NUMERIC_COLUMNS, TARGET_COLUMN


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """300 rows, random (signal-free) ~22% target."""
    rng = np.random.default_rng(0)
    n = 300
    data = {}
    for col in NUMERIC_COLUMNS:
        data[col] = rng.uniform(-5000, 50000, n)
    data["SEX"] = rng.integers(1, 3, n)
    data["EDUCATION"] = rng.integers(1, 5, n)
    data["MARRIAGE"] = rng.integers(1, 4, n)
    df = pd.DataFrame(data)
    df[TARGET_COLUMN] = rng.choice([0, 1], size=n, p=[0.78, 0.22])
    return df


@pytest.fixture
def signal_df(toy_df: pd.DataFrame) -> pd.DataFrame:
    """Like toy_df but the target genuinely depends on LIMIT_BAL, so a
    meaningful precision floor is achievable."""
    rng = np.random.default_rng(1)
    df = toy_df.copy()
    df[TARGET_COLUMN] = (df["LIMIT_BAL"] + rng.normal(0, 6000, len(df)) > 38000).astype(int)
    return df


@pytest.fixture
def signal_df_large() -> pd.DataFrame:
    """600 rows with signal, for end-to-end CLI tests that cross-validate."""
    rng = np.random.default_rng(2)
    n = 600
    data = {c: rng.uniform(-5000, 50000, n) for c in NUMERIC_COLUMNS}
    data["SEX"] = rng.integers(1, 3, n)
    data["EDUCATION"] = rng.integers(1, 5, n)
    data["MARRIAGE"] = rng.integers(1, 4, n)
    df = pd.DataFrame(data)
    df[TARGET_COLUMN] = (df["LIMIT_BAL"] + rng.normal(0, 6000, n) > 38000).astype(int)
    return df

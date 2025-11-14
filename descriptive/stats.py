from __future__ import annotations

import pandas as pd


def generate_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic descriptive statistics for numeric columns.

    Parameters
    ----------
    df:
        Data containing numeric columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``mean``, ``median``, ``std``, ``min`` and ``max`` per
        column.
    """

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()

    stats = numeric_df.agg(["mean", "median", "std", "min", "max"]).T
    return stats

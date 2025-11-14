"""Context-aware missing value detection utilities."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd


def find_contextual_missingness(
    df: pd.DataFrame,
    correlated_fields: Optional[List[Tuple[str, str]]] = None,
    suspicious_placeholders: Optional[List[str]] = None,
    rules: Optional[Dict[str, Callable[[pd.Series, pd.DataFrame], pd.Series]]] = None,
    zero_thresholds: Optional[Dict[str, float]] = None,
) -> Dict:
    """Flag likely missing values based on contextual cues.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to analyze.
    correlated_fields : list of tuple, optional
        Pairs of columns expected to be populated together.
    suspicious_placeholders : list, optional
        Values indicating potential missingness (case-insensitive).
    rules : dict, optional
        Mapping of column names to callables returning boolean Series of
        suspicious values.
    zero_thresholds : dict, optional
        Numeric thresholds per column below which values are considered
        suspicious.
    """

    if suspicious_placeholders is None:
        suspicious_placeholders = []
    if correlated_fields is None:
        correlated_fields = []
    if rules is None:
        rules = {}
    if zero_thresholds is None:
        zero_thresholds = {}

    placeholders = {str(v).strip().lower() for v in suspicious_placeholders}

    flagged = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.columns:
        series = df[col]
        mask = series.astype(str).str.strip().str.lower().isin(placeholders)
        if col in zero_thresholds:
            num = pd.to_numeric(series, errors="coerce")
            mask |= num.lt(zero_thresholds[col])
        flagged[col] |= mask

    for left, right in correlated_fields:
        if left not in df.columns or right not in df.columns:
            continue
        a = df[left]
        b = df[right]
        a_missing = a.isna() | a.astype(str).str.strip().str.lower().isin(placeholders)
        b_missing = b.isna() | b.astype(str).str.strip().str.lower().isin(placeholders)
        if left in zero_thresholds:
            a_num = pd.to_numeric(a, errors="coerce")
            a_missing |= a_num.lt(zero_thresholds[left])
        if right in zero_thresholds:
            b_num = pd.to_numeric(b, errors="coerce")
            b_missing |= b_num.lt(zero_thresholds[right])
        flagged[left] |= a_missing & ~b_missing
        flagged[right] |= b_missing & ~a_missing

    for col, func in rules.items():
        if col not in df.columns:
            continue
        try:
            mask = func(df[col], df)
            if isinstance(mask, pd.Series):
                mask = mask.reindex(df.index, fill_value=False).astype(bool)
                flagged[col] |= mask
        except Exception:
            # Rules should not raise during advisory mode
            pass

    flagged_cells = [
        (idx, col) for col in df.columns for idx in df.index[flagged[col]]
    ]
    column_summary = flagged.sum().to_dict()
    flagged_df = df.copy(deep=False)
    flagged_df["is_context_missing"] = flagged.any(axis=1)

    return {
        "flagged_cells": flagged_cells,
        "column_summary": column_summary,
        "flagged_df": flagged_df,
    }


if __name__ == "__main__":
    demo = pd.DataFrame({
        "revenue": ["100", "0", "nan", "50", "N/A"],
        "units_sold": [10, 5, 0, 0, 20],
        "category": ["A", "unknown", "B", "--", "C"],
    })

    result = find_contextual_missingness(
        demo,
        correlated_fields=[("revenue", "units_sold")],
        suspicious_placeholders=["0", "nan", "n/a", "--", "unknown"],
        zero_thresholds={"revenue": 1.0},
    )

    print("Flagged cells:", result["flagged_cells"])
    print("Column summary:", result["column_summary"])  # noqa: T201

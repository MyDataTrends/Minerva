"""Misaligned row detection utilities."""

from __future__ import annotations

import re
from typing import Callable, Optional, Dict, List

import pandas as pd


# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------

def _is_type(value, typ) -> bool:
    """Return ``True`` if ``value`` can be interpreted as ``typ``."""
    if pd.isna(value):
        return True
    try:
        if typ is int:
            int(value)
        elif typ is float:
            float(value)
        elif typ is str:
            str(value)
        else:
            return isinstance(value, typ)
    except Exception:
        return False
    return True


def _implausible_str(value: str) -> bool:
    """Return ``True`` if ``value`` looks implausible for a text field."""
    if not isinstance(value, str):
        return False
    token = value.strip()
    return bool(re.fullmatch(r"\d{5,}", token))


def _count_mismatches(values: List, columns: List[str], schema: Dict[str, type]) -> int:
    mismatches = 0
    for val, col in zip(values, columns):
        typ = schema[col]
        if not _is_type(val, typ) or (typ is str and _implausible_str(str(val))):
            mismatches += 1
    return mismatches


# -------------------------------------------------------------
# Core detector
# -------------------------------------------------------------

def detect_misaligned_rows(
    df: pd.DataFrame,
    expected_schema: Dict[str, type],
    use_llm: bool = False,
    llm_fn: Optional[Callable[[str], Dict]] = None,
) -> Dict:
    """Detect rows that are likely misaligned.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to validate.
    expected_schema : dict
        Mapping of column names to expected types.
    use_llm : bool, optional
        Whether to use ``llm_fn`` to suggest fixes.
    llm_fn : callable, optional
        Callable that takes a JSON row and returns a corrected mapping.
    """

    columns = list(expected_schema.keys())
    issues: Dict[int, List[str]] = {}
    suggestions: Dict[int, Dict] = {}

    for row in df.itertuples(index=True):
        idx = row.Index
        row_dict = row._asdict()
        row_issues = [
            f"{col}: type mismatch"
            if not _is_type(row_dict.get(col), expected_schema[col])
            else (
                f"{col}: implausible string"
                if expected_schema[col] is str and _implausible_str(str(row_dict.get(col)))
                else None
            )
            for col in columns
        ]
        row_issues = [issue for issue in row_issues if issue]

        if row_issues:
            values = [row_dict[c] for c in columns]
            baseline = _count_mismatches(values, columns, expected_schema)
            left_values = values[1:] + [None]
            right_values = [None] + values[:-1]
            left_m = _count_mismatches(left_values, columns, expected_schema)
            right_m = _count_mismatches(right_values, columns, expected_schema)
            if left_m < baseline and left_m <= right_m:
                row_issues += ["possible shift left"]
                if not use_llm:
                    suggestions[idx] = dict(zip(columns, left_values))
            elif right_m < baseline and right_m < left_m:
                row_issues += ["possible shift right"]
                if not use_llm:
                    suggestions[idx] = dict(zip(columns, right_values))

            if use_llm and llm_fn is not None:
                try:
                    proposal = llm_fn(df.loc[idx].to_json())
                    if isinstance(proposal, dict):
                        suggestions[idx] = proposal
                except Exception:
                    row_issues += ["llm_fn failed"]

            issues[idx] = row_issues

    suggested_df = None
    misaligned = list(issues.keys())

    if suggestions:
        corrected = df.copy(deep=False)
        for idx, mapping in suggestions.items():
            for col in columns:
                if col in mapping:
                    corrected.at[idx, col] = mapping[col]
        suggested_df = corrected

    return {
        "misaligned_rows": misaligned,
        "issues": issues,
        "suggested_fix": suggested_df,
    }


if __name__ == "__main__":  # Basic CLI harness
    sample = pd.DataFrame({
        "name": ["Alice", "Bob", "12345"],
        "age": [30, 25, "Charlie"],
        "zip": [11111, 22222, 33333],
    })
    schema = {"name": str, "age": int, "zip": int}
    report = detect_misaligned_rows(sample, schema)
    print("Misaligned rows:", report["misaligned_rows"])
    print("Issues:", report["issues"])
    if report["suggested_fix"] is not None:
        print("Suggested fix:\n", report["suggested_fix"])

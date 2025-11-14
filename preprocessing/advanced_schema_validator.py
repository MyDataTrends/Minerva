"""Advanced DataFrame schema validation utilities."""

from __future__ import annotations

import re
from typing import Optional, Literal, Dict, List, Any

import pandas as pd

# Known invalid placeholders often used in spreadsheets
_INVALID_PLACEHOLDERS = {"n/a", "---", "none", "na", "null", "nan"}

# Mapping from semantic labels to Python types
_TYPE_MAP = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "category": str,  # categories are stored as strings during validation
}

_NUMERIC_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def _looks_numeric(value: str) -> bool:
    return bool(_NUMERIC_RE.match(value.replace(",", "").strip()))


def _coerce(value: Any, typ_label: str) -> Any:
    try:
        if typ_label == "int":
            if isinstance(value, bool):
                return int(value)
            return int(float(str(value).replace(",", "")))
        if typ_label == "float":
            return float(str(value).replace(",", ""))
        if typ_label == "bool":
            if isinstance(value, str):
                val = value.strip().lower()
                if val in {"true", "1", "yes"}:
                    return True
                if val in {"false", "0", "no"}:
                    return False
                raise ValueError
            return bool(value)
        if typ_label in {"str", "category"}:
            return str(value)
    except Exception:
        raise
    return value


def _is_valid_type(value: Any, typ_label: str) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip().lower() in _INVALID_PLACEHOLDERS:
        return False

    if typ_label == "int":
        if isinstance(value, int) and not isinstance(value, bool):
            return True
        if isinstance(value, (float, str)):
            try:
                int(float(str(value).replace(",", "")))
                return True
            except Exception:
                return False
        return False

    if typ_label == "float":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if isinstance(value, str):
            try:
                float(str(value).replace(",", ""))
                return True
            except Exception:
                return False
        return False

    if typ_label == "bool":
        if isinstance(value, bool):
            return True
        if isinstance(value, str):
            return value.strip().lower() in {"true", "false", "1", "0", "yes", "no"}
        return False

    if typ_label in {"str", "category"}:
        return isinstance(value, str)

    py_type = _TYPE_MAP.get(typ_label)
    if py_type is None:
        return True
    return isinstance(value, py_type)


def _validate_categories(value: Any, valid: set[str]) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip().lower() in _INVALID_PLACEHOLDERS:
        return False
    return str(value) in valid


def validate_schema(
    df: pd.DataFrame,
    expected_schema: Dict[str, Literal["int", "float", "str", "bool", "category"]],
    valid_categories: Optional[Dict[str, set[str]]] = None,
    quarantine_invalid: bool = False,
    coerce: bool = False,
) -> Dict[str, Any]:
    """Validate ``df`` against an expected schema.

    Parameters
    ----------
    df : pd.DataFrame
        Data to validate.
    expected_schema : dict
        Mapping of column names to semantic types.
    valid_categories : dict, optional
        Allowed categorical values for each column.
    quarantine_invalid : bool, default False
        Move rows containing mismatched values to a quarantine DataFrame.
    coerce : bool, default False
        Attempt safe conversion of values to the expected type.
    """

    report: Dict[str, List[int]] = {}
    mismatch_counts: Dict[str, int] = {}
    invalid_rows: set[int] = set()
    corrected = df.copy(deep=False)

    for col, typ_label in expected_schema.items():
        if col not in corrected.columns:
            continue
        values = corrected[col]
        invalid_idx = [
            idx
            for idx, val in values.items()
            if not _is_valid_type(val, typ_label)
            or (
                typ_label == "category"
                and valid_categories
                and col in valid_categories
                and not _validate_categories(val, valid_categories[col])
            )
        ]
        if coerce and invalid_idx:
            for idx in invalid_idx:
                try:
                    corrected.at[idx, col] = _coerce(values.at[idx], typ_label)
                except Exception:
                    continue
            values = corrected[col]
            invalid_idx = [
                idx
                for idx, val in values.items()
                if not _is_valid_type(val, typ_label)
                or (
                    typ_label == "category"
                    and valid_categories
                    and col in valid_categories
                    and not _validate_categories(val, valid_categories[col])
                )
            ]
        if invalid_idx:
            report[col] = invalid_idx
            mismatch_counts[col] = len(invalid_idx)
            invalid_rows.update(invalid_idx)

    quarantined_df = None
    if quarantine_invalid and invalid_rows:
        quarantined_df = corrected.loc[sorted(invalid_rows)].copy(deep=False)
        corrected = corrected.drop(index=invalid_rows).reset_index(drop=True)

    # Tag columns as blocking or advisory based on heuristics
    tagged_report = {}
    for col, idxs in report.items():
        role = "blocking" if any(k in col.lower() for k in ["id", "target", "key"]) else "advisory"
        tagged_report[col] = {"role": role, "rows": idxs}

    return {
        "validation_report": tagged_report,
        "mismatch_counts": mismatch_counts,
        "quarantined": quarantined_df,
        "corrected_df": corrected,
    }


if __name__ == "__main__":  # basic demo
    sample = pd.DataFrame({
        "id": [1, 2, "three"],
        "score": ["10", "bad", 5.5],
        "color": ["red", "blue", "unknown"],
    })
    schema = {"id": "int", "score": "float", "color": "category"}
    categories = {"color": {"red", "green", "blue"}}
    result = validate_schema(sample, schema, valid_categories=categories, quarantine_invalid=True, coerce=True)
    print(result["validation_report"])
    print("Corrected:\n", result["corrected_df"])
    if result["quarantined"] is not None:
        print("Quarantined:\n", result["quarantined"])

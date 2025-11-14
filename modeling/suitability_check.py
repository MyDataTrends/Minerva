from __future__ import annotations

from typing import Literal, Dict, Any

import pandas as pd

from config import MAX_ROWS_FIRST_PASS, MAX_FEATURES_FIRST_PASS

Task = Literal["regression", "classification", "unsure"]


def assess_modelability(df: pd.DataFrame, target: str, task_hint: Task | None = None) -> Dict[str, Any]:
    """Evaluate whether a dataset is suitable for heavy modeling.

    The checks performed are intentionally lightweight so this can be called
    before any expensive training routines.  It returns a dictionary with a
    boolean ``is_modelable`` along with basic diagnostics explaining any
    issues that might prevent modeling.
    """
    result: Dict[str, Any] = {
        "is_modelable": True,
        "reason": "ok",
        "task": "unsure",
        "n_rows_used": int(min(len(df), MAX_ROWS_FIRST_PASS)),
        "n_features_used": int(min(max(0, df.shape[1] - 1), MAX_FEATURES_FIRST_PASS)),
        "cardinality_flags": {},
    }

    # Target existence
    if target not in df.columns:
        result["is_modelable"] = False
        result["reason"] = "missing_target"
        return result

    y = df[target]

    # Determine task
    if task_hint in {"regression", "classification"}:
        task: Task = task_hint  # type: ignore[assignment]
    else:
        if pd.api.types.is_numeric_dtype(y):
            task = "classification" if y.nunique(dropna=True) <= 20 else "regression"
        else:
            task = "classification"
    result["task"] = task

    # Target missingness
    target_missing = y.isna().mean()
    if target_missing > 0.5:
        result["is_modelable"] = False
        result["reason"] = "target_missing"
        return result
    if target_missing > 0.2:
        result["reason"] = "borderline_target_missing"

    n_rows = len(df)
    n_features = df.shape[1] - 1

    # Row/feature counts
    if n_rows < 30:
        result["is_modelable"] = False
        result["reason"] = "too_few_rows"
        return result
    if n_rows > MAX_ROWS_FIRST_PASS:
        result["reason"] = "borderline_too_many_rows"

    if n_features <= 0:
        result["is_modelable"] = False
        result["reason"] = "no_features"
        return result
    if n_features > MAX_FEATURES_FIRST_PASS:
        result["reason"] = "borderline_too_many_features"

    # Missingness rates for features
    feature_missing = df.drop(columns=[target]).isna().mean()
    if feature_missing.max() > 0.8:
        result["is_modelable"] = False
        result["reason"] = "feature_missing"
        return result
    if feature_missing.max() > 0.4:
        result["reason"] = "borderline_feature_missing"

    # Cardinality flags for categorical features
    card_flags: Dict[str, bool] = {}
    for col in df.drop(columns=[target]).columns[:MAX_FEATURES_FIRST_PASS]:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            uniq_ratio = s.nunique(dropna=True) / n_rows if n_rows else 0.0
            card_flags[col] = uniq_ratio > 0.5
    result["cardinality_flags"] = card_flags

    # Task specific checks
    if task == "regression":
        if y.nunique(dropna=True) <= 1 or y.var(skipna=True) == 0:
            result["is_modelable"] = False
            result["reason"] = "low_variance"
            return result
    elif task == "classification":
        counts = y.dropna().value_counts()
        if len(counts) < 2:
            result["is_modelable"] = False
            result["reason"] = "single_class"
            return result
        minority_ratio = counts.min() / counts.sum()
        if minority_ratio < 0.05:
            result["reason"] = "borderline_class_imbalance"

    return result

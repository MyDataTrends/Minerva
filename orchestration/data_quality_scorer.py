"""Unified data quality scoring module.

Computes standard data safety metrics for every analysis run, providing
visibility into data quality issues that may affect model reliability.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def compute_safety_metrics(
    df: pd.DataFrame,
    df_original: Optional[pd.DataFrame] = None,
    imputed_mask: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Compute standard data safety metrics.
    
    Args:
        df: The cleaned/processed DataFrame
        df_original: Original DataFrame before processing (for drift detection)
        imputed_mask: Boolean mask indicating which values were imputed
        
    Returns:
        Dictionary containing quality metrics and warnings
    """
    metrics: Dict[str, Any] = {}
    warnings: List[str] = []
    
    # === Completeness Metrics ===
    total_cells = df.size
    non_null_cells = df.notna().sum().sum()
    metrics["completeness_score"] = round((non_null_cells / total_cells) * 100, 2) if total_cells > 0 else 100
    metrics["columns_missing_data"] = [col for col in df.columns if df[col].isna().any()]
    metrics["rows_complete_pct"] = round((df.notna().all(axis=1).sum() / len(df)) * 100, 2) if len(df) > 0 else 100
    
    # Per-column missing percentages
    missing_pct = {}
    for col in df.columns:
        pct = (df[col].isna().sum() / len(df)) * 100 if len(df) > 0 else 0
        if pct > 0:
            missing_pct[col] = round(pct, 2)
    metrics["missing_by_column"] = missing_pct
    
    # === Consistency Metrics ===
    metrics["type_consistency"] = _check_type_consistency(df)
    metrics["value_ranges"] = _check_value_ranges(df)
    
    # === Imputation Metrics ===
    if imputed_mask is not None:
        imputed_count = imputed_mask.sum().sum()
        metrics["imputation_count"] = int(imputed_count)
        metrics["imputation_pct"] = round((imputed_count / total_cells) * 100, 2) if total_cells > 0 else 0
    
    # === Outlier Detection ===
    outlier_info = _count_outliers(df)
    metrics["outlier_count"] = outlier_info["total"]
    metrics["outliers_by_column"] = outlier_info["by_column"]
    
    # === Drift Detection ===
    if df_original is not None:
        drift_info = _check_drift(df, df_original)
        metrics["drift_detected"] = drift_info["detected"]
        metrics["drift_columns"] = drift_info["columns"]
        metrics["drift_rate"] = drift_info["rate"]
    
    # === High Risk Column Detection ===
    high_risk = _identify_high_risk_columns(df, missing_pct)
    metrics["high_risk_columns"] = high_risk
    
    # === Overall Quality Score ===
    quality_score = _compute_overall_quality_score(metrics)
    metrics["data_quality_score"] = quality_score
    
    # === Generate Warnings ===
    if quality_score < 60:
        warnings.append(f"Low data quality score ({quality_score}%). Results may be unreliable.")
    
    for col, pct in missing_pct.items():
        if pct > 30:
            warnings.append(f"Column '{col}' has {pct}% missing values")
    
    if high_risk:
        warnings.append(f"High-risk columns detected: {', '.join(high_risk)}")
    
    metrics["warnings"] = warnings
    metrics["proceed_with_caution"] = quality_score < 70 or len(high_risk) > 0
    
    return metrics


def _check_type_consistency(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for type consistency issues in each column."""
    issues = {}
    for col in df.columns:
        if df[col].dtype == object:
            # Check if object column has mixed types
            types = df[col].dropna().apply(type).unique()
            if len(types) > 1:
                issues[col] = [t.__name__ for t in types]
    return {"mixed_type_columns": issues, "is_consistent": len(issues) == 0}


def _check_value_ranges(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Get value ranges for numeric columns."""
    ranges = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        ranges[col] = {
            "min": float(df[col].min()) if not df[col].isna().all() else None,
            "max": float(df[col].max()) if not df[col].isna().all() else None,
            "mean": float(df[col].mean()) if not df[col].isna().all() else None,
            "std": float(df[col].std()) if not df[col].isna().all() else None,
        }
    return ranges


def _count_outliers(df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, Any]:
    """Count outliers using z-score method."""
    total = 0
    by_column = {}
    
    for col in df.select_dtypes(include=[np.number]).columns:
        col_data = df[col].dropna()
        if len(col_data) < 3:
            continue
        mean = col_data.mean()
        std = col_data.std()
        if std == 0:
            continue
        z_scores = np.abs((col_data - mean) / std)
        outlier_count = int((z_scores > threshold).sum())
        if outlier_count > 0:
            by_column[col] = outlier_count
            total += outlier_count
    
    return {"total": total, "by_column": by_column}


def _check_drift(df: pd.DataFrame, df_original: pd.DataFrame) -> Dict[str, Any]:
    """Check for distribution drift between original and processed data."""
    drift_columns = []
    
    common_cols = set(df.columns) & set(df_original.columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in common_cols:
        if col not in numeric_cols:
            continue
        if col not in df_original.columns:
            continue
            
        try:
            orig_mean = df_original[col].mean()
            new_mean = df[col].mean()
            orig_std = df_original[col].std()
            
            if orig_std > 0:
                # Check if mean shifted by more than 0.5 std
                if abs(new_mean - orig_mean) / orig_std > 0.5:
                    drift_columns.append(col)
        except Exception:
            continue
    
    rate = len(drift_columns) / len(common_cols) if common_cols else 0
    return {
        "detected": len(drift_columns) > 0,
        "columns": drift_columns,
        "rate": round(rate, 3),
    }


def _identify_high_risk_columns(
    df: pd.DataFrame, 
    missing_pct: Dict[str, float]
) -> List[str]:
    """Identify columns that may cause model reliability issues."""
    high_risk = []
    
    # Columns with >50% missing
    for col, pct in missing_pct.items():
        if pct > 50:
            high_risk.append(col)
    
    # Columns with very low variance (near-constant)
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in high_risk:
            continue
        unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
        if unique_ratio < 0.01 and df[col].nunique() < 3:
            high_risk.append(col)
    
    return high_risk


def _compute_overall_quality_score(metrics: Dict[str, Any]) -> int:
    """Compute an overall quality score from 0-100."""
    score = 100.0
    
    # Penalize for missing data
    completeness = metrics.get("completeness_score", 100)
    if completeness < 100:
        score -= (100 - completeness) * 0.3
    
    # Penalize for high-risk columns
    high_risk_count = len(metrics.get("high_risk_columns", []))
    score -= high_risk_count * 5
    
    # Penalize for outliers
    outlier_count = metrics.get("outlier_count", 0)
    if outlier_count > 10:
        score -= min(10, outlier_count * 0.5)
    
    # Penalize for type inconsistency
    if not metrics.get("type_consistency", {}).get("is_consistent", True):
        score -= 10
    
    # Penalize for drift
    if metrics.get("drift_detected", False):
        score -= 15
    
    return max(0, min(100, int(score)))


def summarize_for_display(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Create a user-friendly summary of quality metrics for dashboard display."""
    score = metrics.get("data_quality_score", 100)
    
    # Determine status
    if score >= 80:
        status = "good"
        status_emoji = "✅"
        status_text = "Good"
    elif score >= 60:
        status = "warning"
        status_emoji = "⚠️"
        status_text = "Fair"
    else:
        status = "critical"
        status_emoji = "❌"
        status_text = "Poor"
    
    return {
        "score": score,
        "status": status,
        "status_emoji": status_emoji,
        "status_text": status_text,
        "completeness_pct": metrics.get("completeness_score", 100),
        "rows_complete_pct": metrics.get("rows_complete_pct", 100),
        "missing_columns": metrics.get("columns_missing_data", []),
        "high_risk_columns": metrics.get("high_risk_columns", []),
        "outlier_count": metrics.get("outlier_count", 0),
        "warnings": metrics.get("warnings", []),
        "proceed_with_caution": metrics.get("proceed_with_caution", False),
    }

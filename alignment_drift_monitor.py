import pandas as pd
import re
from collections import Counter
from typing import Any, Dict, List, Optional


def generate_historical_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate baseline statistics from a clean DataFrame."""
    row_length_dist = df.notna().sum(axis=1).tolist()
    col_dtype_map = df.dtypes.to_dict()

    col_token_patterns: Dict[str, Optional[re.Pattern]] = {}
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().astype(str).head(100)
        if sample.empty:
            col_token_patterns[col] = None
            continue
        simple_pattern = re.compile(r"^[A-Za-z0-9_\-\s]+$")
        if sample.map(lambda x: bool(simple_pattern.match(x))).all():
            col_token_patterns[col] = simple_pattern
        else:
            col_token_patterns[col] = None

    return {
        "row_length_dist": row_length_dist,
        "col_dtype_map": col_dtype_map,
        "col_token_patterns": col_token_patterns,
    }


def monitor_alignment_drift(
    df_new: pd.DataFrame,
    historical_stats: Dict[str, Any],
    drift_threshold: float = 0.2,
    return_drifted_rows: bool = False,
) -> Dict[str, Any]:
    """Check new dataframe for alignment drift against historical statistics."""

    total_rows = len(df_new)
    flagged_rows: set[int] = set()

    # Row length comparison
    expected_counts = historical_stats.get("row_length_dist", [])
    expected_mode = None
    if expected_counts:
        expected_mode = Counter(expected_counts).most_common(1)[0][0]

    row_lengths = df_new.notna().sum(axis=1)
    mism_len = row_lengths[row_lengths != expected_mode] if expected_mode is not None else pd.Series([], dtype=int)
    flagged_rows.update(mism_len.index)

    drift_metrics: Dict[str, Any] = {
        "row_length_mismatches": int(len(mism_len))
    }

    # Column dtype mismatches
    type_mismatches: List[str] = []
    for col, expected_dtype in historical_stats.get("col_dtype_map", {}).items():
        if col not in df_new.columns:
            type_mismatches.append(col)
            flagged_rows.update(df_new.index)
            continue
        if not pd.api.types.is_dtype_equal(df_new[col].dtype, expected_dtype):
            type_mismatches.append(col)
            flagged_rows.update(df_new.index[df_new[col].notna()])
    drift_metrics["type_mismatches"] = type_mismatches

    # Pattern mismatches for text columns
    pattern_mismatches: Dict[str, int] = {}
    for col, pattern in historical_stats.get("col_token_patterns", {}).items():
        if pattern is None or col not in df_new.columns:
            continue
        mism = df_new[col].dropna().astype(str).apply(lambda x: not bool(pattern.match(x)))
        if mism.any():
            pattern_mismatches[col] = int(mism.sum())
            flagged_rows.update(mism[mism].index)
    drift_metrics["pattern_mismatches"] = pattern_mismatches

    drift_rate = len(flagged_rows) / total_rows if total_rows > 0 else 0.0
    drift_detected = drift_rate > drift_threshold

    result: Dict[str, Any] = {
        "drift_detected": drift_detected,
        "drift_rate": drift_rate,
        "drift_metrics": drift_metrics,
    }
    if return_drifted_rows:
        result["drifted_rows"] = sorted(flagged_rows)

    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Alignment Drift Monitor test harness")
    parser.add_argument("--baseline", required=True, help="Path to clean baseline CSV")
    parser.add_argument("--new", required=True, help="Path to new CSV batch")
    parser.add_argument("--threshold", type=float, default=0.2, help="Drift threshold")
    parser.add_argument("--block-on-drift", action="store_true", help="Exit with error if drift detected")
    args = parser.parse_args()

    baseline_df = pd.read_csv(args.baseline)
    stats = generate_historical_stats(baseline_df)

    new_df = pd.read_csv(args.new)
    result = monitor_alignment_drift(new_df, stats, drift_threshold=args.threshold, return_drifted_rows=True)

    print(json.dumps(result, indent=2))

    if result["drift_detected"] and args.block_on_drift:
        raise SystemExit("Alignment drift detected beyond threshold")

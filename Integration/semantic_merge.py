from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from catalog.semantic_index import (
    find_tables_by_roles,
    get_table_metadata,
    fetch_semantic_index_from_s3,
)
from config import USE_CLOUD, BUCKET_NAME, SEMANTIC_INDEX_KEY
from preprocessing.column_meta import ColumnMeta
from utils.key_mappers import hash_columns, zip_to_fips


_DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"
_INDEX_DB = Path(__file__).resolve().parents[1] / "catalog" / "semantic_index.db"
if USE_CLOUD and not _INDEX_DB.exists():
    _INDEX_DB = fetch_semantic_index_from_s3(BUCKET_NAME, SEMANTIC_INDEX_KEY)


def _load_dataset(name: str, datasets_dir: Path = _DATASETS_DIR) -> pd.DataFrame | None:
    path = datasets_dir / name
    if not path.exists():
        return None
    try:
        if name.endswith(".csv"):
            return pd.read_csv(path)
        if name.endswith(".json"):
            return pd.read_json(path)
        if name.endswith((".xls", ".xlsx")):
            return pd.read_excel(path)
    except Exception:
        return None
    return None


def find_candidate_tables(column_meta: List[ColumnMeta]) -> List[str]:
    roles = {m.role for m in column_meta if m.role != "unknown"}
    if not roles:
        return []
    return find_tables_by_roles(roles, db_path=_INDEX_DB)


def synthesise_join_keys(
    df_user: pd.DataFrame,
    df_table: pd.DataFrame,
    user_meta: List[ColumnMeta],
    table_meta: List[Tuple[str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[str, str]], str]:
    user_roles: Dict[str, str] = {m.role: m.name for m in user_meta}
    table_roles: Dict[str, str] = {role: name for name, role in table_meta}

    shared = [(user_roles[r], table_roles[r]) for r in set(user_roles) & set(table_roles)]
    if shared:
        return df_user, df_table, shared, "direct"

    if "zip_code" in user_roles and "fips_code" in table_roles:
        df_user = df_user.copy(deep=False)
        df_user["_join_key"] = df_user[user_roles["zip_code"]].map(zip_to_fips)
        df_table = df_table.copy(deep=False)
        df_table["_join_key"] = df_table[table_roles["fips_code"]]
        return df_user, df_table, [("_join_key", "_join_key")], "zip_to_fips"

    if {"city", "state"} <= set(user_roles) and {"city", "state"} <= set(table_roles):
        df_user = hash_columns(
            df_user.copy(deep=False), [user_roles["city"], user_roles["state"]], "_join_key"
        )
        df_table = hash_columns(
            df_table.copy(deep=False), [table_roles["city"], table_roles["state"]], "_join_key"
        )
        return df_user, df_table, [("_join_key", "_join_key")], "city_state_hash"

    shared_roles = set(user_roles) & set(table_roles)
    if shared_roles:
        df_user = hash_columns(
            df_user.copy(deep=False), [user_roles[r] for r in shared_roles], "_join_key"
        )
        df_table = hash_columns(
            df_table.copy(deep=False), [table_roles[r] for r in shared_roles], "_join_key"
        )
        return df_user, df_table, [("_join_key", "_join_key")], "hash"

    return df_user, df_table, [], "none"


def _attempt_merge(
    df_user: pd.DataFrame,
    table_df: pd.DataFrame,
    left_on: List[str],
    right_on: List[str],
) -> pd.DataFrame | None:
    """Attempt a merge with multiple type coercion strategies.
    
    Returns the merged DataFrame on success, None on failure.
    Tries progressively more aggressive coercion strategies.
    """
    u_df = df_user
    t_df = table_df
    
    # Strategy 1: Direct merge (no coercion)
    try:
        return pd.merge(u_df, t_df, left_on=left_on, right_on=right_on, how="left")
    except (ValueError, TypeError):
        pass
    
    # Strategy 2: Coerce mismatched types to string
    try:
        u_df = df_user.copy()
        t_df = table_df.copy()
        for l_col, r_col in zip(left_on, right_on):
            if u_df[l_col].dtype != t_df[r_col].dtype:
                u_df[l_col] = u_df[l_col].astype(str).str.strip().str.lower()
                t_df[r_col] = t_df[r_col].astype(str).str.strip().str.lower()
        return pd.merge(u_df, t_df, left_on=left_on, right_on=right_on, how="left")
    except (ValueError, TypeError):
        pass
    
    # Strategy 3: Force all join keys to string
    try:
        u_df = df_user.copy()
        t_df = table_df.copy()
        for l_col, r_col in zip(left_on, right_on):
            u_df[l_col] = u_df[l_col].astype(str)
            t_df[r_col] = t_df[r_col].astype(str)
        return pd.merge(u_df, t_df, left_on=left_on, right_on=right_on, how="left")
    except (ValueError, TypeError):
        pass
    
    # Strategy 4: Try numeric coercion for both
    try:
        u_df = df_user.copy()
        t_df = table_df.copy()
        for l_col, r_col in zip(left_on, right_on):
            u_df[l_col] = pd.to_numeric(u_df[l_col], errors="coerce")
            t_df[r_col] = pd.to_numeric(t_df[r_col], errors="coerce")
        return pd.merge(u_df, t_df, left_on=left_on, right_on=right_on, how="left")
    except (ValueError, TypeError):
        pass
    
    # All strategies failed
    return None


def rank_and_merge(
    df_user: pd.DataFrame,
    column_meta: List[ColumnMeta],
    datasets_dir: Path | None = None,
    max_merges: int = 3,
) -> Tuple[pd.DataFrame, Dict]:
    """Find and merge the best matching public datasets.
    
    Tries all candidate tables, gracefully skipping failures.
    Can perform multiple successful merges up to max_merges.
    
    Args:
        df_user: User's input DataFrame
        column_meta: Inferred column metadata
        datasets_dir: Directory containing public datasets
        max_merges: Maximum number of successful merges to perform
        
    Returns:
        Tuple of (enriched DataFrame, merge report)
    """
    datasets_dir = Path(datasets_dir or _DATASETS_DIR)
    candidates = find_candidate_tables(column_meta)
    report_details = []
    current_df = df_user
    successful_merges = []
    failed_merges = []

    for name in candidates:
        # Stop if we've done enough merges
        if len(successful_merges) >= max_merges:
            break
            
        table_df = _load_dataset(name, datasets_dir)
        if table_df is None:
            failed_merges.append({"table": name, "reason": "load_failed"})
            continue
            
        table_meta = get_table_metadata(name, db_path=_INDEX_DB)
        u_df, t_df, join_keys, method = synthesise_join_keys(
            current_df, table_df, column_meta, table_meta
        )
        
        if not join_keys:
            failed_merges.append({"table": name, "method": method, "reason": "no_join_keys"})
            continue
            
        left_on, right_on = zip(*join_keys)
        
        # Attempt merge with multiple strategies
        merged = _attempt_merge(u_df, t_df, list(left_on), list(right_on))
        
        if merged is None:
            failed_merges.append({
                "table": name,
                "method": method,
                "reason": "merge_failed",
                "join_keys": list(zip(left_on, right_on)),
            })
            continue
        
        # Check if merge actually added useful columns
        new_cols = set(merged.columns) - set(current_df.columns)
        # Filter out join key artifacts
        new_cols = {c for c in new_cols if not c.startswith("_join_key")}
        
        if not new_cols:
            failed_merges.append({
                "table": name,
                "method": method,
                "reason": "no_new_columns",
            })
            continue
        
        # Check merge quality - did we get actual matches?
        # If all new columns are NaN, the merge didn't really work
        sample_col = list(new_cols)[0]
        non_null_ratio = merged[sample_col].notna().mean()
        
        if non_null_ratio < 0.01:  # Less than 1% matched
            failed_merges.append({
                "table": name,
                "method": method,
                "reason": "poor_match_rate",
                "match_rate": f"{non_null_ratio:.1%}",
            })
            continue
        
        # Success! Update current_df and record
        gain = len(new_cols)
        current_df = merged
        successful_merges.append({
            "table": name,
            "method": method,
            "new_columns": gain,
            "column_names": list(new_cols),
            "match_rate": f"{non_null_ratio:.1%}",
        })
        
        report_details.append({
            "table": name,
            "method": method,
            "joined": True,
            "new_columns": gain,
            "match_rate": f"{non_null_ratio:.1%}",
        })
    
    # Add failed attempts to report
    for fail in failed_merges:
        report_details.append({
            "table": fail["table"],
            "method": fail.get("method", "unknown"),
            "joined": False,
            "reason": fail.get("reason", "unknown"),
        })
    
    # Build summary report
    total_new_cols = sum(m["new_columns"] for m in successful_merges)
    report = {
        "tables_considered": len(candidates),
        "successful_merges": len(successful_merges),
        "failed_attempts": len(failed_merges),
        "total_new_columns": total_new_cols,
        "merged_tables": [m["table"] for m in successful_merges],
        "details": report_details,
    }
    
    return current_df, report

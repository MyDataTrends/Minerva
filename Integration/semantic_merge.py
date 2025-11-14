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


def rank_and_merge(
    df_user: pd.DataFrame,
    column_meta: List[ColumnMeta],
    datasets_dir: Path | None = None,
) -> Tuple[pd.DataFrame, Dict]:
    datasets_dir = Path(datasets_dir or _DATASETS_DIR)
    candidates = find_candidate_tables(column_meta)
    report_details = []
    best_df = df_user
    best_gain = -1
    chosen = None

    for name in candidates:
        table_df = _load_dataset(name, datasets_dir)
        if table_df is None:
            continue
        table_meta = get_table_metadata(name, db_path=_INDEX_DB)
        u_df, t_df, join_keys, method = synthesise_join_keys(df_user, table_df, column_meta, table_meta)
        if not join_keys:
            report_details.append({"table": name, "method": method, "joined": False})
            continue
        left_on, right_on = zip(*join_keys)
        merged = pd.merge(u_df, t_df, left_on=list(left_on), right_on=list(right_on), how="left")
        gain = merged.shape[1] - df_user.shape[1]
        report_details.append({
            "table": name,
            "method": method,
            "joined": True,
            "new_columns": gain,
        })
        if gain > best_gain:
            best_gain = gain
            best_df = merged
            chosen = name

    report = {"tables_considered": candidates, "chosen_table": chosen, "details": report_details}
    return best_df, report

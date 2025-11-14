"""Compatibility shim for legacy imports.

This module is deprecated. Please import from ``integration.semantic_merge`` instead.
It remains as a thin wrapper to preserve backward compatibility while the
package layout is being reorganized.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from preprocessing.column_meta import ColumnMeta

from Integration.semantic_merge import (
    find_candidate_tables as _find_candidate_tables,
    synthesise_join_keys as _synthesise_join_keys,
    rank_and_merge as _rank_and_merge,
)
import Integration.semantic_merge as _sm

warnings.warn(
    "Integration.semantic_integration is deprecated; use integration.semantic_merge.",
    DeprecationWarning,
    stacklevel=2,
)

# Mirror configurable module-level paths so tests can override them here
_DATASETS_DIR = _sm._DATASETS_DIR
_INDEX_DB = _sm._INDEX_DB


# Re-exported wrapper functions for backward compatibility
def find_candidate_tables(column_meta: List[ColumnMeta]) -> List[str]:
    return _find_candidate_tables(column_meta)


def synthesise_join_keys(
    df_user: pd.DataFrame,
    df_table: pd.DataFrame,
    user_meta: List[ColumnMeta],
    table_meta: List[Tuple[str, str]],
):
    return _synthesise_join_keys(df_user, df_table, user_meta, table_meta)


def rank_and_merge(
    df_user: pd.DataFrame,
    column_meta: List[ColumnMeta],
    datasets_dir: Path | None = None,
):
    # Ensure the underlying implementation uses the overridden DB path
    _sm._INDEX_DB = _INDEX_DB
    return _rank_and_merge(df_user, column_meta, datasets_dir or _DATASETS_DIR)


__all__ = [
    "find_candidate_tables",
    "synthesise_join_keys",
    "rank_and_merge",
]

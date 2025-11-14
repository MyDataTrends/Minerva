from __future__ import annotations

from typing import Optional, Dict
import pandas as pd

from preprocessing.save_meta import save_column_roles, load_column_roles, _hash_df


class MetadataCache:
    """Load or save metadata roles based on column hashes."""

    def load(self, df: pd.DataFrame) -> Optional[Dict[str, str]]:
        return load_column_roles(df)  # type: ignore[arg-type]

    def save(self, df: pd.DataFrame, meta, run_id: str) -> str:
        return save_column_roles(df, meta, identifier=run_id)

    def hash_df(self, df: pd.DataFrame) -> str:
        return _hash_df(df)

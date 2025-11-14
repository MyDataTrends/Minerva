from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .column_meta import ColumnMeta
from utils.security import secure_join


def _hash_df(df: pd.DataFrame) -> str:
    data = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha1(data).hexdigest()


def save_column_roles(
    df: pd.DataFrame,
    meta: List[ColumnMeta],
    dest_dir: str = "metadata",
    identifier: str | None = None,
) -> str:
    """Persist column roles for ``df`` to ``dest_dir`` and return file path.

    If ``identifier`` is provided, it is prepended to the file name so multiple
    versions can coexist.
    """
    os.makedirs(dest_dir, exist_ok=True)
    h = _hash_df(df)
    prefix = f"{identifier}_" if identifier else ""
    path = secure_join(Path(dest_dir), f"{prefix}{h}_roles.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                m.name: {"role": m.role, "description": m.description}
                for m in meta
            },
            f,
            indent=2,
        )
    return path


def load_column_roles(
    df: pd.DataFrame,
    dest_dir: str = "metadata",
    identifier: str | None = None,
) -> Optional[Dict[str, str]]:
    """Return stored column roles for ``df``.

    If ``identifier`` is ``None`` the most recently modified matching file is
    loaded when available.
    """
    h = _hash_df(df)
    base = Path(dest_dir)
    if identifier:
        path = secure_join(base, f"{identifier}_{h}_roles.json")
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    # Find the newest versioned file or fall back to legacy naming
    pattern = f"*_{h}_roles.json"
    matches = list(base.glob(pattern))
    if matches:
        latest = max(matches, key=lambda p: p.stat().st_mtime)
        with open(latest, "r", encoding="utf-8") as f:
            return json.load(f)
    legacy = secure_join(base, f"{h}_roles.json")
    if legacy.exists():
        with open(legacy, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_column_descriptions(
    df: pd.DataFrame,
    descriptions: Dict[str, str],
    dest_dir: str = "metadata",
    identifier: str | None = None,
) -> str:
    """Persist column descriptions for ``df`` to ``dest_dir``.

    ``identifier`` works the same as in :func:`save_column_roles`.
    """
    os.makedirs(dest_dir, exist_ok=True)
    h = _hash_df(df)
    prefix = f"{identifier}_" if identifier else ""
    path = secure_join(Path(dest_dir), f"{prefix}{h}_descriptions.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, indent=2)
    return path


def load_column_descriptions(
    df: pd.DataFrame,
    dest_dir: str = "metadata",
    identifier: str | None = None,
) -> Optional[Dict[str, str]]:
    """Return stored column descriptions for ``df`` if available."""
    h = _hash_df(df)
    base = Path(dest_dir)
    if identifier:
        path = secure_join(base, f"{identifier}_{h}_descriptions.json")
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    pattern = f"*_{h}_descriptions.json"
    matches = list(base.glob(pattern))
    if matches:
        latest = max(matches, key=lambda p: p.stat().st_mtime)
        with open(latest, "r", encoding="utf-8") as f:
            return json.load(f)
    legacy = secure_join(base, f"{h}_descriptions.json")
    if legacy.exists():
        with open(legacy, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

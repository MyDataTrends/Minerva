import json
import os
from pathlib import Path
from typing import Dict

import pandas as pd

from utils.security import secure_join
from preprocessing.save_meta import _hash_df


def store_role_corrections(
    df: pd.DataFrame | None,
    corrections: Dict[str, str],
    file_path: str = "feedback/role_corrections.json",
) -> None:
    """Persist ``corrections`` keyed by the DataFrame hash."""
    dest_dir = Path(file_path).parent
    safe_path = secure_join(dest_dir, Path(file_path).name)
    os.makedirs(dest_dir, exist_ok=True)
    try:
        with open(safe_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    h = _hash_df(df) if df is not None else None
    if h is None:
        return
    if h not in data:
        data[h] = {}
    data[h].update(corrections)

    with open(safe_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_role_corrections(
    df: pd.DataFrame | None,
    file_path: str = "feedback/role_corrections.json",
) -> Dict[str, str]:
    """Return stored corrections for ``df`` if available."""
    dest_dir = Path(file_path).parent
    safe_path = secure_join(dest_dir, Path(file_path).name)
    try:
        with open(safe_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}

    h = _hash_df(df) if df is not None else None
    if h is None:
        return {}
    return data.get(h, {})


def store_role_corrections_by_hash(
    df_hash: str, corrections: Dict[str, str], file_path: str = "feedback/role_corrections.json"
) -> None:
    """Persist ``corrections`` under precomputed ``df_hash``."""
    dest_dir = Path(file_path).parent
    safe_path = secure_join(dest_dir, Path(file_path).name)
    os.makedirs(dest_dir, exist_ok=True)
    try:
        with open(safe_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    if df_hash not in data:
        data[df_hash] = {}
    data[df_hash].update(corrections)

    with open(safe_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_role_corrections_by_hash(
    df_hash: str, file_path: str = "feedback/role_corrections.json"
) -> Dict[str, str]:
    """Return stored corrections under ``df_hash``."""
    dest_dir = Path(file_path).parent
    safe_path = secure_join(dest_dir, Path(file_path).name)
    try:
        with open(safe_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    return data.get(df_hash, {})

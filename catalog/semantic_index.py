import os
import sqlite3
import logging
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from preprocessing.metadata_parser import infer_column_meta
from config import USE_CLOUD, BUCKET_NAME, SEMANTIC_INDEX_KEY
import boto3

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "semantic_index.db"


def fetch_semantic_index_from_s3(bucket_name: str, file_key: str, dest: Path | str = DEFAULT_DB_PATH) -> Path:
    """Download the semantic index file from S3 to ``dest`` and return the path."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
    s3.download_file(bucket_name, file_key, str(dest))
    return dest


def _load_dataframe(path: str) -> pd.DataFrame | None:
    try:
        if path.endswith(".csv"):
            return pd.read_csv(path)
        if path.endswith(".json"):
            return pd.read_json(path)
        if path.endswith(('.xls', '.xlsx')):
            return pd.read_excel(path)
    except Exception as e:
        logger.warning(f"Failed to load dataframe from {path}: {e}")
        return None
    return None


def build_index(datasets_dir: str = "datasets", db_path: str | os.PathLike = DEFAULT_DB_PATH) -> None:
    """Scan ``datasets_dir`` and build a semantic index SQLite DB."""
    os.makedirs(Path(db_path).parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS column_stats (table_name TEXT, column_name TEXT, role TEXT, uniqueness REAL)"
    )
    cur.execute("DELETE FROM column_stats")
    for fname in os.listdir(datasets_dir):
        fpath = os.path.join(datasets_dir, fname)
        if not os.path.isfile(fpath):
            continue
        df = _load_dataframe(fpath)
        if df is None:
            continue
        try:
            meta = infer_column_meta(df)
        except Exception as e:
            logger.warning(f"Failed to infer column metadata for {fname}: {e}")
            continue
        for m in meta:
            if len(df) == 0:
                uniq = 0.0
            else:
                uniq = df[m.name].nunique(dropna=False) / len(df)
            cur.execute(
                "INSERT INTO column_stats VALUES (?,?,?,?)",
                (fname, m.name, m.role, float(uniq)),
            )
    conn.commit()
    conn.close()
    if USE_CLOUD:
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
        s3.upload_file(str(db_path), BUCKET_NAME, SEMANTIC_INDEX_KEY)


def find_tables_by_roles(roles: Iterable[str], db_path: str | os.PathLike = DEFAULT_DB_PATH) -> List[str]:
    """Return table names that contain columns matching any of ``roles`` sorted by count."""
    roles = list(roles)
    if not roles:
        return []
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    qmarks = ",".join("?" for _ in roles)
    query = f"SELECT table_name, COUNT(*) as cnt FROM column_stats WHERE role IN ({qmarks}) GROUP BY table_name ORDER BY cnt DESC"
    cur.execute(query, roles)
    results = [row[0] for row in cur.fetchall()]
    conn.close()
    return results


def get_table_metadata(table_name: str, db_path: str | os.PathLike = DEFAULT_DB_PATH) -> List[tuple[str, str]]:
    """Return list of (column_name, role) for ``table_name`` from the index."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT column_name, role FROM column_stats WHERE table_name=?", (table_name,))
    rows = cur.fetchall()
    conn.close()
    return rows

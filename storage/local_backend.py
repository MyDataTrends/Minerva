from __future__ import annotations

import io
import json
import os
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
from utils.security import secure_join

import pandas as pd
from utils.logging import configure_logging, get_logger
from categorization.categorize import categorize_dataset
from config import (
    CSV_READ_CHUNKSIZE,
    DESCRIBE_SAMPLE_ROWS,
    LARGE_FILE_BYTES,
)

from .base import StorageBackend

configure_logging()
logger = get_logger(__name__)


class LocalStorage(StorageBackend):
    """Storage backend that reads and writes from the local filesystem."""

    def __init__(self, base_dir: str | Path | None = None):
        self._base_dir = Path(base_dir) if base_dir is not None else None

    def _dir(self) -> Path:
        return Path(self._base_dir or os.getenv("LOCAL_DATA_DIR", "local_data"))

    def get_file(self, path: str) -> Path:
        file_path = secure_join(self._dir(), path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        return file_path

    def put_file(self, src: Path, dest: str) -> None:
        dest_path = secure_join(self._dir(), dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(Path(src).read_bytes())

    def list_files(self, prefix: str = "") -> list[str]:
        base_path = secure_join(self._dir(), prefix)
        if not base_path.exists():
            return []
        if base_path.is_file():
            return [prefix]
        return [
            str(p.relative_to(self._dir()).as_posix())
            for p in base_path.glob("**/*")
            if p.is_file()
        ]

    def read_df(self, path: str, *, sample_only: bool = False) -> pd.DataFrame:
        """Read a DataFrame from ``path`` with optional sampling.

        For CSV files, this will stream the file in chunks when the file is
        large or ``CSV_READ_CHUNKSIZE`` is configured to a positive value.
        A sample up to ``DESCRIBE_SAMPLE_ROWS`` rows can be returned by setting
        ``sample_only`` to ``True``.
        """
        file_path = self.get_file(path)
        suffix = file_path.suffix.lower().lstrip(".")

        if suffix != "csv":
            df = parse_file(
                file_path.read_bytes(), file_name=file_path.name, file_type=suffix
            )
            return df if not sample_only else df.head(DESCRIBE_SAMPLE_ROWS)

        file_size = file_path.stat().st_size
        chunk_size = CSV_READ_CHUNKSIZE or 50000

        if file_size > LARGE_FILE_BYTES or CSV_READ_CHUNKSIZE > 0:
            reader = pd.read_csv(file_path, chunksize=chunk_size)
            if sample_only:
                chunks: list[pd.DataFrame] = []
                total = 0
                for chunk in reader:
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= DESCRIBE_SAMPLE_ROWS:
                        break
                return pd.concat(chunks, ignore_index=True).head(DESCRIBE_SAMPLE_ROWS)
            return pd.concat(reader, ignore_index=True)

        df = pd.read_csv(file_path)
        return df if not sample_only else df.head(DESCRIBE_SAMPLE_ROWS)


def parse_file(
    data: bytes, file_name: Optional[str] = None, file_type: Optional[str] = None
) -> pd.DataFrame | dict:
    """Parse raw file bytes into a DataFrame and validate content."""
    if file_type is None and file_name:
        file_type = Path(file_name).suffix.lstrip(".")
    file_type = (file_type or "csv").lower()

    try:
        if file_type == "csv":
            df = pd.read_csv(io.BytesIO(data))
        elif file_type in {"xls", "xlsx"}:
            df = pd.read_excel(io.BytesIO(data))
        elif file_type == "json":
            df = pd.read_json(io.BytesIO(data))
        elif file_type == "tsv":
            df = pd.read_csv(io.BytesIO(data), delimiter="\t")
        elif file_type == "parquet":
            df = pd.read_parquet(io.BytesIO(data))
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        if df.empty:
            raise ValueError("The file is empty")
        if not any(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
            raise ValueError("The file must contain at least one numeric column")
        return df
    except ValueError as exc:
        logger.error("ValueError parsing file: %s", exc)
        return {"error": str(exc)}
    except KeyError as exc:
        logger.error("KeyError parsing file: %s", exc)
        return {
            "error": "The data is missing expected columns. Please ensure the file contains the correct columns."
        }
    except Exception as exc:
        logger.error("Error parsing file: %s", exc)
        return {
            "error": "Failed to parse file. Please check the file format and content."
        }


def load_datalake_dfs(base_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load all datasets from the local datalake directory into a dict of DataFrames."""
    storage_dir = (
        Path(base_dir or os.getenv("LOCAL_DATA_DIR", "local_data")) / "datasets"
    )
    datalake_dfs: dict[str, pd.DataFrame] = {}
    if not storage_dir.is_dir():
        return datalake_dfs

    for path in storage_dir.iterdir():
        if not path.is_file():
            continue
        try:
            if path.suffix == ".csv":
                df = pd.read_csv(path)
            elif path.suffix == ".json":
                df = pd.read_json(path)
            elif path.suffix in {".xls", ".xlsx"}:
                df = pd.read_excel(path)
            elif path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                continue
            df.attrs["tags"] = categorize_dataset(df)
            datalake_dfs[path.name] = df
        except Exception as exc:  # pragma: no cover - best effort loading
            print(f"Failed to load {path.name}: {exc}")
    return datalake_dfs

def _resolve_history_path(path: str | Path | None) -> Path:
    """Return full path to the ``run_history.json`` file.

    ``path`` can be either a directory or a file. If ``None`` it defaults to the
    ``LOCAL_DATA_DIR`` environment variable.
    """
    base = Path(path or Path(os.getenv("LOCAL_DATA_DIR", "local_data")))
    if base.is_dir() or not base.suffix:
        return secure_join(base, "run_history.json")
    return secure_join(base.parent, base.name)

DB_PATH = os.getenv("SESSION_HISTORY_DB", "session_history.db")


def log_run_metadata(
    run_id: str,
    score_ok: bool,
    needs_role_review: bool,
    *,
    dataset_path: str | None = None,
    target_column: str | None = None,
    model_path: str | None = None,
    column_roles: dict | None = None,
    file_name: str | None = None,
    user_id: str | None = None,
    model_name: str | None = None,
    model_type: str | None = None,
    metadata_path: str | None = None,
    output_path: str | None = None,
    roles_path: str | None = None,
    descriptions_path: str | None = None,
    file_path: str | None = None,
    db_path: str | None = None,
    **extra: Any,
) -> None:
    """Persist metadata for a pipeline run."""

    entry: dict[str, Any] = {
        "run_id": run_id,
        "score_ok": bool(score_ok),
        "needs_role_review": bool(needs_role_review),
    }

    optional_fields = {
        "dataset_path": dataset_path,
        "target_column": target_column,
        "model_path": model_path,
        "file_name": file_name,
        "user_id": user_id,
        "model_name": model_name,
        "model_type": model_type,
        "metadata_path": metadata_path,
        "output_path": output_path,
        "column_roles": column_roles,
        "roles_path": roles_path,
        "descriptions_path": descriptions_path,
    }

    for key, val in optional_fields.items():
        if val is not None:
            entry[key] = val

    if extra:
        entry.update({k: v for k, v in extra.items() if v is not None})

    if os.getenv("DYNAMO_SESSIONS_TABLE"):
        from . import session_db

        session_db.save_run_to_dynamo(entry)
        return

    target = _resolve_history_path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(target.read_text())
    except FileNotFoundError:
        data = []
    data.append(entry)
    target.write_text(json.dumps(data))

    db_file = db_path or DB_PATH
    if db_file:
        conn = sqlite3.connect(db_file)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS session_history (
            run_id TEXT,
            user_id TEXT,
            file_name TEXT,
            model_name TEXT,
            metadata_path TEXT,
            output_path TEXT,
            score_ok INTEGER,
            needs_role_review INTEGER
        )"""
        )
        conn.execute(
            "INSERT INTO session_history VALUES (?,?,?,?,?,?,?,?)",
            (
                run_id,
                user_id,
                file_name,
                model_name,
                metadata_path,
                output_path,
                int(score_ok),
                int(needs_role_review),
            ),
        )
        conn.commit()
        conn.close()
    
def load_run_metadata(run_id: str, file_path: str | Path | None = None) -> dict:
    """Return stored metadata for ``run_id`` when available."""

    if os.getenv("DYNAMO_SESSIONS_TABLE"):
        from . import session_db

        res = session_db.get_run_by_id(run_id)
        return res or {}

    target = _resolve_history_path(file_path)
    try:
        data = json.loads(target.read_text())
    except FileNotFoundError:
        return {}
    for entry in data:
        if entry.get("run_id") == run_id:
            return entry
    return {}

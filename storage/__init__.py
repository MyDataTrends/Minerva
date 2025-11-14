from pathlib import Path

from .get_backend import backend
from .local_backend import (
    parse_file,
    load_datalake_dfs,
    log_run_metadata,
    load_run_metadata,
)

__all__ = [
    "backend",
    "parse_file",
    "load_datalake_dfs",
    "log_run_metadata",
    "load_run_metadata",
    "load_user_file",
    "load_dataset_file",
    "load_user_dataframe",
    "load_dataset_dataframe",
]

ROOT_DIR = Path(__file__).resolve().parents[1]


def load_user_file(user_id: str, filename: str) -> bytes:
    path = backend.get_file(f"User_Data/{user_id}/{filename}")
    return Path(path).read_bytes()


def load_dataset_file(filename: str) -> bytes:
    path = backend.get_file(f"datasets/{filename}")
    return Path(path).read_bytes()


def load_user_dataframe(user_id: str, filename: str, file_type: str | None = None):
    data = load_user_file(user_id, filename)
    if file_type is None:
        file_type = Path(filename).suffix.lstrip(".")
    return parse_file(data, file_name=filename, file_type=file_type)


def load_dataset_dataframe(filename: str, file_type: str | None = None):
    data = load_dataset_file(filename)
    if file_type is None:
        file_type = Path(filename).suffix.lstrip(".")
    return parse_file(data, file_name=filename, file_type=file_type)

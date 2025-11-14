"""Data intake utilities."""

try:
    from storage.local_backend import parse_file
    download_user_file = None
except Exception:  # pragma: no cover - optional dependency may be missing
    parse_file = None
    download_user_file = None

try:
    from .datalake_ingestion import sync_from_s3, fetch_from_api, main as ingest_cli
except Exception:  # pragma: no cover - optional dependency may be missing
    sync_from_s3 = None
    fetch_from_api = None
    def ingest_cli(*_args, **_kwargs):
        raise RuntimeError("datalake ingestion dependencies are missing")

__all__ = [
    "parse_file",
    "sync_from_s3",
    "fetch_from_api",
    "ingest_cli",
]

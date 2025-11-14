from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

from config import MAX_GB_FREE, MAX_REQUESTS_FREE, get_bool

from utils.security import secure_join

__all__ = [
    "increment_request",
    "usage_info",
    "increment_usage",
    "get_usage",
    "check_quota",
]

_USAGELock = Lock()


def _usage_dir() -> Path:
    base_dir = Path(os.getenv("LOCAL_DATA_DIR", "local_data"))
    return Path(os.getenv("USAGE_DIR", str(base_dir / "usage")))


def _usage_file(user_id: str) -> Path:
    return secure_join(_usage_dir(), f"{user_id}.json")


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {"requests": 0, "bytes": 0}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _get_table():
    table_name = os.getenv("USAGE_TRACKING_TABLE")
    if table_name and get_bool("USE_CLOUD", False):
        import boto3

        region = os.getenv("AWS_REGION", "us-east-1")
        return boto3.resource("dynamodb", region_name=region).Table(table_name)
    return None

def _increment_file(user_id: str, bytes_uploaded: int) -> None:
    path = _usage_file(user_id)
    with _USAGELock:
        info = _load(path)
        info["requests"] = info.get("requests", 0) + 1
        info["bytes"] = info.get("bytes", 0) + int(bytes_uploaded)
        _save(path, info)


def _get_file(user_id: str) -> dict:
    path = _usage_file(user_id)
    with _USAGELock:
        return _load(path)


def increment_usage(user_id: str, bytes_uploaded: int) -> None:
    """Increment usage counters for ``user_id``."""
    table = _get_table()
    if table is not None:
        table.update_item(
            Key={"user_id": user_id},
            UpdateExpression="ADD #r :inc1, #b :inc2",
            ExpressionAttributeNames={"#r": "requests", "#b": "bytes"},
            ExpressionAttributeValues={":inc1": 1, ":inc2": int(bytes_uploaded)},
        )
    else:
        _increment_file(user_id, bytes_uploaded)


def get_usage(user_id: str) -> dict:
    table = _get_table()
    if table is not None:
        res = table.get_item(Key={"user_id": user_id})
        item = res.get("Item")
        if not item:
            return {"requests": 0, "bytes": 0}
        return {
            "requests": int(item.get("requests", 0)),
            "bytes": int(item.get("bytes", 0)),
        }
    else:
        return _get_file(user_id)


# Backwards compatibility
increment_request = increment_usage
usage_info = get_usage


def check_quota(user_id: str) -> tuple[bool, dict | None, int]:
    """Validate usage quotas for ``user_id``.

    Parameters
    ----------
    user_id:
        The user identifier to check usage for.

    Returns
    -------
    (allowed, message, status_code)
        ``allowed`` indicates if the request is permitted.
        ``message`` contains warning or error details.
        ``status_code`` is the HTTP status to return when blocked.
    """

    info = get_usage(user_id)
    max_bytes = MAX_GB_FREE * 1024 * 1024 * 1024
    over_requests = info.get("requests", 0) >= MAX_REQUESTS_FREE
    over_bytes = info.get("bytes", 0) >= max_bytes

    if over_requests or over_bytes:
        if os.getenv("LOCAL_DEV_LENIENT", "1") == "1":
            return True, {
                "warning": "Quota exceeded. Simulated allowance in local dev mode."}, 200
        status = 429 if over_requests else 402
        return False, {"error": "quota_exceeded"}, status
    return True, None, 200

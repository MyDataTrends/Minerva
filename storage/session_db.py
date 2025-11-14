import os
import sqlite3
import json
from datetime import datetime
from typing import Any, Dict, Optional, List

import boto3

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
TABLE_NAME = os.getenv("DYNAMO_SESSIONS_TABLE")


def _table():
    if not TABLE_NAME:
        raise RuntimeError("DYNAMO_SESSIONS_TABLE not configured")
    return boto3.resource("dynamodb", region_name=AWS_REGION).Table(TABLE_NAME)

DB_PATH = os.getenv("SESSIONS_DB_PATH", "sessions.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize local SQLite database when Dynamo is not configured."""
    if TABLE_NAME:
        return
    with _get_conn() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (run_id TEXT PRIMARY KEY, timestamp TEXT, params TEXT, result TEXT)"
        )


def record_session(run_id: str, params: Optional[Dict[str, Any]], result: Optional[Dict[str, Any]]) -> None:
    """Persist session details to Dynamo or local SQLite."""
    if TABLE_NAME:
        _table().update_item(
            Key={"run_id": run_id},
            UpdateExpression="SET #p=:p, #r=:r, #ts=:t",
            ExpressionAttributeNames={"#p": "params", "#r": "result", "#ts": "timestamp"},
            ExpressionAttributeValues={
                ":p": json.dumps(params or {}),
                ":r": json.dumps(result or {}),
                ":t": datetime.utcnow().isoformat(),
            },
        )
        return
    init_db()
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions (run_id, timestamp, params, result) VALUES (?, ?, ?, ?)",
            (run_id, datetime.utcnow().isoformat(), json.dumps(params or {}), json.dumps(result or {})),
        )


def list_sessions(limit: int = 20) -> List[Dict[str, Any]]:
    """Return recent session ids."""
    if TABLE_NAME:
        resp = _table().scan()
        items = resp.get("Items", [])
        items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return [{"run_id": i["run_id"], "timestamp": i.get("timestamp")} for i in items[:limit]]
    init_db()
    with _get_conn() as conn:
        cur = conn.execute(
            "SELECT run_id, timestamp FROM sessions ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]


def get_session(run_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a stored session by id."""
    if TABLE_NAME:
        resp = _table().get_item(Key={"run_id": run_id})
        return resp.get("Item")
    init_db()
    with _get_conn() as conn:
        cur = conn.execute("SELECT run_id, timestamp, params, result FROM sessions WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def save_run_to_dynamo(run_metadata: Dict[str, Any]) -> None:
    """Save run metadata to the Dynamo table."""
    if not TABLE_NAME:
        raise RuntimeError("DYNAMO_SESSIONS_TABLE not configured")
    meta = json.dumps(run_metadata)
    _table().update_item(
        Key={"run_id": run_metadata["run_id"]},
        UpdateExpression="SET run_metadata=:m, #ts=:t",
        ExpressionAttributeNames={"#ts": "timestamp"},
        ExpressionAttributeValues={
            ":m": meta,
            ":t": datetime.utcnow().isoformat(),
        },
    )


def get_run_by_id(run_id: str) -> Optional[Dict[str, Any]]:
    """Fetch run metadata from the Dynamo table."""
    if not TABLE_NAME:
        return None
    resp = _table().get_item(Key={"run_id": run_id})
    item = resp.get("Item")
    if not item:
        return None
    meta = item.get("run_metadata")
    return json.loads(meta) if meta else None

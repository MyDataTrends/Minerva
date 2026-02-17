"""
Operational Memory — Tier 1 persistence for agent state.

Each agent gets its own SQLite database storing actions, escalations,
and run history. This is the real-time layer that gets summarized
into the Tier 2 knowledge base periodically.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logging import get_logger

logger = get_logger(__name__)

STATE_DIR = Path(__file__).parent.parent / "state"


class OperationalMemory:
    """
    SQLite-backed operational memory for a single agent.

    Stores:
    - Action log (what the agent did)
    - Escalation log (items needing human attention)
    - Run history (success/failure, duration, summary)
    """

    def __init__(self, agent_name: str, db_dir: Optional[Path] = None):
        self.agent_name = agent_name
        self.db_dir = db_dir or STATE_DIR
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_dir / f"{agent_name}.db"
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    detail TEXT,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS escalations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    title TEXT NOT NULL,
                    detail TEXT,
                    resolved INTEGER DEFAULT 0,
                    resolved_at TEXT
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    duration_seconds REAL,
                    summary TEXT,
                    actions_count INTEGER,
                    escalations_count INTEGER,
                    error TEXT
                );
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    # ── Actions ──────────────────────────────────────────────────────

    def log_action(self, action: str, detail: str = "", metadata: Optional[Dict] = None) -> None:
        """Record an action taken by the agent."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO actions (timestamp, action, detail, metadata) VALUES (?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), action, detail, json.dumps(metadata or {})),
            )

    def get_recent_actions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get the most recent actions."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT timestamp, action, detail, metadata FROM actions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"timestamp": r[0], "action": r[1], "detail": r[2], "metadata": json.loads(r[3])}
            for r in rows
        ]

    # ── Escalations ──────────────────────────────────────────────────

    def log_escalation(self, priority: str, title: str, detail: str = "") -> None:
        """Record an escalation needing human attention."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO escalations (timestamp, priority, title, detail) VALUES (?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), priority, title, detail),
            )

    def get_pending_escalations(self) -> List[Dict[str, Any]]:
        """Get all unresolved escalations."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, timestamp, priority, title, detail FROM escalations WHERE resolved = 0 ORDER BY id DESC",
            ).fetchall()
        return [
            {"id": r[0], "timestamp": r[1], "priority": r[2], "title": r[3], "detail": r[4]}
            for r in rows
        ]

    def resolve_escalation(self, escalation_id: int) -> None:
        """Mark an escalation as resolved."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE escalations SET resolved = 1, resolved_at = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), escalation_id),
            )

    # ── Run History ──────────────────────────────────────────────────

    def log_run(self, success: bool, duration: float, summary: str,
                actions_count: int = 0, escalations_count: int = 0,
                error: Optional[str] = None) -> None:
        """Record a completed agent run."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO runs
                   (timestamp, success, duration_seconds, summary, actions_count, escalations_count, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (datetime.utcnow().isoformat(), int(success), duration, summary,
                 actions_count, escalations_count, error),
            )

    def get_run_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent run history."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT timestamp, success, duration_seconds, summary, error FROM runs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"timestamp": r[0], "success": bool(r[1]), "duration": r[2], "summary": r[3], "error": r[4]}
            for r in rows
        ]

    def get_daily_summary(self) -> Dict[str, Any]:
        """Get a summary of today's activity."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        with self._connect() as conn:
            actions_count = conn.execute(
                "SELECT COUNT(*) FROM actions WHERE timestamp LIKE ?", (f"{today}%",)
            ).fetchone()[0]
            escalations_count = conn.execute(
                "SELECT COUNT(*) FROM escalations WHERE timestamp LIKE ?", (f"{today}%",)
            ).fetchone()[0]
            runs = conn.execute(
                "SELECT COUNT(*), SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) FROM runs WHERE timestamp LIKE ?",
                (f"{today}%",),
            ).fetchone()

        return {
            "agent": self.agent_name,
            "date": today,
            "actions_today": actions_count,
            "escalations_today": escalations_count,
            "runs_today": runs[0] or 0,
            "successful_runs": runs[1] or 0,
        }

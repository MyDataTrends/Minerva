"""
Interaction Logger - Track all LLM interactions for learning and analytics.

Features:
- Log every prompt → response → outcome
- Track user ratings and corrections
- Sentiment/frustration detection
- Gamification metrics (examples collected, milestones)
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class InteractionType(Enum):
    CHAT = "chat"
    VISUALIZATION = "visualization"
    ANALYSIS = "analysis"
    ACTION = "action"
    SUMMARY = "summary"


class Outcome(Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"
    CORRECTED = "corrected"  # User modified the output
    RETRIED = "retried"  # User asked similar question again


@dataclass
class Interaction:
    """A single LLM interaction."""
    id: Optional[int] = None
    session_id: str = ""
    interaction_type: str = ""
    prompt: str = ""
    response: str = ""
    code_generated: str = ""
    execution_success: Optional[bool] = None
    rating: Optional[float] = None
    outcome: str = "pending"
    retry_count: int = 0
    response_time_ms: int = 0
    dataset_name: str = ""
    created_at: str = ""


# Gamification milestones
MILESTONES = [
    (10, "🌱 Getting Started", "You've provided 10 examples!"),
    (50, "📚 Building Knowledge", "50 examples - your LLM is learning!"),
    (100, "🧠 Growing Smarter", "100 examples collected!"),
    (250, "🚀 Power User", "250 examples - impressive dedication!"),
    (500, "🏆 LLM Master", "500 examples - your model is highly tuned!"),
    (1000, "⭐ Legendary", "1000 examples - maximum learning achieved!"),
]


class InteractionLogger:
    """
    Log and analyze LLM interactions.
    
    Provides:
    - Interaction recording for all LLM calls
    - Rating and correction tracking
    - Gamification metrics and milestones
    - Frustration detection (multiple retries)
    """
    
    _instance = None
    
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path.home() / ".minerva" / "interactions.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    interaction_type TEXT,
                    prompt TEXT NOT NULL,
                    response TEXT,
                    code_generated TEXT,
                    execution_success INTEGER,
                    rating REAL,
                    outcome TEXT DEFAULT 'pending',
                    retry_count INTEGER DEFAULT 0,
                    response_time_ms INTEGER,
                    dataset_name TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS milestones_achieved (
                    milestone_count INTEGER PRIMARY KEY,
                    achieved_at TEXT
                )
            """)
            
            # Session summaries for two-tier retention (memory management)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_summaries (
                    session_id TEXT PRIMARY KEY,
                    interaction_count INTEGER,
                    success_count INTEGER,
                    failure_count INTEGER,
                    avg_response_time_ms REAL,
                    interaction_types TEXT,  -- JSON list
                    datasets_used TEXT,      -- JSON list
                    outcome_distribution TEXT,  -- JSON dict
                    created_at TEXT,
                    summarized_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    is_pinned INTEGER DEFAULT 0  -- Keep raw logs for failures/novel cases
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_interactions_session 
                ON interactions(session_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_interactions_rating 
                ON interactions(rating)
            """)
            
            conn.commit()
    
    def log(
        self,
        prompt: str,
        response: str,
        interaction_type: InteractionType = InteractionType.CHAT,
        code_generated: str = "",
        execution_success: Optional[bool] = None,
        response_time_ms: int = 0,
        dataset_name: str = "",
        session_id: str = ""
    ) -> int:
        """
        Log an interaction.
        
        Returns the interaction ID.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO interactions 
                (session_id, interaction_type, prompt, response, code_generated,
                 execution_success, response_time_ms, dataset_name, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                interaction_type.value if isinstance(interaction_type, InteractionType) else interaction_type,
                prompt,
                response,
                code_generated,
                1 if execution_success else (0 if execution_success is False else None),
                response_time_ms,
                dataset_name,
                datetime.now().isoformat()
            ))
            conn.commit()
            
            interaction_id = cursor.lastrowid
            logger.debug(f"Logged interaction {interaction_id}: {prompt[:50]}...")
            
            return interaction_id
    
    def update_rating(self, interaction_id: int, rating: float):
        """Update the rating for an interaction."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE interactions SET rating = ?, outcome = 'success' WHERE id = ?",
                (rating, interaction_id)
            )
            conn.commit()
            
            # Check for new milestones
            self._check_milestones(conn)
    
    def mark_corrected(self, interaction_id: int):
        """Mark that the user corrected this output."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE interactions SET outcome = 'corrected' WHERE id = ?",
                (interaction_id,)
            )
            conn.commit()
    
    def mark_retry(self, session_id: str, prompt: str):
        """Mark similar prompts in session as retries (frustration signal)."""
        with sqlite3.connect(self.db_path) as conn:
            # Find similar recent prompts in session
            conn.execute("""
                UPDATE interactions 
                SET retry_count = retry_count + 1, outcome = 'retried'
                WHERE session_id = ? AND prompt LIKE ?
                AND created_at > datetime('now', '-5 minutes')
            """, (session_id, f"%{prompt[:50]}%"))
            conn.commit()
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """
        Get gamification metrics for UI display.
        
        Returns progress bar data and milestone info.
        """
        with sqlite3.connect(self.db_path) as conn:
            # Count high-quality interactions (rated or successful)
            quality_count = conn.execute("""
                SELECT COUNT(*) FROM interactions 
                WHERE rating >= 4.0 OR execution_success = 1
            """).fetchone()[0]
            
            total_count = conn.execute(
                "SELECT COUNT(*) FROM interactions"
            ).fetchone()[0]
            
            # Get achieved milestones
            achieved = conn.execute(
                "SELECT milestone_count FROM milestones_achieved"
            ).fetchall()
            achieved_counts = {row[0] for row in achieved}
            
            # Find current and next milestone
            current_milestone = None
            next_milestone = None
            
            for count, name, message in MILESTONES:
                if quality_count >= count:
                    current_milestone = {"count": count, "name": name, "message": message}
                elif next_milestone is None:
                    next_milestone = {"count": count, "name": name, "message": message}
                    break
            
            # Calculate progress to next milestone
            if next_milestone:
                prev_count = current_milestone["count"] if current_milestone else 0
                progress_pct = ((quality_count - prev_count) / (next_milestone["count"] - prev_count)) * 100
            else:
                progress_pct = 100  # All milestones achieved
            
            return {
                "quality_examples": quality_count,
                "total_interactions": total_count,
                "current_milestone": current_milestone,
                "next_milestone": next_milestone,
                "progress_pct": min(100, round(progress_pct, 1)),
                "milestones_achieved": len(achieved_counts),
                "total_milestones": len(MILESTONES)
            }
    
    def _check_milestones(self, conn: sqlite3.Connection):
        """Check and record newly achieved milestones."""
        quality_count = conn.execute("""
            SELECT COUNT(*) FROM interactions 
            WHERE rating >= 4.0 OR execution_success = 1
        """).fetchone()[0]
        
        for count, name, message in MILESTONES:
            if quality_count >= count:
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO milestones_achieved 
                        (milestone_count, achieved_at)
                        VALUES (?, ?)
                    """, (count, datetime.now().isoformat()))
                except:
                    pass
    
    def get_frustration_signals(self) -> List[Dict]:
        """Get interactions with high retry counts (user frustration)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM interactions 
                WHERE retry_count >= 2 OR outcome = 'corrected'
                ORDER BY created_at DESC
                LIMIT 20
            """).fetchall()
            
            return [dict(row) for row in rows]
    
    def get_success_rate(self, days: int = 7) -> Dict[str, float]:
        """Get success rates by interaction type."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT interaction_type,
                       COUNT(*) as total,
                       SUM(CASE WHEN execution_success = 1 THEN 1 ELSE 0 END) as successes
                FROM interactions
                WHERE created_at > datetime('now', ?)
                GROUP BY interaction_type
            """, (f"-{days} days",)).fetchall()
            
            return {
                row[0]: round((row[2] / row[1]) * 100, 1) if row[1] > 0 else 0
                for row in rows
            }

    def summarize_session(self, session_id: str, pin: bool = False) -> Dict[str, Any]:
        """
        Summarize a session's interactions into a compact summary.
        
        Args:
            session_id: The session to summarize
            pin: If True, keep raw logs even after purge (for failures/novel cases)
            
        Returns:
            The summary dict
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM interactions WHERE session_id = ?", (session_id,)
            ).fetchall()
            
            if not rows:
                return {}
            
            interactions = [dict(r) for r in rows]
            
            # Build summary statistics
            summary = {
                "session_id": session_id,
                "interaction_count": len(interactions),
                "success_count": sum(1 for i in interactions if i.get("execution_success")),
                "failure_count": sum(1 for i in interactions if i.get("execution_success") == 0),
                "avg_response_time_ms": sum(i.get("response_time_ms", 0) or 0 for i in interactions) / len(interactions) if interactions else 0,
                "interaction_types": list(set(i.get("interaction_type", "") for i in interactions)),
                "datasets_used": list(set(i.get("dataset_name", "") for i in interactions if i.get("dataset_name"))),
                "outcome_distribution": {},
                "created_at": min(i.get("created_at", "") for i in interactions),
                "is_pinned": 1 if pin else 0,
            }
            
            # Count outcomes
            for i in interactions:
                outcome = i.get("outcome", "pending")
                summary["outcome_distribution"][outcome] = summary["outcome_distribution"].get(outcome, 0) + 1
            
            # Store summary
            conn.execute("""
                INSERT OR REPLACE INTO session_summaries 
                (session_id, interaction_count, success_count, failure_count, avg_response_time_ms,
                 interaction_types, datasets_used, outcome_distribution, created_at, is_pinned)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                summary["session_id"],
                summary["interaction_count"],
                summary["success_count"],
                summary["failure_count"],
                summary["avg_response_time_ms"],
                json.dumps(summary["interaction_types"]),
                json.dumps(summary["datasets_used"]),
                json.dumps(summary["outcome_distribution"]),
                summary["created_at"],
                summary["is_pinned"],
            ))
            conn.commit()
            
            logger.info(f"Summarized session {session_id}: {summary['interaction_count']} interactions")
            return summary
    
    def purge_old_interactions(self, keep_days: int = 7, keep_pinned: bool = True) -> int:
        """
        Two-tier retention: purge raw interactions older than keep_days,
        but preserve summaries and optionally pinned sessions.
        
        Args:
            keep_days: Keep raw interactions from the last N days
            keep_pinned: If True, don't purge interactions from pinned sessions
            
        Returns:
            Number of interactions deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            # First, ensure all old sessions have summaries
            old_sessions = conn.execute("""
                SELECT DISTINCT session_id FROM interactions
                WHERE created_at < datetime('now', ?)
                AND session_id IS NOT NULL AND session_id != ''
            """, (f"-{keep_days} days",)).fetchall()
            
            for (session_id,) in old_sessions:
                # Check if summary exists
                exists = conn.execute(
                    "SELECT 1 FROM session_summaries WHERE session_id = ?", (session_id,)
                ).fetchone()
                if not exists:
                    self.summarize_session(session_id)
            
            # Now delete old interactions, respecting pinned flag
            if keep_pinned:
                cursor = conn.execute("""
                    DELETE FROM interactions 
                    WHERE created_at < datetime('now', ?)
                    AND (session_id IS NULL OR session_id NOT IN (
                        SELECT session_id FROM session_summaries WHERE is_pinned = 1
                    ))
                """, (f"-{keep_days} days",))
            else:
                cursor = conn.execute("""
                    DELETE FROM interactions 
                    WHERE created_at < datetime('now', ?)
                """, (f"-{keep_days} days",))
            
            deleted = cursor.rowcount
            conn.commit()
            
            if deleted > 0:
                logger.info(f"Purged {deleted} old interactions (kept summaries)")
            return deleted
    
    def get_db_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring memory growth."""
        with sqlite3.connect(self.db_path) as conn:
            interactions_count = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            summaries_count = conn.execute("SELECT COUNT(*) FROM session_summaries").fetchone()[0]
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return {
                "interactions_count": interactions_count,
                "summaries_count": summaries_count,
                "db_size_bytes": db_size,
                "db_size_mb": round(db_size / (1024 * 1024), 2),
            }


# Singleton access
_logger_instance: Optional[InteractionLogger] = None


def get_interaction_logger() -> InteractionLogger:
    """Get the global interaction logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = InteractionLogger()
    return _logger_instance

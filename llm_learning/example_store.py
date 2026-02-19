"""
Example Store - SQLite-based storage for successful LLM interactions.

Features:
- Store prompt/code/result triplets with ratings
- Configurable storage limits with auto-pruning
- Vector embeddings for similarity search (optional)
- Action outcome tracking for weighted retrieval
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)

# Default storage limits
DEFAULT_MAX_EXAMPLES = 10000
DEFAULT_MAX_SIZE_MB = 500


@dataclass
class Example:
    """A stored example of a successful LLM interaction."""
    id: Optional[int] = None
    prompt: str = ""
    code: str = ""
    result: str = ""
    rating: float = 3.0
    intent: str = ""
    action_id: Optional[str] = None
    action_success: Optional[bool] = None
    dataset_context: str = ""
    created_at: str = ""
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d.pop('embedding', None)  # Don't include embedding in dict
        return d


class ExampleStore:
    """
    SQLite-based store for LLM interaction examples.
    
    Used for:
    - RAG: Retrieve similar past successes as few-shot examples
    - Fine-tuning: Export high-quality examples for LoRA training
    - Analytics: Track what prompts work well
    """
    
    _instance = None
    
    def __init__(
        self, 
        db_path: Optional[Path] = None,
        max_examples: int = DEFAULT_MAX_EXAMPLES,
        max_size_mb: int = DEFAULT_MAX_SIZE_MB
    ):
        if db_path is None:
            # Default: ~/.assay/llm_learning.db
            db_path = Path.home() / ".assay" / "llm_learning.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_examples = max_examples
        self.max_size_mb = max_size_mb
        
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT NOT NULL,
                    code TEXT,
                    result TEXT,
                    rating REAL DEFAULT 3.0,
                    intent TEXT,
                    action_id TEXT,
                    action_success INTEGER,
                    dataset_context TEXT,
                    prompt_hash TEXT UNIQUE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    embedding BLOB
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_examples_rating 
                ON examples(rating DESC)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_examples_intent 
                ON examples(intent)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            conn.commit()
    
    def _hash_prompt(self, prompt: str) -> str:
        """Create hash for deduplication."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:32]
    
    def add(self, example: Example) -> int:
        """
        Add an example to the store.
        
        Returns the example ID, or -1 if duplicate.
        """
        prompt_hash = self._hash_prompt(example.prompt)
        
        with sqlite3.connect(self.db_path) as conn:
            try:
                cursor = conn.execute("""
                    INSERT INTO examples 
                    (prompt, code, result, rating, intent, action_id, 
                     action_success, dataset_context, prompt_hash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    example.prompt,
                    example.code,
                    example.result,
                    example.rating,
                    example.intent,
                    example.action_id,
                    1 if example.action_success else (0 if example.action_success is False else None),
                    example.dataset_context,
                    prompt_hash,
                    example.created_at or datetime.now().isoformat()
                ))
                conn.commit()
                
                example_id = cursor.lastrowid
                logger.debug(f"Added example {example_id}: {example.prompt[:50]}...")
                
                # Check limits and prune if needed
                self._enforce_limits(conn)
                
                return example_id
                
            except sqlite3.IntegrityError:
                # Duplicate prompt
                logger.debug(f"Duplicate prompt, skipping: {example.prompt[:50]}...")
                return -1
    
    def update_rating(self, example_id: int, rating: float):
        """Update the rating for an example."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE examples SET rating = ? WHERE id = ?",
                (rating, example_id)
            )
            conn.commit()
    
    def update_action_outcome(self, action_id: str, success: bool):
        """Update action success for examples with this action_id."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE examples SET action_success = ? WHERE action_id = ?",
                (1 if success else 0, action_id)
            )
            conn.commit()
            logger.info(f"Updated action {action_id} success={success}")
    
    def get_by_intent(self, intent: str, limit: int = 10) -> List[Example]:
        """Get top examples for an intent type."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM examples 
                WHERE intent = ? AND rating >= 4.0
                ORDER BY 
                    CASE WHEN action_success = 1 THEN 0 ELSE 1 END,
                    rating DESC,
                    created_at DESC
                LIMIT ?
            """, (intent, limit)).fetchall()
            
            return [self._row_to_example(row) for row in rows]
    
    def get_top_examples(self, limit: int = 100) -> List[Example]:
        """Get the highest-rated examples."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM examples 
                WHERE rating >= 4.0
                ORDER BY 
                    CASE WHEN action_success = 1 THEN 0 ELSE 1 END,
                    rating DESC
                LIMIT ?
            """, (limit,)).fetchall()
            
            return [self._row_to_example(row) for row in rows]
    
    def search_similar(self, prompt: str, limit: int = 5) -> List[Example]:
        """
        Search for similar prompts (basic text matching).
        
        For proper semantic search, use embeddings (see embeddings.py).
        """
        # Basic keyword matching for now
        keywords = prompt.lower().split()[:5]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Build LIKE conditions
            conditions = " OR ".join(["LOWER(prompt) LIKE ?" for _ in keywords])
            params = [f"%{kw}%" for kw in keywords]
            params.append(limit)
            
            rows = conn.execute(f"""
                SELECT * FROM examples 
                WHERE rating >= 3.5 AND ({conditions})
                ORDER BY rating DESC
                LIMIT ?
            """, params).fetchall()
            
            return [self._row_to_example(row) for row in rows]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM examples").fetchone()[0]
            high_rated = conn.execute(
                "SELECT COUNT(*) FROM examples WHERE rating >= 4.0"
            ).fetchone()[0]
            with_actions = conn.execute(
                "SELECT COUNT(*) FROM examples WHERE action_success = 1"
            ).fetchone()[0]
            
            # Get DB size
            size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
            size_mb = size_bytes / (1024 * 1024)
            
            return {
                "total_examples": total,
                "high_rated": high_rated,
                "successful_actions": with_actions,
                "size_mb": round(size_mb, 2),
                "max_examples": self.max_examples,
                "max_size_mb": self.max_size_mb,
                "capacity_pct": round((total / self.max_examples) * 100, 1)
            }
    
    def _enforce_limits(self, conn: sqlite3.Connection):
        """Prune old/low-rated examples if limits exceeded."""
        stats = self.get_stats()
        
        if stats["total_examples"] > self.max_examples:
            # Delete lowest rated examples
            excess = stats["total_examples"] - self.max_examples
            conn.execute("""
                DELETE FROM examples WHERE id IN (
                    SELECT id FROM examples 
                    ORDER BY rating ASC, created_at ASC
                    LIMIT ?
                )
            """, (excess,))
            conn.commit()
            logger.info(f"Pruned {excess} low-rated examples")
    
    def _row_to_example(self, row: sqlite3.Row) -> Example:
        """Convert database row to Example."""
        return Example(
            id=row["id"],
            prompt=row["prompt"],
            code=row["code"],
            result=row["result"],
            rating=row["rating"],
            intent=row["intent"],
            action_id=row["action_id"],
            action_success=bool(row["action_success"]) if row["action_success"] is not None else None,
            dataset_context=row["dataset_context"],
            created_at=row["created_at"]
        )
    
    def export_for_training(self, min_rating: float = 4.0) -> List[Dict]:
        """Export high-quality examples for fine-tuning."""
        examples = []
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM examples 
                WHERE rating >= ?
                ORDER BY rating DESC
            """, (min_rating,)).fetchall()
            
            for row in rows:
                examples.append({
                    "instruction": row["prompt"],
                    "input": row["dataset_context"] or "",
                    "output": row["code"]
                })
        
        return examples


# Singleton access
_store_instance: Optional[ExampleStore] = None


def get_example_store() -> ExampleStore:
    """Get the global example store instance."""
    global _store_instance
    if _store_instance is None:
        _store_instance = ExampleStore()
    return _store_instance

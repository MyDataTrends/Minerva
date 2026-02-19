"""
Session Persistence Module for Assay.

Provides state checkpointing and restoration for cross-session continuity:
- Session snapshots (save current state for later restoration)
- Structured summary schema (intent, tool sequence, outcome, deltas)
- Weighted memory retrieval (blend long-term summaries with short-term logs)
- Optional state encryption for sensitive raw traces

Usage:
    from ui.session_persistence import get_persistence, SessionSnapshot
    
    # Save a snapshot
    persistence = get_persistence()
    snapshot_id = persistence.save_snapshot("My analysis session")
    
    # List available snapshots
    snapshots = persistence.list_snapshots()
    
    # Restore a snapshot
    persistence.restore_snapshot(snapshot_id)
"""
import json
import hashlib
import base64
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
import streamlit as st

logger = logging.getLogger(__name__)

# Import diagnostics for logging
try:
    from ui.diagnostics import capture_error, get_diagnostics
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False
    from contextlib import nullcontext as capture_error


@dataclass
class StructuredSummary:
    """
    Structured summary schema for learned interactions.
    
    Stores actionable information rather than prose:
    - Intent label (what user wanted)
    - Tool sequence (what operations were performed)
    - Outcome (success/failure + details)
    - Learned deltas (what changed in knowledge)
    """
    intent_label: str  # e.g., "create_visualization", "data_cleaning", "model_training"
    intent_description: str  # Brief human-readable description
    tool_sequence: List[str]  # e.g., ["load_csv", "filter_data", "create_chart"]
    outcome: str  # "success", "partial_success", "failure"
    outcome_details: str  # What happened
    learned_deltas: Dict[str, Any] = field(default_factory=dict)  # Knowledge changes
    dataset_context: Optional[str] = None  # Dataset used
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StructuredSummary":
        return cls(**d)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class SessionSnapshot:
    """
    A checkpoint of session state for later restoration.
    """
    snapshot_id: str
    name: str
    description: str
    created_at: str
    
    # Captured state
    chat_history: List[Dict[str, Any]]
    action_log: List[Dict[str, Any]]
    discoveries: List[Dict[str, Any]]
    preferences: Dict[str, Any]
    
    # Metadata
    dataset_names: List[str]  # We don't store actual data, just names
    interaction_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionSnapshot":
        return cls(**d)


class SessionPersistence:
    """
    Manages session state persistence with SQLite backend.
    
    Features:
    - Save/restore session snapshots
    - Structured summary storage
    - Weighted memory retrieval for RAG
    - Optional encryption for raw traces
    """
    
    DB_PATH = Path("local_data/session_persistence.db")
    
    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize persistence manager.
        
        Args:
            encryption_key: If provided, enables encryption for raw traces
        """
        self.encryption_key = encryption_key
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.DB_PATH) as conn:
            # Snapshots table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    is_encrypted INTEGER DEFAULT 0
                )
            """)
            
            # Structured summaries table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS structured_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    intent_label TEXT NOT NULL,
                    intent_description TEXT,
                    tool_sequence TEXT,
                    outcome TEXT NOT NULL,
                    outcome_details TEXT,
                    learned_deltas TEXT,
                    dataset_context TEXT,
                    created_at TEXT NOT NULL,
                    weight REAL DEFAULT 1.0
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_summaries_intent ON structured_summaries(intent_label)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_summaries_outcome ON structured_summaries(outcome)")
            conn.commit()
    
    # =========================================================================
    # Session Snapshots
    # =========================================================================
    
    def save_snapshot(
        self,
        name: str,
        description: str = "",
    ) -> str:
        """
        Save current session state as a snapshot.
        
        Args:
            name: User-friendly name for the snapshot
            description: Optional description
            
        Returns:
            snapshot_id for later restoration
        """
        if DIAGNOSTICS_AVAILABLE:
            diag = get_diagnostics()
            diag.log_info("Saving snapshot", f"Creating snapshot: {name}")
        
        # Generate unique ID
        snapshot_id = hashlib.md5(
            f"{name}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Capture current state from session_context
        try:
            from ui.session_context import get_context, KEYS
            ctx = get_context()
            
            snapshot = SessionSnapshot(
                snapshot_id=snapshot_id,
                name=name,
                description=description,
                created_at=datetime.now().isoformat(),
                chat_history=list(ctx.chat_history),
                action_log=list(ctx.action_log),
                discoveries=list(ctx.discoveries),
                preferences=dict(ctx.preferences),
                dataset_names=list(ctx.datasets.keys()),
                interaction_count=len(ctx.chat_history),
            )
            
            # Serialize state
            state_json = json.dumps(snapshot.to_dict())
            
            # Optionally encrypt
            is_encrypted = 0
            if self.encryption_key:
                state_json = self._encrypt(state_json)
                is_encrypted = 1
            
            # Store
            with sqlite3.connect(self.DB_PATH) as conn:
                conn.execute("""
                    INSERT INTO snapshots (snapshot_id, name, description, created_at, state_json, is_encrypted)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (snapshot_id, name, description, snapshot.created_at, state_json, is_encrypted))
                conn.commit()
            
            logger.info(f"Saved snapshot '{name}' with ID {snapshot_id}")
            return snapshot_id
            
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            if DIAGNOSTICS_AVAILABLE:
                get_diagnostics().log_error("Saving snapshot", e, context={"name": name})
            raise
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots."""
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT snapshot_id, name, description, created_at
                FROM snapshots
                ORDER BY created_at DESC
            """).fetchall()
            return [dict(row) for row in rows]
    
    def restore_snapshot(self, snapshot_id: str) -> bool:
        """
        Restore session state from a snapshot.
        
        Args:
            snapshot_id: ID of snapshot to restore
            
        Returns:
            True if successful
        """
        if DIAGNOSTICS_AVAILABLE:
            diag = get_diagnostics()
            diag.log_info("Restoring snapshot", f"Restoring snapshot: {snapshot_id}")
        
        try:
            with sqlite3.connect(self.DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
                ).fetchone()
                
                if not row:
                    logger.warning(f"Snapshot {snapshot_id} not found")
                    return False
                
                state_json = row["state_json"]
                
                # Decrypt if needed
                if row["is_encrypted"]:
                    if not self.encryption_key:
                        raise ValueError("Snapshot is encrypted but no encryption key provided")
                    state_json = self._decrypt(state_json)
                
                snapshot_data = json.loads(state_json)
                snapshot = SessionSnapshot.from_dict(snapshot_data)
                
                # Restore to session context
                from ui.session_context import get_context, KEYS
                ctx = get_context()
                
                st.session_state[KEYS["chat_history"]] = snapshot.chat_history
                st.session_state[KEYS["action_log"]] = snapshot.action_log
                st.session_state[KEYS["discoveries"]] = snapshot.discoveries
                st.session_state[KEYS["user_preferences"]] = snapshot.preferences
                
                logger.info(f"Restored snapshot '{snapshot.name}'")
                return True
                
        except Exception as e:
            logger.error(f"Failed to restore snapshot: {e}")
            if DIAGNOSTICS_AVAILABLE:
                get_diagnostics().log_error("Restoring snapshot", e, context={"snapshot_id": snapshot_id})
            return False
    
    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot."""
        with sqlite3.connect(self.DB_PATH) as conn:
            cursor = conn.execute("DELETE FROM snapshots WHERE snapshot_id = ?", (snapshot_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    # =========================================================================
    # Structured Summaries
    # =========================================================================
    
    def save_structured_summary(
        self,
        summary: StructuredSummary,
        session_id: Optional[str] = None,
        weight: float = 1.0,
    ) -> int:
        """
        Save a structured summary for later retrieval.
        
        Args:
            summary: The structured summary to save
            session_id: Optional session identifier
            weight: Importance weight for retrieval (default 1.0)
            
        Returns:
            Row ID of inserted summary
        """
        with sqlite3.connect(self.DB_PATH) as conn:
            cursor = conn.execute("""
                INSERT INTO structured_summaries 
                (session_id, intent_label, intent_description, tool_sequence, outcome,
                 outcome_details, learned_deltas, dataset_context, created_at, weight)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                summary.intent_label,
                summary.intent_description,
                json.dumps(summary.tool_sequence),
                summary.outcome,
                summary.outcome_details,
                json.dumps(summary.learned_deltas),
                summary.dataset_context,
                summary.timestamp,
                weight,
            ))
            conn.commit()
            logger.debug(f"Saved structured summary: {summary.intent_label}")
            return cursor.lastrowid
    
    def get_weighted_memory(
        self,
        intent_filter: Optional[str] = None,
        limit: int = 20,
        recency_weight: float = 0.7,
        success_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve summaries with weighted scoring for RAG context.
        
        Blends:
        - Recency (more recent = higher weight)
        - Outcome success (success = higher weight)
        - Explicit weights stored per summary
        
        Args:
            intent_filter: Optional filter by intent label
            limit: Max number of summaries to return
            recency_weight: Weight for recency factor (0-1)
            success_weight: Weight for success factor (0-1)
            
        Returns:
            List of summaries with computed scores
        """
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            
            query = """
                SELECT *,
                    julianday('now') - julianday(created_at) as days_ago
                FROM structured_summaries
            """
            params = []
            
            if intent_filter:
                query += " WHERE intent_label LIKE ?"
                params.append(f"%{intent_filter}%")
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit * 2)  # Fetch more for scoring
            
            rows = conn.execute(query, params).fetchall()
            
            # Score and rank
            results = []
            for row in rows:
                data = dict(row)
                
                # Parse JSON fields
                data["tool_sequence"] = json.loads(data["tool_sequence"] or "[]")
                data["learned_deltas"] = json.loads(data["learned_deltas"] or "{}")
                
                # Compute weighted score
                days_ago = data.get("days_ago", 0) or 0
                recency_score = max(0, 1 - (days_ago / 30))  # Decay over 30 days
                
                success_score = 1.0 if data["outcome"] == "success" else (
                    0.5 if data["outcome"] == "partial_success" else 0.1
                )
                
                base_weight = data.get("weight", 1.0)
                
                # Blend scores
                final_score = (
                    recency_weight * recency_score +
                    success_weight * success_score
                ) * base_weight
                
                data["computed_score"] = round(final_score, 3)
                results.append(data)
            
            # Sort by score and limit
            results.sort(key=lambda x: x["computed_score"], reverse=True)
            return results[:limit]
    
    def search_by_intent(self, intent: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search summaries by intent label."""
        return self.get_weighted_memory(intent_filter=intent, limit=limit)
    
    # =========================================================================
    # Encryption (Simple XOR-based for local use)
    # =========================================================================
    
    def _encrypt(self, plaintext: str) -> str:
        """Simple encryption for local storage (not for high-security use)."""
        if not self.encryption_key:
            return plaintext
        
        key_bytes = self.encryption_key.encode()
        data_bytes = plaintext.encode()
        
        encrypted = bytes([
            data_bytes[i] ^ key_bytes[i % len(key_bytes)]
            for i in range(len(data_bytes))
        ])
        
        return base64.b64encode(encrypted).decode()
    
    def _decrypt(self, ciphertext: str) -> str:
        """Decrypt data encrypted with _encrypt."""
        if not self.encryption_key:
            return ciphertext
        
        key_bytes = self.encryption_key.encode()
        encrypted = base64.b64decode(ciphertext.encode())
        
        decrypted = bytes([
            encrypted[i] ^ key_bytes[i % len(key_bytes)]
            for i in range(len(encrypted))
        ])
        
        return decrypted.decode()
    
    # =========================================================================
    # Stats and Cleanup
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        with sqlite3.connect(self.DB_PATH) as conn:
            snapshots = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
            summaries = conn.execute("SELECT COUNT(*) FROM structured_summaries").fetchone()[0]
            
            success_rate = conn.execute("""
                SELECT 
                    COUNT(CASE WHEN outcome = 'success' THEN 1 END) * 100.0 / COUNT(*)
                FROM structured_summaries
                WHERE outcome IS NOT NULL
            """).fetchone()[0] or 0
            
            return {
                "snapshots_count": snapshots,
                "summaries_count": summaries,
                "success_rate": round(success_rate, 1),
                "db_size_bytes": self.DB_PATH.stat().st_size if self.DB_PATH.exists() else 0,
            }


# =============================================================================
# Global Access
# =============================================================================

_persistence: Optional[SessionPersistence] = None


def get_persistence(encryption_key: Optional[str] = None) -> SessionPersistence:
    """Get the global session persistence instance."""
    global _persistence
    if _persistence is None:
        _persistence = SessionPersistence(encryption_key=encryption_key)
    return _persistence


# =============================================================================
# UI Components
# =============================================================================

def render_snapshot_manager():
    """Render a UI for managing session snapshots."""
    st.subheader("📸 Session Snapshots")
    
    persistence = get_persistence()
    
    # Save new snapshot
    with st.expander("Save Current Session", expanded=False):
        name = st.text_input("Snapshot Name", placeholder="My analysis session")
        description = st.text_area("Description (optional)", placeholder="What this session contains...")
        
        if st.button("💾 Save Snapshot", disabled=not name):
            try:
                snapshot_id = persistence.save_snapshot(name, description)
                st.success(f"Saved snapshot: {snapshot_id}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save: {e}")
    
    # List existing snapshots
    snapshots = persistence.list_snapshots()
    if snapshots:
        st.write(f"**{len(snapshots)} snapshots available:**")
        
        for snap in snapshots:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{snap['name']}** ({snap['snapshot_id']})")
                if snap.get('description'):
                    st.caption(snap['description'])
                st.caption(f"Created: {snap['created_at'][:16]}")
            with col2:
                if st.button("🔄 Restore", key=f"restore_{snap['snapshot_id']}"):
                    if persistence.restore_snapshot(snap['snapshot_id']):
                        st.success("Restored!")
                        st.rerun()
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{snap['snapshot_id']}"):
                    persistence.delete_snapshot(snap['snapshot_id'])
                    st.rerun()
    else:
        st.info("No snapshots saved yet. Save a snapshot to preserve your session state.")
    
    # Stats
    with st.expander("📊 Persistence Stats"):
        stats = persistence.get_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Snapshots", stats["snapshots_count"])
            st.metric("Summaries", stats["summaries_count"])
        with col2:
            st.metric("Success Rate", f"{stats['success_rate']}%")
            st.metric("DB Size", f"{stats['db_size_bytes'] / 1024:.1f} KB")

"""
Run Artifacts for Cascade Planner.

Provides deterministic storage of execution runs:
- Save inputs, schema, code snippets, outputs per run
- Replay failed runs for debugging
- Export runs for analysis

Usage:
    from orchestration.run_artifacts import ArtifactStore
    
    store = ArtifactStore()
    store.save(plan, result, context)
    artifact = store.load(run_id)
    replay_result = store.replay(run_id)
"""
import json
import logging
import hashlib
import pickle
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Default storage location
_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "local_data" / "run_artifacts"


@dataclass
class DataSnapshot:
    """Snapshot of input data for reproducibility."""
    shape: tuple
    columns: List[str]
    dtypes: Dict[str, str]
    sample_hash: str  # Hash of first 100 rows for validation
    sample_data: Optional[List[Dict]] = None  # First 5 rows for debugging
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, include_sample: bool = True) -> "DataSnapshot":
        """Create snapshot from DataFrame."""
        sample_hash = hashlib.md5(
            df.head(100).to_json().encode()
        ).hexdigest()
        
        sample_data = None
        if include_sample:
            sample_data = df.head(5).to_dict(orient="records")
        
        return cls(
            shape=df.shape,
            columns=list(df.columns),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            sample_hash=sample_hash,
            sample_data=sample_data,
        )


@dataclass
class RunArtifact:
    """Complete artifact for a single execution run."""
    run_id: str
    timestamp: str
    query: str
    intent: str
    
    # Input snapshot
    data_snapshot: Optional[DataSnapshot] = None
    context_keys: List[str] = field(default_factory=list)
    
    # Execution details
    plan_json: Dict[str, Any] = field(default_factory=dict)
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Output
    success: bool = False
    output_summary: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.data_snapshot:
            d["data_snapshot"] = asdict(self.data_snapshot)
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunArtifact":
        """Reconstruct from dictionary."""
        snapshot_data = d.pop("data_snapshot", None)
        snapshot = None
        if snapshot_data:
            snapshot = DataSnapshot(**snapshot_data)
        
        return cls(data_snapshot=snapshot, **d)


class ArtifactStore:
    """
    Persistent storage for execution artifacts.
    
    Enables:
    - Saving complete execution context
    - Loading and replaying failed runs
    - Analyzing execution patterns
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or _ARTIFACT_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.storage_dir / "index.json"
        self._index = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict]:
        """Load artifact index."""
        if self._index_file.exists():
            try:
                with open(self._index_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load artifact index: {e}")
        return {}
    
    def _save_index(self):
        """Save artifact index."""
        try:
            with open(self._index_file, "w") as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save artifact index: {e}")
    
    def save(
        self,
        plan: "ExecutionPlan",
        result: "ExecutionResult",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a complete execution artifact.
        
        Args:
            plan: The executed plan
            result: The execution result
            context: Original execution context
            
        Returns:
            The run_id
        """
        from orchestration.cascade_planner import ExecutionPlan, ExecutionResult
        
        context = context or {}
        run_id = plan.plan_id
        
        # Create data snapshot
        data_snapshot = None
        if "df" in context and isinstance(context["df"], pd.DataFrame):
            data_snapshot = DataSnapshot.from_dataframe(context["df"])
        
        # Create artifact
        artifact = RunArtifact(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            query=plan.query,
            intent=plan.intent.value,
            data_snapshot=data_snapshot,
            context_keys=list(context.keys()),
            plan_json=plan.to_dict(),
            step_results=[s.to_dict() for s in plan.steps],
            success=result.success,
            output_summary=str(result.output)[:500] if result.output else None,
            error=result.error,
            execution_time_ms=result.execution_time_ms,
            metadata={
                "steps_completed": result.steps_completed,
                "total_steps": result.total_steps,
            },
        )
        
        # Save to file
        artifact_file = self.storage_dir / f"{run_id}.json"
        try:
            with open(artifact_file, "w") as f:
                json.dump(artifact.to_dict(), f, indent=2, default=str)
            
            # Update index
            self._index[run_id] = {
                "timestamp": artifact.timestamp,
                "query": artifact.query[:100],
                "intent": artifact.intent,
                "success": artifact.success,
                "file": artifact_file.name,
            }
            self._save_index()
            
            logger.info(f"Saved artifact: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to save artifact: {e}")
            raise
    
    def load(self, run_id: str) -> Optional[RunArtifact]:
        """Load an artifact by run_id."""
        if run_id not in self._index:
            logger.warning(f"Artifact not found: {run_id}")
            return None
        
        artifact_file = self.storage_dir / self._index[run_id]["file"]
        if not artifact_file.exists():
            logger.warning(f"Artifact file missing: {artifact_file}")
            return None
        
        try:
            with open(artifact_file, "r") as f:
                data = json.load(f)
            return RunArtifact.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load artifact: {e}")
            return None
    
    def list_recent(self, limit: int = 10, failed_only: bool = False) -> List[Dict[str, Any]]:
        """List recent artifacts."""
        artifacts = list(self._index.values())
        
        if failed_only:
            artifacts = [a for a in artifacts if not a.get("success")]
        
        # Sort by timestamp descending
        artifacts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return artifacts[:limit]
    
    def delete(self, run_id: str) -> bool:
        """Delete an artifact."""
        if run_id not in self._index:
            return False
        
        artifact_file = self.storage_dir / self._index[run_id]["file"]
        try:
            if artifact_file.exists():
                artifact_file.unlink()
            del self._index[run_id]
            self._save_index()
            logger.info(f"Deleted artifact: {run_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete artifact: {e}")
            return False
    
    def replay(
        self,
        run_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional["ExecutionResult"]:
        """
        Replay a saved execution.
        
        Args:
            run_id: The run to replay
            context: Override context (use original if not provided)
            
        Returns:
            New ExecutionResult
        """
        from orchestration.cascade_planner import (
            CascadePlanner,
            ExecutionPlan,
            PlanStep,
            Intent,
            PlanStatus,
        )
        
        artifact = self.load(run_id)
        if not artifact:
            return None
        
        # Reconstruct plan
        plan_data = artifact.plan_json
        steps = [
            PlanStep(
                step_id=s["step_id"],
                action=s["action"],
                tool=s["tool"],
                inputs=s["inputs"],
                expected_output=s["expected_output"],
                fallback_tool=s.get("fallback_tool"),
                depends_on=s.get("depends_on", []),
            )
            for s in plan_data.get("steps", [])
        ]
        
        plan = ExecutionPlan(
            plan_id=f"{run_id}_replay_{datetime.now().strftime('%H%M%S')}",
            intent=Intent(plan_data.get("intent", "unknown")),
            query=plan_data.get("query", ""),
            steps=steps,
            metadata={"replayed_from": run_id},
        )
        
        # Execute with new or original context
        planner = CascadePlanner()
        result = planner.execute(plan, context=context or {})
        
        # Save the replay as new artifact
        self.save(plan, result, context or {})
        
        logger.info(f"Replayed {run_id} -> {plan.plan_id}")
        return result
    
    def get_failed_runs(self, limit: int = 5) -> List[RunArtifact]:
        """Get the most recent failed runs."""
        failed = self.list_recent(limit=limit * 2, failed_only=True)
        artifacts = []
        for info in failed[:limit]:
            run_id = info.get("file", "").replace(".json", "")
            if run_id:
                artifact = self.load(run_id)
                if artifact:
                    artifacts.append(artifact)
        return artifacts
    
    def cleanup_old(self, days: int = 7) -> int:
        """Remove artifacts older than specified days."""
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        removed = 0
        
        for run_id, info in list(self._index.items()):
            try:
                ts = datetime.fromisoformat(info.get("timestamp", ""))
                if ts < cutoff:
                    self.delete(run_id)
                    removed += 1
            except Exception:
                continue
        
        logger.info(f"Cleaned up {removed} old artifacts")
        return removed


# =============================================================================
# Global Access
# =============================================================================

_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    """Get the global artifact store instance."""
    global _store
    if _store is None:
        _store = ArtifactStore()
    return _store

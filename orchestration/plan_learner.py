"""
Active Learning System for Cascade Planner.

Captures successful executions and converts patterns into:
1. Learned intent patterns (from successful LLM fallbacks)
2. Tool success weights (for routing optimization)
3. Column/operation mappings (for auto-filling placeholders)

Usage:
    from orchestration.plan_learner import get_learner
    
    learner = get_learner()
    learner.learn_from_execution(plan, result, context)
    
    # Get learned patterns for future planning
    patterns = learner.get_learned_patterns(intent)
    weights = learner.get_tool_weights()
"""
import json
import logging
import sqlite3
import hashlib
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# Storage location
_LEARNING_DB = Path(__file__).resolve().parents[1] / "local_data" / "cascade_learning.db"


@dataclass
class LearnedPattern:
    """A pattern learned from successful execution."""
    pattern_id: str
    intent: str
    query_pattern: str  # Regex pattern derived from query
    tool_sequence: List[str]  # Tools used in order
    input_mappings: Dict[str, str]  # Column/value mappings
    success_count: int = 1
    fail_count: int = 0
    confidence: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolWeight:
    """Weight/score for a tool based on historical performance."""
    tool_name: str
    intent: str
    total_uses: int = 0
    successes: int = 0
    failures: int = 0
    avg_execution_time_ms: int = 0
    
    @property
    def success_rate(self) -> float:
        return self.successes / self.total_uses if self.total_uses > 0 else 0.5
    
    @property
    def weight(self) -> float:
        """Higher is better. Combines success rate with usage confidence."""
        # Bayesian-ish weighting: prior of 0.5, grows with observations
        prior_weight = 2  # Equivalent to 2 prior observations
        adj_successes = self.successes + prior_weight * 0.5
        adj_total = self.total_uses + prior_weight
        return adj_successes / adj_total


class PlanLearner:
    """
    Active learning system for the cascade planner.
    
    Learns from:
    1. Successful tool executions → increases tool weights
    2. Failed executions → decreases weights, records patterns to avoid
    3. LLM fallback successes → extracts patterns for deterministic rules
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _LEARNING_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
        # In-memory caches
        self._pattern_cache: Dict[str, List[LearnedPattern]] = {}
        self._weight_cache: Dict[str, ToolWeight] = {}
        self._cache_loaded = False
    
    def _init_db(self):
        """Initialize learning database."""
        with sqlite3.connect(self.db_path) as conn:
            # Learned patterns table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    intent TEXT NOT NULL,
                    query_pattern TEXT NOT NULL,
                    tool_sequence TEXT NOT NULL,
                    input_mappings TEXT,
                    success_count INTEGER DEFAULT 1,
                    fail_count INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT,
                    last_used TEXT
                )
            """)
            
            # Tool weights table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_weights (
                    tool_name TEXT,
                    intent TEXT,
                    total_uses INTEGER DEFAULT 0,
                    successes INTEGER DEFAULT 0,
                    failures INTEGER DEFAULT 0,
                    avg_execution_time_ms INTEGER DEFAULT 0,
                    PRIMARY KEY (tool_name, intent)
                )
            """)
            
            # Intent pattern aliases (new patterns discovered from queries)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS intent_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    intent TEXT NOT NULL,
                    regex_pattern TEXT NOT NULL,
                    specificity REAL DEFAULT 0.5,
                    hit_count INTEGER DEFAULT 1,
                    created_at TEXT
                )
            """)
            
            conn.commit()
    
    def _load_cache(self):
        """Load patterns and weights into memory."""
        if self._cache_loaded:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Load patterns
                cursor = conn.execute("SELECT * FROM learned_patterns")
                for row in cursor:
                    pattern = LearnedPattern(
                        pattern_id=row["pattern_id"],
                        intent=row["intent"],
                        query_pattern=row["query_pattern"],
                        tool_sequence=json.loads(row["tool_sequence"]),
                        input_mappings=json.loads(row["input_mappings"] or "{}"),
                        success_count=row["success_count"],
                        fail_count=row["fail_count"],
                        confidence=row["confidence"],
                        created_at=row["created_at"],
                        last_used=row["last_used"],
                    )
                    if pattern.intent not in self._pattern_cache:
                        self._pattern_cache[pattern.intent] = []
                    self._pattern_cache[pattern.intent].append(pattern)
                
                # Load weights
                cursor = conn.execute("SELECT * FROM tool_weights")
                for row in cursor:
                    key = f"{row['tool_name']}:{row['intent']}"
                    self._weight_cache[key] = ToolWeight(
                        tool_name=row["tool_name"],
                        intent=row["intent"],
                        total_uses=row["total_uses"],
                        successes=row["successes"],
                        failures=row["failures"],
                        avg_execution_time_ms=row["avg_execution_time_ms"],
                    )
            
            self._cache_loaded = True
            logger.debug(f"Loaded {len(self._pattern_cache)} intent patterns and {len(self._weight_cache)} tool weights")
            
        except Exception as e:
            logger.warning(f"Failed to load learning cache: {e}")
    
    def learn_from_execution(
        self,
        plan: "ExecutionPlan",
        result: "ExecutionResult",
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Learn from a completed execution.
        
        Updates:
        1. Tool weights based on success/failure
        2. Learned patterns if this was a novel successful execution
        3. Intent patterns if query matched a new pattern
        """
        self._load_cache()
        context = context or {}
        
        # 1. Update tool weights
        for step in plan.steps:
            self._update_tool_weight(
                tool_name=step.tool,
                intent=plan.intent.value,
                success=step.status.value == "success",
                execution_time_ms=step.execution_time_ms,
            )
        
        # 2. Learn pattern from successful execution
        if result.success:
            self._learn_pattern(plan, context)
        
        # 3. Learn new intent pattern from query
        self._learn_intent_pattern(plan.query, plan.intent.value)
        
        logger.debug(f"Learned from execution: {plan.plan_id} (success: {result.success})")
    
    def _update_tool_weight(
        self,
        tool_name: str,
        intent: str,
        success: bool,
        execution_time_ms: int,
    ):
        """Update tool weight based on execution outcome."""
        key = f"{tool_name}:{intent}"
        
        if key not in self._weight_cache:
            self._weight_cache[key] = ToolWeight(tool_name=tool_name, intent=intent)
        
        weight = self._weight_cache[key]
        weight.total_uses += 1
        
        if success:
            weight.successes += 1
        else:
            weight.failures += 1
        
        # Rolling average of execution time
        if weight.avg_execution_time_ms == 0:
            weight.avg_execution_time_ms = execution_time_ms
        else:
            weight.avg_execution_time_ms = int(
                0.8 * weight.avg_execution_time_ms + 0.2 * execution_time_ms
            )
        
        # Persist
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO tool_weights 
                    (tool_name, intent, total_uses, successes, failures, avg_execution_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    weight.tool_name,
                    weight.intent,
                    weight.total_uses,
                    weight.successes,
                    weight.failures,
                    weight.avg_execution_time_ms,
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to update tool weight: {e}")
    
    def _learn_pattern(self, plan: "ExecutionPlan", context: Dict[str, Any]):
        """Extract and store a learned pattern from successful execution."""
        # Generate pattern from query
        query_pattern = self._query_to_pattern(plan.query)
        pattern_id = hashlib.md5(f"{plan.intent.value}:{query_pattern}".encode()).hexdigest()[:12]
        
        # Extract input mappings from context
        input_mappings = {}
        df = context.get("df")
        if df is not None:
            import pandas as pd
            if isinstance(df, pd.DataFrame):
                input_mappings["columns"] = list(df.columns)
                input_mappings["numeric_cols"] = df.select_dtypes(include=["number"]).columns.tolist()
                input_mappings["categorical_cols"] = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Tool sequence
        tool_sequence = [step.tool for step in plan.steps]
        
        # Check if pattern already exists
        intent = plan.intent.value
        existing = self._find_pattern(intent, query_pattern)
        
        if existing:
            # Update existing pattern
            existing.success_count += 1
            existing.last_used = datetime.now().isoformat()
            existing.confidence = min(1.0, existing.success_rate + 0.05)
        else:
            # Create new pattern
            pattern = LearnedPattern(
                pattern_id=pattern_id,
                intent=intent,
                query_pattern=query_pattern,
                tool_sequence=tool_sequence,
                input_mappings=input_mappings,
            )
            if intent not in self._pattern_cache:
                self._pattern_cache[intent] = []
            self._pattern_cache[intent].append(pattern)
            existing = pattern
        
        # Persist
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO learned_patterns
                    (pattern_id, intent, query_pattern, tool_sequence, input_mappings,
                     success_count, fail_count, confidence, created_at, last_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    existing.pattern_id,
                    existing.intent,
                    existing.query_pattern,
                    json.dumps(existing.tool_sequence),
                    json.dumps(existing.input_mappings),
                    existing.success_count,
                    existing.fail_count,
                    existing.confidence,
                    existing.created_at,
                    existing.last_used,
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to save learned pattern: {e}")
    
    def _learn_intent_pattern(self, query: str, intent: str):
        """Learn a new regex pattern for intent classification."""
        # Extract key tokens from query
        tokens = self._extract_key_tokens(query)
        if len(tokens) < 2:
            return
        
        # Build regex pattern
        pattern_parts = [r"\b" + re.escape(t) + r"\b" for t in tokens[:3]]
        regex_pattern = r".*".join(pattern_parts)
        
        pattern_id = hashlib.md5(f"{intent}:{regex_pattern}".encode()).hexdigest()[:12]
        specificity = len(tokens) / 10  # More tokens = more specific
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO intent_patterns (pattern_id, intent, regex_pattern, specificity, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(pattern_id) DO UPDATE SET hit_count = hit_count + 1
                """, (pattern_id, intent, regex_pattern, specificity, datetime.now().isoformat()))
                conn.commit()
        except Exception as e:
            logger.debug(f"Failed to save intent pattern: {e}")
    
    def _query_to_pattern(self, query: str) -> str:
        """Convert a query to a generalizable regex pattern."""
        # Normalize: lowercase, strip punctuation
        normalized = query.lower().strip()
        normalized = re.sub(r"[^\w\s]", "", normalized)
        
        # Replace specific values with wildcards
        # Numbers -> \d+
        normalized = re.sub(r"\d+", r"\\d+", normalized)
        
        # Common column name patterns -> (\w+)
        normalized = re.sub(r"\b(column|field|col)\b", r"(\\w+)", normalized)
        
        return normalized
    
    def _extract_key_tokens(self, query: str) -> List[str]:
        """Extract key tokens for intent pattern learning."""
        # Remove stopwords and short words
        stopwords = {"the", "a", "an", "my", "your", "me", "to", "for", "of", "in", "on", "is", "are"}
        tokens = query.lower().split()
        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
        return tokens[:5]
    
    def _find_pattern(self, intent: str, query_pattern: str) -> Optional[LearnedPattern]:
        """Find existing pattern by intent and query pattern."""
        patterns = self._pattern_cache.get(intent, [])
        for p in patterns:
            if p.query_pattern == query_pattern:
                return p
        return None
    
    # =========================================================================
    # Public Query API
    # =========================================================================
    
    def get_learned_patterns(self, intent: str) -> List[LearnedPattern]:
        """Get learned patterns for an intent, sorted by confidence."""
        self._load_cache()
        patterns = self._pattern_cache.get(intent, [])
        return sorted(patterns, key=lambda p: p.confidence, reverse=True)
    
    def get_tool_weight(self, tool_name: str, intent: str) -> float:
        """Get the weight for a tool in a given intent context."""
        self._load_cache()
        key = f"{tool_name}:{intent}"
        weight = self._weight_cache.get(key)
        return weight.weight if weight else 0.5  # Default weight
    
    def get_tool_weights(self, intent: Optional[str] = None) -> Dict[str, float]:
        """Get all tool weights, optionally filtered by intent."""
        self._load_cache()
        result = {}
        for key, weight in self._weight_cache.items():
            if intent is None or weight.intent == intent:
                result[weight.tool_name] = weight.weight
        return result
    
    def get_best_tool_for_intent(self, intent: str, candidates: List[str]) -> Optional[str]:
        """Get the best-performing tool from a list of candidates."""
        self._load_cache()
        
        best_tool = None
        best_weight = 0.0
        
        for tool_name in candidates:
            weight = self.get_tool_weight(tool_name, intent)
            if weight > best_weight:
                best_weight = weight
                best_tool = tool_name
        
        return best_tool
    
    def get_suggested_inputs(self, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get suggested input values based on learned patterns."""
        self._load_cache()
        patterns = self.get_learned_patterns(intent)
        
        if not patterns:
            return {}
        
        # Use the most confident pattern
        best = patterns[0]
        return best.input_mappings
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning system statistics."""
        self._load_cache()
        
        total_patterns = sum(len(p) for p in self._pattern_cache.values())
        total_tools = len(self._weight_cache)
        
        # Calculate average success rate
        success_rates = [w.success_rate for w in self._weight_cache.values() if w.total_uses > 0]
        avg_success = sum(success_rates) / len(success_rates) if success_rates else 0.0
        
        return {
            "total_patterns": total_patterns,
            "total_tool_weights": total_tools,
            "avg_success_rate": round(avg_success, 3),
            "intents_learned": list(self._pattern_cache.keys()),
        }


# =============================================================================
# Global Access
# =============================================================================

_learner: Optional[PlanLearner] = None


def get_learner() -> PlanLearner:
    """Get the global plan learner instance."""
    global _learner
    if _learner is None:
        _learner = PlanLearner()
    return _learner

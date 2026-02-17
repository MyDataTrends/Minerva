"""
Base classes for the Minerva agent infrastructure.

All agents extend BaseAgent and implement run() to perform their work.
Results are captured in AgentResult for aggregation by the Conductor.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from utils.logging import get_logger

logger = get_logger(__name__)


class TriggerType(Enum):
    """How an agent is activated."""
    CRON = "cron"           # Scheduled (daily, weekly)
    EVENT = "event"         # Triggered by an external event (new issue, PR, etc.)
    MANUAL = "manual"       # Run on-demand via CLI


class Priority(Enum):
    """Escalation priority levels for the daily digest."""
    URGENT = "urgent"       # 🔴 Requires human action today
    REVIEW = "review"       # 🟡 Needs review within 48 hours
    FYI = "fyi"             # 🟢 Handled autonomously, informational
    METRIC = "metric"       # 📊 Data point for the metrics snapshot


@dataclass
class Escalation:
    """An item that needs human attention."""
    priority: Priority
    title: str
    detail: str
    source_agent: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AgentResult:
    """Captures the outcome of an agent run."""
    agent_name: str
    success: bool
    actions_taken: List[str] = field(default_factory=list)
    escalations: List[Escalation] = field(default_factory=list)
    summary: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def add_action(self, action: str) -> None:
        self.actions_taken.append(action)

    def add_escalation(self, priority: Priority, title: str, detail: str) -> None:
        self.escalations.append(Escalation(
            priority=priority,
            title=title,
            detail=detail,
            source_agent=self.agent_name,
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "success": self.success,
            "actions_taken": self.actions_taken,
            "escalations": [
                {
                    "priority": e.priority.value,
                    "title": e.title,
                    "detail": e.detail,
                    "source_agent": e.source_agent,
                    "timestamp": e.timestamp,
                }
                for e in self.escalations
            ],
            "summary": self.summary,
            "metrics": self.metrics,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    name: str
    enabled: bool = True
    dry_run: bool = True          # Safe default: agents report but don't act
    schedule: str = "daily"       # "daily", "weekly", "on_event"
    llm_model: str = "claude-3-5-sonnet"
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base for all Minerva agents.

    Subclasses must implement:
        - run(**kwargs) -> AgentResult
    """

    name: str = "base"
    trigger_type: TriggerType = TriggerType.CRON

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig(name=self.name)
        self.logger = get_logger(f"agents.{self.name}")

    @property
    def is_enabled(self) -> bool:
        return self.config.enabled

    @property
    def is_dry_run(self) -> bool:
        return self.config.dry_run

    @abstractmethod
    def run(self, **kwargs) -> AgentResult:
        """Execute the agent's primary workflow. Must be implemented by subclasses."""
        ...

    def _make_result(self, success: bool = True, **kwargs) -> AgentResult:
        """Convenience to create an AgentResult pre-filled with this agent's name."""
        return AgentResult(agent_name=self.name, success=success, **kwargs)

    def __repr__(self) -> str:
        status = "enabled" if self.is_enabled else "disabled"
        mode = "dry-run" if self.is_dry_run else "live"
        return f"<{self.__class__.__name__} name={self.name} {status} {mode}>"

"""
Tests for the agent framework base classes and configuration.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from agents.base import (
    AgentConfig, AgentResult, BaseAgent, Escalation, Priority, TriggerType,
)


# ── BaseAgent Tests ──────────────────────────────────────────────────


class DummyAgent(BaseAgent):
    """Concrete agent for testing."""
    name = "dummy"
    trigger_type = TriggerType.MANUAL

    def run(self, **kwargs):
        result = self._make_result()
        result.success = True
        result.summary = "Test run complete"
        result.add_action("test_action")
        return result


class FailingAgent(BaseAgent):
    """Agent that always fails."""
    name = "failing"

    def run(self, **kwargs):
        result = self._make_result(success=False, error="intentional failure")
        result.summary = "Failed on purpose"
        return result


class TestBaseAgent:

    def test_subclass_instantiation(self):
        agent = DummyAgent()
        assert agent.name == "dummy"
        assert agent.trigger_type == TriggerType.MANUAL

    def test_run_returns_result(self):
        agent = DummyAgent()
        result = agent.run()
        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.agent_name == "dummy"
        assert "test_action" in result.actions_taken

    def test_dry_run_default(self):
        agent = DummyAgent()
        assert agent.is_dry_run is True  # Safe default

    def test_enabled_default(self):
        agent = DummyAgent()
        assert agent.is_enabled is True

    def test_config_override(self):
        config = AgentConfig(name="dummy", enabled=False, dry_run=False)
        agent = DummyAgent(config=config)
        assert agent.is_enabled is False
        assert agent.is_dry_run is False

    def test_repr(self):
        agent = DummyAgent()
        r = repr(agent)
        assert "DummyAgent" in r
        assert "dummy" in r
        assert "enabled" in r
        assert "dry-run" in r

    def test_failing_agent(self):
        agent = FailingAgent()
        result = agent.run()
        assert result.success is False
        assert result.error == "intentional failure"


# ── AgentResult Tests ────────────────────────────────────────────────


class TestAgentResult:

    def test_add_action(self):
        result = AgentResult(agent_name="test", success=True)
        result.add_action("action_1")
        result.add_action("action_2")
        assert result.actions_taken == ["action_1", "action_2"]

    def test_add_escalation(self):
        result = AgentResult(agent_name="test", success=True)
        result.add_escalation(Priority.URGENT, "Title", "Detail")
        assert len(result.escalations) == 1
        assert result.escalations[0].priority == Priority.URGENT
        assert result.escalations[0].source_agent == "test"

    def test_to_dict(self):
        result = AgentResult(agent_name="test", success=True, summary="Test summary")
        result.add_action("action_1")
        result.add_escalation(Priority.FYI, "Info", "Details")

        d = result.to_dict()
        assert d["agent_name"] == "test"
        assert d["success"] is True
        assert len(d["actions_taken"]) == 1
        assert len(d["escalations"]) == 1
        assert d["escalations"][0]["priority"] == "fyi"

    def test_default_timestamp(self):
        result = AgentResult(agent_name="test", success=True)
        assert result.timestamp is not None
        assert "T" in result.timestamp  # ISO format


# ── AgentConfig Tests ────────────────────────────────────────────────


class TestAgentConfig:

    def test_defaults(self):
        config = AgentConfig(name="test")
        assert config.enabled is True
        assert config.dry_run is True
        assert config.schedule == "daily"
        assert config.llm_model == "claude-3-5-sonnet"

    def test_custom_values(self):
        config = AgentConfig(
            name="custom",
            enabled=False,
            dry_run=False,
            schedule="weekly",
            llm_model="gpt-4",
        )
        assert config.enabled is False
        assert config.dry_run is False
        assert config.schedule == "weekly"


# ── Config Loader Tests ─────────────────────────────────────────────


class TestConfigLoader:

    def test_load_agent_configs(self):
        from agents.config import load_agent_configs
        configs = load_agent_configs()
        assert isinstance(configs, dict)
        # Should have at least the agents defined in agents_config.yaml
        assert "conductor" in configs

    def test_get_agent_config_existing(self):
        from agents.config import get_agent_config
        config = get_agent_config("conductor")
        assert config.name == "conductor"
        assert isinstance(config.enabled, bool)

    def test_get_agent_config_unknown(self):
        from agents.config import get_agent_config
        config = get_agent_config("nonexistent_agent")
        assert config.name == "nonexistent_agent"
        assert config.dry_run is True  # Safe default

    def test_github_config(self):
        from agents.config import get_github_config
        gh = get_github_config()
        assert "token" in gh
        assert "repo" in gh

    def test_global_dry_run_override(self, monkeypatch):
        monkeypatch.setenv("AGENT_DRY_RUN", "1")
        from agents.config import get_agent_config
        config = get_agent_config("conductor")
        assert config.dry_run is True


# ── Priority and TriggerType ─────────────────────────────────────────


class TestEnums:

    def test_priority_values(self):
        assert Priority.URGENT.value == "urgent"
        assert Priority.REVIEW.value == "review"
        assert Priority.FYI.value == "fyi"
        assert Priority.METRIC.value == "metric"

    def test_trigger_type_values(self):
        assert TriggerType.CRON.value == "cron"
        assert TriggerType.EVENT.value == "event"
        assert TriggerType.MANUAL.value == "manual"

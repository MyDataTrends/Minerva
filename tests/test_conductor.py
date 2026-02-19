"""
Tests for the Conductor agent.
"""

import pytest
from unittest.mock import patch, MagicMock

from agents.base import AgentConfig, AgentResult, Priority
from agents.conductor import ConductorAgent


class TestConductorAgent:

    @pytest.fixture
    def conductor(self, tmp_path):
        config = AgentConfig(name="conductor", dry_run=True)
        agent = ConductorAgent(config=config)
        # Use temp dir for state
        agent.memory.db_dir = tmp_path
        agent.memory.db_path = tmp_path / "conductor.db"
        agent.memory._init_db()
        return agent

    def test_instantiation(self, conductor):
        assert conductor.name == "conductor"
        assert conductor.is_dry_run is True
        assert conductor.is_enabled is True

    @patch("agents.conductor.ConductorAgent._scan_github")
    @patch("agents.conductor.ConductorAgent._run_sub_agents")
    def test_produces_digest(self, mock_sub, mock_gh, conductor, tmp_path):
        """Conductor produces a valid markdown digest."""
        mock_gh.return_value = {"stars": 10, "open_issues": 2}
        mock_sub.return_value = [
            AgentResult(
                agent_name="engineer",
                success=True,
                summary="Gap analysis done",
                actions_taken=["scored_subsystems"],
            ),
        ]

        # Override digest dir
        import agents.conductor as mod
        original_dir = mod.DIGEST_DIR
        mod.DIGEST_DIR = tmp_path / "digests"

        try:
            result = conductor.run()

            assert result.success is True
            assert "compiled_digest" in result.actions_taken
            assert result.metrics.get("stars") == 10

            # Check digest file was created
            digest_files = list((tmp_path / "digests").glob("*.md"))
            assert len(digest_files) == 1

            content = digest_files[0].read_text(encoding="utf-8")
            assert "Assay Daily Digest" in content
            assert "📊 Metrics" in content
        finally:
            mod.DIGEST_DIR = original_dir

    @patch("agents.conductor.ConductorAgent._scan_github")
    @patch("agents.conductor.ConductorAgent._run_sub_agents")
    def test_aggregates_escalations(self, mock_sub, mock_gh, conductor, tmp_path):
        """Conductor aggregates escalations from sub-agents."""
        mock_gh.return_value = {}

        sub_result = AgentResult(agent_name="advocate", success=True, summary="Done")
        sub_result.add_escalation(Priority.URGENT, "Bug!", "Critical bug found")
        sub_result.add_escalation(Priority.FYI, "Info", "Handled automatically")
        mock_sub.return_value = [sub_result]

        import agents.conductor as mod
        original_dir = mod.DIGEST_DIR
        mod.DIGEST_DIR = tmp_path / "digests"

        try:
            result = conductor.run()

            assert len(result.escalations) == 2
            urgent = [e for e in result.escalations if e.priority == Priority.URGENT]
            assert len(urgent) == 1

            digest_content = list((tmp_path / "digests").glob("*.md"))[0].read_text(encoding="utf-8")
            assert "🔴 Urgent" in digest_content
            assert "🟢 FYI" in digest_content
        finally:
            mod.DIGEST_DIR = original_dir

    @patch("agents.conductor.ConductorAgent._scan_github")
    @patch("agents.conductor.ConductorAgent._run_sub_agents")
    def test_respects_enabled_flags(self, mock_sub, mock_gh, conductor, tmp_path):
        """Conductor only runs enabled agents."""
        mock_gh.return_value = {}
        mock_sub.return_value = []  # No agents ran

        import agents.conductor as mod
        original_dir = mod.DIGEST_DIR
        mod.DIGEST_DIR = tmp_path / "digests"

        try:
            result = conductor.run()
            assert result.success is True
            assert "ran_0_sub_agents" in result.actions_taken
        finally:
            mod.DIGEST_DIR = original_dir

    def test_compile_digest_format(self, conductor):
        """Verify digest markdown structure."""
        sub_results = [
            AgentResult(agent_name="engineer", success=True, summary="Analysis done"),
            AgentResult(agent_name="advocate", success=False, summary="Failed", error="API timeout"),
        ]
        sub_results[0].add_escalation(Priority.REVIEW, "PR ready", "Engineer PR #1")

        digest = conductor._compile_digest(sub_results, {"stars": 42, "open_prs": 1})

        assert "# Assay Daily Digest" in digest
        assert "📊 Metrics" in digest
        assert "stars" in digest
        assert "🟡 Needs Review" in digest
        assert "engineer" in digest
        assert "advocate" in digest
        assert "API timeout" in digest

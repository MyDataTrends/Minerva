"""
Tests for the Engineer agent.
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from agents.base import AgentConfig, Priority
from agents.engineer import EngineerAgent


class TestEngineerAgent:

    @pytest.fixture
    def engineer(self, tmp_path):
        config = AgentConfig(name="engineer", dry_run=True)
        agent = EngineerAgent(config=config)
        # Use temp dirs for state and knowledge base
        agent.memory.db_dir = tmp_path / "state"
        agent.memory.db_dir.mkdir()
        agent.memory.db_path = agent.memory.db_dir / "engineer.db"
        agent.memory._init_db()
        agent.kb.base_dir = tmp_path / "kb"
        agent.kb._ensure_structure()
        return agent

    def test_instantiation(self, engineer):
        assert engineer.name == "engineer"
        assert engineer.is_dry_run is True

    def test_subsystems_defined(self, engineer):
        """Engineer knows about all subsystems from the vision doc."""
        assert len(engineer.SUBSYSTEMS) > 0
        assert "cascade_planner" in engineer.SUBSYSTEMS
        assert "mcp_server" in engineer.SUBSYSTEMS

    def test_run_produces_gap_analysis(self, engineer, tmp_path):
        """Engineer produces a gap analysis report."""
        result = engineer.run()

        assert result.success is True
        assert result.agent_name == "engineer"
        assert "scored" in result.actions_taken[0]
        assert "identified_top_gap" in result.actions_taken[1]

        # Verify gap analysis was written to knowledge base
        gap_report = engineer.kb.read_doc("product/vision_gap_analysis.md")
        assert gap_report is not None
        assert "Vision Gap Analysis" in gap_report
        assert "Subsystem Maturity Scores" in gap_report

    def test_selects_highest_priority(self, engineer):
        """Engineer identifies the subsystem with the largest gap."""
        result = engineer.run()

        # Should have an escalation about the top improvement
        review_escalations = [
            e for e in result.escalations if e.priority == Priority.REVIEW
        ]
        assert len(review_escalations) >= 1
        assert "improvement target" in review_escalations[0].title.lower() or "engineer" in review_escalations[0].title.lower()

    def test_maturity_scoring(self, engineer):
        """Test the heuristic maturity scoring."""
        scores = engineer._score_subsystems()

        for name, info in scores.items():
            assert 1 <= info["score"] <= 4
            assert info["maturity"] in engineer.MATURITY_LEVELS.values()
            assert "recommendation" in info

    def test_gap_report_format(self, engineer):
        """Gap report contains expected markdown structure."""
        result = engineer.run()
        report = engineer.kb.read_doc("product/vision_gap_analysis.md")

        assert "| Subsystem |" in report
        assert "Top Priority Improvement" in report
        assert "Maturity Scale" in report
        assert "█" in report  # Progress bars

    def test_metrics_in_result(self, engineer):
        """Result includes maturity metrics."""
        result = engineer.run()

        assert "subsystems_scored" in result.metrics
        assert "average_maturity" in result.metrics
        assert "top_gap" in result.metrics
        assert result.metrics["subsystems_scored"] == len(engineer.SUBSYSTEMS)

    def test_count_lines(self, engineer, tmp_path):
        """Line counting utility works."""
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\n\nline4\n", encoding="utf-8")
        assert engineer._count_lines(test_file) == 3  # Excludes empty lines

    def test_count_lines_missing_file(self, engineer, tmp_path):
        """Missing files return 0 lines."""
        assert engineer._count_lines(tmp_path / "nonexistent.py") == 0

    def test_find_top_gap_all_perfect(self, engineer):
        """When all subsystems are perfect, returns 'none'."""
        scores = {
            name: {"score": 4, "maturity": "best-in-class", "has_tests": True,
                   "recommendation": "Already best-in-class"}
            for name in ["a", "b", "c"]
        }
        top = engineer._find_top_gap(scores)
        assert top["name"] == "none"
        assert top["gap"] == 0

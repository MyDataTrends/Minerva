"""
Tests for the Sentinel (QA) agent.
"""

import pytest
from unittest.mock import patch, MagicMock

from agents.base import AgentConfig, Priority
from agents.sentinel import SentinelAgent


class TestSentinelAgent:

    @pytest.fixture
    def sentinel(self, tmp_path):
        config = AgentConfig(
            name="sentinel",
            dry_run=True,
            extra={"min_confidence_score": 7},
        )
        agent = SentinelAgent(config=config)
        agent.memory.db_dir = tmp_path / "state"
        agent.memory.db_dir.mkdir()
        agent.memory.db_path = agent.memory.db_dir / "sentinel.db"
        agent.memory._init_db()

        # Use temp file for history
        import agents.sentinel as mod
        mod.QA_HISTORY_FILE = tmp_path / "qa_history.json"

        return agent

    def test_instantiation(self, sentinel):
        assert sentinel.name == "sentinel"
        assert sentinel.min_confidence == 7

    def test_parse_pytest_output_all_pass(self, sentinel):
        output = "========================= 56 passed in 12.34s ========================="
        passed, failed, errors, total = sentinel._parse_pytest_output(output)
        assert passed == 56
        assert failed == 0
        assert errors == 0
        assert total == 56

    def test_parse_pytest_output_mixed(self, sentinel):
        output = "============ 50 passed, 4 failed, 2 error in 15.0s ============"
        passed, failed, errors, total = sentinel._parse_pytest_output(output)
        assert passed == 50
        assert failed == 4
        assert errors == 2
        assert total == 56

    def test_confidence_all_pass(self, sentinel):
        test_results = {"all_passed": True, "passed": 56, "failed": 0, "errors": 0, "total": 56}
        lint_results = {"clean": True, "issues": 0}
        score = sentinel._calculate_confidence(test_results, lint_results)
        assert score >= 7  # Should be high confidence

    def test_confidence_some_failures(self, sentinel):
        test_results = {"all_passed": False, "passed": 40, "failed": 16, "errors": 0, "total": 56}
        lint_results = {"clean": False, "issues": 10}
        score = sentinel._calculate_confidence(test_results, lint_results)
        assert score < 7  # Should be lower confidence

    def test_confidence_zero_tests(self, sentinel):
        test_results = {"all_passed": False, "passed": 0, "failed": 0, "errors": 0, "total": 0}
        lint_results = {"clean": True, "issues": 0}
        score = sentinel._calculate_confidence(test_results, lint_results)
        assert score <= 4  # Low confidence with no tests

    def test_report_format_passing(self, sentinel):
        test_results = {"all_passed": True, "passed": 56, "failed": 0, "errors": 0, "total": 56}
        lint_results = {"clean": True, "issues": 0}
        report = sentinel._generate_report(test_results, lint_results, 9, pr_number=42)

        assert "🛡️ Sentinel QA Report" in report
        assert "PR" in report
        assert "#42" in report
        assert "56/56" in report
        assert "9/10" in report
        assert "Ready for human review" in report

    def test_report_format_failing(self, sentinel):
        test_results = {"all_passed": False, "passed": 40, "failed": 16, "errors": 0, "total": 56}
        lint_results = {"clean": False, "issues": 5}
        report = sentinel._generate_report(test_results, lint_results, 4)

        assert "Needs iteration" in report

    def test_report_format_severe_failure(self, sentinel):
        test_results = {"all_passed": False, "passed": 10, "failed": 46, "errors": 5, "total": 61}
        lint_results = {"clean": False, "issues": 20}
        report = sentinel._generate_report(test_results, lint_results, 2)

        assert "Rejected" in report

    @patch("agents.sentinel.SentinelAgent._run_tests")
    @patch("agents.sentinel.SentinelAgent._run_linter")
    def test_high_score_routes_to_human(self, mock_lint, mock_tests, sentinel):
        mock_tests.return_value = {
            "all_passed": True, "passed": 56, "failed": 0, "errors": 0,
            "total": 56, "returncode": 0, "output_tail": ""
        }
        mock_lint.return_value = {"clean": True, "issues": 0, "output_tail": ""}

        result = sentinel.run()

        review_escalations = [e for e in result.escalations if e.priority == Priority.REVIEW]
        assert len(review_escalations) >= 1
        assert "ready for review" in review_escalations[0].title.lower()

    @patch("agents.sentinel.SentinelAgent._run_tests")
    @patch("agents.sentinel.SentinelAgent._run_linter")
    def test_low_score_returns_to_engineer(self, mock_lint, mock_tests, sentinel):
        mock_tests.return_value = {
            "all_passed": False, "passed": 30, "failed": 26, "errors": 5,
            "total": 61, "returncode": 1, "output_tail": ""
        }
        mock_lint.return_value = {"clean": False, "issues": 15, "output_tail": ""}

        result = sentinel.run()

        fyi_escalations = [e for e in result.escalations if e.priority == Priority.FYI]
        assert len(fyi_escalations) >= 1
        assert "sent back to engineer" in fyi_escalations[0].title.lower()

    def test_history_tracking(self, sentinel, tmp_path):
        """QA history is persisted and loaded correctly."""
        import agents.sentinel as mod
        sentinel._update_history(8, {"passed": 56, "total": 56}, {"issues": 0})
        sentinel._update_history(5, {"passed": 40, "total": 56}, {"issues": 3})

        history = sentinel._load_history()
        assert len(history) == 2
        assert history[0]["score"] == 8
        assert history[1]["score"] == 5

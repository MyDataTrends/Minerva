"""
Tests for the Advocate agent.
"""

import pytest
from unittest.mock import patch, MagicMock

from agents.base import AgentConfig, Priority
from agents.advocate import AdvocateAgent, ISSUE_TYPES


class TestAdvocateAgent:

    @pytest.fixture
    def advocate(self, tmp_path):
        config = AgentConfig(
            name="advocate",
            dry_run=True,
            extra={"stale_days": 14},
        )
        agent = AdvocateAgent(config=config)
        agent.memory.db_dir = tmp_path / "state"
        agent.memory.db_dir.mkdir()
        agent.memory.db_path = agent.memory.db_dir / "advocate.db"
        agent.memory._init_db()
        return agent

    def test_instantiation(self, advocate):
        assert advocate.name == "advocate"
        assert advocate.stale_days == 14

    @patch("llm_manager.llm_interface.is_llm_available", return_value=False)
    def test_classify_bug(self, mock_llm, advocate):
        result = advocate._classify_issue(
            "App crashes when loading large CSV",
            "I get a traceback error when trying to load a 500MB file"
        )
        assert result == "bug"

    @patch("llm_manager.llm_interface.is_llm_available", return_value=False)
    def test_classify_feature(self, mock_llm, advocate):
        result = advocate._classify_issue(
            "Feature request: PDF export",
            "It would be nice to add PDF export for reports"
        )
        assert result == "feature"

    @patch("llm_manager.llm_interface.is_llm_available", return_value=False)
    def test_classify_question(self, mock_llm, advocate):
        result = advocate._classify_issue(
            "How do I connect to PostgreSQL?",
            "I want to use Minerva with my existing setup"
        )
        assert result == "question"

    @patch("llm_manager.llm_interface.is_llm_available", return_value=False)
    def test_classify_docs(self, mock_llm, advocate):
        result = advocate._classify_issue(
            "Missing documentation for API endpoints",
            "The README doesn't cover the REST API"
        )
        assert result == "docs"

    @patch("llm_manager.llm_interface.is_llm_available", return_value=False)
    def test_classify_security(self, mock_llm, advocate):
        result = advocate._classify_issue(
            "Potential SQL injection vulnerability",
            "Found a security issue in the query builder"
        )
        assert result == "security"

    def test_issue_types_have_labels(self):
        """All issue types map to GitHub labels."""
        for type_name, info in ISSUE_TYPES.items():
            assert "labels" in info
            assert len(info["labels"]) >= 1
            assert "emoji" in info

    @patch("agents.advocate.AdvocateAgent._classify_issue")
    def test_run_without_github(self, mock_classify, advocate):
        """Advocate handles missing GitHub client gracefully."""
        with patch("agents.tools.github_client.GitHubClient") as MockClient:
            instance = MockClient.return_value
            instance.is_available = False

            result = advocate.run()
            assert result.success is True
            assert "skipped_no_github" in result.actions_taken

    @patch("agents.tools.github_client.GitHubClient")
    def test_run_with_issues(self, MockGH, advocate):
        """Advocate processes issues correctly in dry-run mode."""
        mock_client = MagicMock()
        mock_client.is_available = True
        mock_client.list_issues.return_value = [
            {
                "number": 1,
                "title": "Bug: crash on startup",
                "body": "The app fails to start",
                "labels": [],
                "updated_at": "2026-02-16T00:00:00Z",
            },
            {
                "number": 2,
                "title": "How to configure LLM?",
                "body": "I need help setting up the LLM",
                "labels": [],
                "updated_at": "2026-02-16T00:00:00Z",
            },
        ]
        MockGH.return_value = mock_client

        result = advocate.run()
        assert result.success is True
        assert result.metrics["open_issues"] == 2

        # Bug should generate an escalation
        bug_escalations = [
            e for e in result.escalations if e.priority == Priority.REVIEW
        ]
        assert len(bug_escalations) >= 1

    @patch("agents.tools.github_client.GitHubClient")
    def test_skips_already_labeled(self, MockGH, advocate):
        """Advocate skips issues that already have classification labels."""
        mock_client = MagicMock()
        mock_client.is_available = True
        mock_client.list_issues.return_value = [
            {
                "number": 1,
                "title": "Already labeled bug",
                "body": "This was labeled before",
                "labels": [{"name": "bug"}],
                "updated_at": "2026-02-16T00:00:00Z",
            },
        ]
        MockGH.return_value = mock_client

        result = advocate.run()
        assert result.success is True
        # Should not have classified any issues
        classified_actions = [a for a in result.actions_taken if "label" in a.lower()]
        assert len(classified_actions) == 0

    @patch("agents.tools.github_client.GitHubClient")
    def test_skips_pull_requests(self, MockGH, advocate):
        """Advocate skips PRs (GitHub API returns them as issues)."""
        mock_client = MagicMock()
        mock_client.is_available = True
        mock_client.list_issues.return_value = [
            {
                "number": 1,
                "title": "PR: fix typo",
                "body": "",
                "labels": [],
                "pull_request": {"url": "..."},
                "updated_at": "2026-02-16T00:00:00Z",
            },
        ]
        MockGH.return_value = mock_client

        result = advocate.run()
        assert result.success is True
        # Should not have classified any
        classified_actions = [a for a in result.actions_taken if "label" in a.lower()]
        assert len(classified_actions) == 0

    def test_security_escalation_is_urgent(self, advocate):
        """Security issues should escalate as URGENT."""
        # This tests the classification → escalation flow
        result = advocate._classify_issue(
            "Critical CVE found",
            "There's an exploit vulnerability in the auth module"
        )
        assert result == "security"

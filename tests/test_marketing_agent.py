"""
Tests for the Marketing Agent.

All LLM calls and git subprocess calls are mocked so these run offline
with no external dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.base import AgentConfig, Priority, TriggerType
from agents.marketing import MarketingAgent


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_agent(tmp_path, monkeypatch):
    """Return a MarketingAgent whose digest and KB dirs point to tmp_path."""
    monkeypatch.setattr("agents.marketing.DIGEST_DIR", tmp_path / "marketing")
    monkeypatch.setattr("agents.marketing.KB_PRODUCT_DIR", tmp_path / "product")
    monkeypatch.setattr(
        "agents.marketing.PROJECT_ROOT", tmp_path
    )
    # Redirect OperationalMemory DB to tmp_path
    monkeypatch.setattr(
        "agents.memory.operational.STATE_DIR", tmp_path / "state"
    )
    config = AgentConfig(name="marketing", dry_run=True)
    return MarketingAgent(config=config)


# ── Class / init ─────────────────────────────────────────────────────


class TestMarketingAgentInit:

    def test_name_and_trigger(self, tmp_agent):
        assert tmp_agent.name == "marketing"
        assert tmp_agent.trigger_type == TriggerType.CRON

    def test_is_dry_run(self, tmp_agent):
        assert tmp_agent.is_dry_run is True

    def test_digest_dir_created(self, tmp_path, tmp_agent):
        assert (tmp_path / "marketing").exists()


# ── Git log helper ────────────────────────────────────────────────────


class TestGetRecentCommits:

    def test_returns_list(self, tmp_agent):
        with patch("subprocess.check_output", return_value=b"feat: add thing\nfix: bug"):
            commits = tmp_agent._get_recent_commits(7)
        assert isinstance(commits, list)
        assert "feat: add thing" in commits

    def test_empty_on_git_error(self, tmp_agent):
        import subprocess
        with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "git")):
            commits = tmp_agent._get_recent_commits(7)
        assert commits == []

    def test_empty_on_no_git(self, tmp_agent):
        with patch("subprocess.check_output", side_effect=FileNotFoundError):
            commits = tmp_agent._get_recent_commits(7)
        assert commits == []


# ── Changelog summarizer ──────────────────────────────────────────────


class TestSummarizeChangelog:

    def test_empty_commits_returns_empty(self, tmp_agent):
        result = tmp_agent._summarize_changelog([])
        assert result == ""

    def test_llm_response_used_when_available(self, tmp_agent):
        with patch("llm_manager.llm_interface.get_llm_completion", return_value="- New feature X"):
            result = tmp_agent._summarize_changelog(["feat: X"])
        assert "New feature X" in result

    def test_fallback_to_bullet_list(self, tmp_agent):
        with patch("llm_manager.llm_interface.get_llm_completion", return_value=""):
            result = tmp_agent._summarize_changelog(["feat: X", "fix: Y"])
        assert "feat: X" in result
        assert "fix: Y" in result


# ── Productizer doc reader ────────────────────────────────────────────


class TestReadProductizerDocs:

    def test_returns_empty_when_no_dir(self, tmp_agent, tmp_path):
        # KB dir exists but no sales_kits subdir
        result = tmp_agent._read_productizer_docs()
        assert result == []

    def test_reads_recent_md_files(self, tmp_agent, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.marketing.KB_PRODUCT_DIR", tmp_path / "product")
        sales_dir = tmp_path / "product" / "sales_kits"
        sales_dir.mkdir(parents=True)
        (sales_dir / "kit_retail_20260101.md").write_text("# Retail Kit\nContent here.", encoding="utf-8")

        tmp_agent2 = MarketingAgent.__new__(MarketingAgent)
        tmp_agent2.__init__(AgentConfig(name="marketing", dry_run=True))
        # Re-patch
        import agents.marketing as m_mod
        orig = m_mod.KB_PRODUCT_DIR
        m_mod.KB_PRODUCT_DIR = tmp_path / "product"
        try:
            docs = tmp_agent2._read_productizer_docs()
        finally:
            m_mod.KB_PRODUCT_DIR = orig

        assert len(docs) >= 1


# ── Draft generation ──────────────────────────────────────────────────


class TestDraftGeneration:

    def _agent_with_llm(self, tmp_agent, response="Draft content here"):
        patch_target = "llm_manager.llm_interface.get_llm_completion"
        return patch(patch_target, return_value=response)

    def test_draft_hn(self, tmp_agent):
        with patch("llm_manager.llm_interface.get_llm_completion", return_value="TITLE: Show HN\n\nBODY: Cool"):
            result = tmp_agent._draft_hn("Some context")
        assert result != ""

    def test_draft_hn_fallback(self, tmp_agent):
        with patch("llm_manager.llm_interface.get_llm_completion", return_value=""):
            result = tmp_agent._draft_hn("fallback context")
        assert "Assay" in result

    def test_draft_reddit(self, tmp_agent):
        with patch("llm_manager.llm_interface.get_llm_completion", return_value="TITLE: Reddit\n\nBODY: Stuff"):
            result = tmp_agent._draft_reddit("Some context")
        assert result != ""

    def test_draft_reddit_fallback(self, tmp_agent):
        with patch("llm_manager.llm_interface.get_llm_completion", return_value=""):
            result = tmp_agent._draft_reddit("fallback context")
        assert "Assay" in result

    def test_draft_twitter(self, tmp_agent):
        with patch("llm_manager.llm_interface.get_llm_completion", return_value="1/ Tweet one\n2/ Tweet two"):
            result = tmp_agent._draft_twitter("Some context")
        assert "Tweet" in result

    def test_draft_twitter_fallback(self, tmp_agent):
        with patch("llm_manager.llm_interface.get_llm_completion", return_value=""):
            result = tmp_agent._draft_twitter("fallback context")
        assert "1/" in result

    def test_generate_drafts_returns_all_platforms(self, tmp_agent):
        with patch("llm_manager.llm_interface.get_llm_completion", return_value="Some draft"):
            drafts = tmp_agent._generate_drafts("Changelog text", ["Doc content"])
        assert set(drafts.keys()) == {"hackernews", "reddit", "twitter"}


# ── Save draft ────────────────────────────────────────────────────────


class TestSaveDraft:

    def test_file_written(self, tmp_agent, tmp_path):
        path = tmp_agent._save_draft("twitter", "1/ Hello world", "2026-01-01")
        assert path.exists()
        content = path.read_text()
        assert "Hello world" in content
        assert "twitter" in path.name.lower()

    def test_file_contains_header(self, tmp_agent):
        path = tmp_agent._save_draft("hackernews", "# Show HN post", "2026-01-01")
        content = path.read_text()
        assert "Auto-generated" in content


# ── Full run() ────────────────────────────────────────────────────────


class TestMarketingAgentRun:

    def test_run_no_content(self, tmp_agent):
        """When git is unavailable and no productizer docs exist, run succeeds gracefully."""
        with patch("subprocess.check_output", side_effect=FileNotFoundError):
            result = tmp_agent.run(days=7)
        assert result.success is True
        assert result.agent_name == "marketing"

    def test_run_with_commits(self, tmp_agent):
        with (
            patch("subprocess.check_output", return_value=b"feat: new chart type\nfix: CSV parsing"),
            patch("llm_manager.llm_interface.get_llm_completion", return_value="- New chart type added"),
        ):
            result = tmp_agent.run(days=7)

        assert result.success is True
        assert result.metrics.get("commits_last_7d", 0) >= 2
        assert result.metrics.get("drafts_generated", 0) == 3
        # Each draft should be escalated for review
        review_escs = [e for e in result.escalations if e.priority == Priority.REVIEW]
        assert len(review_escs) == 3

    def test_run_creates_digest_files(self, tmp_agent, tmp_path):
        with (
            patch("subprocess.check_output", return_value=b"feat: something cool"),
            patch("llm_manager.llm_interface.get_llm_completion", return_value="Draft content"),
        ):
            result = tmp_agent.run(days=7)

        assert result.success is True
        digest_dir = tmp_path / "marketing"
        md_files = list(digest_dir.glob("*.md"))
        assert len(md_files) == 3  # HN, Reddit, Twitter

    def test_run_error_handled_gracefully(self, tmp_agent, monkeypatch):
        """An unexpected exception should not propagate — result.success=False."""
        monkeypatch.setattr(tmp_agent, "_get_recent_commits", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        result = tmp_agent.run(days=7)
        assert result.success is False
        assert "boom" in result.error

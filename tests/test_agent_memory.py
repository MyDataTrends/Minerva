"""
Tests for the agent memory system (operational + knowledge base).
"""

import os
import tempfile
from pathlib import Path

import pytest

from agents.memory.operational import OperationalMemory
from agents.memory.knowledge import KnowledgeBase


# ── Operational Memory Tests ─────────────────────────────────────────


class TestOperationalMemory:

    @pytest.fixture
    def memory(self, tmp_path):
        """Create a memory instance with a temp directory."""
        return OperationalMemory("test_agent", db_dir=tmp_path)

    def test_creates_db_on_init(self, memory, tmp_path):
        assert (tmp_path / "test_agent.db").exists()

    def test_log_action(self, memory):
        memory.log_action("test_action", "test detail", {"key": "val"})
        actions = memory.get_recent_actions(limit=1)
        assert len(actions) == 1
        assert actions[0]["action"] == "test_action"
        assert actions[0]["detail"] == "test detail"
        assert actions[0]["metadata"] == {"key": "val"}

    def test_recent_actions_ordering(self, memory):
        memory.log_action("first")
        memory.log_action("second")
        memory.log_action("third")
        actions = memory.get_recent_actions(limit=2)
        assert len(actions) == 2
        assert actions[0]["action"] == "third"   # Most recent first
        assert actions[1]["action"] == "second"

    def test_log_escalation(self, memory):
        memory.log_escalation("urgent", "Test urgent", "Something broke")
        pending = memory.get_pending_escalations()
        assert len(pending) == 1
        assert pending[0]["priority"] == "urgent"
        assert pending[0]["title"] == "Test urgent"

    def test_resolve_escalation(self, memory):
        memory.log_escalation("review", "Review this", "Details")
        pending = memory.get_pending_escalations()
        assert len(pending) == 1

        memory.resolve_escalation(pending[0]["id"])
        pending_after = memory.get_pending_escalations()
        assert len(pending_after) == 0

    def test_log_run(self, memory):
        memory.log_run(
            success=True,
            duration=1.5,
            summary="Test run",
            actions_count=3,
            escalations_count=1,
        )
        history = memory.get_run_history(limit=1)
        assert len(history) == 1
        assert history[0]["success"] is True
        assert history[0]["duration"] == 1.5
        assert history[0]["summary"] == "Test run"

    def test_daily_summary(self, memory):
        memory.log_action("action_1")
        memory.log_action("action_2")
        memory.log_escalation("fyi", "FYI item", "Details")
        memory.log_run(True, 0.5, "Quick run")

        summary = memory.get_daily_summary()
        assert summary["agent"] == "test_agent"
        assert summary["actions_today"] >= 2
        assert summary["escalations_today"] >= 1
        assert summary["runs_today"] >= 1


# ── Knowledge Base Tests ─────────────────────────────────────────────


class TestKnowledgeBase:

    @pytest.fixture
    def kb(self, tmp_path):
        """Create a knowledge base with temp directory."""
        return KnowledgeBase(base_dir=tmp_path)

    def test_creates_directory_structure(self, kb, tmp_path):
        assert (tmp_path / "product").is_dir()
        assert (tmp_path / "customers").is_dir()
        assert (tmp_path / "market").is_dir()
        assert (tmp_path / "operations").is_dir()

    def test_write_and_read_doc(self, kb):
        content = "# Test Document\n\nHello, world!"
        assert kb.write_doc("product/test.md", content) is True

        result = kb.read_doc("product/test.md")
        assert result == content

    def test_read_nonexistent_doc(self, kb):
        assert kb.read_doc("product/nonexistent.md") is None

    def test_append_doc(self, kb):
        kb.write_doc("product/append_test.md", "# Header")
        kb.append_doc("product/append_test.md", "\n## Section 2")

        content = kb.read_doc("product/append_test.md")
        assert "# Header" in content
        assert "## Section 2" in content

    def test_append_creates_new_doc(self, kb):
        kb.append_doc("product/new_append.md", "# New Doc")
        content = kb.read_doc("product/new_append.md")
        assert "# New Doc" in content

    def test_list_docs(self, kb):
        kb.write_doc("product/doc1.md", "Content 1")
        kb.write_doc("product/doc2.md", "Content 2")
        kb.write_doc("market/doc3.md", "Content 3")

        all_docs = kb.list_docs()
        assert len(all_docs) >= 3

        product_docs = kb.list_docs("product")
        assert len(product_docs) >= 2

    def test_search_docs(self, kb):
        kb.write_doc("product/searchable.md", "This document contains UNIQUE_KEYWORD_XYZ here")
        kb.write_doc("market/other.md", "This doc has different content")

        results = kb.search_docs("UNIQUE_KEYWORD_XYZ")
        assert len(results) == 1
        assert "searchable.md" in results[0]["path"]
        assert "UNIQUE_KEYWORD_XYZ" in results[0]["snippet"]

    def test_search_case_insensitive(self, kb):
        kb.write_doc("product/case.md", "Contains MiXeD CaSe Content")
        results = kb.search_docs("mixed case")
        assert len(results) == 1

    def test_write_creates_subdirs(self, kb):
        kb.write_doc("product/nested/deep/file.md", "Deep content")
        assert kb.read_doc("product/nested/deep/file.md") == "Deep content"

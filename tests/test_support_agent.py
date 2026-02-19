"""
Tests for the Support Agent.

LLM calls and vector store lookups are mocked throughout.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.base import AgentConfig, Priority, TriggerType
from agents.support import SupportAgent, FAQ_PATH, HIGH_CONFIDENCE, LOW_CONFIDENCE


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_faq_path(tmp_path):
    """Return a temporary FAQ path and patch the module-level constant."""
    return tmp_path / "faq.json"


@pytest.fixture()
def agent(tmp_path, monkeypatch):
    """Return a SupportAgent with all file I/O redirected to tmp_path."""
    faq = tmp_path / "faq.json"
    monkeypatch.setattr("agents.support.FAQ_PATH", faq)
    monkeypatch.setattr("agents.memory.operational.STATE_DIR", tmp_path / "state")
    config = AgentConfig(name="support", dry_run=True)
    return SupportAgent(config=config)


# ── Init / defaults ───────────────────────────────────────────────────


class TestSupportAgentInit:

    def test_name_and_trigger(self, agent):
        assert agent.name == "support"
        assert agent.trigger_type == TriggerType.EVENT

    def test_seeds_default_faq(self, agent):
        """On first run, a default FAQ should be seeded."""
        assert len(agent._faq) >= 5

    def test_faq_file_created(self, tmp_path, agent):
        faq_file = tmp_path / "faq.json"
        assert faq_file.exists()
        data = json.loads(faq_file.read_text())
        assert isinstance(data, list)
        assert len(data) >= 5


# ── FAQ search ────────────────────────────────────────────────────────


class TestSearchFaq:

    def test_exact_match_scores_high(self, agent):
        # Seed a known entry
        agent._faq = [{"question": "How do I install Minerva?", "answer": "pip install ..."}]
        entry, score = agent._search_faq("How do I install Minerva?")
        assert score > HIGH_CONFIDENCE
        assert entry["answer"] == "pip install ..."

    def test_fuzzy_match_returns_something(self, agent):
        agent._faq = [{"question": "What file formats does Minerva support?", "answer": "CSV, Excel"}]
        entry, score = agent._search_faq("which formats can I upload")
        assert entry is not None
        assert 0 <= score <= 1.0

    def test_empty_faq_returns_zero(self, agent):
        agent._faq = []
        entry, score = agent._search_faq("anything")
        assert entry is None
        assert score == 0.0

    def test_keyword_overlap_boosts_score(self, agent):
        agent._faq = [{"question": "How to install local LLM model?", "answer": "Place GGUF in ..."}]
        _, score_related = agent._search_faq("install local model LLM")
        _, score_unrelated = agent._search_faq("why is my CSV broken")
        assert score_related > score_unrelated


# ── Vector store search ───────────────────────────────────────────────


class TestSearchVectorStore:

    def test_returns_zero_when_unavailable(self, agent):
        # VectorStore is imported locally inside the method — patch at source
        with patch("learning.vector_store.VectorStore", side_effect=ImportError):
            text, score = agent._search_vector_store("some question")
        assert score == 0.0
        assert text == ""

    def test_returns_zero_on_exception(self, agent):
        with patch("learning.vector_store.VectorStore") as MockVS:
            MockVS.return_value.search.side_effect = RuntimeError("db error")
            text, score = agent._search_vector_store("something")
        assert score == 0.0

    def test_returns_result_when_available(self, agent):
        mock_vs = MagicMock()
        mock_vs.search.return_value = [{"similarity": 0.82, "intent": "plot a histogram"}]
        with patch("learning.vector_store.VectorStore", return_value=mock_vs):
            text, score = agent._search_vector_store("make a histogram")
        assert score == pytest.approx(0.82)
        assert "histogram" in text


# ── Simple embedding ──────────────────────────────────────────────────


class TestSimpleEmbedding:

    def test_returns_unit_vector(self, agent):
        import numpy as np
        vec = agent._simple_embedding("show me a bar chart")
        assert vec.shape == (256,)
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5 or norm == pytest.approx(0.0)

    def test_different_texts_differ(self, agent):
        import numpy as np
        v1 = agent._simple_embedding("install Minerva")
        v2 = agent._simple_embedding("purple elephant dancing")
        assert not np.allclose(v1, v2)


# ── Response generation ───────────────────────────────────────────────


class TestGenerateResponse:

    def test_uses_llm_when_available(self, agent):
        with patch("llm_manager.llm_interface.get_llm_completion", return_value="Sure, here's how."):
            resp = agent._generate_response("How do I install?", "pip install minerva")
        assert "Sure, here's how." in resp

    def test_fallback_returns_faq_answer(self, agent):
        with patch("llm_manager.llm_interface.get_llm_completion", return_value=""):
            resp = agent._generate_response("How do I install?", "pip install minerva")
        assert "pip install minerva" in resp
        assert "Minerva Support Bot" in resp


# ── FAQ management ────────────────────────────────────────────────────


class TestFaqManagement:

    def test_add_entry_persists(self, agent, tmp_path):
        initial_count = len(agent._faq)
        result = agent.add_faq_entry(
            "What is Minerva?",
            "Minerva is a local-first AI data analyst.",
            tags=["general"],
        )
        assert result is True
        assert len(agent._faq) == initial_count + 1
        assert agent._faq[-1]["question"] == "What is Minerva?"

    def test_add_entry_saved_to_file(self, agent, tmp_path):
        agent.add_faq_entry("Q?", "A.", tags=["test"])
        faq_file = tmp_path / "faq.json"
        data = json.loads(faq_file.read_text())
        questions = [e["question"] for e in data]
        assert "Q?" in questions

    def test_get_faq_returns_list(self, agent):
        faq = agent.get_faq()
        assert isinstance(faq, list)
        assert len(faq) >= 5

    def test_load_existing_faq(self, tmp_path, monkeypatch):
        """Agent should load an existing valid FAQ file."""
        faq_file = tmp_path / "faq.json"
        entries = [{"question": "Loaded Q?", "answer": "Loaded A.", "tags": [], "added_at": "2026-01-01"}]
        faq_file.write_text(json.dumps(entries))
        monkeypatch.setattr("agents.support.FAQ_PATH", faq_file)
        monkeypatch.setattr("agents.memory.operational.STATE_DIR", tmp_path / "state")
        a = SupportAgent(AgentConfig(name="support", dry_run=True))
        assert any(e["question"] == "Loaded Q?" for e in a._faq)


# ── Full run() ────────────────────────────────────────────────────────


class TestSupportAgentRun:

    def test_run_no_question(self, agent):
        result = agent.run()
        assert result.success is True
        assert "No question" in result.summary

    def test_run_high_confidence_produces_fyi(self, agent):
        """A question closely matching a FAQ entry should produce a FYI escalation."""
        agent._faq = [{
            "question": "How do I install Minerva?",
            "answer": "pip install -r requirements.txt",
        }]
        with (
            patch.object(agent, "_search_vector_store", return_value=("", 0.0)),
            patch("llm_manager.llm_interface.get_llm_completion", return_value="Here's how to install."),
        ):
            result = agent.run(question="How do I install Minerva?")

        assert result.success is True
        fyi_escs = [e for e in result.escalations if e.priority == Priority.FYI]
        assert len(fyi_escs) == 1
        assert result.metrics.get("auto_responded") is True

    def test_run_low_confidence_escalates_review(self, agent):
        """An unknown question should produce a REVIEW escalation."""
        agent._faq = []
        with patch.object(agent, "_search_vector_store", return_value=("", 0.0)):
            result = agent.run(question="What is the airspeed velocity of an unladen swallow?")

        assert result.success is True
        review_escs = [e for e in result.escalations if e.priority == Priority.REVIEW]
        assert len(review_escs) == 1
        assert result.metrics.get("auto_responded") is False

    def test_run_with_issue_number(self, agent):
        """issue_number should appear in escalation title."""
        agent._faq = []
        with patch.object(agent, "_search_vector_store", return_value=("", 0.0)):
            result = agent.run(question="Unknown thing", issue_number=42)
        assert result.success is True
        assert any("#42" in e.title for e in result.escalations)

    def test_run_metrics_populated(self, agent):
        with (
            patch.object(agent, "_search_faq", return_value=(None, 0.1)),
            patch.object(agent, "_search_vector_store", return_value=("", 0.05)),
        ):
            result = agent.run(question="test question")
        assert "faq_score" in result.metrics
        assert "vs_score" in result.metrics
        assert "combined_score" in result.metrics

    def test_run_handles_exception(self, agent, monkeypatch):
        monkeypatch.setattr(agent, "_search_faq", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        result = agent.run(question="Will this blow up?")
        assert result.success is False
        assert "boom" in result.error

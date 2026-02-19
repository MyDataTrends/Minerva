"""
Support Agent — Intelligent FAQ-backed issue responder.

Extends the Advocate's capabilities: maintains a structured FAQ knowledge base,
performs fuzzy string matching + vector-store lookup, and either auto-drafts a
response (FYI) or escalates for human review.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.base import AgentConfig, AgentResult, BaseAgent, Priority, TriggerType
from agents.memory.operational import OperationalMemory
from utils.logging import get_logger

logger = get_logger(__name__)

# Where the FAQ JSON lives
FAQ_PATH = Path(__file__).parent / "knowledge_base" / "faq.json"
# Confidence thresholds
HIGH_CONFIDENCE = 0.65   # auto-draft response
LOW_CONFIDENCE = 0.35    # escalate to human


class SupportAgent(BaseAgent):
    """
    Support automation agent.

    Workflow:
    1. Receive a question or issue body.
    2. Search the FAQ for high-confidence matches (fuzzy string similarity).
    3. Search the vector store for semantically similar past interactions.
    4. If combined confidence ≥ HIGH_CONFIDENCE → draft response (Priority.FYI).
    5. If confidence < HIGH_CONFIDENCE → escalate (Priority.REVIEW).
    6. Expose `add_faq_entry()` for adding resolved-issue answers to the FAQ.
    """

    name = "support"
    trigger_type = TriggerType.EVENT

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self.memory = OperationalMemory("support")
        FAQ_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._faq: List[Dict[str, str]] = self._load_faq()

    # ── Public entry point ────────────────────────────────────────────

    def run(self, **kwargs) -> AgentResult:
        """
        Handle a support question or issue.

        kwargs:
            question (str): The user's question or issue body.
            issue_number (int, optional): GitHub issue number for reference.
            title (str, optional): Issue/question title for context.
        """
        start = time.time()
        result = self._make_result()

        question: str = kwargs.get("question", "")
        title: str = kwargs.get("title", "")
        issue_num: Optional[int] = kwargs.get("issue_number")

        if not question and not title:
            result.success = True
            result.summary = "No question provided — nothing to support."
            result.add_action("skipped_no_question")
            result.duration_seconds = time.time() - start
            self.memory.log_run(
                result.success, result.duration_seconds, result.summary,
                len(result.actions_taken), len(result.escalations), result.error,
            )
            return result

        full_text = f"{title} {question}".strip()
        result.add_action("received_question")
        result.metrics["question_length"] = len(full_text)

        try:
            # 1. Search FAQ
            faq_match, faq_score = self._search_faq(full_text)
            result.add_action(f"faq_search_score:{faq_score:.2f}")
            result.metrics["faq_score"] = round(faq_score, 3)

            # 2. Search vector store
            vs_match, vs_score = self._search_vector_store(full_text)
            result.add_action(f"vector_store_score:{vs_score:.2f}")
            result.metrics["vs_score"] = round(vs_score, 3)

            # 3. Combine scores (FAQ weighted higher — curated ground truth)
            combined_score = max(faq_score * 0.7 + vs_score * 0.3, faq_score, vs_score * 0.5)
            result.metrics["combined_score"] = round(combined_score, 3)

            # 4. Choose best source material for the response draft
            best_answer = (faq_match or {}).get("answer", "") if faq_score >= vs_score else vs_match
            best_question = (faq_match or {}).get("question", full_text) if faq_score >= vs_score else full_text

            if combined_score >= HIGH_CONFIDENCE and best_answer:
                # High-confidence: draft a response
                draft = self._generate_response(full_text, best_answer)
                issue_ref = f"#{issue_num} " if issue_num else ""
                result.add_escalation(
                    Priority.FYI,
                    f"Support auto-response drafted {issue_ref}(confidence: {combined_score:.0%})",
                    f"Q: {full_text[:200]}\n\nMatched: {best_question[:120]}\n\nDraft response:\n{draft}",
                )
                result.add_action("drafted_auto_response")
                result.metrics["auto_responded"] = True

            else:
                # Low-confidence: escalate for human review
                hint = f"Closest FAQ match: {best_question[:120]}\nAnswer hint: {best_answer[:200]}" if best_answer else "No close FAQ match found."
                issue_ref = f"#{issue_num} " if issue_num else ""
                result.add_escalation(
                    Priority.REVIEW,
                    f"Support question needs human review {issue_ref}(confidence: {combined_score:.0%})",
                    f"Q: {full_text[:300]}\n\n{hint}",
                )
                result.add_action("escalated_for_review")
                result.metrics["auto_responded"] = False

            self.memory.log_action(
                "support_query",
                f"score={combined_score:.2f} issue={issue_num or 'n/a'}",
            )
            result.success = True
            result.summary = (
                f"Support query processed. Combined confidence: {combined_score:.0%}. "
                f"{'Auto-drafted response.' if result.metrics.get('auto_responded') else 'Escalated for review.'}"
            )

        except Exception as exc:
            result.success = False
            result.error = str(exc)
            result.summary = f"Support agent failed: {exc}"
            logger.exception("Support agent run failed")

        result.duration_seconds = time.time() - start
        self.memory.log_run(
            result.success, result.duration_seconds, result.summary,
            len(result.actions_taken), len(result.escalations), result.error,
        )
        return result

    # ── FAQ management ────────────────────────────────────────────────

    def add_faq_entry(self, question: str, answer: str, tags: Optional[List[str]] = None) -> bool:
        """
        Add a new entry to the FAQ knowledge base.

        Call this from resolved GitHub issues or manually curated Q&A pairs.

        Args:
            question: The canonical question text.
            answer: The authoritative answer.
            tags: Optional list of topic tags (e.g. ["installation", "llm"]).

        Returns:
            True if saved successfully.
        """
        entry: Dict[str, Any] = {
            "question": question.strip(),
            "answer": answer.strip(),
            "tags": tags or [],
            "added_at": datetime.utcnow().isoformat(),
        }
        self._faq.append(entry)
        return self._save_faq()

    def get_faq(self) -> List[Dict[str, Any]]:
        """Return the full in-memory FAQ list."""
        return list(self._faq)

    # ── Search helpers ────────────────────────────────────────────────

    def _search_faq(self, query: str) -> Tuple[Optional[Dict], float]:
        """
        Fuzzy-search the FAQ for the best matching entry.

        Returns (best_entry | None, best_score 0-1).
        """
        if not self._faq:
            return None, 0.0

        query_lower = query.lower()
        best_entry: Optional[Dict] = None
        best_score = 0.0

        for entry in self._faq:
            candidate = entry.get("question", "").lower()
            # SequenceMatcher ratio
            ratio = SequenceMatcher(None, query_lower, candidate).ratio()
            # Also boost if significant words overlap
            boost = self._keyword_overlap(query_lower, candidate)
            score = min(1.0, ratio + boost * 0.2)
            if score > best_score:
                best_score = score
                best_entry = entry

        return best_entry, best_score

    def _keyword_overlap(self, a: str, b: str) -> float:
        """Return fraction of non-trivial words from `a` that appear in `b`."""
        stopwords = {"the", "a", "an", "is", "it", "to", "i", "how", "do", "can", "my", "and", "or", "not"}
        words_a = set(w for w in a.split() if w not in stopwords and len(w) > 2)
        words_b = set(w for w in b.split() if w not in stopwords and len(w) > 2)
        if not words_a:
            return 0.0
        return len(words_a & words_b) / len(words_a)

    def _search_vector_store(self, query: str) -> Tuple[str, float]:
        """
        Search the vector store for semantically similar past interactions.

        Returns (best_matching_intent_text, confidence_score).
        Falls back gracefully if the vector store is unavailable.
        """
        try:
            from learning.vector_store import VectorStore
            import numpy as np

            vs = VectorStore()
            # Create a simple TF-IDF-style embedding via hashing trick
            embedding = self._simple_embedding(query)
            results = vs.search(embedding, top_k=1)
            if results:
                top = results[0]
                score = float(top.get("similarity", 0.0))
                intent = top.get("intent", "")
                return intent, score
        except Exception as exc:
            logger.debug("Vector store search skipped: %s", exc)

        return "", 0.0

    def _simple_embedding(self, text: str):
        """
        Minimal bag-of-words embedding (256-dim) for vector store queries.

        The real system uses proper sentence embeddings; this is a safe fallback
        when no embedding model is loaded, giving the vector store something to
        work with for cosine similarity.
        """
        import numpy as np
        import hashlib

        vec = [0.0] * 256
        for word in text.lower().split():
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % 256
            vec[idx] += 1.0
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr /= norm
        return arr

    # ── Response generation ───────────────────────────────────────────

    def _generate_response(self, question: str, faq_answer: str) -> str:
        """Use the LLM to personalize a FAQ answer for the specific question."""
        from llm_manager.llm_interface import get_llm_completion
        prompt = (
            "You are a helpful support bot for Minerva, an AI-powered local data analysis tool.\n\n"
            f"User question: {question[:400]}\n\n"
            f"Reference answer from knowledge base:\n{faq_answer[:600]}\n\n"
            "Write a friendly, concise support response (≤150 words) using the reference answer. "
            "Personalize it to the user's specific question. "
            "End with: — Minerva Support Bot"
        )
        response = get_llm_completion(prompt, max_tokens=300, temperature=0.5)
        if response:
            return response.strip()

        # Deterministic fallback: return the FAQ answer verbatim
        return faq_answer + "\n\n— Minerva Support Bot"

    # ── FAQ persistence ───────────────────────────────────────────────

    def _load_faq(self) -> List[Dict[str, Any]]:
        """Load FAQ from JSON file, returning empty list if missing."""
        if not FAQ_PATH.exists():
            return self._seed_default_faq()
        try:
            data = json.loads(FAQ_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            logger.warning("FAQ file has unexpected structure, resetting.")
            return self._seed_default_faq()
        except Exception as exc:
            logger.error("Failed to load FAQ: %s", exc)
            return []

    def _save_faq(self) -> bool:
        """Persist the in-memory FAQ to disk."""
        try:
            FAQ_PATH.write_text(
                json.dumps(self._faq, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("FAQ saved (%d entries)", len(self._faq))
            return True
        except Exception as exc:
            logger.error("Failed to save FAQ: %s", exc)
            return False

    def _seed_default_faq(self) -> List[Dict[str, Any]]:
        """Create and persist a starter FAQ with common Minerva questions."""
        entries: List[Dict[str, Any]] = [
            {
                "question": "How do I install Minerva?",
                "answer": (
                    "Clone the repo and install dependencies with `pip install -r requirements.txt`. "
                    "Then run `streamlit run ui/dashboard.py` to launch the dashboard. "
                    "See the README for full setup instructions including optional LLM model download."
                ),
                "tags": ["installation", "setup"],
                "added_at": datetime.utcnow().isoformat(),
            },
            {
                "question": "What file formats does Minerva support?",
                "answer": (
                    "Minerva supports CSV, Excel (.xlsx/.xls), and Parquet files. "
                    "Upload via the dashboard or provide a path via the CLI."
                ),
                "tags": ["data", "formats"],
                "added_at": datetime.utcnow().isoformat(),
            },
            {
                "question": "Does Minerva require an internet connection or API key?",
                "answer": (
                    "No. Minerva is local-first and runs fully offline with a local GGUF model. "
                    "An Anthropic API key is optional — it enables cloud LLM features but is not required "
                    "for core data analysis functionality."
                ),
                "tags": ["privacy", "offline", "api"],
                "added_at": datetime.utcnow().isoformat(),
            },
            {
                "question": "How do I add a local LLM model?",
                "answer": (
                    "Place a GGUF model file in `adm/llm_backends/local_model/`. "
                    "Minerva auto-discovers and registers GGUF files at startup. "
                    "You can also configure a model via the LLM Manager UI tab."
                ),
                "tags": ["llm", "model", "setup"],
                "added_at": datetime.utcnow().isoformat(),
            },
            {
                "question": "Why is my CSV not loading?",
                "answer": (
                    "Check that the file is UTF-8 or Latin-1 encoded and has a header row. "
                    "Very large files (>500 MB) are automatically chunked. "
                    "If you see a parsing error, try opening the file in a text editor to inspect delimiters."
                ),
                "tags": ["csv", "error", "data"],
                "added_at": datetime.utcnow().isoformat(),
            },
            {
                "question": "How do I report a bug or request a feature?",
                "answer": (
                    "Open an issue on GitHub at https://github.com/rme0722/Minerva/issues. "
                    "For bugs, include your OS, Python version, and a minimal reproducer. "
                    "For features, describe the use case and expected behavior."
                ),
                "tags": ["bug", "feature", "github"],
                "added_at": datetime.utcnow().isoformat(),
            },
            {
                "question": "What machine learning models does Minerva use?",
                "answer": (
                    "Minerva runs an automated model sweep over LightGBM, XGBoost, Ridge Regression, "
                    "and Logistic Regression. The best-scoring model (by cross-validated metric) is "
                    "selected automatically. SHAP explanations are computed for the winner."
                ),
                "tags": ["modeling", "ml", "algorithms"],
                "added_at": datetime.utcnow().isoformat(),
            },
        ]
        # Persist defaults
        try:
            FAQ_PATH.write_text(
                json.dumps(entries, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
        return entries

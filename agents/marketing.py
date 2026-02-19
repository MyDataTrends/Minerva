"""
Marketing Agent — Changelog summarizer and social media draft generator.

Reads recent git commits, picks up new Productizer outputs, and drafts
platform-specific posts (HN, Reddit, Twitter/X) for human review.
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from agents.base import AgentConfig, AgentResult, BaseAgent, Priority, TriggerType
from agents.memory.operational import OperationalMemory
from utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIGEST_DIR = Path(__file__).parent / "digests" / "marketing"
KB_PRODUCT_DIR = Path(__file__).parent / "knowledge_base" / "product"


class MarketingAgent(BaseAgent):
    """
    Marketing automation agent.

    Workflow:
    1. Pull recent git log (last 7 days) — summarize into a human-readable changelog.
    2. Read new Productizer sales kit outputs from the knowledge base.
    3. Generate platform-specific social media drafts (HN, Reddit, Twitter/X).
    4. Save drafts to agents/digests/marketing/ for human review.
    5. Escalate each draft as Priority.REVIEW so the Conductor surfaces them.
    """

    name = "marketing"
    trigger_type = TriggerType.CRON

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self.memory = OperationalMemory("marketing")
        DIGEST_DIR.mkdir(parents=True, exist_ok=True)
        KB_PRODUCT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public entry point ────────────────────────────────────────────

    def run(self, **kwargs) -> AgentResult:
        """Execute the marketing draft generation workflow."""
        start = time.time()
        result = self._make_result()

        try:
            days = int(kwargs.get("days", 7))

            # 1. Build changelog from git log
            commits = self._get_recent_commits(days)
            result.add_action(f"fetched_{len(commits)}_commits")
            result.metrics["commits_last_7d"] = len(commits)

            changelog_summary = self._summarize_changelog(commits)
            result.add_action("generated_changelog_summary")

            # 2. Read Productizer outputs
            productizer_docs = self._read_productizer_docs()
            result.add_action(f"loaded_{len(productizer_docs)}_productizer_docs")
            result.metrics["productizer_docs"] = len(productizer_docs)

            if not changelog_summary and not productizer_docs:
                result.success = True
                result.summary = "Nothing new to market this cycle — no commits or productizer docs."
                result.add_action("skipped_no_content")
                result.duration_seconds = time.time() - start
                self.memory.log_run(
                    result.success, result.duration_seconds, result.summary,
                    len(result.actions_taken), len(result.escalations), result.error,
                )
                return result

            # 3. Generate social media drafts
            drafts = self._generate_drafts(changelog_summary, productizer_docs)
            result.add_action(f"generated_{len(drafts)}_drafts")
            result.metrics["drafts_generated"] = len(drafts)

            # 4. Save drafts and escalate for review
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            saved_paths = []
            for platform, content in drafts.items():
                if not content:
                    continue
                path = self._save_draft(platform, content, date_str)
                saved_paths.append(str(path))
                result.add_action(f"saved_draft:{platform}")

                result.add_escalation(
                    Priority.REVIEW,
                    f"Marketing draft ready: {platform} ({date_str})",
                    f"Review and edit before posting:\n\n{content[:400]}…\n\nFull draft: {path}",
                )
                self.memory.log_action("draft_created", f"{platform}:{path}")

            result.success = True
            result.summary = (
                f"Generated {len(drafts)} social media drafts from "
                f"{len(commits)} commits and {len(productizer_docs)} productizer docs. "
                f"Saved to {DIGEST_DIR.name}/."
            )
            result.metrics["saved_paths"] = saved_paths

        except Exception as exc:
            result.success = False
            result.error = str(exc)
            result.summary = f"Marketing agent failed: {exc}"
            logger.exception("Marketing agent run failed")

        result.duration_seconds = time.time() - start
        self.memory.log_run(
            result.success, result.duration_seconds, result.summary,
            len(result.actions_taken), len(result.escalations), result.error,
        )
        return result

    # ── Git helpers ───────────────────────────────────────────────────

    def _get_recent_commits(self, days: int = 7) -> List[str]:
        """Return commit subjects from the last N days."""
        since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        try:
            output = subprocess.check_output(
                ["git", "log", f"--since={since}", "--pretty=format:%s", "--no-merges"],
                cwd=str(PROJECT_ROOT),
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
            lines = [l.strip() for l in output.decode("utf-8", errors="replace").splitlines() if l.strip()]
            return lines
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("git log unavailable, returning empty commit list")
            return []

    def _summarize_changelog(self, commits: List[str]) -> str:
        """Use LLM to produce a user-facing changelog summary, or fall back to a bullet list."""
        if not commits:
            return ""

        bullet_list = "\n".join(f"- {c}" for c in commits[:40])

        from llm_manager.llm_interface import get_llm_completion
        prompt = (
            "You are a developer advocate writing release notes for Assay, "
            "an AI-powered local data analysis tool.\n\n"
            "Summarize the following raw git commit messages into a concise, "
            "user-friendly changelog (3-6 bullet points, no jargon, present tense). "
            "Focus on user value, not implementation details. "
            "Omit minor/internal changes.\n\n"
            f"Raw commits:\n{bullet_list}\n\n"
            "Changelog:"
        )
        summary = get_llm_completion(prompt, max_tokens=400, temperature=0.5)
        if summary:
            return summary.strip()

        # Deterministic fallback: return the bullet list directly
        return bullet_list

    # ── Knowledge base helpers ────────────────────────────────────────

    def _read_productizer_docs(self) -> List[str]:
        """Read markdown content from the productizer sales kit directory."""
        sales_kit_dir = KB_PRODUCT_DIR / "sales_kits"
        if not sales_kit_dir.exists():
            return []

        docs = []
        cutoff = datetime.utcnow() - timedelta(days=30)  # last 30 days

        for md_file in sorted(sales_kit_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                mtime = datetime.utcfromtimestamp(md_file.stat().st_mtime)
                if mtime < cutoff:
                    continue
                content = md_file.read_text(encoding="utf-8")
                docs.append(content[:2000])  # cap per-doc to avoid token blow-up
            except Exception as exc:
                logger.debug("Could not read %s: %s", md_file, exc)

        return docs[:5]  # at most 5 docs

    # ── Draft generation ──────────────────────────────────────────────

    def _generate_drafts(
        self,
        changelog: str,
        productizer_docs: List[str],
    ) -> dict:
        """Generate platform-specific post drafts. Returns {platform: content}."""
        context = self._build_context(changelog, productizer_docs)

        return {
            "hackernews": self._draft_hn(context),
            "reddit": self._draft_reddit(context),
            "twitter": self._draft_twitter(context),
        }

    def _build_context(self, changelog: str, productizer_docs: List[str]) -> str:
        parts = []
        if changelog:
            parts.append(f"CHANGELOG:\n{changelog}")
        if productizer_docs:
            combined = "\n\n---\n\n".join(productizer_docs[:3])
            parts.append(f"VERTICAL SALES KITS (excerpts):\n{combined[:1500]}")
        return "\n\n".join(parts)

    def _draft_hn(self, context: str) -> str:
        """Draft a Hacker News Show HN post."""
        from llm_manager.llm_interface import get_llm_completion
        prompt = (
            "You are writing a 'Show HN' submission for Hacker News.\n\n"
            "Project: Assay — local-first AI data analyst. Upload a CSV, get autonomous "
            "cleaning, profiling, modeling, and insights. Zero config. Runs as Streamlit dashboard, "
            "FastAPI backend, CLI, and MCP server.\n\n"
            f"Context:\n{context}\n\n"
            "Write a Show HN title + body (≤300 words). HN style: plain, honest, technical. "
            "No marketing fluff. Highlight what's new or interesting.\n\n"
            "Format:\n"
            "TITLE: ...\n\n"
            "BODY:\n..."
        )
        draft = get_llm_completion(prompt, max_tokens=500, temperature=0.6)
        if draft:
            return draft.strip()
        # Fallback
        return f"Show HN: Assay — local AI data analyst\n\n{context[:300]}"

    def _draft_reddit(self, context: str) -> str:
        """Draft a Reddit post (r/MachineLearning, r/datascience, or r/programming)."""
        from llm_manager.llm_interface import get_llm_completion
        prompt = (
            "You are writing a Reddit post for r/MachineLearning or r/datascience.\n\n"
            "Project: Assay — local-first AI data analyst. Upload a CSV, get autonomous "
            "cleaning, profiling, modeling (LightGBM/XGBoost/Ridge), SHAP explanations, "
            "and NL insights. No cloud required.\n\n"
            f"Context:\n{context}\n\n"
            "Write a post title and body (≤250 words). Reddit style: conversational, genuine, "
            "invite discussion. Include a brief description and what's new.\n\n"
            "Format:\n"
            "TITLE: ...\n\n"
            "BODY:\n..."
        )
        draft = get_llm_completion(prompt, max_tokens=450, temperature=0.6)
        if draft:
            return draft.strip()
        return f"Assay — autonomous local AI data analyst update\n\n{context[:300]}"

    def _draft_twitter(self, context: str) -> str:
        """Draft a Twitter/X thread (up to 3 tweets)."""
        from llm_manager.llm_interface import get_llm_completion
        prompt = (
            "You are writing a Twitter/X thread (up to 3 tweets) about Assay updates.\n\n"
            "Project: Assay — local-first AI data analyst. Upload CSV → autonomous insights.\n\n"
            f"Context:\n{context}\n\n"
            "Write 2-3 tweets. Keep each under 280 characters. Use relevant hashtags sparingly "
            "(e.g. #OpenSource #DataScience #AI). Make it punchy and share-worthy.\n\n"
            "Format each tweet as:\n"
            "1/ ...\n"
            "2/ ...\n"
            "3/ ..."
        )
        draft = get_llm_completion(prompt, max_tokens=350, temperature=0.7)
        if draft:
            return draft.strip()
        return f"1/ Assay update: {context[:200]}\n\n#OpenSource #DataScience"

    # ── Persistence ───────────────────────────────────────────────────

    def _save_draft(self, platform: str, content: str, date_str: str) -> Path:
        """Write a draft markdown file and return its path."""
        filename = f"{date_str}_{platform}.md"
        path = DIGEST_DIR / filename
        header = (
            f"# Marketing Draft — {platform.title()} ({date_str})\n\n"
            f"> Auto-generated by MarketingAgent. Review and edit before posting.\n\n"
            "---\n\n"
        )
        path.write_text(header + content + "\n", encoding="utf-8")
        logger.info("Saved marketing draft: %s", path)
        return path

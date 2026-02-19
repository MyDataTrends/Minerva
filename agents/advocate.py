"""
Advocate Agent — GitHub community management.

Monitors issues, classifies them using LLM, auto-labels,
and generates responses for common questions.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from agents.base import AgentConfig, AgentResult, BaseAgent, Priority, TriggerType
from agents.memory.operational import OperationalMemory
from utils.logging import get_logger

logger = get_logger(__name__)


# Issue classification categories
ISSUE_TYPES = {
    "bug": {"labels": ["bug"], "emoji": "🐛"},
    "feature": {"labels": ["enhancement"], "emoji": "✨"},
    "question": {"labels": ["question"], "emoji": "❓"},
    "docs": {"labels": ["documentation"], "emoji": "📚"},
    "security": {"labels": ["security"], "emoji": "🔒"},
}


class AdvocateAgent(BaseAgent):
    """
    Community management agent.

    Workflow:
    1. Fetch new issues from GitHub
    2. Classify each issue using LLM (bug/feature/question/docs)
    3. Auto-label via GitHub API
    4. Generate responses for questions
    5. Flag stale issues
    """

    name = "advocate"
    trigger_type = TriggerType.CRON

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self.memory = OperationalMemory("advocate")
        self.stale_days = (config.extra or {}).get("stale_days", 14) if config else 14

    def run(self, **kwargs) -> AgentResult:
        """Execute the issue triage workflow."""
        start = time.time()
        result = self._make_result()

        try:
            from agents.tools.github_client import GitHubClient
            client = GitHubClient()

            if not client.is_available:
                result.success = True
                result.summary = "GitHub client unavailable — skipped issue triage."
                result.add_action("skipped_no_github")
                return result

            # 1. Fetch open issues
            issues = client.list_issues(state="open")
            result.add_action(f"fetched_{len(issues)}_issues")
            result.metrics["open_issues"] = len(issues)

            # 2. Process each issue
            classified = 0
            stale_flagged = 0

            for issue in issues:
                # Skip pull requests (GitHub API returns them as issues too)
                if "pull_request" in issue:
                    continue

                issue_num = issue["number"]
                title = issue.get("title", "")
                body = issue.get("body", "")
                labels = [l["name"] for l in issue.get("labels", [])]
                updated_at = issue.get("updated_at", "")

                # Skip already-labeled issues (already processed)
                agent_labels = {"bug", "enhancement", "question", "documentation", "security"}
                if any(l in agent_labels for l in labels):
                    continue

                # Classify
                issue_type = self._classify_issue(title, body)
                classified += 1

                # Apply labels
                type_info = ISSUE_TYPES.get(issue_type, ISSUE_TYPES["question"])
                if not self.is_dry_run:
                    client.add_labels(issue_num, type_info["labels"])
                    result.add_action(f"labeled:#{issue_num}:{issue_type}")
                else:
                    result.add_action(f"[dry-run]would_label:#{issue_num}:{issue_type}")

                # Generate response for questions
                if issue_type == "question":
                    response = self._generate_response(title, body)
                    if response and not self.is_dry_run:
                        client.create_comment(issue_num, response)
                        result.add_action(f"responded:#{issue_num}")
                    elif response:
                        result.add_action(f"[dry-run]would_respond:#{issue_num}")

                # Escalate bugs and security issues
                if issue_type == "bug":
                    result.add_escalation(
                        Priority.REVIEW,
                        f"Bug reported: #{issue_num} — {title[:60]}",
                        f"Issue #{issue_num}: {title}\n{body[:200]}",
                    )
                elif issue_type == "security":
                    result.add_escalation(
                        Priority.URGENT,
                        f"Security issue: #{issue_num} — {title[:60]}",
                        f"Issue #{issue_num}: {title}\n{body[:200]}",
                    )

                self.memory.log_action("classified_issue", f"#{issue_num}: {issue_type}")

            # 3. Check for stale issues
            stale_flagged = self._check_stale_issues(client, issues, result)

            client.close()

            result.success = True
            result.summary = (
                f"Processed {len(issues)} issues. "
                f"Classified {classified} new. "
                f"Flagged {stale_flagged} stale."
            )
            result.metrics["classified"] = classified
            result.metrics["stale_flagged"] = stale_flagged

        except Exception as exc:
            result.success = False
            result.error = str(exc)
            result.summary = f"Issue triage failed: {exc}"
            logger.exception("Advocate run failed")

        result.duration_seconds = time.time() - start
        self.memory.log_run(
            result.success, result.duration_seconds, result.summary,
            len(result.actions_taken), len(result.escalations), result.error,
        )
        return result

    def _classify_issue(self, title: str, body: str) -> str:
        """Classify an issue using LLM or keyword heuristics."""
        # Try LLM classification first
        try:
            from llm_manager.llm_interface import get_llm_chat, is_llm_available
            if is_llm_available():
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a GitHub issue classifier. Classify the issue into exactly one category: "
                            "bug, feature, question, docs, security. Respond with ONLY the category name, "
                            "nothing else."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Title: {title}\n\nBody: {body[:500]}",
                    },
                ]
                response = get_llm_chat(messages, max_tokens=10, temperature=0.1)
                category = response.strip().lower()
                if category in ISSUE_TYPES:
                    return category
        except Exception as exc:
            logger.debug("LLM classification failed, using heuristics: %s", exc)

        # Fallback: keyword heuristics
        text = (title + " " + body).lower()
        if any(w in text for w in ["bug", "error", "crash", "broken", "fail", "traceback"]):
            return "bug"
        elif any(w in text for w in ["security", "vulnerability", "cve", "exploit"]):
            return "security"
        elif any(w in text for w in ["feature", "request", "add", "would be nice", "suggestion"]):
            return "feature"
        elif any(w in text for w in ["docs", "documentation", "readme", "example"]):
            return "docs"
        else:
            return "question"

    def _generate_response(self, title: str, body: str) -> Optional[str]:
        """Generate a helpful response for a question-type issue."""
        try:
            from llm_manager.llm_interface import get_llm_chat, is_llm_available
            if not is_llm_available():
                return None

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful project maintainer for Assay, an autonomous data analysis tool. "
                        "Write a brief, friendly response to this user question. Keep it under 200 words. "
                        "If you're unsure about something, say so honestly. "
                        "Sign off with: — Assay Bot 🤖"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Title: {title}\n\nBody: {body[:500]}",
                },
            ]
            return get_llm_chat(messages, max_tokens=300, temperature=0.7)
        except Exception as exc:
            logger.debug("Response generation failed: %s", exc)
            return None

    def _check_stale_issues(self, client, issues: List[Dict], result: AgentResult) -> int:
        """Flag issues with no activity for stale_days."""
        stale_count = 0
        cutoff = datetime.utcnow() - timedelta(days=self.stale_days)

        for issue in issues:
            if "pull_request" in issue:
                continue

            updated_str = issue.get("updated_at", "")
            labels = [l["name"] for l in issue.get("labels", [])]

            if "stale" in labels:
                continue

            try:
                updated_at = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                if updated_at.replace(tzinfo=None) < cutoff:
                    stale_count += 1
                    issue_num = issue["number"]

                    if not self.is_dry_run:
                        client.add_labels(issue_num, ["stale"])
                        client.create_comment(
                            issue_num,
                            f"This issue has had no activity for {self.stale_days} days. "
                            "It will be closed in 7 days if no further activity occurs.\n\n"
                            "— Assay Bot 🤖"
                        )
                    result.add_action(f"stale_flagged:#{issue_num}")
            except (ValueError, TypeError):
                continue

        return stale_count

"""
Conductor Agent — Central orchestrator for the Minerva agent system.

Runs daily: scans inputs, delegates to other agents, compiles digests.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from agents.base import AgentConfig, AgentResult, BaseAgent, Priority, TriggerType
from agents.config import load_agent_configs
from agents.memory.operational import OperationalMemory
from utils.logging import get_logger

logger = get_logger(__name__)

DIGEST_DIR = Path(__file__).parent / "digests"


class ConductorAgent(BaseAgent):
    """
    Central orchestrator that runs daily:
    1. Scans GitHub, filesystem, and other inputs
    2. Delegates tasks to sub-agents
    3. Compiles a daily digest with 🔴/🟡/🟢/📊 sections
    4. Delivers digest to file (and optionally email)
    """

    name = "conductor"
    trigger_type = TriggerType.CRON

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self.memory = OperationalMemory("conductor")

    def run(self, **kwargs) -> AgentResult:
        """Execute the daily orchestration workflow."""
        start = time.time()
        result = self._make_result()
        sub_results: List[AgentResult] = []

        try:
            # 1. Gather context
            repo_info = self._scan_github()
            result.metrics.update(repo_info)
            result.add_action("scanned_github")

            # 2. Run enabled sub-agents
            sub_results = self._run_sub_agents()
            result.add_action(f"ran_{len(sub_results)}_sub_agents")

            # 3. Compile digest
            digest = self._compile_digest(sub_results, repo_info)
            result.add_action("compiled_digest")

            # 4. Deliver
            digest_path = self._deliver_digest(digest)
            result.add_action(f"digest_saved:{digest_path}")

            # 5. Log to memory
            self.memory.log_action("daily_run", f"Ran {len(sub_results)} agents")

            # Aggregate escalations from sub-agents
            for sr in sub_results:
                result.escalations.extend(sr.escalations)

            result.success = True
            result.summary = f"Daily run complete. {len(sub_results)} agents ran, {len(result.escalations)} escalations."

        except Exception as exc:
            result.success = False
            result.error = str(exc)
            result.summary = f"Daily run failed: {exc}"
            logger.exception("Conductor run failed")

        result.duration_seconds = time.time() - start
        self.memory.log_run(
            result.success, result.duration_seconds, result.summary,
            len(result.actions_taken), len(result.escalations), result.error,
        )
        return result

    def _scan_github(self) -> Dict:
        """Scan GitHub for repo stats. Returns metrics dict."""
        try:
            from agents.tools.github_client import GitHubClient
            client = GitHubClient()
            if not client.is_available:
                return {"github": "unavailable"}

            info = client.get_repo_info() or {}
            issues = client.list_issues(state="open")
            prs = client.list_pull_requests(state="open")
            client.close()

            return {
                "stars": info.get("stargazers_count", 0),
                "forks": info.get("forks_count", 0),
                "open_issues": len(issues),
                "open_prs": len(prs),
            }
        except Exception as exc:
            logger.warning("GitHub scan failed: %s", exc)
            return {"github_error": str(exc)}

    def _run_sub_agents(self) -> List[AgentResult]:
        """Run all enabled sub-agents (excluding self)."""
        results = []
        configs = load_agent_configs()

        # Import lazily to avoid circular imports
        from agents.cli import _register_agents, _AGENT_CLASSES
        _register_agents()

        for name, agent_cls in _AGENT_CLASSES.items():
            if name == "conductor":
                continue
            config = configs.get(name)
            if config and not config.enabled:
                continue

            try:
                agent = agent_cls(config=config)
                self.logger.info("Running sub-agent: %s", name)
                sub_result = agent.run()
                results.append(sub_result)
            except Exception as exc:
                self.logger.error("Sub-agent %s failed: %s", name, exc)
                failed = AgentResult(agent_name=name, success=False, error=str(exc))
                results.append(failed)

        return results

    def _compile_digest(self, sub_results: List[AgentResult], metrics: Dict) -> str:
        """Compile all results into a markdown daily digest."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        lines = [
            f"# Minerva Daily Digest — {today}",
            "",
        ]

        # Metrics snapshot
        lines.append("## 📊 Metrics")
        lines.append("")
        for key, val in metrics.items():
            lines.append(f"- **{key}**: {val}")
        lines.append("")

        # Collect all escalations by priority
        all_escalations = []
        for sr in sub_results:
            all_escalations.extend(sr.escalations)

        urgent = [e for e in all_escalations if e.priority == Priority.URGENT]
        review = [e for e in all_escalations if e.priority == Priority.REVIEW]
        fyi = [e for e in all_escalations if e.priority == Priority.FYI]

        if urgent:
            lines.append("## 🔴 Urgent (requires your action today)")
            lines.append("")
            for e in urgent:
                lines.append(f"- **[{e.source_agent}]** {e.title}: {e.detail}")
            lines.append("")

        if review:
            lines.append("## 🟡 Needs Review (within 48 hours)")
            lines.append("")
            for e in review:
                lines.append(f"- **[{e.source_agent}]** {e.title}: {e.detail}")
            lines.append("")

        if fyi:
            lines.append("## 🟢 FYI (handled autonomously)")
            lines.append("")
            for e in fyi:
                lines.append(f"- **[{e.source_agent}]** {e.title}: {e.detail}")
            lines.append("")

        # Agent summaries
        lines.append("## Agent Reports")
        lines.append("")
        for sr in sub_results:
            status = "✅" if sr.success else "❌"
            lines.append(f"### {status} {sr.agent_name}")
            lines.append(f"- {sr.summary}")
            if sr.actions_taken:
                lines.append(f"- Actions: {', '.join(sr.actions_taken)}")
            if sr.error:
                lines.append(f"- ⚠️ Error: {sr.error}")
            lines.append("")

        return "\n".join(lines)

    def _deliver_digest(self, digest: str) -> str:
        """Save digest to file. Returns the file path."""
        DIGEST_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.utcnow().strftime("%Y-%m-%d")
        filepath = DIGEST_DIR / f"{today}.md"

        if self.is_dry_run:
            self.logger.info("[DRY RUN] Would save digest to %s (%d chars)", filepath, len(digest))
            filepath.write_text(digest, encoding="utf-8")
            return str(filepath)

        filepath.write_text(digest, encoding="utf-8")
        self.logger.info("Digest saved to %s", filepath)

        # Optional email delivery
        email = os.getenv("AGENT_DIGEST_EMAIL")
        if email:
            self.logger.info("Email delivery not yet implemented for: %s", email)

        return str(filepath)

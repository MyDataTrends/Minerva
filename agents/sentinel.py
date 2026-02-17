"""
Sentinel Agent — QA validation for Engineer PRs.

Validates code changes by running tests, checking lint, generating
edge cases, and assigning confidence scores. Only PRs scoring ≥7
are surfaced to the human for review.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.base import AgentConfig, AgentResult, BaseAgent, Priority, TriggerType
from agents.memory.operational import OperationalMemory
from utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
QA_HISTORY_FILE = Path(__file__).parent / "state" / "qa_history.json"


class SentinelAgent(BaseAgent):
    """
    QA validation agent — the Engineer's gatekeeper.

    Workflow:
    1. Run pytest suite and capture results
    2. Run ruff linter on changed files
    3. Calculate confidence score (1-10)
    4. Generate QA report
    5. Route: ≥7 to human, <7 back to Engineer
    """

    name = "sentinel"
    trigger_type = TriggerType.EVENT

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self.memory = OperationalMemory("sentinel")
        self.min_confidence = (config.extra or {}).get("min_confidence_score", 7) if config else 7

    def run(self, **kwargs) -> AgentResult:
        """
        Run QA validation.

        kwargs:
            pr_number (int): PR number to validate (if validating a specific PR)
            changed_files (list): List of changed file paths (for targeted testing)
        """
        start = time.time()
        result = self._make_result()
        pr_number = kwargs.get("pr_number")
        changed_files = kwargs.get("changed_files", [])

        try:
            # 1. Run pytest
            test_results = self._run_tests()
            result.add_action("ran_pytest")
            result.metrics["tests"] = test_results

            # 2. Run linter
            lint_results = self._run_linter(changed_files)
            result.add_action("ran_linter")
            result.metrics["lint"] = lint_results

            # 3. Calculate confidence score
            score = self._calculate_confidence(test_results, lint_results)
            result.metrics["confidence_score"] = score

            # 4. Generate QA report
            report = self._generate_report(test_results, lint_results, score, pr_number)
            result.add_action("generated_qa_report")

            # 5. Route based on score
            if score >= self.min_confidence:
                result.add_escalation(
                    Priority.REVIEW,
                    f"PR ready for review (score: {score}/10)",
                    report,
                )
                result.add_action(f"flagged_for_human_review:score={score}")
            else:
                result.add_escalation(
                    Priority.FYI,
                    f"PR sent back to Engineer (score: {score}/10)",
                    report,
                )
                result.add_action(f"returned_to_engineer:score={score}")

            # 6. Post as PR comment if we have a PR number and GitHub access
            if pr_number and not self.is_dry_run:
                self._post_pr_comment(pr_number, report)
                result.add_action(f"posted_comment:pr#{pr_number}")

            # 7. Update QA history
            self._update_history(score, test_results, lint_results)
            result.add_action("updated_qa_history")

            result.success = True
            result.summary = (
                f"QA complete. Score: {score}/10. "
                f"Tests: {test_results['passed']}/{test_results['total']}. "
                f"Lint: {lint_results['issues']} issues. "
                f"{'→ Ready for review' if score >= self.min_confidence else '→ Sent back to Engineer'}"
            )

        except Exception as exc:
            result.success = False
            result.error = str(exc)
            result.summary = f"QA validation failed: {exc}"
            logger.exception("Sentinel run failed")

        result.duration_seconds = time.time() - start
        self.memory.log_run(
            result.success, result.duration_seconds, result.summary,
            len(result.actions_taken), len(result.escalations), result.error,
        )
        return result

    def _run_tests(self) -> Dict[str, Any]:
        """Run pytest and capture results."""
        try:
            proc = subprocess.run(
                ["python", "-m", "pytest", "--tb=short", "-q"],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=300,
            )
            output = proc.stdout + proc.stderr

            # Parse pytest output for pass/fail counts
            passed, failed, errors, total = self._parse_pytest_output(output)

            return {
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "total": total,
                "returncode": proc.returncode,
                "all_passed": proc.returncode == 0,
                "output_tail": output[-500:] if len(output) > 500 else output,
            }
        except subprocess.TimeoutExpired:
            return {"passed": 0, "failed": 0, "errors": 1, "total": 0,
                    "returncode": -1, "all_passed": False, "output_tail": "TIMEOUT"}
        except Exception as exc:
            return {"passed": 0, "failed": 0, "errors": 1, "total": 0,
                    "returncode": -1, "all_passed": False, "output_tail": str(exc)}

    def _parse_pytest_output(self, output: str) -> Tuple[int, int, int, int]:
        """Extract pass/fail/error counts from pytest output."""
        import re
        passed = failed = errors = 0

        # Match patterns like "56 passed", "2 failed", "1 error"
        for match in re.finditer(r"(\d+) (passed|failed|error)", output):
            count = int(match.group(1))
            kind = match.group(2)
            if kind == "passed":
                passed = count
            elif kind == "failed":
                failed = count
            elif kind == "error":
                errors = count

        total = passed + failed + errors
        return passed, failed, errors, total

    def _run_linter(self, changed_files: List[str] = None) -> Dict[str, Any]:
        """Run ruff linter and capture results."""
        try:
            cmd = ["python", "-m", "ruff", "check"]
            if changed_files:
                cmd.extend(changed_files)
            else:
                cmd.append(".")

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=60,
            )

            issues = proc.stdout.count("\n") if proc.stdout else 0

            return {
                "issues": issues,
                "clean": proc.returncode == 0,
                "output_tail": proc.stdout[-300:] if proc.stdout and len(proc.stdout) > 300 else (proc.stdout or ""),
            }
        except FileNotFoundError:
            return {"issues": -1, "clean": True, "output_tail": "ruff not installed"}
        except Exception as exc:
            return {"issues": -1, "clean": True, "output_tail": str(exc)}

    def _calculate_confidence(self, test_results: Dict, lint_results: Dict) -> int:
        """
        Calculate confidence score (1-10).

        Scoring:
        - Tests all pass: +4
        - Tests mostly pass (>90%): +3
        - Tests somewhat pass (>70%): +2
        - Lint clean: +2
        - No errors: +2
        - Previously reliable (from history): +2
        """
        score = 0

        # Test score (0-4)
        if test_results.get("all_passed"):
            score += 4
        elif test_results["total"] > 0:
            pass_rate = test_results["passed"] / test_results["total"]
            if pass_rate > 0.9:
                score += 3
            elif pass_rate > 0.7:
                score += 2
            else:
                score += 1

        # Lint score (0-2)
        if lint_results.get("clean"):
            score += 2
        elif lint_results.get("issues", 0) < 5:
            score += 1

        # Error score (0-2)
        if test_results.get("errors", 0) == 0:
            score += 2
        elif test_results.get("errors", 0) <= 2:
            score += 1

        # History bonus (0-2)
        history = self._load_history()
        if history:
            recent = history[-5:]
            avg_score = sum(h.get("score", 0) for h in recent) / len(recent)
            if avg_score >= 7:
                score += 2
            elif avg_score >= 5:
                score += 1

        return min(score, 10)

    def _generate_report(self, test_results: Dict, lint_results: Dict,
                         score: int, pr_number: Optional[int] = None) -> str:
        """Generate a markdown QA report."""
        lines = [
            "## 🛡️ Sentinel QA Report",
            "",
        ]

        if pr_number:
            lines.append(f"**PR**: #{pr_number}")
            lines.append("")

        # Tests
        test_emoji = "✅" if test_results.get("all_passed") else "❌"
        lines.append(f"{test_emoji} **Tests**: {test_results['passed']}/{test_results['total']} passed")
        if test_results.get("failed"):
            lines.append(f"  - ❌ {test_results['failed']} failed")
        if test_results.get("errors"):
            lines.append(f"  - ⚠️ {test_results['errors']} errors")

        # Lint
        lint_emoji = "✅" if lint_results.get("clean") else "⚠️"
        lines.append(f"{lint_emoji} **Lint**: {lint_results.get('issues', '?')} issues")

        # Score
        score_emoji = "✅" if score >= self.min_confidence else "⚠️" if score >= 4 else "❌"
        bar = "█" * score + "░" * (10 - score)
        lines.extend([
            "",
            f"{score_emoji} **Confidence Score**: {bar} **{score}/10**",
            "",
        ])

        if score >= self.min_confidence:
            lines.append("**Verdict**: ✅ Ready for human review")
        elif score >= 4:
            lines.append("**Verdict**: ⚠️ Needs iteration — sent back to Engineer")
        else:
            lines.append("**Verdict**: ❌ Rejected — significant issues found")

        return "\n".join(lines)

    def _post_pr_comment(self, pr_number: int, report: str) -> None:
        """Post the QA report as a PR comment."""
        try:
            from agents.tools.github_client import GitHubClient
            client = GitHubClient()
            if client.is_available:
                client.create_comment(pr_number, report)
                client.close()
        except Exception as exc:
            logger.warning("Failed to post PR comment: %s", exc)

    def _load_history(self) -> List[Dict]:
        """Load QA history from JSON file."""
        if not QA_HISTORY_FILE.exists():
            return []
        try:
            return json.loads(QA_HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _update_history(self, score: int, test_results: Dict, lint_results: Dict) -> None:
        """Append to QA history."""
        QA_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        history = self._load_history()
        history.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "score": score,
            "tests_passed": test_results.get("passed", 0),
            "tests_total": test_results.get("total", 0),
            "lint_issues": lint_results.get("issues", 0),
        })
        # Keep last 100 entries
        history = history[-100:]
        QA_HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")

"""
Telemetry Agent — Weekly usage analytics and trend reporter.

Data sources (all with graceful fallback):
  1. InteractionLogger (~/.assay/interactions.db)
       Written by cascade_planner, dashboard chat, nl_query, feedback_handler.
       Contains: prompts, interaction types, success/failure, response times,
       retry counts, user ratings, dataset names.
  2. Agent OperationalMemory (agents/state/<name>.db)
       One DB per agent. Contains run history, action log, escalation log.
       Used to build a cross-agent health table.
  3. UsageTracker (local_data/usage/*.json)
       Counts file-upload requests and bytes per user_id.

The agent aggregates these sources, computes trends vs the previous period,
feeds a compact insight doc to the knowledge base for the Productizer,
and writes a human-readable digest report.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.base import AgentConfig, AgentResult, BaseAgent, Priority, TriggerType
from agents.memory.operational import OperationalMemory
from utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIGEST_DIR = Path(__file__).parent / "digests" / "telemetry"
KB_OPERATIONS_DIR = Path(__file__).parent / "knowledge_base" / "operations"
TELEMETRY_KB_PATH = KB_OPERATIONS_DIR / "telemetry_insights.md"

# Known agents whose OperationalMemory we inspect for health stats
KNOWN_AGENTS = ["conductor", "engineer", "sentinel", "advocate", "productizer",
                "marketing", "support", "telemetry"]


class TelemetryAgent(BaseAgent):
    """
    Usage analytics agent.

    Workflow:
    1. Pull LLM interaction metrics from InteractionLogger.
    2. Pull agent health metrics from each agent's OperationalMemory DB.
    3. Pull file-upload counts from UsageTracker.
    4. Compute trends (current period vs previous same-length period).
    5. Detect frustration signals (high retry counts, corrected outputs).
    6. Write a compact insights doc to knowledge_base/operations/ for Productizer.
    7. Write a full human-readable report to agents/digests/telemetry/.
    8. Escalate summary as Priority.METRIC; spike error rates as Priority.REVIEW.
    """

    name = "telemetry"
    trigger_type = TriggerType.CRON

    # If overall error rate exceeds this, raise a REVIEW escalation
    ERROR_RATE_THRESHOLD = 0.30

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self.memory = OperationalMemory("telemetry")
        DIGEST_DIR.mkdir(parents=True, exist_ok=True)
        KB_OPERATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public entry point ────────────────────────────────────────────

    def run(self, **kwargs) -> AgentResult:
        """Execute the telemetry analysis workflow."""
        start = time.time()
        result = self._make_result()
        days = int(kwargs.get("days", 7))

        try:
            logger.info("TelemetryAgent starting (window=%dd)", days)

            # 1. LLM interaction metrics
            interaction_metrics = self._collect_interaction_metrics(days)
            result.add_action(f"collected_interaction_metrics_last_{days}d")

            # 2. Agent health metrics
            agent_health = self._collect_agent_health(days)
            result.add_action(f"collected_health_for_{len(agent_health)}_agents")

            # 3. File-upload usage
            upload_counts = self._collect_upload_counts()
            result.add_action("collected_upload_counts")

            # 4. Trends vs previous period
            trends = self._compute_trends(days)
            result.add_action("computed_trends")

            # 5. Frustration signals (already inside interaction_metrics)
            frustration = interaction_metrics.get("frustration", [])
            error_rate = interaction_metrics["summary"].get("error_rate", 0.0)

            # 6. Write KB artifact for Productizer
            kb_content = self._build_kb_doc(
                interaction_metrics, agent_health, upload_counts, trends, days
            )
            self._write_kb(kb_content)
            result.add_action("wrote_kb_artifact")

            # 7. Write digest report
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            report = self._build_report(
                interaction_metrics, agent_health, upload_counts,
                trends, frustration, days, date_str,
            )
            report_path = self._save_report(report, date_str)
            result.add_action(f"saved_report:{report_path.name}")

            # 8. Escalations
            total = interaction_metrics["summary"].get("total_queries", 0)
            result.add_escalation(
                Priority.METRIC,
                f"Telemetry: {total:,} queries last {days}d | error rate {error_rate:.0%}",
                self._metrics_snippet(interaction_metrics, agent_health, upload_counts)
                + f"\n\nFull report: {report_path}",
            )

            if error_rate > self.ERROR_RATE_THRESHOLD:
                samples = "\n".join(
                    f"  - [{s.get('interaction_type', '?')}] retries={s.get('retry_count', 0)}"
                    f" — {str(s.get('prompt', ''))[:80]}"
                    for s in frustration[:5]
                )
                result.add_escalation(
                    Priority.REVIEW,
                    f"High error rate: {error_rate:.0%} (threshold {self.ERROR_RATE_THRESHOLD:.0%})",
                    f"Error rate exceeded threshold in the last {days} days.\n\n"
                    f"Top frustration signals:\n{samples or '  (none logged)'}",
                )
                result.add_action("escalated_high_error_rate")

            # Populate result metrics for Conductor digest
            result.metrics.update(interaction_metrics["summary"])
            result.metrics["agents_healthy"] = sum(
                1 for h in agent_health.values() if h.get("last_run_success") is True
            )
            result.metrics["total_uploads"] = upload_counts.get("total_requests", 0)

            self.memory.log_action(
                "telemetry_run",
                f"total={total} error_rate={error_rate:.2f} agents={len(agent_health)}",
            )

            result.success = True
            result.summary = (
                f"Telemetry: {total:,} queries, {error_rate:.0%} error rate, "
                f"{len(frustration)} frustration signals, "
                f"{len(agent_health)} agents checked. "
                f"Report → {report_path.name}"
            )

        except Exception as exc:
            result.success = False
            result.error = str(exc)
            result.summary = f"Telemetry agent failed: {exc}"
            logger.exception("Telemetry agent run failed")

        result.duration_seconds = time.time() - start
        self.memory.log_run(
            result.success, result.duration_seconds, result.summary,
            len(result.actions_taken), len(result.escalations), result.error,
        )
        return result

    # ── Source 1: InteractionLogger ───────────────────────────────────

    def _collect_interaction_metrics(self, days: int) -> Dict[str, Any]:
        """
        Pull metrics from InteractionLogger (~/.assay/interactions.db).

        Populated by: cascade_planner (ACTION), dashboard chat (CHAT),
        nl_query (ANALYSIS/CHAT), feedback_handler (ratings/corrections).
        """
        db_path = Path.home() / ".assay" / "interactions.db"

        empty = {
            "summary": {
                "total_queries": 0,
                "success_count": 0,
                "failure_count": 0,
                "error_rate": 0.0,
                "avg_response_time_ms": 0.0,
                "unique_sessions": 0,
                "unique_datasets": 0,
                "avg_rating": None,
            },
            "type_breakdown": {},
            "popular_intents": [],
            "dataset_usage": [],
            "frustration": [],
        }

        if not db_path.exists():
            logger.info("InteractionLogger DB not found at %s — returning zeros", db_path)
            return empty

        try:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            with sqlite3.connect(str(db_path)) as conn:
                conn.row_factory = sqlite3.Row

                # Aggregate summary
                row = conn.execute("""
                    SELECT
                        COUNT(*)                                                   AS total,
                        SUM(CASE WHEN execution_success = 1 THEN 1 ELSE 0 END)    AS successes,
                        SUM(CASE WHEN execution_success = 0 THEN 1 ELSE 0 END)    AS failures,
                        AVG(response_time_ms)                                      AS avg_ms,
                        COUNT(DISTINCT session_id)                                 AS sessions,
                        COUNT(DISTINCT dataset_name)                               AS datasets,
                        AVG(rating)                                                AS avg_rating
                    FROM interactions
                    WHERE created_at >= ?
                """, (cutoff,)).fetchone()

                total = row["total"] or 0
                successes = row["successes"] or 0
                failures = row["failures"] or 0
                avg_ms = row["avg_ms"] or 0.0
                sessions = row["sessions"] or 0
                datasets = row["datasets"] or 0
                avg_rating = row["avg_rating"]

                # Interaction type breakdown
                type_rows = conn.execute("""
                    SELECT interaction_type, COUNT(*) AS cnt
                    FROM interactions
                    WHERE created_at >= ?
                    GROUP BY interaction_type
                    ORDER BY cnt DESC
                """, (cutoff,)).fetchall()
                type_breakdown = {r["interaction_type"] or "unknown": r["cnt"]
                                  for r in type_rows}

                # Top prompts / intents (proxy for user intent)
                intent_rows = conn.execute("""
                    SELECT prompt, COUNT(*) AS cnt
                    FROM interactions
                    WHERE created_at >= ?
                    GROUP BY prompt
                    ORDER BY cnt DESC
                    LIMIT 10
                """, (cutoff,)).fetchall()
                popular_intents = [
                    {"prompt": (r["prompt"] or "")[:100], "count": r["cnt"]}
                    for r in intent_rows
                ]

                # Most-used datasets
                dataset_rows = conn.execute("""
                    SELECT dataset_name, COUNT(*) AS cnt
                    FROM interactions
                    WHERE created_at >= ?
                      AND dataset_name IS NOT NULL
                      AND dataset_name != ''
                    GROUP BY dataset_name
                    ORDER BY cnt DESC
                    LIMIT 5
                """, (cutoff,)).fetchall()
                dataset_usage = [
                    {"dataset": r["dataset_name"], "count": r["cnt"]}
                    for r in dataset_rows
                ]

                # Frustration signals: high retry or corrected outcome
                frustration_rows = conn.execute("""
                    SELECT interaction_type, prompt, retry_count, outcome, created_at
                    FROM interactions
                    WHERE (retry_count >= 2 OR outcome IN ('corrected', 'retried'))
                      AND created_at >= ?
                    ORDER BY retry_count DESC, created_at DESC
                    LIMIT 20
                """, (cutoff,)).fetchall()
                frustration = [dict(r) for r in frustration_rows]

            error_rate = failures / total if total > 0 else 0.0

            return {
                "summary": {
                    "total_queries": total,
                    "success_count": successes,
                    "failure_count": failures,
                    "error_rate": round(error_rate, 4),
                    "avg_response_time_ms": round(avg_ms, 1),
                    "unique_sessions": sessions,
                    "unique_datasets": datasets,
                    "avg_rating": round(avg_rating, 2) if avg_rating is not None else None,
                },
                "type_breakdown": type_breakdown,
                "popular_intents": popular_intents,
                "dataset_usage": dataset_usage,
                "frustration": frustration,
            }

        except Exception as exc:
            logger.warning("Failed to query InteractionLogger: %s", exc)
            return empty

    # ── Source 2: Agent OperationalMemory ─────────────────────────────

    def _collect_agent_health(self, days: int) -> Dict[str, Dict[str, Any]]:
        """
        Read each known agent's OperationalMemory DB and build a health snapshot.

        Returns {agent_name: {runs, successes, failures, last_run_success,
                               last_run_at, avg_duration, escalations}}.
        """
        state_dir = Path(__file__).parent / "state"
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        health: Dict[str, Dict[str, Any]] = {}

        for agent_name in KNOWN_AGENTS:
            db_path = state_dir / f"{agent_name}.db"
            if not db_path.exists():
                health[agent_name] = {"available": False}
                continue

            try:
                with sqlite3.connect(str(db_path)) as conn:
                    conn.row_factory = sqlite3.Row

                    # Run history within window
                    run_row = conn.execute("""
                        SELECT
                            COUNT(*)                                                AS runs,
                            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END)           AS successes,
                            SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END)           AS failures,
                            AVG(duration_seconds)                                   AS avg_dur
                        FROM runs
                        WHERE timestamp >= ?
                    """, (cutoff,)).fetchone()

                    # Most recent run
                    last_row = conn.execute("""
                        SELECT success, timestamp FROM runs
                        ORDER BY id DESC LIMIT 1
                    """).fetchone()

                    # Pending escalations
                    esc_count = conn.execute(
                        "SELECT COUNT(*) FROM escalations WHERE resolved = 0"
                    ).fetchone()[0]

                runs = run_row["runs"] or 0
                successes = run_row["successes"] or 0
                failures = run_row["failures"] or 0
                avg_dur = run_row["avg_dur"] or 0.0

                health[agent_name] = {
                    "available": True,
                    "runs": runs,
                    "successes": successes,
                    "failures": failures,
                    "success_rate": round(successes / runs, 2) if runs > 0 else None,
                    "avg_duration_s": round(avg_dur, 1),
                    "last_run_success": bool(last_row["success"]) if last_row else None,
                    "last_run_at": last_row["timestamp"] if last_row else None,
                    "pending_escalations": esc_count,
                }

            except Exception as exc:
                logger.debug("Could not read health for agent %s: %s", agent_name, exc)
                health[agent_name] = {"available": False, "error": str(exc)}

        return health

    # ── Source 3: UsageTracker ────────────────────────────────────────

    def _collect_upload_counts(self) -> Dict[str, Any]:
        """
        Aggregate file-upload request counts from local_data/usage/*.json.

        Each file is {requests: N, bytes: N} for a user_id.
        """
        usage_dir = PROJECT_ROOT / "local_data" / "usage"
        if not usage_dir.exists():
            return {"total_requests": 0, "total_bytes": 0, "user_count": 0}

        total_requests = 0
        total_bytes = 0
        user_count = 0

        for json_file in usage_dir.glob("*.json"):
            try:
                data = json.loads(json_file.read_text())
                total_requests += int(data.get("requests", 0))
                total_bytes += int(data.get("bytes", 0))
                user_count += 1
            except Exception:
                continue

        return {
            "total_requests": total_requests,
            "total_bytes": total_bytes,
            "total_bytes_mb": round(total_bytes / (1024 * 1024), 2),
            "user_count": user_count,
        }

    # ── Trend computation ─────────────────────────────────────────────

    def _compute_trends(self, days: int) -> Dict[str, Any]:
        """
        Compare current-period stats vs the immediately preceding same-length period.

        Reads from InteractionLogger. Returns a dict of metric → trend info.
        """
        db_path = Path.home() / ".assay" / "interactions.db"
        if not db_path.exists():
            return {}

        try:
            now = datetime.now()
            cur_start = (now - timedelta(days=days)).isoformat()
            prev_start = (now - timedelta(days=days * 2)).isoformat()
            prev_end = cur_start

            def _period(conn, start: str, end: str) -> Dict[str, float]:
                r = conn.execute("""
                    SELECT
                        COUNT(*) AS total,
                        SUM(CASE WHEN execution_success = 1 THEN 1 ELSE 0 END) AS ok,
                        AVG(response_time_ms) AS avg_ms
                    FROM interactions
                    WHERE created_at >= ? AND created_at < ?
                """, (start, end)).fetchone()
                total = r[0] or 0
                ok = r[1] or 0
                return {
                    "total": total,
                    "success_rate": round(ok / total, 4) if total else 0.0,
                    "avg_ms": round(r[2] or 0.0, 1),
                }

            with sqlite3.connect(str(db_path)) as conn:
                cur = _period(conn, cur_start, now.isoformat())
                prev = _period(conn, prev_start, prev_end)

            def _trend(key: str) -> Dict[str, Any]:
                c, p = cur[key], prev[key]
                if p and p != 0:
                    delta = round((c - p) / abs(p) * 100, 1)
                else:
                    delta = 0.0
                direction = "up" if delta > 1 else ("down" if delta < -1 else "flat")
                return {"current": c, "previous": p, "delta_pct": delta, "direction": direction}

            return {
                "query_volume": _trend("total"),
                "success_rate": _trend("success_rate"),
                "avg_response_ms": _trend("avg_ms"),
            }

        except Exception as exc:
            logger.debug("Trend computation failed: %s", exc)
            return {}

    # ── Report / KB generation ────────────────────────────────────────

    def _build_report(
        self,
        im: Dict,
        agent_health: Dict,
        uploads: Dict,
        trends: Dict,
        frustration: List,
        days: int,
        date_str: str,
    ) -> str:
        s = im["summary"]
        dir_sym = {"up": "↑", "down": "↓", "flat": "→"}

        lines = [
            f"# Telemetry Report — {date_str}",
            f"*Window: last {days} days | Generated by TelemetryAgent*",
            "",
            "## LLM Interaction Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total queries | {s['total_queries']:,} |",
            f"| Successful | {s['success_count']:,} |",
            f"| Failed | {s['failure_count']:,} |",
            f"| Error rate | {s['error_rate']:.1%} |",
            f"| Avg response time | {s['avg_response_time_ms']:.0f} ms |",
            f"| Unique sessions | {s['unique_sessions']:,} |",
            f"| Unique datasets | {s['unique_datasets']:,} |",
            f"| Avg user rating | {s['avg_rating'] if s['avg_rating'] is not None else 'n/a'} / 5 |",
            "",
            "## File Upload Activity",
            "",
            f"- Total upload requests: {uploads.get('total_requests', 0):,}",
            f"- Total data uploaded: {uploads.get('total_bytes_mb', 0):.1f} MB",
            f"- Unique uploading users: {uploads.get('user_count', 0):,}",
            "",
            "## Interaction Type Breakdown",
            "",
        ]

        type_bd = im.get("type_breakdown", {})
        if type_bd:
            lines += ["| Type | Count |", "|------|-------|"]
            for t, cnt in sorted(type_bd.items(), key=lambda x: -x[1]):
                lines.append(f"| {t} | {cnt:,} |")
        else:
            lines.append("*No data.*")

        lines += ["", "## Top Prompts / Intents", ""]
        intents = im.get("popular_intents", [])
        if intents:
            for i, item in enumerate(intents[:10], 1):
                lines.append(f"{i}. `{item['prompt']}` — {item['count']}×")
        else:
            lines.append("*No data.*")

        lines += ["", "## Top Datasets Used", ""]
        datasets = im.get("dataset_usage", [])
        if datasets:
            for d in datasets:
                lines.append(f"- `{d['dataset']}` — {d['count']}×")
        else:
            lines.append("*No data.*")

        lines += ["", "## Trends (vs previous period)", ""]
        if trends:
            lines += ["| Metric | Current | Previous | Change |",
                      "|--------|---------|----------|--------|"]
            for key, t in trends.items():
                sym = dir_sym.get(t["direction"], "→")
                lines.append(
                    f"| {key} | {t['current']} | {t['previous']} | {sym} {t['delta_pct']:+.1f}% |"
                )
        else:
            lines.append("*Insufficient data for trends.*")

        lines += ["", "## Agent Health", ""]
        lines += ["| Agent | Runs | Successes | Success Rate | Last Run | Pending Esc. |",
                  "|-------|------|-----------|--------------|----------|--------------|"]
        for agent, h in sorted(agent_health.items()):
            if not h.get("available"):
                lines.append(f"| {agent} | — | — | — | — | — |")
                continue
            sr = f"{h['success_rate']:.0%}" if h.get("success_rate") is not None else "—"
            last = (h.get("last_run_at") or "—")[:10]
            ok_sym = "✅" if h.get("last_run_success") else ("❌" if h.get("last_run_success") is False else "—")
            lines.append(
                f"| {agent} | {h['runs']} | {h['successes']} | {sr} | {ok_sym} {last} | {h.get('pending_escalations', '—')} |"
            )

        lines += ["", "## Frustration Signals", ""]
        if frustration:
            lines.append(f"*{len(frustration)} signal(s) detected in window.*")
            lines.append("")
            for sig in frustration[:8]:
                lines.append(
                    f"- **[{sig.get('interaction_type','?')}]** retries={sig.get('retry_count',0)}"
                    f" outcome={sig.get('outcome','?')} — `{str(sig.get('prompt',''))[:80]}`"
                )
        else:
            lines.append("*No frustration signals.*")

        lines += [
            "",
            "---",
            f"*Generated: {datetime.utcnow().isoformat()} UTC by TelemetryAgent*",
        ]
        return "\n".join(lines)

    def _build_kb_doc(
        self,
        im: Dict,
        agent_health: Dict,
        uploads: Dict,
        trends: Dict,
        days: int,
    ) -> str:
        """Compact KB doc consumed by the Productizer to inform sales kit targeting."""
        s = im["summary"]
        top_intents = ", ".join(
            f"{i['prompt'][:40]} ({i['count']}×)"
            for i in im.get("popular_intents", [])[:5]
        )
        top_types = ", ".join(
            f"{k}: {v}" for k, v in list(im.get("type_breakdown", {}).items())[:4]
        )
        trend_notes = ""
        if trends:
            qv = trends.get("query_volume", {})
            sr = trends.get("success_rate", {})
            trend_notes = (
                f"- Query volume: {qv.get('direction','flat')} ({qv.get('delta_pct',0):+.1f}%)\n"
                f"- Success rate: {sr.get('direction','flat')} ({sr.get('delta_pct',0):+.1f}%)"
            )

        healthy_agents = [a for a, h in agent_health.items() if h.get("last_run_success") is True]
        unhealthy_agents = [a for a, h in agent_health.items()
                            if h.get("available") and h.get("last_run_success") is False]

        return (
            f"# Telemetry Insights (last {days} days)\n\n"
            f"*Updated: {datetime.utcnow().strftime('%Y-%m-%d')}*\n\n"
            "## LLM Usage\n"
            f"- Total queries: {s['total_queries']:,}\n"
            f"- Error rate: {s['error_rate']:.1%}\n"
            f"- Avg response time: {s['avg_response_time_ms']:.0f} ms\n"
            f"- Unique sessions: {s['unique_sessions']:,}\n"
            f"- Avg user rating: {s['avg_rating'] if s['avg_rating'] is not None else 'n/a'} / 5\n\n"
            "## Most-Used Features\n"
            f"- Interaction types: {top_types or 'n/a'}\n"
            f"- Top intents: {top_intents or 'n/a'}\n\n"
            "## File Uploads\n"
            f"- Requests: {uploads.get('total_requests', 0):,} | "
            f"Data: {uploads.get('total_bytes_mb', 0):.1f} MB | "
            f"Users: {uploads.get('user_count', 0)}\n\n"
            "## Trends\n"
            f"{trend_notes or 'Insufficient data for trend comparison.'}\n\n"
            "## Agent Health\n"
            f"- Healthy: {', '.join(healthy_agents) or 'none recorded'}\n"
            f"- Unhealthy: {', '.join(unhealthy_agents) or 'none'}\n\n"
            "## Productizer Signal\n"
            "High-frequency intents above represent validated user pain points. "
            "These should be prioritised in the next Vertical Sales Kit generation run. "
            "Declining features signal low product-market fit for those capabilities.\n"
        )

    def _metrics_snippet(
        self, im: Dict, agent_health: Dict, uploads: Dict
    ) -> str:
        """One-paragraph plain-text summary for escalation detail fields."""
        s = im["summary"]
        healthy = sum(1 for h in agent_health.values() if h.get("last_run_success") is True)
        return (
            f"Queries: {s['total_queries']:,}  |  "
            f"Successes: {s['success_count']:,}  |  "
            f"Failures: {s['failure_count']:,}  |  "
            f"Error rate: {s['error_rate']:.1%}\n"
            f"Avg response: {s['avg_response_time_ms']:.0f} ms  |  "
            f"Sessions: {s['unique_sessions']:,}  |  "
            f"Avg rating: {s['avg_rating'] if s['avg_rating'] is not None else 'n/a'}/5\n"
            f"File uploads: {uploads.get('total_requests', 0):,} requests "
            f"({uploads.get('total_bytes_mb', 0):.1f} MB)\n"
            f"Agent health: {healthy}/{len(agent_health)} agents ran successfully last cycle"
        )

    # ── Persistence ───────────────────────────────────────────────────

    def _save_report(self, content: str, date_str: str) -> Path:
        path = DIGEST_DIR / f"{date_str}_telemetry.md"
        path.write_text(content, encoding="utf-8")
        logger.info("Telemetry report saved: %s", path)
        return path

    def _write_kb(self, content: str) -> None:
        try:
            TELEMETRY_KB_PATH.write_text(content, encoding="utf-8")
            logger.info("Telemetry KB artifact updated")
        except Exception as exc:
            logger.error("Failed to write telemetry KB artifact: %s", exc)

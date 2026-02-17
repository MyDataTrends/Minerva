"""
Engineer Agent — Autonomous codebase improvement.

Reads the vision doc, analyzes the current codebase, identifies gaps,
and proposes improvements. Phase 1: gap analysis + recommendations.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base import AgentConfig, AgentResult, BaseAgent, Priority, TriggerType
from agents.memory.operational import OperationalMemory
from agents.memory.knowledge import KnowledgeBase
from utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


class EngineerAgent(BaseAgent):
    """
    Autonomous developer agent.

    Phase 1 (current): Gap analysis + improvement recommendations.
    Phase 2 (future): Code generation + PR creation.

    Workflow:
    1. Load vision doc as context
    2. Scan codebase modules and test coverage
    3. Score subsystems against endgame maturity targets
    4. Identify highest-priority improvement
    5. Write gap analysis to knowledge base
    """

    name = "engineer"
    trigger_type = TriggerType.CRON

    # Subsystems from the vision doc with their endgame targets
    SUBSYSTEMS = {
        "cascade_planner": {
            "key_file": "orchestration/cascade_planner.py",
            "endgame": "Production-ready orchestration with dependency-aware execution, retry, fallback",
            "test_pattern": "test_robustness.py",
        },
        "plan_learner": {
            "key_file": "orchestration/plan_learner.py",
            "endgame": "Active learning loop with pattern storage and tool weight refinement",
            "test_pattern": "test_closed_loop.py",
        },
        "mcp_server": {
            "key_file": "mcp_server/server.py",
            "endgame": "JSON-RPC 2.0 over stdio + HTTP with full tool exposure",
            "test_pattern": "test_mcp_server.py",
        },
        "api_discovery": {
            "key_file": "mcp_server/discovery_agent.py",
            "endgame": "Autonomous web search, registry lookup, and auto-connect for any API",
            "test_pattern": "test_router.py",
        },
        "dynamic_connectors": {
            "key_file": "mcp_server/dynamic_connector.py",
            "endgame": "LLM-generated connector code with full AST sandboxing",
            "test_pattern": None,
        },
        "model_training": {
            "key_file": "modeling/model_training.py",
            "endgame": "AutoML-grade model selection with hyperparameter tuning and SHAP explanations",
            "test_pattern": "test_model_training.py",
        },
        "smart_charts": {
            "key_file": "visualization/smart_charts.py",
            "endgame": "Column profiling → chart recommendation → interactive Plotly rendering",
            "test_pattern": None,
        },
        "nl_query": {
            "key_file": "ui/nl_query.py",
            "endgame": "RAG-augmented NL code gen with autocomplete and history recall",
            "test_pattern": None,
        },
        "data_fabric": {
            "key_file": "ui/data_fabric.py",
            "endgame": "Full data lineage graph with versioned provenance tracking",
            "test_pattern": None,
        },
        "dashboard": {
            "key_file": "ui/dashboard.py",
            "endgame": "Polished onboarding, error messaging, scheduled analysis, PDF export",
            "test_pattern": "test_dashboard_sections.py",
        },
    }

    MATURITY_LEVELS = {
        1: "prototype",
        2: "functional",
        3: "production",
        4: "best-in-class",
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self.memory = OperationalMemory("engineer")
        self.kb = KnowledgeBase()
        
        # Optimize for speed if API key is present (as requested by user)
        if os.getenv("ANTHROPIC_API_KEY"):
            # This hint tells the LLM interface to prefer a high-performance model
            self.model_hint = "claude-3-5-sonnet" 
        else:
            self.model_hint = "local"

    def run(self, **kwargs) -> AgentResult:
        """Execute the gap analysis workflow."""
        start = time.time()
        result = self._make_result()

        try:
            # 1. Scan codebase
            subsystem_scores = self._score_subsystems()
            result.add_action(f"scored_{len(subsystem_scores)}_subsystems")

            # 2. Find highest-priority gap
            top_gap = self._find_top_gap(subsystem_scores)
            result.add_action(f"identified_top_gap:{top_gap['name']}")

            # 3. Generate gap analysis report
            report = self._generate_gap_report(subsystem_scores, top_gap)
            result.add_action("generated_gap_report")

            # 4. Write to knowledge base
            self.kb.write_doc("product/vision_gap_analysis.md", report)
            result.add_action("wrote_gap_analysis_to_kb")

            # 5. Log and escalate
            self.memory.log_action(
                "gap_analysis",
                f"Top gap: {top_gap['name']} (score: {top_gap['score']}/4)",
            )

            result.add_escalation(
                Priority.REVIEW,
                f"Engineer: top improvement target — {top_gap['name']}",
                top_gap['gap'],
            )

            result.success = True
            result.summary = (
                f"Gap analysis complete. "
                f"Top priority: {top_gap['name']} ({top_gap['score']}/4). "
                f"Gap: {top_gap['gap'][:80]}..."
            )
            result.metrics = {
                "subsystems_scored": len(subsystem_scores),
                "average_maturity": sum(s["score"] for s in subsystem_scores.values()) / len(subsystem_scores),
                "top_gap": top_gap["name"],
            }

        except Exception as exc:
            result.success = False
            result.error = str(exc)
            result.summary = f"Gap analysis failed: {exc}"
            logger.exception("Engineer run failed")

        result.duration_seconds = time.time() - start
        self.memory.log_run(
            result.success, result.duration_seconds, result.summary,
            len(result.actions_taken), len(result.escalations), result.error,
        )
        return result

    def _score_subsystems(self) -> Dict[str, Dict[str, Any]]:
        """Score each subsystem using LLM-based maturity assessment."""
        scores = {}
        for name, info in self.SUBSYSTEMS.items():
            key_file = PROJECT_ROOT / info["key_file"]
            content = ""
            if key_file.exists():
                try:
                    content = key_file.read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning(f"Could not read {key_file}: {e}")

            # Get LLM assessment
            assessment = self._evaluate_maturity(name, info, content)
            
            # Combine assessment with file stats
            scores[name] = {
                "score": assessment["score"],
                "maturity": assessment["maturity"],
                "reason": assessment["reason"],
                "gap": assessment["gap"],
                "file_exists": key_file.exists(),
                "lines": len(content.splitlines()) if content else 0,
                "endgame": info["endgame"],
            }
        return scores

    def _evaluate_maturity(self, subsystem: str, info: dict, content: str) -> dict:
        """
        Evaluate maturity using LLM assessment against best-in-class standards.
        """
        # Quick check for empty/missing files
        if not content.strip():
            return {
                "score": 0, 
                "maturity": "missing", 
                "reason": "File does not exist or is empty", 
                "gap": "Implement the core functionality."
            }
            
        line_count = len(content.splitlines())
        if line_count < 50:
            return {
                "score": 1, 
                "maturity": "prototype", 
                "reason": "Implementation is a skeletal stub (<50 lines).", 
                "gap": "Expand the core logic and add basic functionality."
            }

        # Use LLM for deep analysis
        # Using chat interface for better instruction following
        messages = [
            {"role": "system", "content": "You are a Principal Software Engineer. You evaluate code maturity critically. Return ONLY JSON."},
            {"role": "user", "content": f"""
Vision Endgame: "{info['endgame']}"

Code Snippet (truncated):
```python
{content[:6000]}
```

Evaluate against:
1. PROTOTYPE (Script-like, no types)
2. FUNCTIONAL (Basic, untyped, untested)
3. PRODUCTION (Typed, robust, tested - e.g. FastAPI)
4. BEST-IN-CLASS (Zero-config, optimized, perfect docs - e.g. Pydantic)

Return JSON with keys: score (int), maturity (str), reason (str), gap (str).
JSON:
"""}
        ]
        
        try:
            from llm_manager.llm_interface import get_llm_chat
            import json
            import re
            
            # Use chat interface
            response = get_llm_chat(messages, max_tokens=600, temperature=0.2)
            
            # Robust JSON parsing
            try:
                # 1. Try stripping markdown
                clean_response = response.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_response)
            except json.JSONDecodeError:
                # 2. Try finding JSON block with regex
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(0))
                    except:
                        raise ValueError("Found JSON-like block but failed to parse")
                else:
                    # 3. Regex fallback for key fields
                    score_match = re.search(r'"?score"?:\s*(\d)', response)
                    reason_match = re.search(r'"?reason(?:ing)?"?:\s*"(.*?)"', response) 
                    gap_match = re.search(r'"?gap"?:\s*"(.*?)"', response)
                    
                    if score_match:
                        data = {
                            "score": int(score_match.group(1)),
                            "maturity": "functional" if int(score_match.group(1)) == 2 else "prototype",
                            "reason": reason_match.group(1) if reason_match else "Extracted via regex",
                            "gap": gap_match.group(1) if gap_match else "Extracted via regex"
                        }
                    else:
                         raise ValueError(f"No JSON found. Raw: {response[:100]}...")
            
            return {
                "score": int(data.get("score", 1)),
                "maturity": str(data.get("maturity", "prototype")),
                "reason": str(data.get("reasoning", data.get("reason", "No reason provided"))),
                "gap": str(data.get("gap", "Unknown gap"))
            }
            
        except Exception as e:
            logger.error(f"LLM scoring failed for {subsystem}: {e}")
            return {
                "score": 2, 
                "maturity": "functional", 
                "reason": f"Automated scoring failed: {e}", 
                "gap": "Manual review required."
            }

    def _find_top_gap(self, scores: Dict[str, Dict]) -> Dict[str, Any]:
        """Find the subsystem with the largest gap from best-in-class."""
        gaps = []
        for name, info in scores.items():
            gap_score = 4 - info["score"]
            if gap_score > 0:
                gaps.append({
                    "name": name,
                    "score": info["score"],
                    "gap": info["gap"],  # Text description of the gap
                    "gap_score": gap_score, # Numeric gap
                })

        # Sort by gap size (descending)
        # We could also use lines of code as a tiebreaker (larger complexity = higher priority to fix)
        gaps.sort(key=lambda g: -g["gap_score"])

        return gaps[0] if gaps else {
            "name": "none",
            "score": 4,
            "gap": "All subsystems at best-in-class!",
            "gap_score": 0,
        }

    def _generate_gap_report(self, scores: Dict[str, Dict], top_gap: Dict) -> str:
        """Generate a markdown gap analysis report."""
        lines = [
            "# Vision Gap Analysis",
            "",
            f"*Generated: {time.strftime('%Y-%m-%d %H:%M UTC')}*",
            "",
            "## Subsystem Maturity Scores",
            "",
            "| Subsystem | Score | Maturity | Quality Note | Next Step |",
            "|:---|:---|:---|:---|:---|",
        ]

        for name, info in sorted(scores.items(), key=lambda x: x[1]["score"]):
            bar = "█" * info["score"] + "░" * (4 - info["score"])
            lines.append(
                f"| {name} | {bar} {info['score']}/4 | {info['maturity']} | "
                f"{info['reason']} | {info['gap']} |"
            )

        avg = sum(s["score"] for s in scores.values()) / len(scores)
        lines.extend([
            "",
            f"**Average maturity: {avg:.1f}/4**",
            "",
            "## Top Priority Improvement",
            "",
            f"**{top_gap['name']}** (current: {top_gap['score']}/4)",
            f"> **Gap**: {top_gap['gap']}",
            "",
            "## Maturity Standards",
            "",
            "| Level | Class | Definition | Peers |",
            "|:---|:---|:---|:---|",
            "| 1 | Prototype | Minimal error handling, no types | Scripts |",
            "| 2 | Functional | Usable, basic structure | Flask (basic) |",
            "| 3 | Production | Fully typed, robust, tested | FastAPI, Requests |",
            "| 4 | Best-in-Class | Zero-config, self-optimizing, perfect docs | Pydantic, PyTorch |",
        ])

        return "\n".join(lines)

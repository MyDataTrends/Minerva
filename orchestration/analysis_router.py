from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any, Dict

from config import get_int

from config.feature_flags import ENABLE_LOCAL_LLM, ALLOW_FULL_COMPARE_MODELS
from preprocessing.sanitize import redact
from utils.logging import get_logger

MAX_ROWS_FULL: int = get_int("MAX_ROWS_FULL", 5000)
_logger = get_logger(__name__)


@dataclass
class LLMHint:
    """Structured hint returned from the LLM recommender."""

    task: str | None = None
    complexity: str | None = None
    confidence: float = 0.0


def route_analysis(
    df: Any,
    target: str,
    stats: Dict[str, Any],
    llm_hint: LLMHint | None,
) -> Literal["no_model", "baseline", "full"]:
    """Determine modeling depth based on stats and optional LLM hint."""

    if not stats.get("is_modelable", False):
        _logger.info(
            "analysis_route",
            decision="no_model",
            reason_code="not_modelable",
        )
        return "no_model"

    reason = str(stats.get("reason", ""))
    borderline = reason.startswith("borderline")
    n_rows = getattr(df, "shape", [0])[0] if hasattr(df, "shape") else len(df)

    use_hint = (
        ENABLE_LOCAL_LLM and llm_hint is not None and llm_hint.confidence >= 0.6
    )

    strong_hint = (
        use_hint
        and llm_hint.task in {"regression", "classification"}
        and llm_hint.complexity in {"high", "complex"}
        and llm_hint.confidence >= 0.9
    )

    heuristic = not borderline and reason == "ok"

    decision = "baseline"
    reason_code = "no_signal"

    if (
        n_rows <= MAX_ROWS_FULL
        and ALLOW_FULL_COMPARE_MODELS
        and (strong_hint or heuristic)
    ):
        decision = "full"
        reason_code = "strong_hint" if strong_hint else "heuristic"
    else:
        if n_rows > MAX_ROWS_FULL:
            reason_code = "too_many_rows"
        elif not ALLOW_FULL_COMPARE_MODELS:
            reason_code = "flag_off"
        elif borderline:
            reason_code = "borderline"

    hint_info = {}
    if llm_hint is not None:
        hint_info = {
            "hint_task": redact(llm_hint.task or ""),
            "hint_complexity": redact(llm_hint.complexity or ""),
            "hint_confidence": llm_hint.confidence,
        }

    _logger.info(
        "analysis_route",
        decision=decision,
        reason_code=reason_code,
        n_rows=n_rows,
        allow_full=ALLOW_FULL_COMPARE_MODELS,
        strong_hint=strong_hint,
        heuristic=heuristic,
        **hint_info,
    )

    return decision

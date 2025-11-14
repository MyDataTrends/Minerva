"""Utilities to infer the best visualization for a dataset."""

import json
import logging
from typing import Optional, Tuple

import pandas as pd

from preprocessing.llm_preprocessor import llm_completion
from preprocessing.metadata_parser import parse_metadata
from preprocessing.sanitize import redact
from utils.logging import log_decision
from config import LLM_VISUALIZATION_TOKEN_BUDGET


def _heuristic_visualization(df: pd.DataFrame) -> Tuple[str, float]:
    """Return a visualization guess and confidence using dataframe dtypes."""

    numeric_cols = df.select_dtypes(include="number").columns
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    if datetime_cols and len(numeric_cols) >= 1:
        return "line_chart", 0.9
    if len(categorical_cols) > 0 and len(numeric_cols) >= 1:
        return "bar_chart", 0.9
    if len(numeric_cols) >= 2:
        return "scatter_plot", 0.6
    if len(numeric_cols) == 1:
        return "histogram", 0.6
    return "line_chart", 0.5


def infer_visualization_type(
    question: str, df: pd.DataFrame, model_type: Optional[str] = None, industry: Optional[str] = None
) -> str:
    """Infer a suitable visualization type using heuristics with LLM fallback."""

    red_q = redact(question)
    red_model = redact(model_type) if model_type else model_type
    red_industry = redact(industry) if industry else industry

    # 1. Heuristic guess based on dataframe properties
    vis_type, confidence = _heuristic_visualization(df)
    if confidence >= 0.7:
        log_decision(
            red_q,
            {"visualization_type": vis_type, "confidence": confidence, "reasoning": "heuristic"},
        )
        return vis_type

    # 2. Build LLM prompt only when needed
    metadata = parse_metadata(df)
    prompt = (
        "Determine the best visualization for the provided dataset.\n"
        "Respond only in JSON with keys 'visualization_type' and 'reasoning'.\n"
        f"Question: {red_q}\n"
        f"Model type: {red_model}\n"
        f"Industry: {red_industry}\n"
        f"Columns: {metadata['columns']}\n"
        f"Types: {metadata['dtypes']}"
    )

    vis_type_llm = None
    reasoning = ""
    try:
        resp = llm_completion(prompt, max_tokens=LLM_VISUALIZATION_TOKEN_BUDGET)
        if resp not in {"LLM unavailable"} and not resp.startswith("LLM error"):
            result = json.loads(resp)
            vis_type_llm = result.get("visualization_type")
            reasoning = result.get("reasoning", "")
    except Exception as exc:  # pragma: no cover - runtime issues
        logging.getLogger(__name__).warning("LLM visualization parse failed: %s", exc)

    if vis_type_llm:
        log_decision(red_q, {"visualization_type": vis_type_llm, "reasoning": reasoning})
        return vis_type_llm

    log_decision(
        red_q,
        {"visualization_type": vis_type, "confidence": confidence, "reasoning": "heuristic"},
    )
    return vis_type


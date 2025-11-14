import json
import logging
from functools import lru_cache
from typing import Dict

from preprocessing.llm_preprocessor import load_mistral_model, run_mistral_inference
from preprocessing.sanitize import redact
from config import (
    MISTRAL_MODEL_PATH,
    LLM_INTENT_TOKEN_BUDGET,
    LLM_INTENT_TEMPERATURE,
)

_logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _load_model():
    """Load and cache the local Mistral model."""
    return load_mistral_model(MISTRAL_MODEL_PATH)


def _query_llm(prompt: str):
    """Query the LLM if available, otherwise return an error structure."""
    model = _load_model()
    if model is None:
        return {"status": "error", "reason": "LLM unavailable: model not loaded"}

    resp = run_mistral_inference(
        model,
        input=redact(prompt),
        max_tokens=LLM_INTENT_TOKEN_BUDGET,
        temperature=LLM_INTENT_TEMPERATURE,
    )
    if isinstance(resp, str) and (
        resp == "LLM unavailable" or resp.startswith("LLM error")
    ):
        return {"status": "error", "reason": resp}
    return resp


def modeling_needed(query: str) -> Dict[str, any]:
    """Return a decision on whether modeling is required for ``query``."""
    red_q = redact(query)
    prompt = (
        "Determine if the following user request requires predictive modeling. "
        "Respond only in JSON with keys 'modeling_required' and 'reasoning'.\n"
        f"Query: {red_q}"
    )
    resp = _query_llm(prompt)
    if isinstance(resp, dict) and resp.get("status") == "error":
        return resp
    try:
        result = json.loads(resp)
        return {
            "modeling_required": bool(result.get("modeling_required")),
            "reasoning": result.get("reasoning", ""),
        }
    except Exception as exc:
        _logger.warning("Stage 1 LLM parse failure: %s", exc)
        raise


def classify_modeling_type(query: str) -> Dict[str, str]:
    """Return the modeling type appropriate for ``query``."""
    red_q = redact(query)
    prompt = (
        "If modeling is needed, determine the modeling type (classification, "
        "regression, clustering, etc.). Respond only in JSON with keys "
        "'modeling_type' and 'reasoning'.\n"
        f"Query: {red_q}"
    )
    resp = _query_llm(prompt)
    if isinstance(resp, dict) and resp.get("status") == "error":
        return resp
    try:
        result = json.loads(resp)
        return {
            "modeling_type": result.get("modeling_type"),
            "reasoning": result.get("reasoning", ""),
        }
    except Exception as exc:
        _logger.warning("Stage 2 LLM parse failure: %s", exc)
        raise

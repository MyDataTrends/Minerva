from __future__ import annotations

from typing import Any, Mapping, Tuple

from .intent_parser import parse_intent


def decide_action(user_query: str) -> Tuple[str, Mapping[str, Any]]:
    """Determine high level action from a chat message.

    The function inspects the parsed intent and groups it into one of three
    broad categories:

    - ``"modeling"``: a direct request to run a model or scenario.
    - ``"visualization"``: the user wants a specific chart.
    - ``"analysis"``: no clear request was detected so the system should run
      automatic analysis.
    """

    result = parse_intent(user_query)
    if result is None:
        raise RuntimeError("LLM shard unavailable")
    intent, params = result

    if intent == "visualization":
        return "visualization", params
    if intent in {"modeling", "scenario"}:
        return "modeling", params
    return "analysis", {}

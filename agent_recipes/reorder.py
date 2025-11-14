"""Recipe for suggesting inventory reorder based on predictions."""

from __future__ import annotations

import logging
from typing import Any, Dict

from config import REORDER_API
from utils.net import request_with_retry

logger = logging.getLogger(__name__)


def suggest_reorder(output: Dict[str, Any]) -> None:
    """Send a reorder suggestion using ``REORDER_API``.

    Parameters
    ----------
    output : dict
        The model output to send as the request payload.
    """
    if not REORDER_API:
        logger.info("REORDER_API is not configured")
        return
    try:
        # Placeholder for an API call
        request_with_retry("post", REORDER_API, json=output, timeout=5)
        logger.info("Posted reorder suggestion to %s", REORDER_API)
    except Exception as exc:  # pragma: no cover - demo only
        logger.error("Failed to post reorder suggestion: %s", exc)

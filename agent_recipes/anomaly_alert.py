"""Recipe for alerting when anomalies are detected in metrics."""

from __future__ import annotations

import logging
from typing import Any, Dict

from config import ANOMALY_API
from utils.net import request_with_retry

logger = logging.getLogger(__name__)


def send_anomaly_alert(output: Dict[str, Any]) -> None:
    """Send an anomaly alert using ``ANOMALY_API``.

    Parameters
    ----------
    output : dict
        Model metrics or predictions indicating an anomaly.
    """
    if not ANOMALY_API:
        logger.info("ANOMALY_API is not configured")
        return
    try:
        # Placeholder for an API call
        request_with_retry("post", ANOMALY_API, json=output, timeout=5)
        logger.info("Posted anomaly alert to %s", ANOMALY_API)
    except Exception as exc:  # pragma: no cover - demo only
        logger.error("Failed to post anomaly alert: %s", exc)

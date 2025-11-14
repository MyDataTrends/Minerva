"""Simple agent for notifications and result logging."""

from __future__ import annotations

import json
from utils.logging import configure_logging, get_logger
import os
import re
import sqlite3
import smtplib
from datetime import datetime
from email.message import EmailMessage
from typing import Any, Dict, List
from config import sns_client, ENABLE_RECIPES, get_int


DB_PATH = os.getenv("ACTION_DB_PATH", "action_log.db")

configure_logging()
logger = get_logger(__name__)


def _parse_confidence(confidence: str) -> int:
    """Extract numeric confidence value from a string."""
    match = re.search(r"(\d+)", confidence or "")
    return int(match.group(1)) if match else 0


def execute_actions(result: Dict[str, Any]) -> List[str]:
    """Send a notification and store ``result`` in a SQLite DB.

    Parameters
    ----------
    result : dict
        Dictionary containing predictions and optional metadata.

    Returns
    -------
    list[str]
        A list of actions performed.
    """

    actions: List[str] = []

    predictions = result.get("predictions", [])
    message = f"Generated {len(predictions)} predictions."
    _send_notification(message)
    actions.append("notification_sent")

    confidence = _parse_confidence(result.get("confidence_score", ""))
    if confidence and confidence < 50:
        _send_notification("Low confidence detected in model output")
        actions.append("low_confidence_alert")

    _record_result(result)
    actions.append("record_saved")


    if _publish_sns(predictions, result.get("confidence_score"), actions):
        actions.append("sns_published")

    # Optional recipe execution
    if ENABLE_RECIPES:
        try:
            from agent_recipes import (
                suggest_reorder,
                send_anomaly_alert,
                request_role_review,
            )

            prev_mae = result.get("prev_mae")
            mae = result.get("mae")
            if prev_mae and mae and mae > 1.2 * prev_mae:
                send_anomaly_alert(result)
                actions.append("anomaly_alert")

            if predictions and min(predictions) < 0:
                suggest_reorder(result)
                actions.append("reorder_suggested")

            if result.get("needs_role_review"):
                request_role_review(result)
                actions.append("role_review_requested")
        except Exception as exc:  # pragma: no cover - demo only
            logger.error("Failed to execute agent recipes: %s", exc)


    logger.info("Action agent executed actions: %s", actions)
    return actions


def _send_notification(message: str) -> None:
    """Send an email if configured, otherwise log the message."""

    recipient = os.getenv("NOTIFY_EMAIL")
    if recipient:
        from_addr = os.getenv("SMTP_FROM", "noreply@example.com")
        smtp_server = os.getenv("SMTP_SERVER", "localhost")
        smtp_port = get_int("SMTP_PORT", 25)
        smtp_user = os.getenv("SMTP_USER")
        smtp_pass = os.getenv("SMTP_PASS")

        try:
            email = EmailMessage()
            email["From"] = from_addr
            email["To"] = recipient
            email["Subject"] = "Model Results Notification"
            email.set_content(message)
            with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
                server.starttls()
                if smtp_user and smtp_pass:
                    server.login(smtp_user, smtp_pass)
                server.send_message(email)
            logger.info("Sent email notification to %s", recipient)
            return
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Email notification failed: %s", exc)

    logger.info("Notification: %s", message)


def _record_result(result: Dict[str, Any]) -> None:
    """Write the result to a local SQLite database for demo purposes."""

    try:
        conn = sqlite3.connect(DB_PATH)
        with conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, data TEXT)"
            )
            conn.execute(
                "INSERT INTO results (timestamp, data) VALUES (?, ?)",
                (datetime.utcnow().isoformat(), json.dumps(result)),
            )
    except Exception as exc:  # pragma: no cover - demo only
        logger.error("Failed to record result: %s", exc)


def _publish_sns(predictions: List[Any], confidence: Any, actions: List[str]) -> bool:
    """Publish a summary of ``predictions`` to an SNS topic if configured.

    Returns
    -------
    bool
        ``True`` if the message was published successfully.
    """

    topic_arn = os.getenv("SNS_TOPIC_ARN")
    if sns_client is None or not topic_arn:
        return False

    payload = json.dumps(
        {
            "predictions": predictions,
            "confidence": confidence,
            "actions": actions,
        }
    )

    try:  # pragma: no cover - optional cloud
        sns_client.publish(TopicArn=topic_arn, Message=payload)
        logger.info("Published action summary to SNS topic %s", topic_arn)
        return True
    except Exception as exc:  # pragma: no cover - network failures
        logger.error("Failed to publish to SNS: %s", exc)
    return False

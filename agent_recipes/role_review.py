"""Recipe to trigger a role review workflow."""

from __future__ import annotations

import logging
from typing import Any, Dict

from config import ROLE_REVIEW_API
from utils.net import request_with_retry
from feedback.role_corrections import store_role_corrections_by_hash

logger = logging.getLogger(__name__)


def request_role_review(output: Dict[str, Any]) -> None:
    """Prompt for corrected roles and persist them."""
    if ROLE_REVIEW_API:
        try:
            request_with_retry("post", ROLE_REVIEW_API, json=output, timeout=5)
            logger.info("Requested role review via %s", ROLE_REVIEW_API)
        except Exception as exc:  # pragma: no cover - demo only
            logger.error("Failed to request role review: %s", exc)

    cols = output.get("column_roles")
    data_hash = output.get("data_hash")
    if not cols or not data_hash:
        return

    print("--- Role Review ---")
    print("Enter corrected role for each column (blank to keep current)")
    corrections = {}
    for col, role in cols.items():
        new_role = input(f"{col} [{role}]: ").strip()
        if new_role:
            corrections[col] = new_role
    if corrections:
        store_role_corrections_by_hash(data_hash, corrections)
        logger.info("Stored %d role corrections", len(corrections))

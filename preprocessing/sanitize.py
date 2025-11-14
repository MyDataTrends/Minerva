"""Utilities for redacting PII from text and DataFrames."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from config import get_bool


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]*)?(?:\(\d{3}\)|\d{3})[-.\s]*\d{3}[-.\s]*\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CC_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
ADDRNUM_RE = re.compile(r"\b\d{1,5}(?=\s+[A-Za-z])")


_PATTERNS: Iterable[tuple[re.Pattern[str], str]] = (
    (EMAIL_RE, "[EMAIL]"),
    (PHONE_RE, "[PHONE]"),
    (SSN_RE, "[SSN]"),
    (CC_RE, "[CC]"),
    (IP_RE, "[IP]"),
    (UUID_RE, "[UUID]"),
    (ADDRNUM_RE, "[ADDRNUM]"),
)


def redact(text: str) -> str:
    """Redact PII from ``text`` using compiled regex patterns."""

    if not get_bool("REDACTION_ENABLED", True) or not isinstance(text, str):
        return text

    for pattern, token in _PATTERNS:
        text = pattern.sub(token, text)
    return text


def scrub_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with PII removed from object columns."""

    if not get_bool("REDACTION_ENABLED", True):
        return df

    scrubbed = df.copy(deep=False)
    for col in scrubbed.select_dtypes(include=["object"]).columns:
        scrubbed[col] = scrubbed[col].map(lambda x: redact(x) if isinstance(x, str) else x)
    return scrubbed


__all__ = ["redact", "scrub_df"]


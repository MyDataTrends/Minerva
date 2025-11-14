"""UI package helpers."""

from __future__ import annotations

import streamlit as st

from config import REDACTION_ENABLED


def redaction_banner() -> None:
    """Display a banner when data redaction is enabled."""
    if REDACTION_ENABLED:
        st.info("Redaction active – sensitive values hidden.")


import logging
import re
from typing import Any
import pandas as pd
import structlog

from pathlib import Path
from config import LOG_LEVEL, LOG_DIR, LOG_FILE
from preprocessing.sanitize import redact, scrub_df

_configured = False

_SENSITIVE_KEYS = {"prompt", "question", "email", "phone", "ssn", "card"}
_TOKEN_RE = re.compile(r"^\[[A-Z]+\]$")


def _redact_processor(_logger, _name, event_dict):
    for key in _SENSITIVE_KEYS:
        if key in event_dict:
            value = event_dict[key]
            if isinstance(value, str) and not _TOKEN_RE.fullmatch(value):
                event_dict[key] = redact(value)
    return event_dict

def configure_logging() -> None:
    """Configure structlog for JSON console logging once."""
    global _configured
    if _configured:
        return

    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / LOG_FILE
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_path), encoding="utf-8"),
        ],
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            _redact_processor,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    _configured = True

def get_logger(name: str):
    """Return a configured structlog logger."""
    configure_logging()
    return structlog.get_logger(name)

def log_decision(query: str, decision: dict) -> None:
    """Log the LLM modeling decision, redacting sensitive data when enabled."""
    logger = get_logger("intent")
    scrubbed: dict[str, Any] = {}
    for key, value in decision.items():
        if isinstance(value, pd.DataFrame):
            scrubbed[key] = scrub_df(value.head(50))
        else:
            scrubbed[key] = value
    logger.info("intent_decision", query=redact(query), **scrubbed)

def log_metrics(source: str, metrics: dict) -> None:
    """Log metrics dictionary using structlog."""
    get_logger("metrics").info("metrics", source=source, metrics=metrics)

# expose via the built-in logging module
setattr(logging, "log_metrics", log_metrics)
setattr(logging, "log_decision", log_decision)

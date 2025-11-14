import importlib

from preprocessing.sanitize import redact


def test_redact_email_and_phone():
    text = "Contact me at alice@example.com or 555-123-4567"
    redacted = redact(text)
    assert "[EMAIL]" in redacted
    assert "[PHONE]" in redacted


def test_redact_disabled(monkeypatch):
    monkeypatch.setenv("REDACTION_ENABLED", "0")
    import config
    import preprocessing.sanitize as sanitize

    importlib.reload(config)
    importlib.reload(sanitize)

    text = "Reach me at bob@example.com"
    assert sanitize.redact(text) == text


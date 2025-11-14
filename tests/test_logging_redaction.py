import json
import importlib
import logging as pylogging

from utils import logging as logging_utils


def test_logging_redacts_pii(capsys):
    pylogging.getLogger().handlers.clear()
    importlib.reload(logging_utils)
    logger = logging_utils.get_logger("redact")
    logger.info(
        "event",
        prompt="call 555-123-4567",
        question="my email test@example.com",
        email="test@example.com",
        phone="555-123-4567",
        ssn="123-45-6789",
        card="4111111111111111",
    )
    out = capsys.readouterr().err.strip()
    log = json.loads(out)
    assert log["prompt"] == "call [PHONE]"
    assert log["question"] == "my email [EMAIL]"
    assert log["email"] == "[EMAIL]"
    assert log["phone"] == "[PHONE]"
    assert log["ssn"] == "[SSN]"
    assert log["card"] == "[CC]"
    assert "test@example.com" not in out
    assert "555-123-4567" not in out
    assert "123-45-6789" not in out
    assert "4111111111111111" not in out

    logger.info("event", email="[EMAIL]")
    out = capsys.readouterr().err.strip()
    log = json.loads(out)
    assert log["email"] == "[EMAIL]"

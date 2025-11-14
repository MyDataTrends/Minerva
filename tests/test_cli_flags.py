import os
import sys


def test_cli_flags_set_env(monkeypatch):
    for key in [
        "ENABLE_LOCAL_LLM",
        "ENABLE_PROMETHEUS",
        "REDACTION_ENABLED",
        "LOCAL_DEV_LENIENT",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--no-llm",
            "--enable-prometheus",
            "--safe-logs",
            "--dev-lenient",
        ],
    )
    sys.modules.pop("main", None)
    import main  # noqa: F401 - imported for side effects

    assert os.getenv("ENABLE_LOCAL_LLM") == "0"
    assert os.getenv("ENABLE_PROMETHEUS") == "1"
    assert os.getenv("REDACTION_ENABLED") == "1"
    assert os.getenv("LOCAL_DEV_LENIENT") == "1"

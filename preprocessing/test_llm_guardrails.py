import logging

from preprocessing.llm_preprocessor import llm_completion


class EchoLLM:
    def __init__(self):
        self.prompts = []

    def create_completion(self, prompt, max_tokens):  # pragma: no cover - simple mock
        self.prompts.append(prompt)
        return {"choices": [{"text": prompt}]}


def test_truncate_long_input(monkeypatch, caplog):
    dummy = EchoLLM()
    monkeypatch.setattr("preprocessing.llm_preprocessor.load_local_llm", lambda: dummy)
    monkeypatch.setattr("preprocessing.llm_preprocessor.LLM_MAX_INPUT_CHARS", 10)
    caplog.set_level(logging.WARNING, logger="preprocessing.llm_preprocessor")

    result = llm_completion("a" * 20, max_tokens=5)

    assert "[TRUNCATED]" in result
    assert "[TRUNCATED]" in dummy.prompts[0]
    assert any("truncated" in rec.message for rec in caplog.records)


def test_rejects_binary_input(monkeypatch):
    dummy = EchoLLM()
    monkeypatch.setattr("preprocessing.llm_preprocessor.load_local_llm", lambda: dummy)
    monkeypatch.setattr("preprocessing.llm_preprocessor.LLM_NON_PRINTABLE_THRESHOLD", 0.1)

    payload = "\x00" * 20 + "abc"
    result = llm_completion(payload, max_tokens=5)

    assert result == "LLM input rejected"
    assert dummy.prompts == []


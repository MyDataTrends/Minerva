import threading
import time
from unittest.mock import patch

from config import LLM_MAX_CONCURRENCY
from preprocessing.llm_preprocessor import llm_completion


class DummyLLM:
    def __init__(self):
        self.calls = 0

    def create_completion(self, prompt, max_tokens):
        self.calls += 1
        return {"choices": [{"text": f"{prompt}-{max_tokens}"}]}


def test_llm_completion_caches_within_ttl():
    dummy = DummyLLM()
    with patch("preprocessing.llm_preprocessor.load_local_llm", return_value=dummy):
        first = llm_completion("hello", max_tokens=10)
        second = llm_completion("hello", max_tokens=10)
    assert first == second
    assert dummy.calls == 1


class SlowLLM:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.lock = threading.Lock()

    def create_completion(self, prompt, max_tokens):
        with self.lock:
            self.active += 1
            self.calls += 1
            if self.active > self.max_active:
                self.max_active = self.active
        time.sleep(0.1)
        with self.lock:
            self.active -= 1
        return {"choices": [{"text": prompt}]}


def test_llm_completion_concurrency_cap():
    dummy = SlowLLM()
    with patch("preprocessing.llm_preprocessor.load_local_llm", return_value=dummy):
        def call(p):
            llm_completion(p, max_tokens=5)
        threads = [threading.Thread(target=call, args=(f"p{i}",)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert dummy.calls == 3
    assert dummy.max_active <= LLM_MAX_CONCURRENCY

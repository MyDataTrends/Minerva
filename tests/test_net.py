import requests
import responses
import time
import pytest

from utils.net import request_with_retry


@responses.activate
def test_retry_on_5xx(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    responses.add(responses.GET, "http://example.com", status=500)
    responses.add(responses.GET, "http://example.com", json={"ok": True}, status=200)
    resp = request_with_retry("get", "http://example.com", max_attempts=3, jitter=False)
    assert resp.status_code == 200
    assert len(responses.calls) == 2


@responses.activate
def test_retry_on_timeout(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    responses.add(responses.GET, "http://timeout.com", body=requests.exceptions.Timeout())
    responses.add(responses.GET, "http://timeout.com", json={"ok": True}, status=200)
    resp = request_with_retry("get", "http://timeout.com", max_attempts=2, jitter=False)
    assert resp.status_code == 200
    assert len(responses.calls) == 2


@responses.activate
def test_raises_after_max_attempts(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    responses.add(responses.POST, "http://bad.com", status=500)
    responses.add(responses.POST, "http://bad.com", status=500)
    with pytest.raises(requests.HTTPError):
        request_with_retry("post", "http://bad.com", max_attempts=2, jitter=False)
    assert len(responses.calls) == 2

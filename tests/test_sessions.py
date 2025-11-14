import importlib
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "sessions.db"
    monkeypatch.setenv("SESSIONS_DB_PATH", str(db_path))
    # Reload modules to pick up env var
    import storage.session_db as sd
    importlib.reload(sd)
    import main
    importlib.reload(main)
    client = TestClient(main.app)
    return client, sd


def test_list_and_get_sessions(client):
    client, sd = client
    sd.record_session("r1", {"foo": "bar"}, {"res": 1})
    sd.record_session("r2", {"foo": "baz"}, {"res": 2})

    resp = client.get("/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["run_id"] == "r2"

    resp = client.get("/sessions/r1")
    assert resp.status_code == 200
    detail = resp.json()
    assert json.loads(detail["params"])["foo"] == "bar"


def test_get_session_not_found(client):
    client, _ = client
    resp = client.get("/sessions/does-not-exist")
    assert resp.status_code == 404

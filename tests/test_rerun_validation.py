import importlib
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "sessions.db"
    monkeypatch.setenv("SESSIONS_DB_PATH", str(db_path))
    import storage.session_db as sd
    importlib.reload(sd)
    import main
    importlib.reload(main)
    client = TestClient(main.app)
    return client, sd


def test_rerun_rejects_unknown_field(client):
    client, sd = client
    sd.record_session("r1", {"foo": "bar"}, {"res": 1})
    resp = client.post("/sessions/r1/rerun", json={"foo": "bar"})
    assert resp.status_code == 400
    assert resp.json() == {"detail": "Invalid payload"}


def test_rerun_rejects_long_string(client):
    client, sd = client
    sd.record_session("r1", {"foo": "bar"}, {"res": 1})
    long_str = "a" * 101
    resp = client.post("/sessions/r1/rerun", json={"user_id": long_str})
    assert resp.status_code == 400
    assert resp.json() == {"detail": "Invalid payload"}

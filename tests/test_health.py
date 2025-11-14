from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main

client = TestClient(main.app)

def test_healthz():
    resp = client.get('/healthz')
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

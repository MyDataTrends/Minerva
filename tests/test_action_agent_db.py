import json
import sqlite3
from pathlib import Path

import agents.action_agent as aa


def test_execute_actions_records_extra_fields(tmp_path, monkeypatch):
    db_path = tmp_path / "actions.db"
    monkeypatch.setenv("ACTION_DB_PATH", str(db_path))
    monkeypatch.setattr(aa, "DB_PATH", str(db_path), raising=False)
    payload = {
        "predictions": [1, 2],
        "mae": 0.5,
        "file_name": "sample.csv",
        "model_type": "DummyModel",
        "metadata_file": "meta.json",
        "output_path": "out.csv",
    }
    aa.execute_actions(payload)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT data FROM results")
    row = cur.fetchone()[0]
    data = json.loads(row)
    assert data["file_name"] == "sample.csv"
    assert data["model_type"] == "DummyModel"
    assert data["metadata_file"] == "meta.json"
    assert data["output_path"] == "out.csv"

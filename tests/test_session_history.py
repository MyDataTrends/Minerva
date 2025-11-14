import sqlite3
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from storage.local_backend import log_run_metadata


def test_log_run_metadata(tmp_path):
    db = tmp_path / "history.db"
    os.environ["SESSION_HISTORY_DB"] = str(db)
    log_run_metadata(
        "run123",
        True,
        False,
        user_id="u1",
        file_name="file.csv",
        model_name="ModelA",
        metadata_path="meta.json",
        output_path="model.pkl",
        db_path=str(db),
    )
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT run_id, user_id, file_name, model_name, metadata_path, output_path, score_ok, needs_role_review FROM session_history").fetchone()
    assert row == (
        "run123",
        "u1",
        "file.csv",
        "ModelA",
        "meta.json",
        "model.pkl",
        1,
        0,
    )

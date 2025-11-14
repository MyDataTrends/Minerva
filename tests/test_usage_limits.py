import pandas as pd
from pathlib import Path
import orchestrate_workflow as ow
from storage.local_backend import LocalStorage
from utils import usage_tracker


def _setup_file(tmp_path: Path, rows: int = 5) -> Path:
    user_dir = tmp_path / "User_Data" / "u1"
    user_dir.mkdir(parents=True)
    df = pd.DataFrame({"sales": range(rows), "feat": range(rows)})
    path = user_dir / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_limits_block(monkeypatch, tmp_path: Path):
    _setup_file(tmp_path)
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ow, "backend", LocalStorage(base_dir=tmp_path))
    monkeypatch.setattr(ow, "MAX_REQUESTS_FREE", 1)
    monkeypatch.setattr(ow, "MAX_GB_FREE", 1.0)

    res1 = ow.orchestrate_workflow("u1", "data.csv", {})
    assert "error" not in res1

    res2 = ow.orchestrate_workflow("u1", "data.csv", {})
    assert res2.get("error")
    info = usage_tracker.get_usage("u1")
    assert info["requests"] == 2


def test_dynamo_limits_block(monkeypatch, tmp_path: Path, usage_table):
    _setup_file(tmp_path)
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ow, "backend", LocalStorage(base_dir=tmp_path))
    monkeypatch.setattr(ow, "MAX_REQUESTS_FREE", 1)
    monkeypatch.setattr(ow, "MAX_GB_FREE", 1.0)

    res1 = ow.orchestrate_workflow("u1", "data.csv", {})
    assert "error" not in res1

    res2 = ow.orchestrate_workflow("u1", "data.csv", {})
    assert res2.get("error")
    info = usage_tracker.get_usage("u1")
    assert info["requests"] == 2

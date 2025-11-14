import importlib
import json
from pathlib import Path
import pandas as pd

from storage.local_backend import LocalStorage


def _setup_files(tmp_path: Path) -> None:
    user_dir = tmp_path / "User_Data" / "u1"
    user_dir.mkdir(parents=True)
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_csv(user_dir / "f1.csv", index=False)
    df.to_csv(user_dir / "f2.csv", index=False)


def test_orchestrate_rerun_block_free(monkeypatch, tmp_path: Path):
    profile = tmp_path / "profiles.json"
    profile.write_text(json.dumps({"u1": {"tier": "free"}}))
    monkeypatch.setenv("USER_PROFILE_PATH", str(profile))
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    import orchestrate_workflow as ow
    importlib.reload(ow)
    monkeypatch.setattr(ow, "backend", LocalStorage(base_dir=tmp_path))
    res = ow.orchestrate_workflow("u1", "f1.csv", {}, run_id="r1")
    assert res["error"] == "Feature not available for free-tier users"


def test_orchestrate_rerun_allowed_premium(monkeypatch, tmp_path: Path):
    profile = tmp_path / "profiles.json"
    profile.write_text(json.dumps({"u1": {"tier": "premium"}}))
    monkeypatch.setenv("USER_PROFILE_PATH", str(profile))
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    import orchestrate_workflow as ow
    importlib.reload(ow)
    monkeypatch.setattr(ow, "backend", LocalStorage(base_dir=tmp_path))
    # will error due to missing run metadata but should not hit tier error
    res = ow.orchestrate_workflow("u1", "f1.csv", {}, run_id="r1")
    assert res.get("error") != "Feature not available for free-tier users"


def test_process_blocks_free_bulk(monkeypatch, tmp_path: Path):
    _setup_files(tmp_path)
    profile = tmp_path / "profiles.json"
    profile.write_text(json.dumps({"u1": {"tier": "free"}}))
    monkeypatch.setenv("USER_PROFILE_PATH", str(profile))
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("USE_CLOUD", "False")
    import Purchase_handler.purchase_handler as ph
    importlib.reload(ph)
    monkeypatch.setattr(ph, "load_datalake_dfs", lambda: {})
    monkeypatch.setattr(ph, "orchestrate_workflow", lambda *a, **k: {"ok": True})
    client = ph.app.test_client()
    resp = client.post("/process", json={"user_id": "u1"})
    assert resp.status_code == 403
    assert resp.get_json()["error"] == "Feature not available for free-tier users"


from pathlib import Path
import pandas as pd

import orchestrate_workflow as ow
from storage.local_backend import LocalStorage
import config
from orchestration import analyzer_selector_helper as ash


def _setup_file(tmp_path: Path, rows: int) -> Path:
    user_dir = tmp_path / "User_Data" / "u1"
    user_dir.mkdir(parents=True)
    df = pd.DataFrame({"sales": range(rows), "feat": range(rows)})
    path = user_dir / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_workflow_fallback_success(tmp_path: Path, monkeypatch):
    _setup_file(tmp_path, 5)
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ow, "backend", LocalStorage(base_dir=tmp_path))
    monkeypatch.setattr(config, "MIN_ROWS", 10)
    monkeypatch.setattr(ash, "MIN_ROWS", 10)
    res = ow.orchestrate_workflow("u1", "data.csv", {})
    assert res.get("modeling_skipped") and "error" not in res


def test_workflow_validation_error(tmp_path: Path, monkeypatch):
    _setup_file(tmp_path, 3)
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ow, "backend", LocalStorage(base_dir=tmp_path))
    monkeypatch.setattr(ow, "setup", None)
    res = ow.orchestrate_workflow("u1", "data.csv", {})
    assert res.get("error")


def test_workflow_modeling_failure(tmp_path: Path, monkeypatch):
    _setup_file(tmp_path, 6)
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ow, "backend", LocalStorage(base_dir=tmp_path))
    monkeypatch.setattr(config, "MIN_ROWS", 5)
    monkeypatch.setattr(ash, "MIN_ROWS", 5)
    monkeypatch.setattr(ash, "assess_modelability", lambda *a, **k: {"is_modelable": True, "reason": "ok", "task": "regression"})
    monkeypatch.setattr(ow, "setup", lambda *a, **k: None)

    class DummyModel:
        def predict(self, X):
            return [0] * len(X)

    monkeypatch.setattr(ow, "train_model", lambda *a, **k: DummyModel())
    monkeypatch.setattr(ow, "save_model", lambda *a, **k: None)

    def fake_eval(*_a, **_k):
        return {"r2": 0.0, "mape": 100, "mae": 1}

    monkeypatch.setattr(ow, "evaluate_model", fake_eval)

    called = {"dash": False}

    def fake_dash(*a, **k):
        called["dash"] = True

    monkeypatch.setattr(ow, "orchestrate_dashboard", fake_dash)
    monkeypatch.setattr(ow, "generate_summary", lambda *a, **k: {"summary": ""})

    res = ow.orchestrate_workflow("u1", "data.csv", {})
    assert res.get("modeling_failed")
    assert not called["dash"]
    assert res.get("analysis_type") == "descriptive"

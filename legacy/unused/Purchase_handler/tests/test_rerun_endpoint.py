import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression

import Purchase_handler.purchase_handler as ph
import orchestrate_workflow as ow
from storage.local_backend import LocalStorage, log_run_metadata, load_run_metadata
from preprocessing.metadata_parser import infer_column_meta
from modeling.model_training import save_model, load_model


import pytest


@pytest.mark.xfail(reason="Rerun flow unstable with simplified model")
def test_rerun_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    storage = LocalStorage(base_dir=tmp_path)
    monkeypatch.setattr(ph, "local_storage", storage, raising=False)
    monkeypatch.setattr(ow, "backend", storage)
    monkeypatch.setattr(ph, "load_datalake_dfs", lambda: {})
    monkeypatch.setattr(ph, "get_user_tier", lambda uid: "premium", raising=False)
    monkeypatch.setattr(ow, "get_user_tier", lambda uid: "premium")

    df = pd.DataFrame({"feat": [1, 2, 3], "sales": [2, 4, 6]})
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    dataset_path = dataset_dir / "data.csv"
    df.to_csv(dataset_path, index=False)

    model = LinearRegression().fit(df[["feat"]], df["sales"])
    save_model(model, "model.pkl")
    model_path = str(Path.cwd() / "models" / "model.pkl")

    roles = {m.name: m.role for m in infer_column_meta(df)}
    history = tmp_path / "run_history.json"
    log_run_metadata(
        "abc123",
        True,
        False,
        file_path=history,
        dataset_path=str(dataset_path),
        model_path=model_path,
        column_roles=roles,
        user_id="u1",
        file_name="data.csv",
    )

    monkeypatch.setattr(ph, "load_run_metadata", lambda rid: load_run_metadata(rid, file_path=history), raising=False)
    monkeypatch.setattr(ow, "load_run_metadata", lambda rid: load_run_metadata(rid, file_path=history))
    monkeypatch.setattr(ow, "load_model", load_model)
    monkeypatch.setattr(ow, "generate_summary", lambda **_: {"summary": "", "artifacts": {}})

    client = ph.app.test_client()
    resp = client.get("/rerun/abc123")
    data = resp.get_json()
    assert resp.status_code == 200
    assert "error" not in data

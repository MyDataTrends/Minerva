import json
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression

import orchestrate_workflow as ow
from storage.local_backend import LocalStorage, log_run_metadata, load_run_metadata
from preprocessing.metadata_parser import infer_column_meta
from modeling.model_training import save_model


def test_reproduce_from_run_id(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ow, "backend", LocalStorage(base_dir=tmp_path))
    profile = tmp_path / "profiles.json"
    profile.write_text(json.dumps({"u": {"tier": "premium"}}))
    monkeypatch.setenv("USER_PROFILE_PATH", str(profile))

    df = pd.DataFrame({"feat": [1, 2, 3, 4, 5], "sales": [2, 4, 6, 8, 10]})
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    dataset_path = dataset_dir / "data.csv"
    df.to_csv(dataset_path, index=False)

    model = LinearRegression()
    model.fit(df[["feat"]], df["sales"])
    model_path = Path.cwd() / "models" / "model.pkl"
    save_model(model, "model.pkl")

    meta = infer_column_meta(df)
    roles = {m.name: m.role for m in meta}

    history = tmp_path / "run_history.json"
    log_run_metadata(
        "r1",
        True,
        False,
        file_path=history,
        dataset_path=str(dataset_path),
        model_path=str(model_path),
        column_roles=roles,
    )

    monkeypatch.setattr(ow, "load_run_metadata", lambda rid: load_run_metadata(rid, file_path=history))
    from modeling.model_training import load_model as lm
    monkeypatch.setattr(ow, "load_model", lm)
    monkeypatch.setattr(ow, "generate_summary", lambda **_: {"summary": "", "artifacts": {}})

    res = ow.orchestrate_workflow("u", "x.csv", {}, target_column="sales", run_id="r1")
    assert res["run_id"] == "r1"
    assert res["metrics"]["rerun"]["mae"] == 0.0

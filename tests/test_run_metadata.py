import json

from storage.local_backend import log_run_metadata, load_run_metadata


def test_log_and_load_run_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    run_id = "123"
    dataset_path = str(tmp_path / "User_Data" / "u1" / "data.csv")
    target = "sales"
    model_path = str(tmp_path / "model.pkl")
    roles_path = str(tmp_path / "roles.json")

    log_run_metadata(
        run_id,
        True,
        False,
        dataset_path=dataset_path,
        target_column=target,
        model_path=model_path,
        roles_path=roles_path,
    )

    history_file = tmp_path / "run_history.json"
    assert history_file.exists()
    data = json.loads(history_file.read_text())
    assert data[0]["dataset_path"] == dataset_path
    assert data[0]["model_path"] == model_path

    loaded = load_run_metadata(run_id)
    assert loaded["target_column"] == target
    assert loaded["roles_path"] == roles_path

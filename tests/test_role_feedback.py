import importlib
import pandas as pd
from preprocessing import metadata_parser as mp
from feedback.role_corrections import store_role_corrections


def test_feedback_applied(monkeypatch, tmp_path):
    df = pd.DataFrame({"Amt": [1, 2]})
    path = tmp_path / "corr.json"
    store_role_corrections(df, {"Amt": "sales_amount"}, file_path=str(path))
    import preprocessing.column_meta as cm
    cm = importlib.reload(cm)
    monkeypatch.setattr(cm, "FEEDBACK_PATH", path, raising=False)
    cm.apply_role_feedback.__defaults__ = (path,)
    cm._load_role_corrections.__defaults__ = (path,)
    importlib.reload(mp)
    meta = mp.infer_column_meta(df)
    assert any(m.role == "sales_amount" and m.source == "feedback" for m in meta)

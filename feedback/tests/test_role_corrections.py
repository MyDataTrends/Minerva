import pandas as pd
from feedback.role_corrections import (
    store_role_corrections,
    load_role_corrections,
    store_role_corrections_by_hash,
    load_role_corrections_by_hash,
)
from preprocessing.save_meta import _hash_df


def test_store_and_load(tmp_path):
    df = pd.DataFrame({"A": [1, 2]})
    path = tmp_path / "corr.json"
    store_role_corrections(df, {"A": "amount"}, file_path=str(path))
    out = load_role_corrections(df, file_path=str(path))
    assert out == {"A": "amount"}


def test_store_by_hash(tmp_path):
    df = pd.DataFrame({"B": [1]})
    h = _hash_df(df)
    path = tmp_path / "corr.json"
    store_role_corrections_by_hash(h, {"B": "qty"}, file_path=str(path))
    out = load_role_corrections_by_hash(h, file_path=str(path))
    assert out == {"B": "qty"}

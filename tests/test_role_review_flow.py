import json
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.column_meta import ColumnMeta
from preprocessing.metadata_parser import infer_column_meta, merge_user_labels
from preprocessing.save_meta import save_column_roles
from utils.role_mapper import map_description_to_role


def test_merge_and_save(tmp_path):
    df = pd.DataFrame({"Amt": [1, 2], "Date": ["2024-01-01", "2024-01-02"]})
    meta = infer_column_meta(df)
    merged = merge_user_labels(meta, {"Amt": "sales_amount"})
    file_path = save_column_roles(df, merged, dest_dir=tmp_path)
    assert Path(file_path).exists()
    data = json.loads(Path(file_path).read_text())
    assert data["Amt"]["role"] == "sales_amount"


def test_description_saved(tmp_path):
    df = pd.DataFrame({"Amt": [1, 2]})
    meta = [
        ColumnMeta(
            name="Amt",
            role="amount",
            confidence=1.0,
            description="Total amount",
        )
    ]
    file_path = save_column_roles(df, meta, dest_dir=tmp_path)
    data = json.loads(Path(file_path).read_text())
    assert data["Amt"]["description"] == "Total amount"


def test_description_mapping(monkeypatch):
    def fake_load_roles():
        return ["role_a", "role_b"]

    def fake_encode(texts):
        import numpy as np

        return np.array(
            [
                [1.0, 0.0],  # description
                [0.1, 0.1],  # role_a
                [1.0, 0.0],  # role_b -> exact match
            ]
        )

    monkeypatch.setattr("utils.role_mapper._load_roles", fake_load_roles)
    monkeypatch.setattr("utils.role_mapper._encode", fake_encode)

    role = map_description_to_role("desc")
    assert role == "role_b"

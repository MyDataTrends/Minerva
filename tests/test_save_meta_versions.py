import time
import pandas as pd

from preprocessing.column_meta import ColumnMeta
from preprocessing.save_meta import (
    save_column_roles,
    load_column_roles,
    save_column_descriptions,
    load_column_descriptions,
)


def test_save_and_load_with_identifier(tmp_path):
    df = pd.DataFrame({"A": [1, 2]})
    meta = [ColumnMeta(name="A", role="amount", confidence=1.0)]
    desc = {"A": "Amt"}
    ident = "v1"
    save_column_roles(df, meta, dest_dir=tmp_path, identifier=ident)
    save_column_descriptions(df, desc, dest_dir=tmp_path, identifier=ident)
    roles = load_column_roles(df, dest_dir=tmp_path, identifier=ident)
    descriptions = load_column_descriptions(df, dest_dir=tmp_path, identifier=ident)
    assert roles["A"]["role"] == "amount"
    assert descriptions["A"] == "Amt"


def test_load_latest_version(tmp_path):
    df = pd.DataFrame({"A": [1]})
    meta1 = [ColumnMeta(name="A", role="old", confidence=1.0)]
    meta2 = [ColumnMeta(name="A", role="new", confidence=1.0)]
    save_column_roles(df, meta1, dest_dir=tmp_path, identifier="old")
    time.sleep(0.01)
    save_column_roles(df, meta2, dest_dir=tmp_path, identifier="new")
    roles = load_column_roles(df, dest_dir=tmp_path)
    assert roles["A"]["role"] == "new"


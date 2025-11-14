import pytest
import pandas as pd
from pathlib import Path

pytest.importorskip("sklearn")

from preprocessing.data_categorization import generate_tags, store_tags, get_tags


def test_generate_tags():
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    tags = generate_tags(df)
    assert set(tags.keys()) == {"columns", "dtypes", "llm_tags"}
    assert tags["columns"] == ["A", "B"]


def test_store_and_get_tags(tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    tags = {"columns": ["A"], "dtypes": {"A": "int64"}, "llm_tags": "test"}
    store_tags("sample", tags, path=catalog)
    loaded = get_tags("sample", path=catalog)
    assert loaded == tags

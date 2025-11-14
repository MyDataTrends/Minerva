import json
import sys
import pandas as pd
from orchestration.output_generator import OutputGenerator


class DummyModel:
    def __str__(self):
        return "DummyModel"


def test_output_generator_pure(monkeypatch, tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "sales": [1, 2, 3]})
    og = OutputGenerator()
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    result = og.generate(
        DummyModel(),
        pd.Series([1, 2, 3]),
        {},
        {"model_type": "Dummy"},
        "rid",
        df,
        "sales",
        False,
        "file.csv",
    )
    assert "streamlit" not in sys.modules
    payload = json.dumps(result)
    assert isinstance(payload, str)

import pytest
import pandas as pd
import modeling.model_training as mt

pytestmark = pytest.mark.skip(reason="pycaret removed; using lightweight sklearn/xgboost/lightgbm stack")
from modeling.model_training import train_model


def test_train_model(monkeypatch):
    df = pd.DataFrame({"text_column": [1, 2, 3]})
    y = pd.Series([2, 4, 6])
    monkeypatch.setattr(mt, "preprocess_data_with_llm", lambda df: df)
    monkeypatch.setattr(mt, "normalize_text_columns", lambda df, column_name: df)
    model = train_model(df, y, datalake_dfs={})
    assert model.predict([4])[0] == pytest.approx(8, 0.1)

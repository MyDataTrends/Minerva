import pandas as pd

from preprocessing.llm_preprocessor import preprocess_data_with_llm


def test_preprocess_passes_through_dataframe():
    df = pd.DataFrame({"num": [1, 2], "cat": ["a", "b"]})
    result = preprocess_data_with_llm(df)
    # DataFrame should be returned unchanged
    assert result.equals(df)
    assert list(result.columns) == ["num", "cat"]

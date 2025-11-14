import pytest
import pandas as pd

pytest.importorskip("sklearn")

from preprocessing.llm_preprocessor import score_similarity


def test_score_similarity_with_tags():
    uploaded_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    datalake_dfs = {
        "similar.csv": pd.DataFrame({"A": [10, 20], "B": [30, 40]}),
        "different.csv": pd.DataFrame({"X": [1, 2], "Y": [3, 4]}),
    }

    results = score_similarity(uploaded_df, datalake_dfs)
    assert results[0][0] == "similar.csv"
    assert results[0][1] >= results[1][1]

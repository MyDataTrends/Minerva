import pandas as pd
import numpy as np

from scripts.imputation_confidence import score_imputations


def test_score_imputations_basic():
    df_orig = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [10.0, np.nan, 30.0],
    })

    df = df_orig.copy()
    df.loc[1, "b"] = 1000.0

    imputed_mask = df_orig.isna()

    method_map = {"a": "mean", "b": "mean"}

    result = score_imputations(df, imputed_mask, method_map, df_original=df_orig)

    score = result["confidence_scores"].loc[1, "b"]
    assert 0.0 <= score <= 1.0
    assert score < 0.3
    assert (1, "b") in result["flagged_low_confidence"]
    assert "overall_mean" in result["summary"]

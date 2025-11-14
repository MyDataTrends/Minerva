import pandas as pd
from orchestration.data_preprocessor import DataPreprocessor
from alignment_drift_monitor import generate_historical_stats

def test_clean_with_diagnostics():
    df = pd.DataFrame({"A": [1, 2, None, 4], "B": ["x", "y", "z", "w"]})
    schema = {"A": float, "B": str}
    pre = DataPreprocessor()
    stats = generate_historical_stats(df.fillna(0))
    cleaned, diag = pre.clean(
        df,
        check_misalignment=True,
        misalignment_schema=schema,
        check_context_missing=True,
        score_imputations_flag=True,
        monitor_drift=True,
        baseline_stats=stats,
        return_diagnostics=True,
    )
    assert isinstance(diag, dict)
    assert "misalignment" in diag
    assert "context_missing" in diag
    assert "imputation_confidence" in diag
    assert "alignment_drift" in diag
    assert len(cleaned) == len(df)

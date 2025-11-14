import numpy as np
import pandas as pd

from config import MAX_ROWS_FIRST_PASS
from orchestration.analyzer_selector_helper import AnalyzerSelector


def test_baseline_path_for_borderline_dataset():
    np.random.seed(0)
    n_rows = MAX_ROWS_FIRST_PASS + 1000
    df = pd.DataFrame(
        np.random.randn(n_rows, 5),
        columns=[f"f{i}" for i in range(5)],
    )
    df["target"] = np.random.randn(n_rows)
    selector = AnalyzerSelector()
    res = selector.analyze(df, "target")

    assert res["analysis_type"] == "baseline"
    assert res["suitability"]["reason"].startswith("borderline")
    baseline = res["baseline"]
    metrics = baseline["metrics"]
    assert metrics["sampled"] is True
    assert metrics["n_rows"] <= MAX_ROWS_FIRST_PASS
    assert baseline["timed_out"] is False

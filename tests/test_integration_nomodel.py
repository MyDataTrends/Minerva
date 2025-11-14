import pandas as pd

from orchestration.analyzer_selector_helper import AnalyzerSelector


def test_descriptive_path_for_unmodelable_target():
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5],
        "target": [1, 1, 1, 1, 1],
    })
    selector = AnalyzerSelector()
    res = selector.analyze(df, "target")

    assert res["analysis_type"] == "descriptive"
    assert res.get("modeling_skipped") is True
    assert res["suitability"]["is_modelable"] is False
    assert "baseline" not in res
    assert "_model" not in res
    assert "_preds" not in res

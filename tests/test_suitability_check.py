import pandas as pd

from modeling.suitability_check import assess_modelability
from orchestration.analyzer_selector_helper import AnalyzerSelector


def test_assess_modelability_constant_target():
    df = pd.DataFrame({
        "a": list(range(50)),
        "target": [5] * 50,
    })
    res = assess_modelability(df, "target", task_hint="regression")
    assert res["is_modelable"] is False
    assert res["reason"] == "low_variance"


def test_analyzer_skips_when_unmodelable():
    df = pd.DataFrame({
        "a": list(range(60)),
        "target": [1] * 60,
    })
    selector = AnalyzerSelector()
    res = selector.analyze(df, "target")
    assert res.get("analysis_type") == "descriptive"
    assert res.get("modeling_skipped") is True

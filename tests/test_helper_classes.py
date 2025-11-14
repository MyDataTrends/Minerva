import pandas as pd
import orchestrate_workflow as ow
from orchestration.data_preprocessor import DataPreprocessor
from orchestration.semantic_enricher import SemanticEnricher
from orchestration.analyzer_selector_helper import AnalyzerSelector
from orchestration.output_generator import OutputGenerator
from orchestration.agent_trigger import AgentTrigger


def test_preprocessor_clean_basic():
    df = pd.DataFrame({"A": [1, 2, None, 4, 5], "B": ["a", "b", "c", "d", "e"]})
    pre = DataPreprocessor()
    cleaned = pre.clean(df)
    assert not cleaned.isna().any().any()
    assert len(cleaned) == 5


def test_selector_fallback(monkeypatch):
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "sales": [5, 4, 3, 2, 1]})
    monkeypatch.setattr(ow, "setup", None)
    selector = AnalyzerSelector()
    res = selector.analyze(df, "sales")
    assert "output" in res or "analysis_type" in res


def test_semantic_enricher_basic():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "sales": [5, 4, 3, 2, 1]})
    enricher = SemanticEnricher()
    enriched = enricher.enrich(df)
    assert isinstance(enriched, pd.DataFrame)


class DummyModel:
    def __str__(self):
        return "DummyModel"


def test_output_generator(monkeypatch, tmp_path):
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
    assert "output" in result
    trigger = AgentTrigger()
    trigger.trigger(result, pd.Series([1, 2, 3]), 0.1, "file.csv", DummyModel(), df, {})
    assert "actions" in result

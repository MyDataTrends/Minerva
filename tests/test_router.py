import pandas as pd
import pytest

from orchestration.analysis_router import route_analysis, LLMHint
from modeling.suitability_check import assess_modelability


def make_df(rows: int = 100):
    df = pd.DataFrame({"feature": range(rows), "target": range(rows)})
    return df


def test_route_no_model_when_unmodelable():
    df = make_df(10)
    stats = assess_modelability(df, "target")
    hint = LLMHint(task="regression", complexity="high", confidence=0.95)
    assert route_analysis(df, "target", stats, hint) == "no_model"


def test_baseline_when_borderline_and_hint_ignored(monkeypatch):
    df = make_df(100)
    df.loc[::3, "target"] = None  # introduce ~33% missing -> borderline
    stats = assess_modelability(df, "target")
    assert stats["reason"].startswith("borderline")
    hint = LLMHint(task="regression", complexity="high", confidence=0.95)
    from orchestration import analysis_router as ar
    monkeypatch.setattr(ar, "ENABLE_LOCAL_LLM", False)
    monkeypatch.setattr(ar, "MAX_ROWS_FULL", 500)
    assert route_analysis(df, "target", stats, hint) == "baseline"


def test_full_when_strong_hint_and_flags(monkeypatch):
    df = make_df(100)
    stats = assess_modelability(df, "target")
    hint = LLMHint(task="regression", complexity="high", confidence=0.95)
    from orchestration import analysis_router as ar
    monkeypatch.setattr(ar, "ENABLE_LOCAL_LLM", True)
    monkeypatch.setattr(ar, "ALLOW_FULL_COMPARE_MODELS", True)
    monkeypatch.setattr(ar, "MAX_ROWS_FULL", 500)
    assert route_analysis(df, "target", stats, hint) == "full"


def test_full_borderline_with_strong_hint(monkeypatch):
    df = make_df(100)
    df.loc[::3, "target"] = None
    stats = assess_modelability(df, "target")
    hint = LLMHint(task="regression", complexity="high", confidence=0.95)
    from orchestration import analysis_router as ar
    monkeypatch.setattr(ar, "ENABLE_LOCAL_LLM", True)
    monkeypatch.setattr(ar, "ALLOW_FULL_COMPARE_MODELS", True)
    monkeypatch.setattr(ar, "MAX_ROWS_FULL", 500)
    assert route_analysis(df, "target", stats, hint) == "full"


def test_baseline_when_flag_off(monkeypatch):
    df = make_df(100)
    stats = assess_modelability(df, "target")
    hint = LLMHint(task="regression", complexity="high", confidence=0.95)
    from orchestration import analysis_router as ar
    monkeypatch.setattr(ar, "ENABLE_LOCAL_LLM", True)
    monkeypatch.setattr(ar, "ALLOW_FULL_COMPARE_MODELS", False)
    monkeypatch.setattr(ar, "MAX_ROWS_FULL", 500)
    assert route_analysis(df, "target", stats, hint) == "baseline"


def test_baseline_large_dataset_even_with_hint(monkeypatch):
    df = make_df(1000)
    stats = assess_modelability(df, "target")
    hint = LLMHint(task="regression", complexity="high", confidence=0.95)
    from orchestration import analysis_router as ar
    monkeypatch.setattr(ar, "ENABLE_LOCAL_LLM", True)
    monkeypatch.setattr(ar, "ALLOW_FULL_COMPARE_MODELS", True)
    monkeypatch.setattr(ar, "MAX_ROWS_FULL", 500)
    assert route_analysis(df, "target", stats, hint) == "baseline"


def test_full_small_dataset_without_hint(monkeypatch):
    df = make_df(100)
    stats = assess_modelability(df, "target")
    from orchestration import analysis_router as ar
    monkeypatch.setattr(ar, "ENABLE_LOCAL_LLM", False)
    monkeypatch.setattr(ar, "ALLOW_FULL_COMPARE_MODELS", True)
    monkeypatch.setattr(ar, "MAX_ROWS_FULL", 500)
    assert route_analysis(df, "target", stats, None) == "full"

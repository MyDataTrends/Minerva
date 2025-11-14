from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualization_selector import infer_visualization_type
from config import LLM_VISUALIZATION_TOKEN_BUDGET


def test_llm_override(monkeypatch):
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    called = {"llm": False}

    def fake_llm(prompt, max_tokens=LLM_VISUALIZATION_TOKEN_BUDGET):
        called["llm"] = True
        return '{"visualization_type": "heatmap", "reasoning": "data density"}'

    captured = {}

    def fake_log(query, decision):
        captured["query"] = query
        captured["decision"] = decision

    monkeypatch.setattr("visualization_selector.llm_completion", fake_llm)
    monkeypatch.setattr("visualization_selector.log_decision", fake_log)

    vis = infer_visualization_type("density view", df, model_type="forecasting")

    assert vis == "heatmap"
    assert captured["decision"]["visualization_type"] == "heatmap"
    assert called["llm"] is True


def test_time_series_line_chart(monkeypatch):
    df = pd.DataFrame({"date": pd.date_range("2020", periods=5), "val": range(5)})
    called = {"llm": False}

    def fake_llm(prompt, max_tokens=LLM_VISUALIZATION_TOKEN_BUDGET):
        called["llm"] = True
        return "LLM unavailable"

    monkeypatch.setattr("visualization_selector.llm_completion", fake_llm)
    vis = infer_visualization_type("show trend over time", df)
    assert vis == "line_chart"
    assert called["llm"] is False


def test_categorical_bar_chart(monkeypatch):
    df = pd.DataFrame({"category": ["a", "b", "c"], "val": [1, 2, 3]})
    called = {"llm": False}

    def fake_llm(prompt, max_tokens=LLM_VISUALIZATION_TOKEN_BUDGET):
        called["llm"] = True
        return "LLM unavailable"

    monkeypatch.setattr("visualization_selector.llm_completion", fake_llm)
    vis = infer_visualization_type("compare categories", df)
    assert vis == "bar_chart"
    assert called["llm"] is False


def test_numeric_scatter_default(monkeypatch):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 3, 4]})
    monkeypatch.setattr("visualization_selector.llm_completion", lambda *a, **k: "LLM unavailable")
    vis = infer_visualization_type("", df)
    assert vis == "scatter_plot"


def test_llm_unavailable_fallback(monkeypatch):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    monkeypatch.setattr("visualization_selector.llm_completion", lambda *a, **k: "LLM unavailable")

    vis = infer_visualization_type("random", df)
    assert vis == "scatter_plot"

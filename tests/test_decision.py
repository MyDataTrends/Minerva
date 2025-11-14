import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatbot.decision import decide_action
import pytest


def test_decide_visualization():
    action, params = decide_action("Show a scatter plot of sales")
    assert action == "visualization"
    assert params["type"] == "scatter_plot"


def test_decide_modeling():
    action, params = decide_action("Forecast next month's revenue")
    assert action == "modeling"
    assert params["task"] == "forecast"


def test_decide_auto():
    with pytest.raises(RuntimeError):
        decide_action("Hello there")

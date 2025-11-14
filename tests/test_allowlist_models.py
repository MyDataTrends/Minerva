import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from config.model_allowlist import MODEL_ALLOWLIST
from modeling import model_selector


def _sample_data():
    X = pd.DataFrame({"a": range(20), "b": range(20, 40)})
    y = pd.Series(range(20), name="target")
    return X, y


def test_quick_model_selection(monkeypatch):
    X, y = _sample_data()
    monkeypatch.setattr(model_selector, "ALLOW_FULL_COMPARE_MODELS", False)
    model = model_selector.select_best_model(X, y)
    allowed = MODEL_ALLOWLIST["regression"]
    assert any(name in model.__class__.__name__.lower() for name in allowed)


@pytest.mark.skip(reason="pycaret removed; this test mocked the pycaret.regression module which is no longer used")
def test_full_compare_models_called(monkeypatch):
    X, y = _sample_data()
    monkeypatch.setattr(model_selector, "ALLOW_FULL_COMPARE_MODELS", True)

    recorded: dict = {}

    def fake_setup(**kwargs):
        recorded["setup"] = True

    def fake_compare_models(**kwargs):
        recorded["include"] = kwargs.get("include")
        return "best"

    dummy_module = SimpleNamespace(
        setup=fake_setup,
        compare_models=fake_compare_models,
        add_metric=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "pycaret.regression", dummy_module)

    model = model_selector.select_best_model(X, y, task="regression")
    assert model == "best"
    assert recorded["include"] == MODEL_ALLOWLIST["regression"]

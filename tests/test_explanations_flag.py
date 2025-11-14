import importlib
import sys
import types

import pandas as pd
import numpy as np


def _reload_modules(monkeypatch, enabled: bool):
    """Reload feature flag and model training modules with flag state."""
    if enabled:
        monkeypatch.setenv("ENABLE_HEAVY_EXPLANATIONS", "1")
    else:
        monkeypatch.delenv("ENABLE_HEAVY_EXPLANATIONS", raising=False)

    import config.feature_flags as ff
    import modeling.model_training as mt

    importlib.reload(ff)
    return importlib.reload(mt)


def _dummy_model():
    return types.SimpleNamespace(feature_importances_=[0.5, 0.5])


def test_explanations_disabled(monkeypatch):
    dummy_shap = types.SimpleNamespace()
    state: dict[str, bool] = {"called": False}

    class DummyTreeExplainer:
        def __init__(self, model):
            state["called"] = True

        def shap_values(self, X):  # pragma: no cover - should not be called
            state["shap_values"] = True
            return np.zeros((X.shape[0], X.shape[1]))

    dummy_shap.TreeExplainer = DummyTreeExplainer
    monkeypatch.setitem(sys.modules, "shap", dummy_shap)

    mt = _reload_modules(monkeypatch, enabled=False)

    model = _dummy_model()
    X = pd.DataFrame([[1, 2], [3, 4]])
    explanations = mt.get_model_explanations(model, X)

    assert explanations.get("explanations_disabled") is True
    assert "shap_values" not in explanations
    assert state["called"] is False


def test_explanations_enabled(monkeypatch):
    dummy_shap = types.SimpleNamespace()
    state: dict[str, bool] = {"called": False}

    class DummyTreeExplainer:
        def __init__(self, model):
            state["called"] = True

        def shap_values(self, X):
            return np.zeros((X.shape[0], X.shape[1]))

    dummy_shap.TreeExplainer = DummyTreeExplainer
    monkeypatch.setitem(sys.modules, "shap", dummy_shap)

    mt = _reload_modules(monkeypatch, enabled=True)

    model = _dummy_model()
    X = pd.DataFrame([[1, 2], [3, 4]])
    explanations = mt.get_model_explanations(model, X)

    assert state["called"] is True
    assert "shap_values" in explanations
    assert "explanations_disabled" not in explanations


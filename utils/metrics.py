import pandas as pd

from config import ENABLE_PROMETHEUS


class _NoopCounter:  # pragma: no cover - simple no-op implementation
    """Fallback counter used when metrics are disabled or unavailable."""

    def __init__(self, *args, **kwargs):
        pass

    def labels(self, *args, **kwargs):
        return self

    def inc(self, *args, **kwargs):
        pass


if ENABLE_PROMETHEUS:
    try:
        from prometheus_client import Counter, REGISTRY  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        Counter = _NoopCounter  # type: ignore
else:  # metrics explicitly disabled
    Counter = _NoopCounter  # type: ignore


# Prevent "Duplicated timeseries" error on reload (e.g. Streamlit)
if Counter != _NoopCounter:
    try:
        # We access the private registry to check for existence because the public API 
        # raises duplicate errors instead of returning existing metrics.
        # This is common in hot-reloading environments.
        if "app_requests_total" in REGISTRY._names_to_collectors:
             REGISTRY.unregister(REGISTRY._names_to_collectors["app_requests_total"])
    except Exception:
        pass

REQUESTS = Counter("app_requests_total", "Total requests")


def suitability_score(df: pd.DataFrame, target_col: str | None = None) -> float:
    """Estimate how well the dataset suits model training."""
    # Simple heuristic based on completeness ratio for now
    score = 0.0
    score += df.dropna().shape[0] / df.shape[0]
    return min(score, 1.0)


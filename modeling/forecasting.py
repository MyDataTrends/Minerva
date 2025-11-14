from __future__ import annotations

import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:  # pragma: no cover - optional dependency
    ARIMA = None


def run_forecasting(
    series: pd.Series, periods: int = 12, order: tuple[int, int, int] = (1, 1, 1)
) -> dict:
    """Forecast ``periods`` steps ahead using an ARIMA model.

    Parameters
    ----------
    series:
        Time-indexed numeric series to model.
    periods:
        Number of future periods to forecast.
    order:
        ARIMA model order.

    Returns
    -------
    dict
        ``{"model": ARIMAResults, "forecast": pd.Series}``
    """
    if ARIMA is None:
        raise ImportError("statsmodels is required for forecasting")

    model = ARIMA(series, order=order)
    fit = model.fit()
    forecast = fit.forecast(steps=periods)
    return {"model": fit, "forecast": forecast}

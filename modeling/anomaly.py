from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(data: pd.DataFrame | pd.Series) -> dict:
    """Detect outliers in numeric data using ``IsolationForest``.

    Parameters
    ----------
    data:
        ``DataFrame`` or ``Series`` containing numeric values.

    Returns
    -------
    dict
        ``{"outlier_indices": list[int], "n_outliers": int}``
    """

    if isinstance(data, pd.Series):
        numeric_df = data.to_frame()
    else:
        numeric_df = data.select_dtypes(include="number")

    if numeric_df.empty:
        return {"outlier_indices": [], "n_outliers": 0}

    model = IsolationForest(random_state=0)
    preds = model.fit_predict(numeric_df)
    outlier_mask = preds == -1
    indices = numeric_df.index[outlier_mask].tolist()
    return {"outlier_indices": indices, "n_outliers": len(indices)}

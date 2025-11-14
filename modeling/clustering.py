from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans


def run_clustering(data: pd.DataFrame, k: int = 4) -> dict:
    """Cluster data using ``KMeans``.

    Parameters
    ----------
    data:
        Input numeric features.
    k:
        Number of clusters.

    Returns
    -------
    dict
        ``{"labels": ndarray, "centers": ndarray}``
    """

    numeric_df = data.select_dtypes(include="number")
    model = KMeans(n_clusters=k, random_state=0)
    labels = model.fit_predict(numeric_df)
    return {"labels": labels, "centers": model.cluster_centers_}

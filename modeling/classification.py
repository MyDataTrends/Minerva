from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def _encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return ``X`` with categorical columns one-hot encoded."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    cat_cols = X.select_dtypes(exclude="number").columns
    if cat_cols.any():
        X = pd.get_dummies(X, columns=list(cat_cols), drop_first=True)
    return X


def run_classification(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train a ``RandomForestClassifier`` and compute a report.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Target labels.

    Returns
    -------
    dict
        Dictionary with ``model``, ``report`` and ``predictions``.
    """

    X_processed = _encode_features(X)
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_processed, y)
    preds = clf.predict(X_processed)
    report = classification_report(y, preds, output_dict=True)
    return {"model": clf, "predictions": preds, "report": report}

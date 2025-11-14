import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.datasets import load_iris, make_blobs

from modeling.classification import run_classification
from modeling.clustering import run_clustering


def test_run_classification():
    data = load_iris(as_frame=True)
    df = data.frame
    result = run_classification(df.drop(columns=["target"]), df["target"])
    assert len(result["predictions"]) == len(df)
    assert "accuracy" in result["report"]


def test_classification_with_strings():
    df = pd.DataFrame({
        "num": [1, 2, 3, 4],
        "cat": ["a", "b", "a", "b"],
        "target": ["yes", "no", "yes", "no"],
    })
    res = run_classification(df.drop(columns=["target"]), df["target"])
    assert len(res["predictions"]) == len(df)


def test_run_clustering():
    X, _ = make_blobs(n_samples=50, centers=3, n_features=2, random_state=0)
    df = pd.DataFrame(X, columns=["x", "y"])
    res = run_clustering(df, k=3)
    assert len(res["labels"]) == 50
    assert res["centers"].shape == (3, 2)

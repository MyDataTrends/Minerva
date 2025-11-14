import numpy as np
import pandas as pd

from modeling.baseline_runner import run_baseline


def test_sampling_and_pruning_classification():
    df = pd.DataFrame(
        np.random.randn(2000, 20),
        columns=[f"f{i}" for i in range(20)],
    )
    df["target"] = np.random.randint(0, 2, size=2000)
    res = run_baseline(
        df,
        "target",
        task="classification",
        sample_rows=100,
        max_features=5,
    )
    m = res["metrics"]
    assert m["sampled"] is True
    assert m["features_pruned"] is True
    assert m["n_rows"] == 100
    assert m["n_features"] == 5
    assert res["timed_out"] is False


def test_time_budget_honored():
    df = pd.DataFrame(
        np.random.randn(100, 5),
        columns=[f"f{i}" for i in range(5)],
    )
    df["target"] = np.random.randn(100)
    res = run_baseline(df, "target", task="regression", time_budget_s=0)
    assert res["timed_out"] is True


def test_large_dataset_memory_safe():
    df = pd.DataFrame(
        np.random.randn(10000, 50),
        columns=[f"f{i}" for i in range(50)],
    )
    df["target"] = np.random.randn(10000)
    res = run_baseline(
        df,
        "target",
        task="regression",
        sample_rows=100,
        max_features=10,
    )
    m = res["metrics"]
    assert m["n_rows"] == 100
    assert m["n_features"] == 10
    assert m["sampled"] is True
    assert res["timed_out"] is False

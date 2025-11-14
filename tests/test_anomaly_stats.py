import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from modeling.anomaly import detect_anomalies
from descriptive.stats import generate_descriptives


def test_detect_anomalies_basic():
    data = pd.Series([1, 1, 1, 100])
    result = detect_anomalies(data)
    assert result["n_outliers"] == 1
    assert result["outlier_indices"] == [3]


def test_generate_descriptives():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    stats = generate_descriptives(df)
    assert stats.loc["a", "mean"] == 2
    assert stats.loc["b", "max"] == 6

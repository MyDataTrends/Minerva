import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from analysis_selector import select_analyzer  # noqa: E402
from modeling.regression_analyzer import RegressionAnalyzer  # noqa: E402
from modeling.classification_analyzer import ClassificationAnalyzer  # noqa: E402
from modeling.cluster_analyzer import ClusterAnalyzer  # noqa: E402
from modeling.anomaly_analyzer import AnomalyAnalyzer  # noqa: E402
from modeling import REGISTRY  # noqa: E402
from utils.metrics import suitability_score as dataset_score  # noqa: E402


def numeric_target_df():
    df = pd.DataFrame({f"f{i}": range(50) for i in range(4)})
    df["sales"] = range(50)
    return df


def categorical_target_df():
    df = pd.DataFrame({f"c{i}": range(60) for i in range(3)})
    df["region"] = ["a" if i < 30 else "b" for i in range(60)]
    return df


def cluster_df():
    return pd.DataFrame({f"c{i}": range(40) for i in range(5)})


def missing_values_df():
    df = numeric_target_df()
    df.loc[::2, "sales"] = None
    return df


import pytest

@pytest.mark.parametrize(
    "df,expected",
    [
        (numeric_target_df(), RegressionAnalyzer),
        (categorical_target_df(), ClassificationAnalyzer),
        (cluster_df(), ClusterAnalyzer),
    ],
)
def test_auto_selection(df, expected):
    analyzer = select_analyzer(df)
    assert isinstance(analyzer, expected)
    result = analyzer.run(df)
    assert result["alternatives"]


def test_preferred_override():
    df = numeric_target_df()
    analyzer = select_analyzer(df, preferred="AnomalyAnalyzer")
    assert isinstance(analyzer, AnomalyAnalyzer)


def test_missing_values_reduce_ranking():
    df_full = numeric_target_df()
    df_missing = missing_values_df()

    analyzer_full = select_analyzer(df_full)
    analyzer_missing = select_analyzer(df_missing)

    score_full = dataset_score(df_full) * analyzer_full.suitability_score(df_full)
    score_missing = dataset_score(df_missing) * analyzer_missing.suitability_score(df_missing)

    assert score_missing < score_full

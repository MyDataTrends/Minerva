import pandas as pd

from preprocessing.context_missing_finder import find_contextual_missingness


def test_find_contextual_missingness_basic():
    df = pd.DataFrame({
        "revenue": ["100", "0", "nan", "50", "N/A"],
        "units_sold": [10, 5, 0, 0, 20],
        "category": ["A", "unknown", "B", "--", "C"],
    })
    result = find_contextual_missingness(
        df,
        correlated_fields=[("revenue", "units_sold")],
        suspicious_placeholders=["0", "nan", "n/a", "--", "unknown"],
        zero_thresholds={"revenue": 1.0},
    )
    cells = set(result["flagged_cells"])
    expected = {
        (1, "revenue"),
        (1, "category"),
        (2, "revenue"),
        (2, "units_sold"),
        (3, "units_sold"),
        (3, "category"),
        (4, "revenue"),
    }
    assert expected.issubset(cells)
    summary = result["column_summary"]
    assert summary["revenue"] == 3
    assert summary["units_sold"] == 2
    assert summary["category"] == 2
    assert list(result["flagged_df"]["is_context_missing"]) == [False, True, True, True, True]


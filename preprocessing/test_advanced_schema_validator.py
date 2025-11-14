import pandas as pd
from preprocessing.advanced_schema_validator import validate_schema


def test_validate_schema_no_mismatches():
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    schema = {"A": "int", "B": "category"}
    result = validate_schema(df, schema, valid_categories={"B": {"x", "y"}})
    assert result["mismatch_counts"] == {}
    assert result["validation_report"] == {}


def test_validate_schema_mismatches_with_quarantine_and_coerce():
    df = pd.DataFrame({"id": [1, "two", 3], "score": ["1.0", "bad", "3"]})
    schema = {"id": "int", "score": "float"}
    res = validate_schema(df, schema, quarantine_invalid=True, coerce=True)
    # Expect row index 1 quarantined
    assert res["quarantined"] is not None
    assert len(res["quarantined"]) == 1
    assert 1 in res["validation_report"]["id"]["rows"]
    assert 1 in res["validation_report"]["score"]["rows"]
    assert res["corrected_df"].shape[0] == 2

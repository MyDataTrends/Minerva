import pandas as pd

from preprocessing.misaligned_row_detector import detect_misaligned_rows


def test_detect_misaligned_rows_basic():
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "12345"],
        "age": [30, 25, "Charlie"],
        "zip": [11111, 22222, 33333],
    })
    schema = {"name": str, "age": int, "zip": int}
    report = detect_misaligned_rows(df, schema)
    assert report["misaligned_rows"] == [2]
    assert 2 in report["issues"]


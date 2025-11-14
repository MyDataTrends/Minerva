import pandas as pd
from typing import List


def categorize_dataset(df: pd.DataFrame) -> List[str]:
    """Infer simple tags describing the dataset."""
    tags = set()
    if not isinstance(df, pd.DataFrame):
        return []

    # Check column data types
    if not df.empty:
        if not df.select_dtypes(include="number").empty:
            tags.add("numeric")
        if not df.select_dtypes(include="object").empty:
            tags.add("text")
        if not df.select_dtypes(include="datetime").empty:
            tags.add("datetime")

    # Look for common column names for extra hints
    lower_cols = [c.lower() for c in df.columns]
    if any("date" in c for c in lower_cols):
        tags.add("date")
    if any("id" == c or c.endswith("_id") for c in lower_cols):
        tags.add("id")

    return sorted(tags)

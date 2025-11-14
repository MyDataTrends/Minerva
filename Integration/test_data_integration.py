from Integration.data_integration import merge_data, resolve_conflicts
from storage import load_dataset_dataframe


def test_merge_data():
    df1 = load_dataset_dataframe("merge_df1.csv")
    df2 = load_dataset_dataframe("merge_df2.csv")
    merged_df = merge_data(df1, df2, on_column="ID")
    assert merged_df.shape == (2, 3)


def test_resolve_conflicts():
    df = load_dataset_dataframe("conflict.json")
    conflict_rules = {"group_by": ["key_column"], "Value": "max"}
    resolved_df = resolve_conflicts(df, conflict_rules)
    assert resolved_df.shape == (2, 2)

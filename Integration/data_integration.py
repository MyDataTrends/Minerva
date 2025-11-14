import pandas as pd


def resolve_conflicts(df: pd.DataFrame, conflict_rules: dict) -> pd.DataFrame:
    """Reconcile overlapping fields according to ``conflict_rules``.

    ``conflict_rules`` maps column names to merge strategies (e.g.
    ``"first_non_null"`` or ``"max"``).  The optional ``"group_by"`` key can be
    used to specify columns to group by when aggregating.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing conflicting data.
    conflict_rules : dict
        Mapping of column names to reconciliation strategies.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with conflicts resolved.
    """

    df = df.copy(deep=False)
    group_cols = conflict_rules.get("group_by", [])
    for col, strategy in conflict_rules.items():
        if col == "group_by":
            continue
        if strategy == "first_non_null":
            df[col] = df[col].ffill().bfill()
        elif strategy == "max":
            if group_cols:
                df[col] = df.groupby(group_cols)[col].transform("max")
            else:
                df[col] = df[col].max()
        # add other strategies as needed
    if group_cols:
        df = df.drop_duplicates(subset=group_cols, keep="first")
    return df


def merge_data(df1, df2, on_column, how="inner"):
    return pd.merge(df1, df2, on=on_column, how=how)


def enrich_data(df, enrichment_df, on_column):
    return pd.merge(df, enrichment_df, on=on_column, how="left")

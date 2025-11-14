import pandas as pd
from rapidfuzz import process


def clean_missing_values(df, strategy="drop", fill_value=0):
    if strategy == "drop":
        return df.dropna()
    if strategy == "fill":
        return df.fillna(fill_value)
    raise ValueError("Unsupported strategy")


def normalize_text_columns(df, column_name):
    df[column_name] = df[column_name].str.lower().str.strip()
    return df


def remove_duplicates(df):
    return df.drop_duplicates()


def convert_to_datetime(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
    return df


def remove_outliers(df, column_name, method="IQR"):
    if method == "IQR":
        q1 = df[column_name].quantile(0.25)
        q3 = df[column_name].quantile(0.75)
        iqr = q3 - q1
        df = df[~((df[column_name] < (q1 - 1.5 * iqr)) | (df[column_name] > (q3 + 1.5 * iqr)))]
    elif method == "Z-score":
        from scipy import stats
        df = df[(abs(stats.zscore(df[column_name])) < 3)]
    return df


def encode_categorical_columns(df, column_name):
    df[column_name] = df[column_name].astype("category").cat.codes
    return df


def fuzzy_match_columns(df, column_name, threshold=80):
    unique_values = df[column_name].unique()
    matched_values = {}
    for value in unique_values:
        result = process.extractOne(value, unique_values)
        if result is None:
            continue
        if isinstance(result, (tuple, list)):
            match, score = result[0], result[1]
        else:
            match, score = result.match, result.score
        if score >= threshold:
            matched_values[value] = match
    df[column_name] = df[column_name].apply(lambda x: matched_values.get(x, x))
    return df


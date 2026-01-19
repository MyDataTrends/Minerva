import re
import numpy as np
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


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to snake_case."""
    df = df.copy()
    new_cols = []
    for col in df.columns:
        s = str(col).strip()
        s = re.sub(r'[^a-zA-Z0-9]', '_', s)
        s = re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()
        s = re.sub(r'_+', '_', s)
        new_cols.append(s.strip('_'))
    df.columns = new_cols
    return df


def standardize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to identify and parse date columns."""
    df = df.copy()
    date_patterns = ['date', 'time', 'period', 'year', 'month', 'timestamp', 'day']
    for col in df.columns:
        if any(p in col.lower() for p in date_patterns):
            try:
                if df[col].dtype == 'object' or 'int' in str(df[col].dtype):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass
    return df


def standardize_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to coerce numeric-looking object columns."""
    df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        try:
            sample = df[col].dropna().head(10).astype(str)
            if sample.empty:
                continue
            check_vals = sample.str.replace(r'[$,%]', '', regex=True)
            pd.to_numeric(check_vals)
            clean_col = df[col].astype(str).str.replace(r'[$,%]', '', regex=True)
            df[col] = pd.to_numeric(clean_col, errors='coerce')
        except (ValueError, TypeError):
            continue
    return df


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run full standardization pipeline."""
    df = normalize_column_names(df)
    df = standardize_dates(df)
    df = standardize_numerics(df)
    return df

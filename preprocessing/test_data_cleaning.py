import pandas as pd
from preprocessing.data_cleaning import clean_missing_values, normalize_text_columns, remove_duplicates, convert_to_datetime

def test_clean_missing_values():
    df = pd.DataFrame({'A': [1, None, 2]})
    df_cleaned = clean_missing_values(df)
    assert df_cleaned.isna().sum().sum() == 0

def test_normalize_text_columns():
    df = pd.DataFrame({'text': [' Hello ', 'world ']})
    df_normalized = normalize_text_columns(df, 'text')
    assert df_normalized['text'].tolist() == ['hello', 'world']

def test_remove_duplicates():
    df = pd.DataFrame({'A': [1, 1, 2], 'B': [3, 3, 4]})
    df_unique = remove_duplicates(df)
    assert df_unique.shape[0] == 2

def test_convert_to_datetime():
    df = pd.DataFrame({'date': ['2022-01-01', 'invalid date']})
    df = convert_to_datetime(df, 'date')
    assert pd.notnull(df['date'][0]) and pd.isnull(df['date'][1])

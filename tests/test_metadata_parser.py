import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.metadata_parser import infer_column_meta, parse_metadata
from config import PROFILE_SAMPLE_ROWS


def test_infer_column_meta_basic():
    df = pd.DataFrame({'TransactionDate': [1, 2], 'StoreID': [10, 11], 'Foo': [3, 4]})
    meta = infer_column_meta(df)
    roles = {m.name: m.role for m in meta}
    assert roles['TransactionDate'] == 'transaction_date'
    assert roles['StoreID'] == 'store_id'
    assert roles['Foo'] == 'unknown'


def test_parse_metadata_uses_sample():
    rows = PROFILE_SAMPLE_ROWS + 5000
    df = pd.DataFrame({'a': range(rows)})
    meta = parse_metadata(df)
    assert meta['summary']['a']['count'] == float(PROFILE_SAMPLE_ROWS)

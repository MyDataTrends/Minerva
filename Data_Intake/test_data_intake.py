import pytest
import pandas as pd
from io import BytesIO
import os

os.environ["LOCAL_DATA_DIR"] = "."

# Optional dependencies used by the ingestion utilities
pytest.importorskip("dotenv")
pytest.importorskip("sklearn")

from storage.local_backend import parse_file
from storage import load_user_file

def test_parse_csv():
    sample_data = load_user_file("sample_user", "user_data.csv")
    df = parse_file(sample_data, file_type="csv")
    assert df.shape == (2, 2)


def test_parse_json():
    sample_data = load_user_file("sample_user", "user_data.json")
    df = parse_file(sample_data, file_type="json")
    assert df.shape == (2, 2)

def test_parse_xlsx():
    pytest.importorskip("openpyxl", reason="openpyxl required for excel parsing")
    sample_data = BytesIO()
    pd.DataFrame({"header1": [1, 3], "header2": [2, 4]}).to_excel(sample_data, index=False)
    sample_data.seek(0)  # Move to the beginning of the file
    df = parse_file(sample_data.read(), file_type="xlsx")
    assert df.shape == (2, 2)

def test_parse_tsv():
    sample_data = b"header1\theader2\n1\t2\n3\t4\n"
    df = parse_file(sample_data, file_type='tsv')
    assert df.shape == (2, 2)

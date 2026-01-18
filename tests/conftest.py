import sys
from pathlib import Path

import boto3
import pandas as pd
import pytest
from moto import mock_aws

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from storage.local_backend import LocalStorage
from storage.s3_backend import S3Storage


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "llm: marks tests that require LLM functionality")


@pytest.fixture(scope="session", autouse=True)
def llm_cleanup():
    """
    Session-scoped fixture that ensures the LLM is unloaded after all tests.
    
    This prevents memory accumulation when running the full test suite.
    The LLM singleton is cleared at the end of the test session.
    """
    yield  # Run all tests
    
    # Cleanup: unload LLM to free memory
    try:
        from preprocessing.llm_preprocessor import unload_llm
        unload_llm()
    except ImportError:
        pass


@pytest.fixture
def mock_llm(monkeypatch):
    """
    Fixture that mocks the LLM to return a fixed response.
    
    Use this for tests that call LLM functions but don't actually need
    real LLM inference. Drastically speeds up tests.
    
    Usage:
        def test_something(mock_llm):
            # LLM calls will return "Mocked LLM response"
            result = some_function_that_uses_llm()
    """
    monkeypatch.setattr(
        "preprocessing.llm_preprocessor.load_local_llm",
        lambda *args, **kwargs: None  # Makes LLM appear unavailable
    )
    yield


@pytest.fixture
def mock_llm_with_response(monkeypatch):
    """
    Fixture factory that mocks LLM completion with a custom response.
    
    Usage:
        def test_something(mock_llm_with_response):
            mock_llm_with_response("custom response text")
            result = llm_completion("any prompt")
            assert result == "custom response text"
    """
    def _set_response(response: str):
        monkeypatch.setattr(
            "preprocessing.llm_preprocessor.llm_completion",
            lambda *args, **kwargs: response
        )
    return _set_response


@pytest.fixture
def small_dataframe() -> pd.DataFrame:
    """Create a small DataFrame for quick testing."""
    return pd.DataFrame({
        "id": range(10),
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


@pytest.fixture
def local_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalStorage:
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    return LocalStorage()

@pytest.fixture
def s3_storage(monkeypatch: pytest.MonkeyPatch):
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        bucket = "test-bucket"
        client.create_bucket(Bucket=bucket)
        monkeypatch.setenv("BUCKET_NAME", bucket)
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        yield S3Storage(bucket=bucket)


@pytest.fixture
def usage_table(monkeypatch: pytest.MonkeyPatch):
    with mock_aws():
        client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "UsageTable"
        client.create_table(
            TableName=table_name,
            KeySchema=[{"AttributeName": "user_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "user_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        monkeypatch.setenv("USAGE_TRACKING_TABLE", table_name)
        monkeypatch.setenv("USE_CLOUD", "True")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        yield boto3.resource("dynamodb", region_name="us-east-1").Table(table_name)

import sys
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from storage.local_backend import LocalStorage
from storage.s3_backend import S3Storage

# Import adm package so coverage includes it

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

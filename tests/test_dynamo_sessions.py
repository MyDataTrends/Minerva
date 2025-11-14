import importlib
import json

import boto3
from moto import mock_aws

import storage.session_db as session_db
import storage.local_backend as local_backend


def setup_table():
    client = boto3.client("dynamodb", region_name="us-east-1")
    table_name = "SessionsTable"
    client.create_table(
        TableName=table_name,
        KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "run_id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    return table_name


def test_dynamo_roundtrip(monkeypatch):
    with mock_aws():
        table = setup_table()
        monkeypatch.setenv("DYNAMO_SESSIONS_TABLE", table)
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        importlib.reload(session_db)
        importlib.reload(local_backend)

        local_backend.log_run_metadata("r1", True, False, file_name="f.csv")
        session_db.record_session("r1", {"a": 1}, {"b": 2})

        meta = local_backend.load_run_metadata("r1")
        sess = session_db.get_session("r1")
        assert meta["file_name"] == "f.csv"
        assert json.loads(sess["params"])["a"] == 1

        # reload modules to simulate new run
        importlib.reload(session_db)
        importlib.reload(local_backend)
        meta2 = local_backend.load_run_metadata("r1")
        assert meta2["file_name"] == "f.csv"


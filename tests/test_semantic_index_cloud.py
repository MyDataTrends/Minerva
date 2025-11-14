
import boto3
import pandas as pd
from moto import mock_aws

import catalog.semantic_index as si


def test_semantic_index_s3_roundtrip(monkeypatch, tmp_path):
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        bucket = "test-bucket"
        client.create_bucket(Bucket=bucket)

        monkeypatch.setattr(si, "USE_CLOUD", True, raising=False)
        monkeypatch.setattr(si, "BUCKET_NAME", bucket, raising=False)
        monkeypatch.setattr(si, "SEMANTIC_INDEX_KEY", "semantic_index.db", raising=False)
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        ddir = tmp_path / "data"
        ddir.mkdir()
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        df.to_csv(ddir / "tbl.csv", index=False)

        index_path = tmp_path / "index.db"
        si.build_index(str(ddir), db_path=index_path)
        objects = client.list_objects_v2(Bucket=bucket)
        keys = [obj["Key"] for obj in objects.get("Contents", [])]
        assert "semantic_index.db" in keys

        index_path.unlink()
        assert not index_path.exists()
        fetched = si.fetch_semantic_index_from_s3(bucket, "semantic_index.db", dest=index_path)
        assert fetched.exists()

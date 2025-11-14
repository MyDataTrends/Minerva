from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

import boto3
from .base import StorageBackend
from config import BUCKET_NAME


class S3Storage(StorageBackend):
    """Storage backend that interacts with S3."""

    def __init__(self, bucket: str | None = None):
        self.bucket = bucket or BUCKET_NAME
        self.s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))

    def get_file(self, path: str) -> Path:
        tmp_dir = Path(tempfile.mkdtemp())
        target = tmp_dir / Path(path).name
        self.s3.download_file(self.bucket, path, str(target))
        return target

    def put_file(self, src: Path, dest: str) -> None:
        self.s3.upload_file(str(src), self.bucket, dest)

    def list_files(self, prefix: str = "") -> List[str]:
        keys: List[str] = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return keys


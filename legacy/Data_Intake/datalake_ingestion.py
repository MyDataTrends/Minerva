"""Datalake ingestion utilities and command-line interface.

These helpers make it easy to populate the local ``datasets`` directory with
sample data either from an S3 bucket or a generic HTTP API.  They are kept
light‑weight so they can be executed from local development machines without
requiring additional infrastructure.
The target directory can be overridden with the ``ADM_DATA_PATH`` environment
variable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3

from utils.logging import configure_logging, get_logger
from utils.security import secure_join
from utils.net import request_with_retry

from catalog.semantic_index import build_index

# Base directory for ingested datasets. The ``ADM_DATA_PATH`` environment
# variable can override the default ``datasets/`` path when set.
DATA_PATH = os.getenv("ADM_DATA_PATH", "datasets/")

DEFAULT_DEST = Path(os.getenv("LOCAL_DATA_DIR", "local_data")) / DATA_PATH



configure_logging()
logger = get_logger(__name__)



def sync_from_s3(
    bucket: str, prefix: str = "", dest: str | Path = DEFAULT_DEST
) -> None:
    """Download objects from an S3 bucket into ``dest``.

    Parameters
    ----------
    bucket : str
        Name of the S3 bucket.
    prefix : str, optional
        Key prefix within the bucket to sync from.
    dest : str, optional
        Local directory to sync files into.
    """

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    Path(dest).mkdir(parents=True, exist_ok=True)
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            filename = secure_join(Path(dest), os.path.basename(key))
            logger.info("Downloading %s to %s", key, filename)
            s3.download_file(bucket, key, str(filename))

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            target_path = secure_join(Path(dest), key[len(prefix) :])
            target_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading %s to %s", key, target_path)
            s3.download_file(bucket, key, str(target_path))


def fetch_from_api(
    endpoint: str, dest: str | Path = DEFAULT_DEST, params: dict | None = None
) -> None:
    """Fetch a file from an API endpoint and save it under ``dest``.

    Parameters
    ----------
    endpoint : str
        API URL to query.
    dest : str, optional
        Path to store the downloaded data.
    params : dict, optional
        Query parameters for the API request.
    """

    Path(dest).mkdir(parents=True, exist_ok=True)
    logger.info("Fetching data from API %s with params %s", endpoint, params)
    response = request_with_retry("get", endpoint, params=params, timeout=30)
    filename = secure_join(Path(dest), os.path.basename(endpoint))
    with open(filename, "wb") as f:
        f.write(response.content)
    logger.info("Saved API response to %s", filename)



def main(argv: list[str] | None = None) -> None:
    """Command-line interface for datalake ingestion."""
    parser = argparse.ArgumentParser(description="Ingest datasets into the local datalake")
    subparsers = parser.add_subparsers(dest="command")

    s3_parser = subparsers.add_parser("s3", help="Sync data from an S3 bucket")
    s3_parser.add_argument("--bucket", required=True, help="Name of the S3 bucket")
    s3_parser.add_argument("--prefix", default="", help="Prefix within the bucket")
    s3_parser.add_argument(
        "--dest", default=str(DEFAULT_DEST), help="Destination directory"
    )

    api_parser = subparsers.add_parser("api", help="Fetch data from an API endpoint")
    api_parser.add_argument("endpoint", help="API endpoint URL")
    api_parser.add_argument(
        "--dest", default=str(DEFAULT_DEST), help="Destination directory"
    )
    api_parser.add_argument("--param", action="append", default=[], help="Query parameters in key=value form")

    args = parser.parse_args(argv)

    if args.command == "s3":
        sync_from_s3(args.bucket, args.prefix, args.dest)
        build_index(args.dest)
    elif args.command == "api":
        params = {}
        for item in args.param:
            if '=' in item:
                key, value = item.split('=', 1)
                params[key] = value
        fetch_from_api(args.endpoint, args.dest, params)
        build_index(args.dest)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

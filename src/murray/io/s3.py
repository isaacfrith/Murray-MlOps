"""S3 (SeaweedFS) client. Same environment variables as catalog.py."""

import os

import boto3
from botocore.client import Config

LAKEHOUSE_BUCKET = "lakehouse"


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT", "http://localhost:8333"),
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        # Path-style: http://host:8333/lakehouse/key, not http://lakehouse.host:8333/key
        config=Config(s3={"addressing_style": "path"}),
    )


def list_keys(prefix: str, bucket: str = LAKEHOUSE_BUCKET) -> list[str]:
    """All file keys under a prefix (handles pagination).

    SeaweedFS also lists directories as keys ending in "/"; those are skipped.
    """
    pages = get_s3_client().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
    return [
        obj["Key"]
        for page in pages
        for obj in page.get("Contents", [])
        if not obj["Key"].endswith("/")
    ]


def delete_prefix(prefix: str, bucket: str = LAKEHOUSE_BUCKET) -> int:
    """Delete every file under a prefix. Returns the number deleted."""
    s3 = get_s3_client()
    keys = list_keys(prefix, bucket)
    for i in range(0, len(keys), 1000):  # DeleteObjects takes at most 1000 keys
        s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in keys[i : i + 1000]], "Quiet": True},
        )
    return len(keys)

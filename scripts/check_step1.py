"""Step 1 smoke test: Postgres users/databases and the SeaweedFS `lakehouse` bucket.

Run from the repo root after `docker compose ... up -d`:
    uv run python scripts/check_step1.py
"""

import os
import sys
import uuid

import boto3
import psycopg
from botocore.client import Config
from dotenv import load_dotenv

load_dotenv()

DATABASES = {
    "iceberg_catalog": "ICEBERG_DB_PASSWORD",
    "mlflow": "MLFLOW_DB_PASSWORD",
    "airflow": "AIRFLOW_DB_PASSWORD",
}


def connect(user: str, password: str, dbname: str) -> psycopg.Connection:
    return psycopg.connect(
        host="localhost", port=5432, user=user, password=password, dbname=dbname,
        connect_timeout=5,
    )


def check_postgres() -> None:
    for db, pw_var in DATABASES.items():
        with connect(db, os.environ[pw_var], db) as conn:
            conn.execute("CREATE TABLE _smoke (x int)")
            conn.execute("DROP TABLE _smoke")
        print(f"ok   postgres: {db} can create tables in {db}")

    # Each user must be locked out of the other databases and need a real password.
    for label, user, pw, db in [
        ("cross-database", "mlflow", os.environ["MLFLOW_DB_PASSWORD"], "airflow"),
        ("wrong password", "mlflow", "wrong", "mlflow"),
    ]:
        try:
            connect(user, pw, db).close()
        except psycopg.OperationalError:
            print(f"ok   postgres: {label} login refused")
        else:
            sys.exit(f"FAIL postgres: {label} login was allowed")


def check_s3() -> None:
    s3 = boto3.client(
        "s3",
        endpoint_url="http://localhost:8333",
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        # Path-style: http://localhost:8333/lakehouse/key, not http://lakehouse.localhost:8333/key
        config=Config(s3={"addressing_style": "path"}),
    )
    buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    assert "lakehouse" in buckets, f"lakehouse bucket missing: {buckets}"
    print(f"ok   s3: buckets = {buckets}")

    key = f"_smoke/{uuid.uuid4()}.txt"
    body = b"hello barmah"
    s3.put_object(Bucket="lakehouse", Key=key, Body=body)
    assert s3.get_object(Bucket="lakehouse", Key=key)["Body"].read() == body
    s3.delete_object(Bucket="lakehouse", Key=key)
    print("ok   s3: put/get/delete round trip")


if __name__ == "__main__":
    check_postgres()
    check_s3()
    print("Step 1 checks passed")

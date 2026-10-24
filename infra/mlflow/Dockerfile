# Official MLflow image + the two drivers it lacks:
#   psycopg2 → Postgres backend store, boto3 → S3 (SeaweedFS) artifact store.
FROM ghcr.io/mlflow/mlflow:v3.17.2

RUN pip install --no-cache-dir \
      psycopg2-binary==2.9.13 \
      boto3==1.43.118

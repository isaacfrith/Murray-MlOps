#!/usr/bin/env bash
# Runs ONCE, when the postgres data volume is empty (docker-entrypoint-initdb.d).
# Editing this file later has no effect until the volume is removed:
#   docker compose ... down -v
#
# Creates one database per service, each owned by its own login role.
# Owning the database gives the role CREATE on its public schema (needed since PG15).
set -euo pipefail

create_db() {
  local name="$1" password="$2"
  # Passwords go in as psql variables (:'pw') so they are quoted safely.
  psql -v ON_ERROR_STOP=1 -v name="$name" -v pw="$password" \
       --username "$POSTGRES_USER" --dbname postgres <<'SQL'
CREATE ROLE :"name" LOGIN PASSWORD :'pw';
CREATE DATABASE :"name" OWNER :"name";
-- By default every role may connect to every database; restrict to the owner.
REVOKE CONNECT ON DATABASE :"name" FROM PUBLIC;
SQL
  echo "created database and role: $name"
}

create_db iceberg_catalog "$ICEBERG_DB_PASSWORD"
create_db mlflow          "$MLFLOW_DB_PASSWORD"
create_db airflow         "$AIRFLOW_DB_PASSWORD"

# Murray River Forecast

Forecasts the Murray River's height at Barmah 1–7 days ahead and turns it into a
"good for wakeboarding?" rating. Built with feature / training / inference (FTI)
pipelines that share only a feature store (Feast) and a model registry (MLflow).

## Local stack

```bash
cp .env.example .env      # then replace every change-me (openssl rand -hex 16)
make up                   # = docker compose --env-file .env -f infra/docker-compose.yml up -d --build --wait
uv sync
make test                 # integration tests: Iceberg, MLflow, Valkey
```

| Service   | Address                 | Purpose                                          |
|-----------|-------------------------|--------------------------------------------------|
| Postgres  | `localhost:5432`        | DBs: `iceberg_catalog`, `mlflow`, `airflow` (one user each) |
| SeaweedFS | `http://localhost:8333` | S3 API (path-style), bucket `lakehouse`          |
| MLflow    | `http://localhost:5050` | Tracking + model registry; artifacts in `s3://lakehouse/mlflow` (proxied) |
| Valkey    | `localhost:6379`        | Feast online store (password auth, `noeviction`, RDB snapshots) |
| Airflow   | `http://localhost:8080` | Airflow 3, LocalExecutor. User `admin`, password `AIRFLOW_ADMIN_PASSWORD` in `.env` |

DAGs live in `pipelines/` (mounted live). Tasks run in `/opt/murray/.venv`, built from `uv.lock`,
via `@task.external_python`; Airflow keeps its own Python. After changing `src/` or `uv.lock`,
run `make up` to rebuild the image. Check DAGs with `make dag-errors` and `make dag-test d=smoke_stack`.

Other shortcuts: `make ps`, `make logs s=airflow-scheduler`, `make down`.
Reset everything (re-runs the Postgres init script): `docker compose --env-file .env -f infra/docker-compose.yml down -v`

Notebooks: `02_iceberg_sandbox` (Iceberg internals), `03_mlflow_tour` (MLflow storage, aliases, hostnames).

## Data and attribution

River data: © Commonwealth of Australia, Bureau of Meteorology (CC BY 4.0).
Weather data: [Open-Meteo](https://open-meteo.com/) (CC BY 4.0).
This is a planning aid, not a safety tool.

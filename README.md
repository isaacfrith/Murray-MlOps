# Murray River Forecast

Forecasts the Murray River's height at Barmah 1–7 days ahead and turns it into a
"good for wakeboarding?" rating. Built with feature / training / inference (FTI)
pipelines that share only a feature store (Feast) and a model registry (MLflow).

## Local stack

```bash
cp .env.example .env      # then replace every change-me (openssl rand -hex 16)
docker compose --env-file .env -f infra/docker-compose.yml up -d
uv sync
uv run python scripts/check_step1.py
```

| Service   | Address                 | Purpose                                          |
|-----------|-------------------------|--------------------------------------------------|
| Postgres  | `localhost:5432`        | DBs: `iceberg_catalog`, `mlflow`, `airflow` (one user each) |
| SeaweedFS | `http://localhost:8333` | S3 API (path-style), bucket `lakehouse`          |

Reset everything (re-runs the Postgres init script): `docker compose --env-file .env -f infra/docker-compose.yml down -v`

## Data and attribution

River data: © Commonwealth of Australia, Bureau of Meteorology (CC BY 4.0).
Weather data: [Open-Meteo](https://open-meteo.com/) (CC BY 4.0).
This is a planning aid, not a safety tool.

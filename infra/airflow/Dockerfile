# Airflow 3 + the murray package in its OWN virtualenv, built from uv.lock.
#
# Why a separate venv: Airflow pins ~700 packages (SQLAlchemy 2.0, pandas 3.0, ...);
# our pipelines need their own versions (pyiceberg, mlflow, later feast/lightgbm).
# Tasks run in /opt/murray/.venv via @task.external_python, so the pipeline code runs
# with exactly the same locked dependencies as on the laptop and in tests.
#
# A second venv, /opt/timestral/.venv, adds the `timestral` extra (CPU PyTorch). It can't be the
# same venv: Feast imports torch whenever it's installed, and torch + LightGBM in one process
# means two OpenMP runtimes. Only the second-opinion task runs there.
FROM apache/airflow:3.3.2-python3.12

COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/murray/.venv \
    UV_PYTHON=/usr/local/bin/python3 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

USER root
# libgomp1: OpenMP runtime that LightGBM's Linux wheel links against.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /opt/murray /opt/timestral && chown airflow:0 /opt/murray /opt/timestral
USER airflow
WORKDIR /opt/murray

# 1) Dependencies only (cached until uv.lock changes).
COPY --chown=airflow:0 pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project \
 && UV_PROJECT_ENVIRONMENT=/opt/timestral/.venv uv sync --frozen --no-dev --no-install-project --extra timestral

# 2) The murray package itself (rebuilt when src/ changes).
COPY --chown=airflow:0 src ./src
RUN uv sync --frozen --no-dev --no-editable \
 && UV_PROJECT_ENVIRONMENT=/opt/timestral/.venv uv sync --frozen --no-dev --no-editable --extra timestral

WORKDIR /opt/airflow

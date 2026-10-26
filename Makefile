# Shortcuts for the long compose command. Run from the repo root.
COMPOSE = docker compose --env-file .env -f infra/docker-compose.yml
# Timestral needs PyTorch, which can't share a venv with LightGBM: Feast imports torch whenever
# it's installed, and torch + LightGBM each bring an OpenMP runtime (macOS aborts on two).
TIMESTRAL = UV_PROJECT_ENVIRONMENT=.venv-timestral uv run --extra timestral --env-file .env

.PHONY: up down ps logs test unit dag-errors dag-test publish timestral-venv timestral-backtest second-opinion

up:        ## start (or update) the whole stack and wait until healthy
	$(COMPOSE) up -d --build --wait

down:      ## stop the stack (keeps data volumes)
	$(COMPOSE) down

ps:
	$(COMPOSE) ps -a

logs:      ## make logs s=mlflow
	$(COMPOSE) logs --tail 100 -f $(s)

test:      ## unit + integration tests against the running stack
	uv run pytest -q

unit:      ## unit tests only (no stack needed) — what CI runs
	uv run pytest -q -m "not integration"

dag-errors: ## show DAG files Airflow failed to import
	$(COMPOSE) exec airflow-scheduler airflow dags list-import-errors

dag-test:  ## run one DAG once, in-process, without the scheduler: make dag-test d=smoke_stack
	$(COMPOSE) exec airflow-scheduler airflow dags test $(d)

publish:   ## push site/*.json to the site-data branch now (Airflow does this after each forecast)
	uv run --env-file .env python -c "from murray.io.github_publish import publish; print(publish('site'))"

timestral-venv: ## build .venv-timestral (same uv.lock + the timestral extra)
	UV_PROJECT_ENVIRONMENT=.venv-timestral uv sync --extra timestral

timestral-backtest: ## zero-shot backtest of both Timestral variants, register the better @live
	$(TIMESTRAL) python -c "from murray.pipelines.timestral_pipeline import run; print(run())"

second-opinion: ## add Timestral's forecast to site/forecast.json (run after the main forecast)
	$(TIMESTRAL) python -c "from murray.pipelines.second_opinion_pipeline import run; print(run())"

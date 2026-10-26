# Shortcuts for the long compose command. Run from the repo root.
COMPOSE = docker compose --env-file .env -f infra/docker-compose.yml

.PHONY: up down ps logs test unit dag-errors dag-test publish

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

publish:   ## commit the latest site/*.json and push (GitHub Pages redeploys)
	git add site/forecast.json site/models.json site/monitoring.json
	git diff --cached --quiet || git commit -m "Publish forecast $$(date +%Y-%m-%d)"
	git push

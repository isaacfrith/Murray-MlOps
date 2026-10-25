# Shortcuts for the long compose command. Run from the repo root.
COMPOSE = docker compose --env-file .env -f infra/docker-compose.yml

.PHONY: up down ps logs test

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

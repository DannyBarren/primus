# =============================================================================
# Primus — one-command local & cloud Docker workflows.
#
#   Local (shares ~/.primus + ~/.config/primus with host):
#     make up           build + start, open http://127.0.0.1:7860
#     make logs         follow logs
#     make down         stop
#
#   Cloud (Vast.ai / RunPod / VPS — named volumes, exposed port):
#     make cloud        build + start the production stack
#     make cloud-logs   follow logs
#     make cloud-down   stop
#
#   Host (no Docker, identical to the daily driver):
#     make run          uv run python admin_assistant.py
# =============================================================================

# Match the host UID/GID so local bind-mounted data stays writable.
export UID := $(shell id -u)
export GID := $(shell id -g)

COMPOSE       := docker compose
CLOUD_COMPOSE := docker compose -f docker-compose.cloud.yml

.DEFAULT_GOAL := help

.PHONY: help up down restart build logs ps shell pull-models clean \
        cloud cloud-down cloud-logs cloud-build cloud-ps cloud-clean \
        run validate eval export import test lint ensure-token

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

ensure-token: ## Write PRIMUS_AUTH_TOKEN to .env if missing (gitignored)
	@if ! grep -q '^PRIMUS_AUTH_TOKEN=.\+' .env 2>/dev/null; then \
	  token=$$(openssl rand -hex 16); \
	  echo "PRIMUS_AUTH_TOKEN=$$token" >> .env; \
	  echo "Wrote PRIMUS_AUTH_TOKEN to .env — Gradio user 'primus', that token as password."; \
	else \
	  echo "Using PRIMUS_AUTH_TOKEN from .env"; \
	fi

## ---- Local (bind mounts, 127.0.0.1, CPU) --------------------------------------------------
up: ensure-token ## Build + start local stack (http://127.0.0.1:7860)
	$(COMPOSE) up -d --build
	@echo "Primus starting — open http://127.0.0.1:7860 (sign in as primus / token in .env)."

down: ## Stop local stack (keeps data)
	$(COMPOSE) down

restart: ## Restart just the primus service
	$(COMPOSE) restart primus

build: ## Build the primus image only
	$(COMPOSE) build

logs: ## Follow primus logs
	$(COMPOSE) logs -f primus

ps: ## Show local stack status
	$(COMPOSE) ps

shell: ## Open a shell inside the running primus container
	$(COMPOSE) exec primus /bin/bash

pull-models: ## (Re)pull Ollama models into the volume
	$(COMPOSE) run --rm ollama-init

clean: ## Stop local stack AND delete the ollama-models volume (host data dirs untouched)
	$(COMPOSE) down -v

## ---- Cloud (named volumes, exposed port, GPU-ready) ---------------------------------------
cloud: ensure-token ## Build + start the production stack
	$(CLOUD_COMPOSE) up -d --build
	@echo "Primus (cloud) starting on :7860 — Gradio auth is on (token in .env). Still prefer a proxy."

cloud-build: ## Build the cloud image only
	$(CLOUD_COMPOSE) build

cloud-down: ## Stop the cloud stack (keeps named volumes)
	$(CLOUD_COMPOSE) down

cloud-logs: ## Follow cloud primus logs
	$(CLOUD_COMPOSE) logs -f primus

cloud-ps: ## Show cloud stack status
	$(CLOUD_COMPOSE) ps

cloud-clean: ## Stop cloud stack AND delete all named volumes (DESTROYS persisted data)
	$(CLOUD_COMPOSE) down -v

## ---- Host / utilities ---------------------------------------------------------------------
run: ## Run on the host without Docker (uv run python admin_assistant.py)
	uv run python admin_assistant.py

eval: ## Run the smoke evals (sandboxed; writes examples/out/eval_<ts>.md)
	uv run python scripts/eval_primus.py

export: ## Write a backup zip to APP_DIR/exports (INCLUDE_SECRETS=1 to include tokens)
	uv run python admin_assistant.py --export $(if $(INCLUDE_SECRETS),--include-secrets,)

import: ## Restore a backup zip (FILE=path.zip; INCLUDE_SECRETS=1 to restore tokens)
	@test -n "$(FILE)" || (echo "usage: make import FILE=path.zip" && exit 2)
	uv run python admin_assistant.py --import $(FILE) $(if $(INCLUDE_SECRETS),--include-secrets,)

test: ## Hermetic unit tests (path guards, risk classifier, routing)
	python -m pytest

lint: ## Ruff error-level check
	ruff check primus tests admin_assistant.py

validate: ## Lint Dockerfile + validate both compose files
	docker build --check . || true
	PRIMUS_AUTH_TOKEN=$${PRIMUS_AUTH_TOKEN:-validate-only} $(COMPOSE) config -q && echo "docker-compose.yml OK"
	PRIMUS_AUTH_TOKEN=$${PRIMUS_AUTH_TOKEN:-validate-only} $(CLOUD_COMPOSE) config -q && echo "docker-compose.cloud.yml OK"

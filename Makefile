.DEFAULT_GOAL := help
COMPOSE ?= docker compose
DEV := -f compose.yaml -f compose.dev.yaml

.PHONY: help env build up dev proxy down logs index eval test

help: ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-6s %s\n", $$1, $$2}'

env: ## Create .env from .env.example if missing
	@test -f .env || cp .env.example .env

build: env ## Build the application images
	$(COMPOSE) build

up: env ## Start Qdrant, API and UI (indexes the corpus on first start)
	$(COMPOSE) up -d --build --wait
	@echo "UI: http://localhost:$${UI_PORT:-8501}  API docs: http://localhost:$${API_PORT:-8000}/docs"

dev: env ## Start with source mounted and API auto-reload
	$(COMPOSE) $(DEV) up --build

proxy: env ## Start everything plus the HTTPS proxy (requires DOMAIN)
	$(COMPOSE) --profile proxy up -d --build --wait

down: ## Stop services (the Qdrant volume is kept)
	$(COMPOSE) --profile proxy --profile eval down

logs: ## Follow service logs
	$(COMPOSE) logs -f --tail=100

index: ## Rebuild the index from CORPUS_DIR (default assets/corpus)
	$(COMPOSE) run --rm api python -m src.ingestion.index $${CORPUS_DIR:-assets/corpus}

eval: ## Run the RAGAS evaluation (ARGS="--judge-provider ollama")
	$(COMPOSE) --profile eval run --rm eval python -m src.eval.run_evaluation $(ARGS)

test: ## Run offline tests on the host
	pytest -q --cov

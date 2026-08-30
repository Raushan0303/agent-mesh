.PHONY: help dev infra worker gateway test eval chaos lint clean

PYTHON := python
VENV := venv
ACTIVATE := source $(VENV)/bin/activate

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

infra: ## Start Temporal + Postgres + Jaeger via docker-compose
	docker-compose up -d
	@echo ""
	@echo "Infrastructure started:"
	@echo "  Temporal UI:  http://localhost:8080"
	@echo "  Jaeger UI:    http://localhost:16686"
	@echo "  Postgres:     localhost:5434"
	@echo ""
	@echo "Run 'make init-db' to create the agentmesh database tables."

init-db: ## Initialize the agentmesh database (one-time)
	$(ACTIVATE) && PYTHONPATH=. $(PYTHON) scripts/init_agent_db.py

init-memory: ## Populate the pgvector memory store (one-time)
	$(ACTIVATE) && PYTHONPATH=. $(PYTHON) scripts/init_memory_store.py

worker: ## Start the Temporal worker process
	$(ACTIVATE) && $(PYTHON) -m app.agentmesh.worker_runner

gateway: ## Start the FastAPI gateway (port 8000)
	$(ACTIVATE) && $(PYTHON) server.py

dev: ## Start everything — infra + worker + gateway (requires 3 terminals)
	@echo "Run each in a separate terminal:"
	@echo "  Terminal 1: make infra && make init-db"
	@echo "  Terminal 2: make worker"
	@echo "  Terminal 3: make gateway"
	@echo ""
	@echo "Then: curl http://localhost:8000/health"

test: ## Run the full test suite (excludes chaos test by default)
	$(ACTIVATE) && $(PYTHON) -m pytest tests/ -v --ignore=tests/agentmesh/test_chaos_idempotency.py

test-chaos: ## Run the chaos idempotency test (requires running stack)
	$(ACTIVATE) && $(PYTHON) -m pytest tests/agentmesh/test_chaos_idempotency.py -v -s

eval: ## Run the CI-gated eval suite for the sourcing agent (50 scenarios)
	$(ACTIVATE) && PYTHONPATH=. $(PYTHON) -m app.agents.sourcing_agent.eval_glue

eval-hiring: ## Run the CI-gated eval suite for the hiring agent (20 scenarios)
	$(ACTIVATE) && PYTHONPATH=. $(PYTHON) -m app.agents.hiring_agent.eval_glue

load: ## Run Locust load test (500 concurrent users, 60s)
	$(ACTIVATE) && locust -f tests/agentmesh/locustfile.py --headless \
		--users 500 --spawn-rate 10 --run-time 60s \
		--host http://localhost:8000

clean: ## Stop and remove docker volumes
	docker-compose down -v

stop: ## Stop docker infrastructure
	docker-compose down

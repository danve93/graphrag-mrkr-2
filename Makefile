SHELL := /bin/bash

.PHONY: unit-test start-neo4j stop-neo4j e2e-local

unit-test:
	@echo "Running unit tests..."
	pytest tests/unit/

start-neo4j:
	@echo "Starting Neo4j container (neo4j:5.21) as 'amber_neo4j'..."
	@docker run -d --name amber_neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/test neo4j:5.21 >/dev/null || true

stop-neo4j:
	@echo "Stopping and removing 'amber_neo4j' container (if present)..."
	-@docker stop amber_neo4j >/dev/null 2>&1 || true
	-@docker rm amber_neo4j >/dev/null 2>&1 || true

e2e-local: start-neo4j
	@echo "Waiting for Neo4j to initialize (polling bolt port 7687)..."
	@for i in {1..30}; do \
		if bash -c "cat < /dev/tcp/localhost/7687" >/dev/null 2>&1; then \
			echo "Neo4j bolt port is responsive"; break; \
		fi; \
		echo "Waiting for Neo4j... ($$i/30)"; sleep 2; \
	done; \
	@if ! bash -c "cat < /dev/tcp/localhost/7687" >/dev/null 2>&1; then \
		echo "Neo4j did not start in time"; $(MAKE) stop-neo4j; exit 1; \
	fi
	@echo "Running E2E tests against local Neo4j..."
	@NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=test pytest -q tests/e2e/
	$(MAKE) stop-neo4j

.PHONY: e2e-dc
e2e-dc:
	@echo "Starting Neo4j via docker-compose (docker-compose.e2e.yml)..."
	@docker compose -f docker-compose.e2e.yml up -d neo4j
	@echo "Waiting for Neo4j to initialize (polling bolt port 7687)..."
	@for i in {1..30}; do \
		if bash -c "cat < /dev/tcp/localhost/7687" >/dev/null 2>&1; then \
			echo "Neo4j bolt port is responsive"; break; \
		fi; \
		echo "Waiting for Neo4j... ($$i/30)"; sleep 2; \
	done; \
	@if ! bash -c "cat < /dev/tcp/localhost/7687" >/dev/null 2>&1; then \
		echo "Neo4j did not start in time"; docker compose -f docker-compose.e2e.yml down; exit 1; \
	fi
	@NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=test pytest -q tests/e2e/
	@docker compose -f docker-compose.e2e.yml down

.PHONY: e2e-compose-smoke
e2e-compose-smoke:
	@echo "Starting e2e compose stack with override (use NEO4J_AUTH to set password)"
	@bash -ec '\
if [ -z "${NEO4J_AUTH}" ]; then \
	NEO4J_PW=$$(openssl rand -base64 24 | tr -dc 'A-Za-z0-9' | head -c16); \
  mkdir -p .secrets; printf "%s" "$$NEO4J_PW" > .secrets/neo4j_password; chmod 600 .secrets/neo4j_password; \
  export NEO4J_AUTH="neo4j/$$NEO4J_PW"; \
  echo "Generated temporary NEO4J_AUTH and saved to .secrets/neo4j_password (mode 600)"; \
  echo "To reuse in this shell: export NEO4J_AUTH=neo4j/$$NEO4J_PW"; \
fi; \
if [ -n "$$NEO4J_PW" ]; then \
	NEO4J_AUTH="neo4j/$$NEO4J_PW" docker compose -f docker-compose.e2e.yml up -d --force-recreate; \
else \
	NEO4J_AUTH=${NEO4J_AUTH} docker compose -f docker-compose.e2e.yml up -d --force-recreate; \
fi; \
NEO4J_URI=bolt://localhost:7687 bash scripts/wait_for_services.sh echo ok; \
pytest -q tests/smoke || true; \
docker compose -f docker-compose.e2e.yml down'

.PHONY: e2e-full-pipeline
e2e-full-pipeline:
	@echo "Starting e2e compose stack for full-pipeline test (use NEO4J_AUTH to set password)"
	@bash -ec '\
if [ -z "${NEO4J_AUTH}" ]; then \
	NEO4J_PW=$$(openssl rand -base64 24 | tr -dc 'A-Za-z0-9' | head -c16); \
  mkdir -p .secrets; printf "%s" "$$NEO4J_PW" > .secrets/neo4j_password; chmod 600 .secrets/neo4j_password; \
  export NEO4J_AUTH="neo4j/$$NEO4J_PW"; \
  echo "Generated temporary NEO4J_AUTH and saved to .secrets/neo4j_password (mode 600)"; \
  echo "To reuse in this shell: export NEO4J_AUTH=neo4j/$$NEO4J_PW"; \
fi; \
if [ -n "$$NEO4J_PW" ]; then \
	NEO4J_AUTH="neo4j/$$NEO4J_PW" docker compose -f docker-compose.e2e.yml up -d --force-recreate; \
else \
	NEO4J_AUTH=${NEO4J_AUTH} docker compose -f docker-compose.e2e.yml up -d --force-recreate; \
fi; \
NEO4J_URI=bolt://localhost:7687 bash scripts/wait_for_services.sh echo ok; \
NEO4J_URI=bolt://localhost:7687 pytest -q tests/e2e/test_full_pipeline.py; \
docker compose -f docker-compose.e2e.yml down'

# Benchmark and performance testing targets
.PHONY: bench bench-ttft bench-e2e bench-breakdown bench-compare test-performance

bench:
	@echo "Running full latency benchmark suite..."
	@./scripts/bench_latency.sh --benchmark all

bench-ttft:
	@echo "Running TTFT (Time To First Token) benchmark..."
	@./scripts/bench_latency.sh --benchmark ttft

bench-e2e:
	@echo "Running end-to-end latency benchmark..."
	@./scripts/bench_latency.sh --benchmark e2e

bench-breakdown:
	@echo "Running latency breakdown benchmark (retrieval vs generation)..."
	@./scripts/bench_latency.sh --benchmark breakdown

bench-compare:
	@if [ -z "$(PREV)" ]; then \
		echo "Error: PREV variable not set. Usage: make bench-compare PREV=path/to/previous_results.json"; \
		exit 1; \
	fi
	@echo "Running benchmark and comparing with previous results..."
	@./scripts/bench_latency.sh --benchmark all --compare $(PREV)

test-performance: bench
	@echo "Performance tests complete. Check benchmark_results.json for details."
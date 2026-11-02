PY ?= .venv/bin/python

.PHONY: install lint test test-integration test-models seed ingest run mcp eval-retrieval eval-full docker-up docker-down

install:
	python3.12 -m venv .venv && $(PY) -m pip install -r requirements-dev.txt

lint:
	$(PY) -m ruff check app scripts tests

test:
	$(PY) -m pytest tests/unit -q

test-integration:   ## needs TEST_DATABASE_URL (admin URL of a disposable database)
	$(PY) -m pytest tests/integration -q -m postgres

test-models:        ## downloads ~180 MB of Hugging Face models
	RUN_MODEL_TESTS=1 $(PY) -m pytest tests/integration -q -m models

seed:
	$(PY) -m scripts.seed_db

ingest:
	$(PY) -m scripts.ingest_samples

run:
	$(PY) -m uvicorn app.main:app --reload

mcp:
	$(PY) -m app.mcp_server

eval-retrieval:
	$(PY) -m scripts.evaluate --mode retrieval

eval-full:
	$(PY) -m scripts.evaluate --mode full

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v

.PHONY: test lint run run-dry clean install install-dev help

help:
	@echo "Available targets:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install all dependencies including dev/test"
	@echo "  test         Run all tests (excluding live Ollama)"
	@echo "  test-all     Run all tests including Ollama-dependent ones"
	@echo "  lint         Run ruff linter"
	@echo "  run          Run full pipeline (sangraha + pdf)"
	@echo "  run-dry      Dry run — validate config and paths only"
	@echo "  clean        Remove pytest cache and coverage artifacts"

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v -m "not requires_ollama" \
		--cov=src/slm_hindi \
		--cov-report=term-missing \
		--cov-report=html:htmlcov

test-all:
	pytest tests/ -v \
		--cov=src/slm_hindi \
		--cov-report=term-missing

test-unit:
	pytest tests/unit/ -v \
		--cov=src/slm_hindi \
		--cov-report=term-missing

test-integration:
	pytest tests/integration/ -v -m "not requires_ollama"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

lint-fix:
	ruff check --fix src/ tests/
	ruff format src/ tests/

run:
	python -m slm_hindi.orchestration.run_ingestion \
		--config configs/ingestion_config.yaml \
		--source all

run-sangraha:
	python -m slm_hindi.orchestration.run_ingestion \
		--config configs/ingestion_config.yaml \
		--source sangraha

run-pdf:
	python -m slm_hindi.orchestration.run_ingestion \
		--config configs/ingestion_config.yaml \
		--source pdf

run-dry:
	python -m slm_hindi.orchestration.run_ingestion \
		--config configs/ingestion_config.yaml \
		--dry-run

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache

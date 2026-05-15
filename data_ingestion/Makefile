.PHONY: test test-unit test-all test-int lint lint-fix run run-sangraha run-pdf run-wiki run-dry install install-dev clean help

help:
	@echo "Hindi SLM — Data Ingestion"
	@echo ""
	@echo "Setup:    install | install-dev"
	@echo "Tests:    test | test-unit | test-all | test-int"
	@echo "Lint:     lint | lint-fix"
	@echo "Pipeline: run | run-sangraha | run-pdf | run-wiki | run-dry"
	@echo "Other:    clean"

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

test-int:
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

run-wiki:
	python -m slm_hindi.orchestration.run_ingestion \
		--config configs/ingestion_config.yaml \
		--source wiki

run-dry:
	python -m slm_hindi.orchestration.run_ingestion \
		--config configs/ingestion_config.yaml \
		--dry-run

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache

.PHONY: test lint lint-fix install run run-sample run-train run-validate run-compare run-package run-publish

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=hindi_tokenizer --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v --cov=hindi_tokenizer --cov-report=term-missing

test-integration:
	pytest tests/integration/ -v -m "not requires_hf_hub"

lint:
	ruff check src/ tests/

lint-fix:
	ruff check src/ tests/ --fix

run:
	python -m hindi_tokenizer.orchestration.run_tokenizer --step all

run-sample:
	python -m hindi_tokenizer.orchestration.run_tokenizer --step sample

run-train:
	python -m hindi_tokenizer.orchestration.run_tokenizer --step train

run-validate:
	python -m hindi_tokenizer.orchestration.run_tokenizer --step validate

run-compare:
	python -m hindi_tokenizer.orchestration.run_tokenizer --step compare

run-package:
	python -m hindi_tokenizer.orchestration.run_tokenizer --step package

run-publish:
	python -m hindi_tokenizer.orchestration.run_tokenizer --step publish

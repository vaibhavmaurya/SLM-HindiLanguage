@echo off
pytest tests/ -v --cov=hindi_tokenizer --cov-report=term-missing %*

#!/usr/bin/env bash
set -euo pipefail
pytest tests/ -v --cov=hindi_tokenizer --cov-report=term-missing "$@"

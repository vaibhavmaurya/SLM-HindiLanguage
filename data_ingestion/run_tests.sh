#!/usr/bin/env bash
# Run the data ingestion test suite.
# Usage: ./run_tests.sh [pytest options]
# Examples:
#   ./run_tests.sh                          # all unit tests, no Ollama required
#   ./run_tests.sh tests/unit/test_wiki_crawler.py -v
#   ./run_tests.sh -m requires_ollama       # tests that need live Ollama

set -euo pipefail
cd "$(dirname "$0")"

# Activate virtualenv if found at common locations
if   [ -f "../.venv/Scripts/activate" ]; then source "../.venv/Scripts/activate"
elif [ -f "../.venv/bin/activate" ];     then source "../.venv/bin/activate"
elif [ -f ".venv/Scripts/activate" ];    then source ".venv/Scripts/activate"
elif [ -f ".venv/bin/activate" ];        then source ".venv/bin/activate"
fi

exec pytest tests/ \
    -v \
    -m "not requires_ollama" \
    --cov=src/slm_hindi \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    "$@"

#!/usr/bin/env bash
set -euo pipefail
python -m hindi_tokenizer.orchestration.run_tokenizer "$@"

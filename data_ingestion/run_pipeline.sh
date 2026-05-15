#!/usr/bin/env bash
# Run the Hindi SLM data ingestion pipeline.
# Usage: ./run_pipeline.sh [--source SOURCE] [--config PATH] [--dry-run]
# Examples:
#   ./run_pipeline.sh                           # ingest all sources
#   ./run_pipeline.sh --source wiki             # wiki only
#   ./run_pipeline.sh --source sangraha         # Sangraha only
#   ./run_pipeline.sh --source pdf              # PDFs only
#   ./run_pipeline.sh --dry-run                 # validate config, no writes
#   ./run_pipeline.sh --config path/to/cfg.yaml # custom config

set -euo pipefail
cd "$(dirname "$0")"

# Activate virtualenv if found at common locations
if   [ -f "../.venv/Scripts/activate" ]; then source "../.venv/Scripts/activate"
elif [ -f "../.venv/bin/activate" ];     then source "../.venv/bin/activate"
elif [ -f ".venv/Scripts/activate" ];    then source ".venv/Scripts/activate"
elif [ -f ".venv/bin/activate" ];        then source ".venv/bin/activate"
fi

exec python -m slm_hindi.orchestration.run_ingestion \
    --config configs/ingestion_config.yaml \
    "$@"

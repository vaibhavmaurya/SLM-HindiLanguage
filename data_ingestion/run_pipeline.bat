@echo off
REM Run the Hindi SLM data ingestion pipeline.
REM Usage: run_pipeline.bat [--source SOURCE] [--config PATH] [--dry-run]
REM Examples:
REM   run_pipeline.bat                            -- ingest all sources
REM   run_pipeline.bat --source wiki              -- wiki only
REM   run_pipeline.bat --source sangraha          -- Sangraha only
REM   run_pipeline.bat --source pdf               -- PDFs only
REM   run_pipeline.bat --dry-run                  -- validate config, no writes
REM   run_pipeline.bat --source all               -- explicit all sources

cd /d "%~dp0"

REM Activate virtualenv if found
if exist "..\venv\Scripts\activate.bat"  call "..\venv\Scripts\activate.bat"
if exist ".venv\Scripts\activate.bat"    call ".venv\Scripts\activate.bat"

python -m slm_hindi.orchestration.run_ingestion ^
    --config configs\ingestion_config.yaml ^
    %*

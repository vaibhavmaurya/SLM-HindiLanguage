@echo off
REM Run the data ingestion test suite.
REM Usage: run_tests.bat [pytest options]
REM Examples:
REM   run_tests.bat
REM   run_tests.bat tests\unit\test_wiki_crawler.py -v
REM   run_tests.bat -m requires_ollama

cd /d "%~dp0"

REM Activate virtualenv if found
if exist "..\venv\Scripts\activate.bat"  call "..\venv\Scripts\activate.bat"
if exist ".venv\Scripts\activate.bat"    call ".venv\Scripts\activate.bat"

pytest tests\ -v -m "not requires_ollama" ^
    --cov=src/slm_hindi ^
    --cov-report=term-missing ^
    --cov-report=html:htmlcov ^
    %*

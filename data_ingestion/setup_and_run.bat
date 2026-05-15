@echo off
REM setup_and_run.bat -- Full setup + pipeline runner for Hindi SLM data ingestion
REM
REM Usage (from repo root or data_ingestion\):
REM   setup_and_run.bat                        -- setup + run all sources
REM   setup_and_run.bat --source sangraha      -- Sangraha only
REM   setup_and_run.bat --source wiki          -- Wikipedia only
REM   setup_and_run.bat --source pdf           -- PDFs only
REM   setup_and_run.bat --dry-run              -- validate config, no writes
REM   setup_and_run.bat --skip-setup           -- skip venv/install, just run

setlocal enabledelayedexpansion

REM ---------------------------------------------------------------------------
REM Parse arguments
REM ---------------------------------------------------------------------------
set "SOURCE=all"
set "DRY_RUN="
set "SKIP_SETUP="

:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="--source"     ( set "SOURCE=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--dry-run"    ( set "DRY_RUN=--dry-run" & shift & goto parse_args )
if /i "%~1"=="--skip-setup" ( set "SKIP_SETUP=1" & shift & goto parse_args )
shift
goto parse_args
:end_parse

REM ---------------------------------------------------------------------------
REM Locate data_ingestion\ regardless of where the script is invoked from
REM ---------------------------------------------------------------------------
set "PROJECT_ROOT=%~dp0"
if "!PROJECT_ROOT:~-1!"=="\" set "PROJECT_ROOT=!PROJECT_ROOT:~0,-1!"

if not exist "!PROJECT_ROOT!\pyproject.toml" (
    set "PROJECT_ROOT=!PROJECT_ROOT!\data_ingestion"
)
if not exist "!PROJECT_ROOT!\pyproject.toml" (
    echo FAIL  Cannot locate data_ingestion\pyproject.toml.
    echo       Run this script from SLM_HINDI\ or data_ingestion\.
    exit /b 1
)

cd /d "!PROJECT_ROOT!"

echo.
echo  Hindi SLM - Data Ingestion Pipeline
echo  Working directory : !PROJECT_ROOT!
echo  Source            : !SOURCE!
echo  Dry run           : !DRY_RUN!
echo  Skip setup        : !SKIP_SETUP!
echo.

REM ---------------------------------------------------------------------------
REM Step 1 - Find Python 3.11+
REM ---------------------------------------------------------------------------
echo ===> Step 1: Checking Python version
set "PYTHON_EXE="

for %%C in (python python3 py) do (
    if not defined PYTHON_EXE (
        %%C --version >nul 2>&1
        if !errorlevel! == 0 (
            %%C --version 2>&1 | findstr /r "3\.[1-9][1-9]" >nul
            if !errorlevel! == 0 (
                set "PYTHON_EXE=%%C"
            )
        )
    )
)

if not defined PYTHON_EXE (
    echo FAIL  No Python 3.11+ found on PATH.
    echo       Install from https://python.org and re-run.
    exit /b 1
)

for /f "tokens=*" %%V in ('!PYTHON_EXE! --version 2^>^&1') do echo   OK  %%V  ^(using !PYTHON_EXE!^)

REM ---------------------------------------------------------------------------
REM Step 2 - Locate or create virtual environment
REM ---------------------------------------------------------------------------
echo ===> Step 2: Setting up virtual environment
set "VENV_DIR="

if exist "!PROJECT_ROOT!\.venv\Scripts\python.exe" set "VENV_DIR=!PROJECT_ROOT!\.venv"
if not defined VENV_DIR if exist "!PROJECT_ROOT!\venv\Scripts\python.exe" set "VENV_DIR=!PROJECT_ROOT!\venv"
if not defined VENV_DIR if exist "!PROJECT_ROOT!\..\venv\Scripts\python.exe" (
    for %%F in ("!PROJECT_ROOT!\..\venv") do set "VENV_DIR=%%~fF"
)
if not defined VENV_DIR if exist "!PROJECT_ROOT!\..\.venv\Scripts\python.exe" (
    for %%F in ("!PROJECT_ROOT!\..\.venv") do set "VENV_DIR=%%~fF"
)

if defined SKIP_SETUP (
    if not defined VENV_DIR (
        echo FAIL  No venv found. Remove --skip-setup to create one.
        exit /b 1
    )
    echo   OK  Using existing venv at !VENV_DIR!
    goto run_pipeline
)

if defined VENV_DIR (
    echo   OK  Found existing venv at !VENV_DIR!
) else (
    set "VENV_DIR=!PROJECT_ROOT!\.venv"
    echo       Creating new venv at !VENV_DIR! ...
    !PYTHON_EXE! -m venv "!VENV_DIR!"
    if !errorlevel! neq 0 ( echo FAIL  venv creation failed & exit /b 1 )
    echo   OK  venv created
)

REM ---------------------------------------------------------------------------
REM Step 3 - Upgrade pip
REM ---------------------------------------------------------------------------
echo ===> Step 3: Upgrading pip
"!VENV_DIR!\Scripts\python.exe" -m pip install --upgrade pip --quiet
if !errorlevel! neq 0 ( echo FAIL  pip upgrade failed & exit /b 1 )
echo   OK  pip up to date

REM ---------------------------------------------------------------------------
REM Step 4 - Install production dependencies
REM ---------------------------------------------------------------------------
echo ===> Step 4: Installing production dependencies
"!VENV_DIR!\Scripts\pip.exe" install -r requirements.txt --quiet
if !errorlevel! neq 0 ( echo FAIL  pip install requirements.txt failed & exit /b 1 )
echo   OK  Production dependencies installed

REM ---------------------------------------------------------------------------
REM Step 5 - Install package in editable mode with dev extras
REM ---------------------------------------------------------------------------
echo ===> Step 5: Installing package in editable mode
"!VENV_DIR!\Scripts\pip.exe" install -e ".[dev]" --quiet
if !errorlevel! neq 0 ( echo FAIL  pip install -e .[dev] failed & exit /b 1 )
echo   OK  Package installed in editable mode

REM ---------------------------------------------------------------------------
REM Step 6 - Verify import
REM ---------------------------------------------------------------------------
echo ===> Step 6: Verifying slm_hindi package is importable
"!VENV_DIR!\Scripts\python.exe" -c "import slm_hindi; print('  OK  slm_hindi importable')"
if !errorlevel! neq 0 (
    echo FAIL  Package import failed - remove --skip-setup to reinstall.
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM Step 7 - Run the pipeline
REM ---------------------------------------------------------------------------
:run_pipeline
echo ===> Step 7: Running pipeline  source=!SOURCE!  !DRY_RUN!
echo.

"!VENV_DIR!\Scripts\python.exe" -m slm_hindi.orchestration.run_ingestion ^
    --config configs\ingestion_config.yaml ^
    --source !SOURCE! ^
    !DRY_RUN!

set "EXIT_CODE=!errorlevel!"
echo.
if !EXIT_CODE! == 0 (
    echo  Pipeline finished successfully.
) else (
    echo FAIL  Pipeline exited with code !EXIT_CODE!
)

exit /b !EXIT_CODE!

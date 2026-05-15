# setup_and_run.ps1 — Full setup + pipeline runner for Hindi SLM data ingestion
#
# Usage (from repo root or data_ingestion/):
#   .\setup_and_run.ps1                          # setup + run all sources
#   .\setup_and_run.ps1 -Source sangraha         # Sangraha only
#   .\setup_and_run.ps1 -Source wiki             # Wikipedia only
#   .\setup_and_run.ps1 -Source pdf              # PDFs only
#   .\setup_and_run.ps1 -DryRun                  # validate config, no writes
#   .\setup_and_run.ps1 -SkipSetup               # skip venv/install, just run
#   .\setup_and_run.ps1 -SkipSetup -Source wiki  # skip setup, wiki only

param(
    [ValidateSet("all", "sangraha", "pdf", "wiki")]
    [string]$Source = "all",

    [switch]$DryRun,
    [switch]$SkipSetup
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Step([string]$msg) {
    Write-Host ""
    Write-Host "===> $msg" -ForegroundColor Cyan
}

function Write-OK([string]$msg) {
    Write-Host "  OK  $msg" -ForegroundColor Green
}

function Write-Fail([string]$msg) {
    Write-Host "FAIL  $msg" -ForegroundColor Red
}

# ---------------------------------------------------------------------------
# Locate data_ingestion/ regardless of where the script is invoked from
# ---------------------------------------------------------------------------

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = $ScriptDir

# If invoked from the repo root (SLM_HINDI/), descend into data_ingestion/
if (-not (Test-Path (Join-Path $ProjectRoot "pyproject.toml"))) {
    $ProjectRoot = Join-Path $ProjectRoot "data_ingestion"
}

if (-not (Test-Path (Join-Path $ProjectRoot "pyproject.toml"))) {
    Write-Fail "Cannot locate data_ingestion/pyproject.toml. Run this script from SLM_HINDI/ or data_ingestion/."
    exit 1
}

Set-Location $ProjectRoot
Write-Host ""
Write-Host "Hindi SLM — Data Ingestion Pipeline" -ForegroundColor Yellow
Write-Host "Working directory : $ProjectRoot" -ForegroundColor DarkGray
Write-Host "Source            : $Source"       -ForegroundColor DarkGray
Write-Host "Dry run           : $DryRun"       -ForegroundColor DarkGray
Write-Host "Skip setup        : $SkipSetup"    -ForegroundColor DarkGray

# ---------------------------------------------------------------------------
# Step 1 — Python version check
# ---------------------------------------------------------------------------

Write-Step "Checking Python version"

$PythonExe = $null
foreach ($candidate in @("python", "python3", "py")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 11)) {
                $PythonExe = $candidate
                Write-OK "$ver found as '$candidate'"
                break
            } else {
                Write-Host "  SKIP $candidate — $ver (need 3.11+)" -ForegroundColor DarkYellow
            }
        }
    } catch { }
}

if (-not $PythonExe) {
    Write-Fail "No Python 3.11+ found on PATH. Install it from https://python.org and re-run."
    exit 1
}

# ---------------------------------------------------------------------------
# Step 2 — Create / locate virtual environment
# ---------------------------------------------------------------------------

if (-not $SkipSetup) {
    Write-Step "Setting up virtual environment"

    # Look for an existing venv in common locations
    $VenvCandidates = @(
        (Join-Path $ProjectRoot ".venv"),
        (Join-Path $ProjectRoot "venv"),
        (Join-Path (Split-Path $ProjectRoot -Parent) "venv"),
        (Join-Path (Split-Path $ProjectRoot -Parent) ".venv")
    )

    $VenvDir = $null
    foreach ($c in $VenvCandidates) {
        if (Test-Path (Join-Path $c "Scripts\python.exe")) {
            $VenvDir = $c
            Write-OK "Found existing venv at $VenvDir"
            break
        }
    }

    if (-not $VenvDir) {
        $VenvDir = Join-Path $ProjectRoot ".venv"
        Write-Host "  Creating new venv at $VenvDir …" -ForegroundColor DarkGray
        & $PythonExe -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) { Write-Fail "venv creation failed"; exit 1 }
        Write-OK "venv created"
    }

    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    $VenvPip    = Join-Path $VenvDir "Scripts\pip.exe"

    # ---------------------------------------------------------------------------
    # Step 3 — Upgrade pip + install dependencies
    # ---------------------------------------------------------------------------

    Write-Step "Upgrading pip"
    & $VenvPython -m pip install --upgrade pip --quiet
    if ($LASTEXITCODE -ne 0) { Write-Fail "pip upgrade failed"; exit 1 }
    Write-OK "pip up to date"

    Write-Step "Installing production dependencies (requirements.txt)"
    & $VenvPip install -r requirements.txt --quiet
    if ($LASTEXITCODE -ne 0) { Write-Fail "pip install requirements.txt failed"; exit 1 }
    Write-OK "Production dependencies installed"

    Write-Step "Installing dev dependencies + package in editable mode"
    & $VenvPip install -e ".[dev]" --quiet
    if ($LASTEXITCODE -ne 0) { Write-Fail "pip install -e .[dev] failed"; exit 1 }
    Write-OK "Package installed in editable mode"

} else {
    # SkipSetup — just locate the venv python
    Write-Step "Locating existing virtual environment (--SkipSetup)"

    $VenvCandidates = @(
        (Join-Path $ProjectRoot ".venv"),
        (Join-Path $ProjectRoot "venv"),
        (Join-Path (Split-Path $ProjectRoot -Parent) "venv"),
        (Join-Path (Split-Path $ProjectRoot -Parent) ".venv")
    )

    $VenvDir = $null
    foreach ($c in $VenvCandidates) {
        if (Test-Path (Join-Path $c "Scripts\python.exe")) {
            $VenvDir = $c
            break
        }
    }

    if (-not $VenvDir) {
        Write-Fail "No venv found. Remove -SkipSetup to create one."
        exit 1
    }

    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    Write-OK "Using venv at $VenvDir"
}

# ---------------------------------------------------------------------------
# Step 4 — Verify package import
# ---------------------------------------------------------------------------

Write-Step "Verifying slm_hindi package is importable"
& $VenvPython -c "import slm_hindi; print('  slm_hindi version:', slm_hindi.__version__ if hasattr(slm_hindi, '__version__') else 'ok')"
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Package import failed — run without -SkipSetup to reinstall."
    exit 1
}

# ---------------------------------------------------------------------------
# Step 5 — Run the pipeline
# ---------------------------------------------------------------------------

Write-Step "Running pipeline (source=$Source$(if ($DryRun) { ', dry-run' }))"

$PipelineArgs = @(
    "-m", "slm_hindi.orchestration.run_ingestion",
    "--config", "configs\ingestion_config.yaml",
    "--source", $Source
)

if ($DryRun) {
    $PipelineArgs += "--dry-run"
}

& $VenvPython @PipelineArgs
$ExitCode = $LASTEXITCODE

Write-Host ""
if ($ExitCode -eq 0) {
    Write-Host "Pipeline finished successfully." -ForegroundColor Green
} else {
    Write-Fail "Pipeline exited with code $ExitCode"
}

exit $ExitCode

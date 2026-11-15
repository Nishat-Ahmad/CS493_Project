@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
cd /d "%REPO_ROOT%"

echo ==========================================================================
echo  Building Edge Collision Anticipation Docker Container Image
echo ==========================================================================
docker build -t edge-collision-benchmark:latest -f docker/Dockerfile .
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker build failed.
    exit /b %ERRORLEVEL%
)

if not exist "%REPO_ROOT%\results" mkdir "%REPO_ROOT%\results"

echo ==========================================================================
echo  Executing Benchmark under POSIX cgroups: --cpus=4.0, --memory=4g
echo ==========================================================================
docker run --rm -it ^
  --cpus="4.0" ^
  --memory="4g" ^
  --memory-swap="4g" ^
  -v "%REPO_ROOT%/data:/workspace/data:ro" ^
  -v "%REPO_ROOT%/weights:/workspace/weights:ro" ^
  -v "%REPO_ROOT%/results:/workspace/results" ^
  edge-collision-benchmark:latest python3 benchmark/runner.py

echo ==========================================================================
echo  Benchmark Execution Complete. Outputs written to results/
echo ==========================================================================

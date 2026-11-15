# PowerShell execution script for Dockerized edge hardware simulation
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host " Building Edge Collision Anticipation Docker Container Image" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

docker build -t edge-collision-benchmark:latest -f docker/Dockerfile .

# Ensure results directory exists locally
if (-not (Test-Path "$RepoRoot\results")) {
    New-Item -ItemType Directory -Path "$RepoRoot\results" | Out-Null
}

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host " Executing Benchmark under POSIX cgroups: --cpus=4.0, --memory=4g" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

docker run --rm -it `
  --cpus="4.0" `
  --memory="4g" `
  --memory-swap="4g" `
  -v "${RepoRoot}/data:/workspace/data:ro" `
  -v "${RepoRoot}/weights:/workspace/weights:ro" `
  -v "${RepoRoot}/results:/workspace/results" `
  edge-collision-benchmark:latest python3 benchmark/runner.py

Write-Host "==========================================================================" -ForegroundColor Green
Write-Host " Benchmark Execution Complete. Outputs written to results/" -ForegroundColor Green
Write-Host "==========================================================================" -ForegroundColor Green

#!/usr/bin/env bash
set -e

# Navigate to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "=========================================================================="
echo " Building Edge Collision Anticipation Docker Container Image"
echo "=========================================================================="
docker build -t edge-collision-benchmark:latest -f docker/Dockerfile .

# Ensure results directory exists locally
mkdir -p results

echo "=========================================================================="
echo " Executing Benchmark under POSIX cgroups: --cpus=4.0, --memory=4g"
echo "=========================================================================="
docker run --rm -it \
  --cpus="4.0" \
  --memory="4g" \
  --memory-swap="4g" \
  -v "${REPO_ROOT}/data:/workspace/data:ro" \
  -v "${REPO_ROOT}/weights:/workspace/weights:ro" \
  -v "${REPO_ROOT}/results:/workspace/results" \
  edge-collision-benchmark:latest python3 benchmark/runner.py

echo "=========================================================================="
echo " Benchmark Execution Complete. Outputs written to results/"
echo "=========================================================================="

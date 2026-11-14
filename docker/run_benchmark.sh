#!/usr/bin/env bash
set -e

# Build the benchmark container image
docker build -t edge-collision-benchmark:latest -f docker/Dockerfile .

# Ensure results directory exists locally
mkdir -p results

# Execute the benchmark under simulated edge hardware constraints (4 CPU cores, 4GB RAM)
docker run --rm -it \
  --cpus="4.0" \
  --memory="4g" \
  --memory-swap="4g" \
  -v "$(pwd)/results:/workspace/results" \
  edge-collision-benchmark:latest python3 benchmark/runner.py


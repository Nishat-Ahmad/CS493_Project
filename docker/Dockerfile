# Dockerfile for Edge Collision Benchmark
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install base dependencies and system libraries for OpenCV and ONNX
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python requirements
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy repository source code
COPY . /workspace

# Default command runs the benchmark runner
CMD ["python3", "benchmark/runner.py"]


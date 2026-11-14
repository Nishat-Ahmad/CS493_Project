# Benchmarking Single-Shot Detector Scaling and Quantization Trade-offs for Real-Time Vehicle Collision Anticipation under Simulated Edge Hardware Constraints

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.19+-green.svg)](https://onnxruntime.ai/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

---

## 1. Project Overview & Research Objective

Intelligent Transportation Systems (ITS) and edge Advanced Driver Assistance Systems (ADAS) demand sub-second trajectory forecasting and collision anticipation under strict hardware constraints (e.g., 4 CPU cores, 4GB RAM, no discrete GPU). While modern deep neural networks achieve high detection accuracy on high-end server hardware, deploying unquantized models on edge compute creates severe inference latency bottlenecks that collapse frame rates below the essential **20–30 FPS real-time threshold**, leading to catastrophic latency-induced collision misses.

This research benchmarks **YOLOv11 model scaling** (Nano vs. Small variants) across multiple **numerical precisions** (**FP32**, **FP16**, and **INT8 post-training quantization**) under strict POSIX cgroup resource limits.

### Core Processing Pipeline
```
[Video / Camera Stream] (640x640 @ 30 FPS)
         │
         ▼
[Single-Shot ONNX Detector] (YOLOv11n / YOLOv11s across FP32, FP16, INT8)
         │
         ▼ (Bounding Boxes [x1, y1, x2, y2, score, class])
[SORT Multi-Object Tracker] (7-state Kalman Filter + Hungarian Data Association)
         │
         ▼ (Active Tracks with Coordinates & Velocities: [u, v, vx, vy])
[Kinematic Trajectory Forecaster] (Closing Speed v_c & Time-to-Collision TTC)
         │
         ▼
[Collision Anticipation Alert] (TTC < 2.5s sustained across >= 3 frames)
```

---

## 2. Mathematical Formulations

### 2.1 Kalman Filter Multi-Object Tracking (SORT Algorithm)
Each object state is modeled by a 7-dimensional continuous state vector:
$$\mathbf{x} = \begin{bmatrix} u & v & s & r & \dot{u} & \dot{v} & \dot{s} \end{bmatrix}^T$$
where:
- $(u, v)$ denotes the 2D bounding box center coordinates.
- $s = w \times h$ represents the bounding box scale (area).
- $r = w / h$ is the bounding box aspect ratio.
- $\dot{u}, \dot{v}, \dot{s}$ denote the first-order temporal derivatives (velocities) of center position and scale.

The state transition matrix $\mathbf{F} \in \mathbb{R}^{7 \times 7}$ under a discrete constant-velocity model ($\Delta t = 1$) is:
$$\mathbf{F} = \begin{bmatrix}
1 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}$$

The 4-dimensional observation vector is $\mathbf{z} = \begin{bmatrix} u & v & s & r \end{bmatrix}^T$ with measurement matrix $\mathbf{H} \in \mathbb{R}^{4 \times 7}$:
$$\mathbf{H} = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0
\end{bmatrix}$$

Data association between predicted track states and newly detected bounding boxes is solved via the Hungarian algorithm on an Intersection over Union (IoU) cost matrix:
$$\mathbf{C}_{i,j} = -\mathrm{IoU}(\mathbf{b}_{\text{pred}, i}, \mathbf{b}_{\text{det}, j})$$

### 2.2 Kinematic Time-to-Collision (TTC) Trajectory Forecasting
For any pair of active tracks $i$ and $j$ with center coordinates $\mathbf{p}_i = (x_i, y_i)$, $\mathbf{p}_j = (x_j, y_j)$ and velocities $\mathbf{v}_i = (\dot{x}_i, \dot{y}_i)$, $\mathbf{v}_j = (\dot{x}_j, \dot{y}_j)$:

1. **Displacement Vector**:
   $$\mathbf{d} = \mathbf{p}_j - \mathbf{p}_i = \begin{bmatrix} x_j - x_i \\ y_j - y_i \end{bmatrix}$$

2. **Relative Velocity**:
   $$\mathbf{v}_{\text{rel}} = \mathbf{v}_i - \mathbf{v}_j = \begin{bmatrix} \dot{x}_i - \dot{x}_j \\ \dot{y}_i - \dot{y}_j \end{bmatrix}$$

3. **Closing Speed**:
   $$v_c = \frac{\mathbf{d} \cdot \mathbf{v}_{\text{rel}}}{\|\mathbf{d}\| + 10^{-6}}$$

4. **Time-To-Collision**:
   $$\mathrm{TTC} = \begin{cases}
   \dfrac{\|\mathbf{d}\|}{v_c}, & \text{if } v_c > 0 \\
   \infty, & \text{if } v_c \le 0 \text{ (diverging or parallel)}
   \end{cases}$$

An alert is triggered when $\mathrm{TTC} < \tau_{\text{threshold}}$ (default: $2.5\,\text{s}$) persists across $k \ge 3$ consecutive video frames.

### 2.3 Coordinate Jitter Variance
High-frequency spatial instability introduced by model quantization noise is quantified via discrete acceleration variance across object trajectories:
$$\mathrm{Jitter} = \frac{1}{N - 2} \sum_{t=1}^{N - 2} \left\| \mathbf{c}_{t+1} - 2\mathbf{c}_t + \mathbf{c}_{t-1} \right\|^2$$
where $\mathbf{c}_t = (x_t, y_t)$ represents the estimated bounding box center at frame $t$.

### 2.4 Resolution Time Margin (RTM)
The practical safety utility of collision anticipation is measured by how early the system provides a persistent warning before actual physical impact:
$$\mathrm{RTM} = \frac{\max\left(0,\, F_{\text{accident}} - F_{\text{first\_sustained\_alert}}\right)}{\mathrm{FPS}_{\text{video}}}$$
where $F_{\text{accident}}$ is the ground-truth collision frame, $F_{\text{first\_sustained\_alert}}$ is the first frame index achieving the consecutive alert threshold, and $\mathrm{FPS}_{\text{video}}$ is the sequence framerate (30 FPS).

---

## 3. Directory Structure

```
University/Projects/Comma/
├── docker/
│   ├── Dockerfile             # Ubuntu 24.04 cgroup benchmarking image
│   └── run_benchmark.sh       # Script applying 4 CPUs & 4GB RAM limits
├── models/
│   ├── export_onnx.py         # Exports YOLOv11n/s to static FP32 & FP16 ONNX
│   └── quantize_int8.py       # INT8 QDQ calibration & static quantization
├── tracking/
│   ├── kalman_tracker.py      # Self-contained SORT Kalman filter tracker
│   └── collision_kinematics.py# Kinematic closing speed and TTC forecaster
├── data/
│   ├── dataset_loader.py      # TU-DAT & synthetic video sequence parser
│   └── generate_mock_data.py  # 300-frame head-on crash video & label generator
├── benchmark/
│   ├── metrics.py             # mAP@50, Jitter variance, Latency & RTM calculations
│   └── runner.py              # Benchmarking engine across 6 configurations
├── analysis/
│   └── pareto_plotter.py      # High-DPI Pareto frontier and jitter charts
├── tests/
│   └── test_pipeline.py       # Pytest unit tests verifying math and pipeline
├── requirements.txt           # Python dependencies
├── run_all.py                 # Master orchestration end-to-end script
└── README.md                  # Project documentation and theoretical background
```

---

## 4. Evaluated Target Configurations

The benchmark evaluates six distinct operational configurations:

| # | Configuration Name | Model Variant | Numerical Precision | Target Engine |
|---|--------------------|---------------|---------------------|---------------|
| 1 | `yolo11n_fp32`     | YOLOv11-Nano  | IEEE 32-bit Float   | ONNX Runtime CPU |
| 2 | `yolo11n_fp16`     | YOLOv11-Nano  | IEEE 16-bit Float   | ONNX Runtime CPU |
| 3 | `yolo11n_int8`     | YOLOv11-Nano  | 8-bit Quantized QDQ | ONNX Runtime CPU |
| 4 | `yolo11s_fp32`     | YOLOv11-Small | IEEE 32-bit Float   | ONNX Runtime CPU |
| 5 | `yolo11s_fp16`     | YOLOv11-Small | IEEE 16-bit Float   | ONNX Runtime CPU |
| 6 | `yolo11s_int8`     | YOLOv11-Small | 8-bit Quantized QDQ | ONNX Runtime CPU |

---

## 5. Execution Instructions

### Option A: Local Python Environment
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute the complete end-to-end pipeline:
   ```bash
   python run_all.py
   ```
   This command sequentially generates synthetic datasets (if missing), runs the pytest validation suite, exports and quantizes all models, executes the 6-model benchmark loop, generates Pareto visual plots, and renders a final summary table in the terminal.

### Option B: Isolated Docker Container with Edge Hardware Limits
Simulate strict edge hardware constraints using POSIX cgroups (`--cpus="4.0"`, `--memory="4g"`):
```bash
bash docker/run_benchmark.sh
```

---

## 6. Generated Artifacts & Analysis Outputs

Upon benchmark completion, results are generated in `results/`:
- **`benchmark_summary.csv`**: Tabular benchmark results with latency, FPS, mAP@50, Jitter, and RTM.
- **`benchmark_summary.json`**: Machine-readable JSON summary for CI/CD integration.
- **`pareto_throughput_vs_rtm.png`**: Scatter plot depicting Throughput (FPS) vs. Resolution Time Margin (RTM), scaling markers by mAP@50, and drawing the Pareto optimal frontier alongside the 20 FPS real-time threshold.
- **`jitter_vs_precision.png`**: Comparative grouped bar chart illustrating the relationship between numerical quantization precision and trajectory coordinate jitter variance.


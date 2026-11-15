# Benchmarking Single-Shot Detector Scaling and Quantization Trade-offs for Real-Time Vehicle Collision Anticipation under Simulated Edge Hardware Constraints

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: YOLO26](https://img.shields.io/badge/Model-YOLO26-orange.svg)](https://github.com/ultralytics/ultralytics)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.19+-green.svg)](https://onnxruntime.ai/)
[![Docker](https://img.shields.io/badge/Docker-cgroups%20Simulated-2496ED.svg)](https://www.docker.com/)
[![Course: CS493](https://img.shields.io/badge/Course-CS493%20Scientific%20Communication-purple.svg)](reports/CS493_Final_Comprehensive_Report.pdf)

---

## 1. Project Overview & Research Objective

Roadside Intelligent Transportation Systems (ITS) and edge Advanced Driver Assistance Systems (ADAS) demand sub-second trajectory forecasting and vehicle collision anticipation to dispatch Vehicle-to-Everything (V2X) pre-crash advisories before physical impact occurs. However, roadside edge compute nodes operate under strict thermal, power, and cost constraints (typically restricted to **4 CPU cores, 4GB RAM, and no discrete GPU** inside IP67-sealed, passively cooled enclosures).

While architectural scaling from Nano to Small improves spatial feature representation, deeper neural networks suffer catastrophic latency degradation on embedded CPUs, collapsing framerates below the essential **20–30 FPS real-time threshold** ($\le 33.3\text{--}50.0\,\text{ms/frame}$).

This research benchmarks **YOLO26 model scaling** (Nano: 2.4M parameters vs. Small: 9.5M parameters) across three **numerical precision domains** (**FP32**, **FP16**, and **INT8 post-training quantization**) paired with a 7-state SORT Kalman tracker and a kinematic Time-to-Collision (TTC) forecasting engine. Evaluation is conducted across **10 real-world collision sequences** (~2,720 frames) and **5 negative control sequences** from the TU-DAT benchmark under standardized 4-core, 4GB RAM edge hardware constraints.

```
[Video / Camera Stream] (640x640 @ 30 FPS / 60 FPS)
         │
         ▼
[Single-Shot ONNX Detector] (YOLO26 Nano / Small across FP32, FP16, INT8)
         │
         ▼ (Bounding Boxes: [x1, y1, x2, y2, score, class])
[SORT Multi-Object Tracker] (7-state Kalman Filter + Hungarian Data Association)
         │
         ▼ (Active Tracks with Coordinates & Velocities: [u, v, s, r, u', v', s'])
[Kinematic Trajectory Forecaster] (Closing Speed v_c, Time-to-Collision TTC, CPA d_min)
         │
         ▼
[Collision Anticipation Alert] (TTC < 2.5s, d_min <= R_coll, v >= 30 px/s, k >= 3 frames)
```

---

## 2. Mathematical Formulations

### 2.1 Kalman Filter Multi-Object Tracking (SORT Algorithm)

Each object state is modeled by a 7-dimensional continuous state vector:

$$
\mathbf{x} = \begin{bmatrix} u & v & s & r & \dot{u} & \dot{v} & \dot{s} \end{bmatrix}^T
$$

where:
- $(u, v)$ denotes 2D bounding box center pixel coordinates.
- $s = w \times h$ represents the bounding box scale (area).
- $r = w / h$ is the bounding box aspect ratio.
- $\dot{u}, \dot{v}, \dot{s}$ denote the first-order temporal derivatives (velocities).

The state transition matrix $\mathbf{F} \in \mathbb{R}^{7 \times 7}$ under a discrete constant-velocity model ($\Delta t = 1$) is:

$$
\mathbf{F} = \begin{bmatrix}
1 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
$$

The 4-dimensional observation vector is $\mathbf{z} = \begin{bmatrix} u & v & s & r \end{bmatrix}^T$ with measurement matrix $\mathbf{H} \in \mathbb{R}^{4 \times 7}$:

$$
\mathbf{H} = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0
\end{bmatrix}
$$

Data association between predicted track states and newly detected bounding boxes is solved via the Hungarian algorithm on an Intersection over Union (IoU) cost matrix:

$$
\mathbf{C}_{i,j} = -\mathrm{IoU}(\mathbf{b}_{\text{pred}, i}, \mathbf{b}_{\text{det}, j})
$$

### 2.2 Kinematic Time-to-Collision (TTC) & Lateral Closest Point of Approach (CPA)

For any pair of active tracks $i$ and $j$ with center coordinates $\mathbf{p}_i = (x_i, y_i)$, $\mathbf{p}_j = (x_j, y_j)$ and velocities $\mathbf{v}_i = (\dot{x}_i, \dot{y}_i)$, $\mathbf{v}_j = (\dot{x}_j, \dot{y}_j)$:

1. **Relative Displacement Vector**:
   $$
   \mathbf{d} = \mathbf{p}_j - \mathbf{p}_i = \begin{bmatrix} x_j - x_i \\ y_j - y_i \end{bmatrix}
   $$

2. **Relative Velocity Vector**:
   $$
   \mathbf{v}_{\text{rel}} = \mathbf{v}_i - \mathbf{v}_j = \begin{bmatrix} \dot{x}_i - \dot{x}_j \\ \dot{y}_i - \dot{y}_j \end{bmatrix}
   $$

3. **Line-of-Sight Closing Speed**:
   $$
   v_c = \frac{\mathbf{d} \cdot \mathbf{v}_{\text{rel}}}{\|\mathbf{d}\| + 10^{-6}}
   $$

4. **Time-To-Collision (TTC)**:
   $$
   \mathrm{TTC} = \begin{cases}
   \dfrac{\|\mathbf{d}\|}{v_c}, & \text{if } v_c > 0 \\
   \infty, & \text{if } v_c \le 0 \quad (\text{diverging or parallel})
   \end{cases}
   $$

5. **Closest Point of Approach ($t_{\text{cpa}}$) & Lateral Miss Distance ($d_{\min}$)**:
   $$
   t_{\text{cpa}} = \frac{\mathbf{d} \cdot \mathbf{v}_{\text{rel}}}{\|\mathbf{v}_{\text{rel}}\|^2 + 10^{-6}}, \quad d_{\min} = \|\mathbf{d} - \mathbf{v}_{\text{rel}} t_{\text{cpa}}\|
   $$

A collision alert triggers **if and only if**:
- $v_c > 0$ (actively closing trajectory),
- $\mathrm{TTC} < \tau_{\text{threshold}}$ (default: $2.5\,\text{s}$),
- $d_{\min} \le R_{\text{coll}} = \frac{(w_i + h_i + w_j + h_j)}{8}$ (lateral miss distance within physical bounding radius),
- dynamic vehicle speed $\ge 30\,\text{px/s}$, sustained over $k \ge 3$ consecutive frames.

This multi-condition gating completely eliminates false alarms from parked vehicles and parallel-lane traffic.

### 2.3 Trajectory Coordinate Jitter Variance

High-frequency spatial instability introduced by model quantization noise is quantified via discrete acceleration variance across object trajectories:

$$
\mathrm{Jitter} = \frac{1}{N - 2} \sum_{t=1}^{N - 2} \left\| \mathbf{c}_{t+1} - 2\mathbf{c}_t + \mathbf{c}_{t-1} \right\|^2
$$

where $\mathbf{c}_t = (x_t, y_t)$ represents the estimated bounding box center at frame $t$.

### 2.4 Resolution Time Margin (RTM)

The practical safety utility of collision anticipation is measured by how early the system provides a persistent warning before actual physical impact:

$$
\mathrm{RTM} = \frac{\max\left(0,\, F_{\text{impact}} - F_{\text{alert}}\right)}{\mathrm{FPS}_{\text{video}}}
$$

where $F_{\text{impact}}$ is the ground-truth collision frame, $F_{\text{alert}}$ is the first sustained alarm frame, and $\mathrm{FPS}_{\text{video}}$ is the sequence framerate (30 FPS).

---

## 3. Directory Structure

```
Comms/
├── analysis/
│   └── pareto_plotter.py       # High-DPI Pareto frontier and jitter trade-off plotter
├── benchmark/
│   ├── metrics.py              # Latency, mAP@50, Jitter variance, and RTM calculations
│   ├── runner.py               # Multi-sequence benchmarking engine across 6 configurations
│   └── verify_cgroup_simulation.py # Strict 4-core physical CPU affinity benchmark
├── data/
│   ├── dataset_loader.py       # Optimized sequential video decoder (20x seek speedup)
│   ├── generate_mock_data.py   # Synthetic collision test sequence generator
│   └── tudat/                  # TU-DAT real-world roadside anomaly dataset
├── docker/
│   ├── Dockerfile              # POSIX cgroup benchmarking container
│   ├── run_benchmark.sh        # Linux cgroup execution script (--cpus=4.0 --memory=4g)
│   └── run_benchmark.ps1       # Windows container runner
├── models/
│   ├── export_onnx.py          # Static ONNX graph exporter for YOLO26 (FP32 & FP16)
│   └── quantize_int8.py        # Static INT8 PTQ with FP32 detection head protection
├── reports/
│   ├── CS493_Final_Comprehensive_Report.pdf  # Compiled 11-page academic report with Q&A
│   ├── CS493_Final_Report.md   # Academic markdown report for course evaluation
│   └── generate_pdf_report.py  # Headless Chromium PDF compiler
├── results/
│   ├── benchmark_summary.csv   # Consolidated multi-video benchmark metrics
│   ├── cgroups_hardware_simulation.csv # Strict 4-core affinity hardware numbers
│   ├── pareto_throughput_vs_rtm.png # 300 DPI Pareto frontier chart
│   ├── jitter_vs_precision.png # 300 DPI coordinate jitter variance chart
│   └── latency_vs_throughput.png # 300 DPI latency vs. throughput trade-off chart
├── tests/
│   └── test_pipeline.py        # 9 pytest verification unit and kinematic tests
├── weights/                    # Pretrained PyTorch checkpoints and optimized ONNX graphs
├── requirements.txt            # Python dependencies
├── run_all.py                  # Master orchestration end-to-end pipeline
└── README.md                   # Project documentation & technical specifications
```

---

## 4. Evaluated Target Configurations

The benchmark evaluates six distinct YOLO26 operational configurations:

| # | Configuration Name | Model Variant | Numerical Precision | Storage Size | Target Engine |
|---|--------------------|---------------|---------------------|:------------:|---------------|
| 1 | `yolo26n_fp32`     | YOLO26-Nano   | IEEE 32-bit Float   | 9.47 MB      | ONNX Runtime CPU |
| 2 | `yolo26n_fp16`     | YOLO26-Nano   | IEEE 16-bit Float   | 4.75 MB      | ONNX Runtime CPU |
| 3 | `yolo26n_int8`     | YOLO26-Nano   | 8-bit Quantized QDQ | 3.29 MB      | ONNX Runtime CPU |
| 4 | `yolo26s_fp32`     | YOLO26-Small  | IEEE 32-bit Float   | 36.50 MB     | ONNX Runtime CPU |
| 5 | `yolo26s_fp16`     | YOLO26-Small  | IEEE 16-bit Float   | 18.30 MB     | ONNX Runtime CPU |
| 6 | `yolo26s_int8`     | YOLO26-Small  | 8-bit Quantized QDQ | 11.14 MB     | ONNX Runtime CPU |

---

## 5. Empirical Benchmark Results

### Table 1: Multi-Sequence Aggregation (10 Collision Sequences + 5 Control Clips)

| Configuration | Architecture | Precision | Throughput (FPS) | Latency (ms) | mAP@50 | Jitter ($\text{px}^2$) | RTM (s) | False Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`yolo26n_fp32`** | Nano | FP32 | **$48.34 \pm 1.25$** | $20.70 \pm 0.54$ | $0.0472 \pm 0.0626$ | $81.72 \pm 103.3$ | $0.91 \pm 1.06$ | **0.0%** |
| **`yolo26n_fp16`** | Nano | FP16 | **$43.04 \pm 6.16$** | $23.75 \pm 3.61$ | $0.0472 \pm 0.0625$ | $89.49 \pm 117.9$ | $0.97 \pm 1.02$ | **0.0%** |
| **`yolo26n_int8`** | Nano | INT8 | $23.01 \pm 1.48$ | $43.76 \pm 3.19$ | $0.0416 \pm 0.0478$ | $96.06 \pm 140.7$ | $0.94 \pm 1.03$ | **0.0%** |
| **`yolo26s_fp32`** | Small | FP32 | $18.22 \pm 0.18$ | $54.89 \pm 0.54$ | $0.0559 \pm 0.0602$ | $60.32 \pm 95.8$ | $1.35 \pm 1.29$ | **0.0%** |
| **`yolo26s_fp16`** | Small | FP16 | $17.77 \pm 0.59$ | $56.33 \pm 1.91$ | $0.0531 \pm 0.0558$ | $91.96 \pm 192.5$ | $1.36 \pm 1.31$ | **0.0%** |
| **`yolo26s_int8`** | Small | INT8 | $9.56 \pm 0.94$ | $105.85 \pm 12.49$ | $0.0533 \pm 0.0596$ | $34.15 \pm 38.3$ | $1.25 \pm 1.10$ | **0.0%** |

### Table 2: Edge Hardware Constraints (Strict 4-Core Physical CPU Affinity)

| Configuration | Precision | Pinned Cores | Throughput (FPS) | Mean Latency | Peak RAM | RAM $\le 4\text{GB}$ | Real-Time Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`yolo26n_fp16`** | FP16 | 4 Cores | **$32.05\,\text{FPS}$** | **$31.20\,\text{ms}$** | $716.9\,\text{MB}$ | Satisfied | **REAL-TIME COMPLIANT ($\ge 30\,\text{FPS}$)** |
| **`yolo26n_fp32`** | FP32 | 4 Cores | **$31.87\,\text{FPS}$** | **$31.38\,\text{ms}$** | $699.9\,\text{MB}$ | Satisfied | **REAL-TIME COMPLIANT ($\ge 30\,\text{FPS}$)** |
| **`yolo26n_int8`** | INT8 | 4 Cores | $17.02\,\text{FPS}$ | $58.76\,\text{ms}$ | $716.7\,\text{MB}$ | Satisfied | **BOTTLENECKED ($< 20\,\text{FPS}$)** |
| **`yolo26s_fp32`** | FP32 | 4 Cores | $10.64\,\text{FPS}$ | $94.02\,\text{ms}$ | $814.4\,\text{MB}$ | Satisfied | **BOTTLENECKED ($< 20\,\text{FPS}$)** |
| **`yolo26s_fp16`** | FP16 | 4 Cores | $10.09\,\text{FPS}$ | $99.11\,\text{ms}$ | $829.9\,\text{MB}$ | Satisfied | **BOTTLENECKED ($< 20\,\text{FPS}$)** |
| **`yolo26s_int8`** | INT8 | 4 Cores | $6.24\,\text{FPS}$ | $160.22\,\text{ms}$ | $817.2\,\text{MB}$ | Satisfied | **BOTTLENECKED ($< 20\,\text{FPS}$)** |

---

## 6. Key Scientific Findings (Gap A)

1. **YOLO26 Nano Real-Time Breakthrough:**
   On 4 CPU cores, `yolo26n_fp16` and `yolo26n_fp32` sustain **$31.87\text{--}32.05\,\text{FPS}$** ($31.2\text{--}31.4\,\text{ms}$), comfortably exceeding the $\ge 30\,\text{FPS}$ real-time threshold.
2. **Small Scaling Latency Collapse:**
   Scaling from Nano to Small incurs a **$3.0\times\text{--}5.1\times$ throughput penalty** on edge CPUs ($6.24\text{--}10.64\,\text{FPS}$, up to $160.22\,\text{ms/frame}$). At $60\,\text{km/h}$, uncompressed Small leaves a vehicle blind for **$2.67\,\text{meters}$ per frame**, making it unsafe for edge CPU deployment without dedicated NPUs.
3. **The INT8 CPU Quantization Paradox:**
   Without dedicated hardware vector dot-product pipelines (Intel AVX-512 VNNI or ARM NEON `i8mm`), ONNX Runtime must emulate QDQ integer nodes in software. This emulation penalty negates memory bandwidth savings, collapsing INT8 Nano throughput to **$17.02\,\text{FPS}$** (making it ~47% slower than FP16).
4. **Pareto-Optimal Recommendation:**
   **`yolo26n_fp16`** delivers the optimal balance of throughput ($32.05\,\text{FPS}$), pre-crash safety margin ($0.97\text{--}1.69\,\text{s}$), $0.0\%$ false alarm rate, and a 50% model footprint reduction ($4.75\,\text{MB}$).

---

## 7. Execution & Reproducibility Guide

### Option A: Local Python Environment
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute the master pipeline:
   ```bash
   python run_all.py
   ```
   *Sequentially generates mock data (if missing), executes the pytest suite, exports ONNX graphs, quantizes INT8 models, runs the 6-model benchmark loop, generates publication plots, and displays a summary table.*

3. Run verification tests:
   ```bash
   python -m pytest tests/test_pipeline.py -v
   ```

### Option B: Simulated Edge Hardware via Linux POSIX cgroups
Execute inside an isolated container enforcing strict quad-core CPU and 4GB memory limits:
```bash
bash docker/run_benchmark.sh
```

---

## 8. Academic Deliverables & Reports

* **Official Academic Report (PDF):** [`reports/CS493_Final_Comprehensive_Report.pdf`](reports/CS493_Final_Comprehensive_Report.pdf) *(11-page compiled report with embedded 300 DPI charts and 18-question defense Q&A)*
* **Academic Markdown Report:** [`reports/CS493_Final_Report.md`](reports/CS493_Final_Report.md)
* **Publication Figures (300 DPI):**
  - `results/pareto_throughput_vs_rtm.png` (Pareto Frontier: Throughput vs. RTM)
  - `results/jitter_vs_precision.png` (Coordinate Jitter Variance across Precisions)
  - `results/latency_vs_throughput.png` (Latency vs. Throughput Real-Time Boundaries)

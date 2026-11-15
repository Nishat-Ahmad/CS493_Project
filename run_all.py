"""Master orchestration pipeline for edge collision benchmark.

Sequentially executes:
  1. Dataset verification and mock data generation (if dataset missing).
  2. Pytest execution of unit and mathematical validation test suite.
  3. ONNX model export (FP32, FP16) and INT8 calibration quantization.
  4. Benchmark execution under simulated edge CPU constraints across 6 configurations:
     - yolo26n_fp32
     - yolo26n_fp16
     - yolo26n_int8
     - yolo26s_fp32
     - yolo26s_fp16
     - yolo26s_int8
  5. Pareto and jitter trade-off visualization plotting.
  6. Final formatted console summary table of all metrics.
"""

from __future__ import annotations

import sys
import os
import subprocess
from pathlib import Path
import pandas as pd

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.generate_mock_data import generate_mock_dataset
from models.export_onnx import run_exports
from models.quantize_int8 import run_int8_quantization
from benchmark.runner import EdgeBenchmarkRunner
from analysis.pareto_plotter import generate_all_plots


def print_step_banner(step_num: int, title: str) -> None:
    """Print an eye-catching step banner."""
    print(f"\n{'#'*80}")
    print(f"# STEP {step_num}: {title.upper()}")
    print(f"{'#'*80}\n")


def step_1_prepare_dataset(repo_root: Path) -> Path:
    """Check dataset existence or synthesize mock dataset."""
    print_step_banner(1, "Dataset Preparation")
    tudat_dir = repo_root / "data" / "tudat"
    seq_dir = repo_root / "data" / "synthetic_sequence"
    calib_dir = repo_root / "data" / "calibration"

    if (tudat_dir / "videos").exists() and (tudat_dir / "annotations").exists():
        print(f"[Step 1] Found TU-DAT real-world roadside anomaly dataset in {tudat_dir}.")
        return tudat_dir

    if (seq_dir / "annotations.json").exists() and calib_dir.exists() and len(list(calib_dir.glob("*.jpg"))) >= 30:
        print(f"[Step 1] Found existing sequence in {seq_dir} and calibration frames in {calib_dir}.")
        return seq_dir

    print("[Step 1] Dataset or calibration frames not found. Generating synthetic video sequence...")
    generate_mock_dataset(output_base_dir=repo_root / "data")
    return seq_dir


def step_2_run_tests(repo_root: Path) -> None:
    """Execute test suite using pytest."""
    print_step_banner(2, "Pytest Verification Test Suite")
    test_path = repo_root / "tests" / "test_pipeline.py"

    cmd = [sys.executable, "-m", "pytest", str(test_path), "-v"]
    print(f"[Step 2] Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(repo_root))
    if result.returncode != 0:
        raise RuntimeError(f"Pytest test suite failed with exit code {result.returncode}")
    print("\n[Step 2] All unit and kinematic tests passed successfully.")


def step_3_export_and_quantize(repo_root: Path) -> None:
    """Export YOLOv11 to ONNX (FP32, FP16) and calibrate INT8 models."""
    print_step_banner(3, "ONNX Export and INT8 Quantization")
    weights_dir = repo_root / "weights"
    calib_dir = repo_root / "data" / "calibration"

    print("[Step 3.1] Running FP32 and FP16 ONNX exports...")
    run_exports(output_dir=weights_dir)

    print("\n[Step 3.2] Running INT8 calibration and static quantization...")
    run_int8_quantization(weights_dir=weights_dir, calibration_dir=calib_dir)

    print(f"\n[Step 3] Model weights generated in {weights_dir}:")
    for f in sorted(weights_dir.glob("*.onnx")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name:<22} ({size_mb:6.2f} MB)")


def step_4_execute_benchmark(repo_root: Path) -> Path:
    """Run simulated edge CPU benchmark across 6 model configurations."""
    print_step_banner(4, "Simulated Edge Benchmark Execution")
    tudat_dir = repo_root / "data" / "tudat"
    dataset_path = tudat_dir if tudat_dir.exists() else (repo_root / "data" / "synthetic_sequence")

    runner = EdgeBenchmarkRunner(
        weights_dir=repo_root / "weights",
        dataset_path=dataset_path,
        results_dir=repo_root / "results",
        num_threads=4,
        warmup_frames=10
    )
    runner.run_all()
    return repo_root / "results" / "benchmark_summary.csv"


def step_5_generate_plots(summary_csv_path: Path) -> None:
    """Generate Pareto frontier and jitter analysis plots."""
    print_step_banner(5, "Pareto and Jitter Analysis Visualization")
    generate_all_plots(csv_path=summary_csv_path)


def step_6_display_summary_table(summary_csv_path: Path) -> None:
    """Render structured final console summary table."""
    print_step_banner(6, "Benchmark Summary and Findings")
    if not summary_csv_path.exists():
        print(f"[Step 6] Error: {summary_csv_path} does not exist.")
        return

    df = pd.read_csv(summary_csv_path)

    # Format table for terminal display
    display_cols = [
        "config",
        "variant",
        "precision",
        "fps",
        "std_fps",
        "mean_latency_ms",
        "std_latency_ms",
        "p95_latency_ms",
        "map50",
        "std_map50",
        "jitter_variance",
        "rtm_seconds",
        "std_rtm_seconds",
        "false_positive_rate"
    ]
    sub_df = df[[c for c in display_cols if c in df.columns]].copy()

    # Rename for cleaner table headers
    rename_map = {
        "config": "Configuration",
        "variant": "Variant",
        "precision": "Precision",
        "fps": "FPS",
        "std_fps": "± FPS",
        "mean_latency_ms": "Lat (ms)",
        "std_latency_ms": "± Lat",
        "p95_latency_ms": "p95 (ms)",
        "map50": "mAP@50",
        "std_map50": "± mAP",
        "jitter_variance": "Jitter (px²)",
        "rtm_seconds": "RTM (s)",
        "std_rtm_seconds": "± RTM",
        "false_positive_rate": "FPR (%)"
    }
    sub_df.rename(columns=rename_map, inplace=True)

    header_line = "=" * 120
    print(header_line)
    print("EDGE HARDWARE CONSTRAINTS: 4 CPU Cores | 4GB RAM | 640x640 Input | Real-Time Threshold: 20 FPS")
    print(header_line)
    print(sub_df.to_string(index=False, justify="center"))
    print(header_line)
    print("\nBenchmark outputs stored in 'results/':")
    print("  - results/benchmark_summary.csv")
    print("  - results/benchmark_summary.json")
    print("  - results/benchmark_sequences_detail.csv")
    print("  - results/pareto_throughput_vs_rtm.png")
    print("  - results/jitter_vs_precision.png\n")


def main() -> None:
    repo_root = REPO_ROOT
    print(f"Initiating Collision Anticipation Edge Benchmark in {repo_root}")

    # 1. Dataset
    step_1_prepare_dataset(repo_root)

    # 2. Pytest
    step_2_run_tests(repo_root)

    # 3. Model Export & INT8 Quantization
    step_3_export_and_quantize(repo_root)

    # 4. Benchmark Runner
    csv_path = step_4_execute_benchmark(repo_root)

    # 5. Visualization Plots
    step_5_generate_plots(csv_path)

    # 6. Final Summary Table
    step_6_display_summary_table(csv_path)


if __name__ == "__main__":
    main()


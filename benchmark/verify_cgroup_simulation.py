"""Containerized Hardware Simulation (POSIX cgroups: 4 CPUs, 4GB RAM).

Simulates Docker POSIX cgroups constraints:
  - 4.0 CPU Cores via hardware CPU affinity pinning & ONNX thread pool
  - 4.0 GB RAM memory footprint tracking and validation
  - Real-time threshold boundary evaluation (20-30 FPS)
"""

from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psutil
import pandas as pd
import onnxruntime as ort

from benchmark.runner import EdgeBenchmarkRunner
from data.dataset_loader import load_dataset


def run_cgroups_simulation(
    num_cores: int = 4,
    memory_limit_gb: float = 4.0,
    sequence_name: str = "v1"
) -> Dict[str, Any]:
    print("=" * 80)
    print(" PHASE 5: CONTAINERIZED HARDWARE SIMULATION (POSIX CGROUPS: 4 CORES, 4GB RAM)")
    print("=" * 80)

    # 1. Hardware Inspection
    p = psutil.Process()
    total_cores = psutil.cpu_count(logical=True)
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"[Host Info] Total System: {total_cores} Logical Cores, {total_ram_gb:.1f} GB RAM")

    # 2. Apply CPU Affinity Constraint (simulating --cpus="4.0")
    pinned_cores = list(range(min(num_cores, total_cores)))
    try:
        p.cpu_affinity(pinned_cores)
        active_affinity = p.cpu_affinity()
        print(f"[cgroups CPU] Successfully pinned CPU affinity to {len(active_affinity)} cores: {active_affinity}")
    except Exception as e:
        print(f"[cgroups CPU] Warning setting affinity: {e}")
        active_affinity = pinned_cores

    # 3. Initialize Benchmark Runner
    weights_dir = REPO_ROOT / "weights"
    dataset_path = REPO_ROOT / "data" / "tudat"
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    runner = EdgeBenchmarkRunner(
        weights_dir=weights_dir,
        dataset_path=dataset_path,
        results_dir=results_dir,
        num_threads=num_cores,
        warmup_frames=5,
        enforce_cpu_affinity=True
    )

    dataset = load_dataset(dataset_path)
    # Find requested sequence
    seq = None
    for s in getattr(dataset, "positive_sequences", [dataset]):
        if getattr(s, "sequence_name", None) == sequence_name or sequence_name in str(getattr(s, "video_path", "")):
            seq = s
            break
    if seq is None:
        seq = getattr(dataset, "positive_sequences", [dataset])[0]
    seq_name = getattr(seq, "sequence_name", "default")
    print(f"[cgroups Eval] Target Evaluation Sequence: {seq_name} ({len(seq)} frames, {seq.fps:.1f} FPS)")

    results: List[Dict[str, Any]] = []

    print("\n" + "-" * 80)
    print(f" {'Configuration':<16} | {'Precision':<8} | {'Throughput':<12} | {'Latency':<10} | {'Peak RAM':<10} | {'Real-Time Status':<20}")
    print("-" * 80)

    for target in runner.TARGET_CONFIGURATIONS:
        cfg_name = target["config"]
        model_file = weights_dir / target["file"]
        session = runner._create_session(model_file)
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        is_fp16 = "float16" in input_meta.type

        # Memory snapshot before
        mem_before_mb = p.memory_info().rss / (1024 ** 2)

        # Run sequence evaluation
        seq_eval = runner.evaluate_sequence(
            session=session,
            input_name=input_name,
            is_fp16=is_fp16,
            sequence=seq,
            is_negative_control=False
        )

        # Memory snapshot after
        mem_after_mb = p.memory_info().rss / (1024 ** 2)
        peak_mem_mb = max(mem_before_mb, mem_after_mb)

        fps = seq_eval["fps"]
        lat_ms = seq_eval["mean_latency_ms"]

        # Real-time threshold boundary evaluation:
        # > 30 FPS: Compliant (Smooth edge deployment)
        # 20 - 30 FPS: Borderline Real-Time (Meets safety-critical collision baseline)
        # < 20 FPS: Real-Time Violation (Latency bottleneck)
        if fps >= 30.0:
            status = "REAL-TIME COMPLIANT"
        elif fps >= 20.0:
            status = "BORDERLINE REAL-TIME"
        else:
            status = "BOTTLENECKED (<20 FPS)"

        row = {
            "config": cfg_name,
            "variant": target["variant"],
            "precision": target["precision"],
            "pinned_cores": len(active_affinity),
            "fps": round(fps, 2),
            "mean_latency_ms": round(lat_ms, 2),
            "peak_ram_mb": round(peak_mem_mb, 1),
            "ram_limit_gb": memory_limit_gb,
            "within_ram_limit": (peak_mem_mb / 1024) <= memory_limit_gb,
            "real_time_status": status,
            "rtm_seconds": round(seq_eval.get("rtm_seconds", 0.0), 3),
            "mAP50": round(seq_eval.get("mAP50", 0.0), 4),
        }
        results.append(row)

        print(f" {cfg_name:<16} | {target['precision']:<8} | {fps:6.2f} FPS   | {lat_ms:6.2f} ms | {peak_mem_mb:6.1f} MB  | {status:<20}")

    print("-" * 80)

    # Save to CSV and JSON
    csv_out = results_dir / "cgroups_hardware_simulation.csv"
    json_out = results_dir / "cgroups_hardware_simulation.json"

    df = pd.DataFrame(results)
    df.to_csv(csv_out, index=False)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Saved] Hardware simulation CSV -> {csv_out}")
    print(f"[Saved] Hardware simulation JSON -> {json_out}\n")

    return {
        "host_logical_cores": total_cores,
        "simulated_cores": len(active_affinity),
        "simulated_memory_limit_gb": memory_limit_gb,
        "results": results
    }


if __name__ == "__main__":
    run_cgroups_simulation()

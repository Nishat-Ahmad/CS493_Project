"""Pareto optimality and jitter visualization analysis module.

Generates publication-quality figures:
  1. results/pareto_throughput_vs_rtm.png:
     - X-axis: Throughput (FPS)
     - Y-axis: Resolution Time Margin (RTM, seconds)
     - Point size: Scaled by mAP@50
     - Vertical dashed red reference line: 20 FPS real-time threshold
     - Highlights Pareto-optimal boundary configurations.
  2. results/jitter_vs_precision.png:
     - Grouped bar chart comparing coordinate jitter variance across FP32, FP16, INT8
       for YOLOv11-Nano vs. YOLOv11-Small.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def compute_pareto_frontier(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Identify indices of Pareto-optimal points maximizing both x (FPS) and y (RTM).

    Args:
        xs: 1D array of metric 1 (e.g., Throughput FPS).
        ys: 1D array of metric 2 (e.g., RTM seconds).

    Returns:
        pareto_indices: Boolean mask where True indicates a Pareto-optimal point.
    """
    n_points = len(xs)
    is_pareto = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                # Point j dominates point i if j is strictly better in one and >= in the other
                if (xs[j] >= xs[i] and ys[j] >= ys[i]) and (xs[j] > xs[i] or ys[j] > ys[i]):
                    is_pareto[i] = False
                    break

    return is_pareto


def plot_pareto_throughput_vs_rtm(
    df: pd.DataFrame,
    output_path: Path | str = "results/pareto_throughput_vs_rtm.png",
    real_time_fps_threshold: float = 20.0
) -> None:
    """Generate Pareto throughput vs. RTM scatter plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter successful rows
    valid_df = df[df["fps"] > 0].copy()
    if valid_df.empty:
        print("[Plotter] No valid benchmark rows with FPS > 0. Skipping Pareto plot.")
        return

    fps = valid_df["fps"].to_numpy()
    rtm = valid_df["rtm_seconds"].to_numpy()
    map50 = valid_df["map50"].to_numpy()

    # Identify Pareto frontier
    pareto_mask = compute_pareto_frontier(fps, rtm)

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    # Color mapping for model variants
    color_map = {
        "FP32": "#1f77b4",  # Blue
        "FP16": "#ff7f0e",  # Orange
        "INT8": "#2ca02c",  # Green
    }
    marker_map = {
        "yolo11n": "o",  # Circle for Nano
        "yolo11s": "s",  # Square for Small
    }

    # Plot non-Pareto points
    for idx, row in valid_df.iterrows():
        is_front = pareto_mask[valid_df.index.get_loc(idx)]
        prec = row["precision"]
        variant = row["variant"]
        c = color_map.get(prec, "#7f7f7f")
        m = marker_map.get(variant, "o")
        size = max(80.0, float(row["map50"]) * 400.0)

        edge_color = "red" if is_front else "black"
        edge_width = 2.5 if is_front else 0.8
        alpha = 0.95 if is_front else 0.70

        ax.scatter(
            row["fps"],
            row["rtm_seconds"],
            s=size,
            c=c,
            marker=m,
            edgecolors=edge_color,
            linewidths=edge_width,
            alpha=alpha,
            zorder=4 if is_front else 3
        )

        # Annotate config label
        offset_y = 0.04 if is_front else -0.05
        ax.annotate(
            f"{row['config']}\n(mAP={row['map50']:.2f})",
            (row["fps"], row["rtm_seconds"]),
            textcoords="offset points",
            xytext=(0, 10 if is_front else -18),
            ha="center",
            fontsize=8.5,
            fontweight="bold" if is_front else "normal",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8, lw=0.5)
        )

    # Connect Pareto frontier with dashed line
    frontier_indices = np.where(pareto_mask)[0]
    if len(frontier_indices) > 0:
        frontier_points = sorted(
            [(fps[i], rtm[i]) for i in frontier_indices],
            key=lambda p: p[0]
        )
        fx, fy = zip(*frontier_points)
        ax.plot(
            fx, fy,
            color="#d62728",
            linestyle="-.",
            linewidth=1.8,
            label="Pareto Optimal Frontier",
            zorder=2
        )

    # Real-time threshold reference line (20 FPS)
    ax.axvline(
        x=real_time_fps_threshold,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label=f"Real-Time Constraint ({real_time_fps_threshold:.0f} FPS)",
        zorder=1
    )

    # Formatting
    ax.set_title(
        "Pareto Trade-offs: Inference Throughput vs. Resolution Time Margin (RTM)\n"
        "(Point size scaled by mAP@50; Edge Hardware: 4 Cores, 4GB RAM)",
        fontsize=12,
        fontweight="bold",
        pad=12
    )
    ax.set_xlabel("Throughput (Frames Per Second, FPS)", fontsize=11, labelpad=8)
    ax.set_ylabel("Resolution Time Margin (RTM, Seconds Before Impact)", fontsize=11, labelpad=8)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Custom legend
    legend_elements = [
        plt.Line2D([0], [0], color="#d62728", linestyle="--", lw=2, label="20 FPS Real-time Line"),
        plt.Line2D([0], [0], color="#d62728", linestyle="-.", lw=1.8, label="Pareto Frontier"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555", markersize=9, label="Nano (yolo11n)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#555555", markersize=9, label="Small (yolo11s)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["FP32"], markersize=9, label="FP32"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["FP16"], markersize=9, label="FP16"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["INT8"], markersize=9, label="INT8"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plotter] Saved Pareto chart to {output_path}")


def plot_jitter_vs_precision(
    df: pd.DataFrame,
    output_path: Path | str = "results/jitter_vs_precision.png"
) -> None:
    """Generate grouped bar chart comparing spatial coordinate jitter variance."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    precisions = ["FP32", "FP16", "INT8"]
    variants = ["yolo11n", "yolo11s"]

    nano_jitters: List[float] = []
    small_jitters: List[float] = []

    for p in precisions:
        # Nano
        row_n = df[(df["variant"] == "yolo11n") & (df["precision"] == p)]
        nano_jitters.append(float(row_n["jitter_variance"].values[0]) if not row_n.empty else 0.0)

        # Small
        row_s = df[(df["variant"] == "yolo11s") & (df["precision"] == p)]
        small_jitters.append(float(row_s["jitter_variance"].values[0]) if not row_s.empty else 0.0)

    x = np.arange(len(precisions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    rects1 = ax.bar(x - width / 2.0, nano_jitters, width, label="YOLOv11-Nano", color="#3b82f6", edgecolor="black", alpha=0.85)
    rects2 = ax.bar(x + width / 2.0, small_jitters, width, label="YOLOv11-Small", color="#ef4444", edgecolor="black", alpha=0.85)

    ax.set_title(
        "Trajectory Coordinate Jitter Variance Across Numerical Precisions\n"
        r"$\mathrm{Jitter} = \frac{1}{N-2} \sum \|c_{t+1} - 2 c_t + c_{t-1}\|^2$",
        fontsize=12,
        fontweight="bold",
        pad=12
    )
    ax.set_xlabel("Numerical Precision Format", fontsize=11, labelpad=8)
    ax.set_ylabel("Bounding Box Center Jitter Variance (px²)", fontsize=11, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(precisions, fontsize=10, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    # Bar labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold"
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plotter] Saved Jitter chart to {output_path}")


def generate_all_plots(csv_path: Optional[Path | str] = None) -> None:
    """Read CSV summary and generate all analysis plots."""
    if csv_path is None:
        csv_file = REPO_ROOT / "results" / "benchmark_summary.csv"
    else:
        csv_file = Path(csv_path)

    if not csv_file.exists():
        raise FileNotFoundError(f"Benchmark summary file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    results_dir = csv_file.parent

    plot_pareto_throughput_vs_rtm(
        df=df,
        output_path=results_dir / "pareto_throughput_vs_rtm.png",
        real_time_fps_threshold=20.0
    )

    plot_jitter_vs_precision(
        df=df,
        output_path=results_dir / "jitter_vs_precision.png"
    )


if __name__ == "__main__":
    generate_all_plots()

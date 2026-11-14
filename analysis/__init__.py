"""Pareto and statistical plotting analysis subpackage."""

from analysis.pareto_plotter import (
    plot_pareto_throughput_vs_rtm,
    plot_jitter_vs_precision,
    compute_pareto_frontier,
    generate_all_plots,
)

__all__ = [
    "plot_pareto_throughput_vs_rtm",
    "plot_jitter_vs_precision",
    "compute_pareto_frontier",
    "generate_all_plots",
]


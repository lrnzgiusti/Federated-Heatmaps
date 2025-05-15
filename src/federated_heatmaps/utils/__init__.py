"""Utility functions for data simulation, heatmap reconstruction, and evaluation.

This module provides support for:
- Simulating user-location datasets (uniform, clustered, multi-location).
- Computing ground-truth heatmaps based on a prefix tree structure.
- Reconstructing dense grid heatmaps from hierarchical histograms.
- Evaluating estimates using MSE and L1 distance metrics.
"""

from .utils import (
    calculate_l1_dist,
    calculate_mse,
    compute_true_heatmap,
    generate_simulated_user_data,
    reconstruct_flat_heatmap_from_tree
)

__all__ = [
    "calculate_l1_dist",
    "calculate_mse",
    "compute_true_heatmap",
    "generate_simulated_user_data",
    "reconstruct_flat_heatmap_from_tree"
]

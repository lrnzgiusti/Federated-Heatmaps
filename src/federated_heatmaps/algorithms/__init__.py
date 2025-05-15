"""Federated histogram algorithms for differentially private heatmaps.

Includes:
- Algorithm 1: Histogram with secure aggregation and client-side noise.
- Algorithm 2: Adaptive hierarchical histogram using prefix trees.
"""

from .algorithm1 import Histogram
from .algorithm2 import AdaptiveHist

__all__ = [
    "AdaptiveHist",
    "Histogram",
]

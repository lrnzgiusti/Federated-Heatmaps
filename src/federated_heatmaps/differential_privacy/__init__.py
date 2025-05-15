"""Differential privacy utilities for the secure aggregation protocol.

This module aggregates noise generation, privacy budget calibration, and
target noise standard deviation estimation functions used in federated
histogram algorithms under differential privacy.
"""

from .differential_privacy import (
    gamma_adjust_eps,
    phi_eps_from_std,
    sigma_tilde_target_std,
    stdgeo_tau_noise_std,
)
from .noise import (
    add_noise_vectorized,
    modulo_clip,
    sample_polya,
)

__all__ = [
    # Core DP computations
    "stdgeo_tau_noise_std",
    "sigma_tilde_target_std",
    "phi_eps_from_std",
    "gamma_adjust_eps",

    # Client-side noise generation
    "sample_polya",
    "add_noise_vectorized",
    "modulo_clip",
]

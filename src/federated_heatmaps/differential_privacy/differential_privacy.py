"""Differential privacy utilities for the secure aggregation protocol."""

import math

import numpy as np

from federated_heatmaps.config import Config
from federated_heatmaps.quad_tree import PrefixTree


def stdgeo_tau_noise_std(epsilon: float, 
                         sensitivity_delta: float) -> float:
    r"""Compute standard deviation of noise to be added to the histogram.

    Implements Equation (1) from the paper:
        \sigma_{\text{rem}} = \tau(\epsilon, \Delta)

    Args:
        epsilon (float): Privacy budget for the current sub-query.
        sensitivity_delta (float): L1-sensitivity \Delta of the query.

    Returns
    -------
        Standard deviation \sigma_{\text{rem}} for the noise to be added.
    """
    if epsilon == 0:
        return float("inf")

    beta = np.exp(-epsilon / sensitivity_delta)
    if beta == 1.0:
        return float("inf")

    return np.sqrt(2 * beta / (1 - beta) ** 2)


def sigma_tilde_target_std(
    cfg: Config,
    tree: PrefixTree,
    u: int,
    s_max: int,
) -> float:
    r"""Compute target standard deviation for current sub-query (Eq. 2).

    Implements Equation (2) from the paper:
        \tilde{\sigma} = c \cdot \frac{u}{t} \cdot \sqrt{\lceil u / s_{\text{max}} \rceil}

    Args
    ------
        cfg (Config): Configuration object.
        tree (PrefixTree): Current prefix tree.
        u (int): Number of users in the current sub-query.
        s_max (int): Maximum number of users per shard.

    Returns
    -------
        Target \tilde{\sigma} to calibrate the DP noise.
    
    Raises
    ------
        ValueError: If s_max <= 0.
    """
    if s_max <= 0:
        msg = f"Maximum number of users per shard (s_max) must be > 0, got {s_max}"
        raise ValueError(msg)
   
    t = len(tree.get_reporting_nodes_ordered())
    if t == 0 or u == 0:
        return float("inf")

    k = math.ceil(u / s_max)
    return cfg.algorithm.c_alg2 * (u / t) * math.sqrt(k)


def phi_eps_from_std(sigma: float, delta: float = 1.0) -> float:
    r"""Convert noise standard deviation to equivalent privacy budget (Eq. 3).

    Implements Equation (3) from the paper:
        \tilde{\epsilon} = -\Delta \cdot \log \beta,
    where
        \beta = \frac{(\sigma^2 + 1) - \sqrt{2\sigma^2 + 1}}{\sigma^2}

    Args
    ------
        sigma (float): Noise standard deviation.
        delta (float): Privacy budget for the current sub-query.
            Default is 1.0.

    Returns
    -------
        Equivalent \tilde{\epsilon}. Returns:
            - 0.0 if \sigma = \infty (no privacy cost),
            - \infty if \sigma \to 0 (infinite privacy cost),
            - Finite positive value otherwise.
    """
    if math.isinf(sigma):
        return 0.0
    if sigma <= 0.0:
        return float("inf")

    s2 = sigma * sigma
    num = (s2 + 1.0) - math.sqrt(2.0 * s2 + 1.0)

    if num <= 0.0:
        return float("inf")

    beta = num / s2
    beta = min(max(beta, 1e-15), 1.0 - 1e-15)

    return -delta * math.log(beta)


def gamma_adjust_eps(eps_remaining: float, eps_tilde: float, b_factor: float) -> float:
    r"""Adjust query budget based on remaining global budget (Eq. 4).

    Implements Equation (4) from the paper:
        \epsilon_q = \min(\tilde{\epsilon}, b \cdot \epsilon_{\text{rem}})

    Args
    ------
        eps_remaining (float): Remaining global privacy budget.
        eps_tilde (float): Privacy budget for the current sub-query.
        b_factor (float): Budget factor for the current sub-query.
            Default is 1.0.
            This is the maximum fraction of the remaining budget
            that can be used for this sub-query.
            If b_factor = 1.0, the sub-query can use the entire remaining budget.
            If b_factor = 0.5, the sub-query can use half of the remaining budget.
            If b_factor <= 0.0, the sub-query cannot use any of the remaining budget.

    Returns
    -------
        \epsilon_q, adjusted budget for this sub-query.
    """
    return eps_tilde if b_factor * eps_tilde <= eps_remaining else eps_remaining

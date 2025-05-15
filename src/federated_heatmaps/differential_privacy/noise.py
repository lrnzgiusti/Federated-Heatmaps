"""Client-side noise generation for differential privacy: Pólya and discrete-Laplace."""


import numpy as np
from numpy.typing import NDArray
from numpy.random import default_rng
from functools import lru_cache

generator = default_rng()

@lru_cache(maxsize=256)
def _get_scale(beta: float) -> float:
    """Retrieve the Gamma scale = beta/(1-beta), cached for speed."""
    return beta / (1.0 - beta)

def sample_polya(alpha_param: float, beta_param: float) -> int:
    """Sample from Pólya(alpha, beta), equivalent to a NegativeBinomial.

    This is a mixture of Poisson distributions with Gamma mixing
    distribution.
    Args
    ------
        alpha_param (float): Shape parameter for Gamma.
        beta_param (float): Rate control (used in scale for Gamma).

    Returns
    -------
        int: Sampled integer from the Pólya distribution with parameters
            alpha_param and beta_param.
    """
    # Clamp parameters to valid range
    alpha = max(alpha_param, 1e-9)
    beta  = min(max(beta_param, 1e-7), 1 - 1e-7)
    scale = _get_scale(beta)

    # Draw mixing rate and Poisson sample
    lam = generator.gamma(alpha, scale=scale)
    return int(generator.poisson(lam))


@lru_cache(maxsize=256)
def add_noise_vectorized(
    vector_dim: int,
    alpha_param: float,
    beta_param: float
) -> NDArray[np.int_]:
    """Generate a noise vector for a given dimension using symmetric Poisson-Gamma mechanism.

    Args
    ------
        vector_dim (int): Dimension of the noise vector.
        alpha_param (float): Shape parameter for Gamma distribution.
        beta_param (float): Rate control (used in scale for Gamma).

    Returns
    -------
        NDArray[np.int_]: Vector of noise samples.
    """
    # Precompute scale for Gamma
    scale = _get_scale(beta_param)

    # 1) Draw gamma mixture parameters
    lam = generator.gamma(alpha_param, scale=scale, size=vector_dim)

    # 2) Two independent Poisson draws
    noise1 = generator.poisson(lam, size=vector_dim)
    noise2 = generator.poisson(lam, size=vector_dim)

    # 3) In-place difference for minimal overhead
    noise1 -= noise2
    return noise1



def modulo_clip(noisy_vector: NDArray[np.int_], m_precision: int) -> NDArray[np.int_]:
    """
    Apply element-wise modulo operation to ensure values lie in [0, m_precision).

    Args:
        noisy_vector: Vector of integers (may be negative).
        m_precision: Modulus value for clipping.

    Returns
    -------
        Vector with elements modulo m_precision.
    """
    return np.mod(noisy_vector, m_precision, out=noisy_vector)

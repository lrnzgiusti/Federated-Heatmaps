import numpy as np
import pytest

from federated_heatmaps.differential_privacy import add_noise_vectorized, modulo_clip, sample_polya


def test_sample_polya_valid():
    """Test Polya sampling with valid parameters."""
    sample = sample_polya(alpha_param=5.0, beta_param=0.3)
    assert isinstance(sample, int)
    assert sample >= 0


@pytest.mark.parametrize("alpha,beta", [
    (0.0, 0.3),      # invalid alpha
    (5.0, 0.0),      # invalid beta low
    (5.0, 1.0),      # invalid beta high
    (-1.0, 0.5),     # invalid alpha
    (2.0, -0.2),     # invalid beta
])
def test_sample_polya_clamps_invalid(alpha, beta):
    """Ensure Polya handles invalid parameters gracefully."""
    sample = sample_polya(alpha, beta)
    assert isinstance(sample, int)
    assert sample >= 0


def test_add_noise_vectorized_basic():
    """Test that noise vector returns integer array of correct size."""
    dim = 10
    noise = add_noise_vectorized(vector_dim=dim, alpha_param=5.0, beta_param=0.5)
    assert isinstance(noise, np.ndarray)
    assert noise.shape == (dim,)
    assert np.issubdtype(noise.dtype, np.integer)


def test_add_noise_vectorized_symmetric():
    """Test that the distribution is approximately symmetric over large samples."""
    vec = add_noise_vectorized(vector_dim=10000, alpha_param=2.0, beta_param=0.5)
    mean = np.mean(vec)
    # Expect roughly zero mean, allow small deviation
    assert abs(mean) < 0.2


def test_modulo_clip_behavior():
    """Test that modulo_clip returns values in [0, m)."""
    raw = np.array([-5, -1, 0, 1, 5, 10, 100])
    m = 7
    result = modulo_clip(raw, m)
    assert isinstance(result, np.ndarray)
    assert all(0 <= val < m for val in result)


def test_modulo_clip_identity_when_all_positive():
    """Test modulo_clip is identity for values already in range."""
    vec = np.array([0, 1, 2, 3])
    clipped = modulo_clip(vec, m_precision=4)
    assert np.array_equal(clipped, vec)

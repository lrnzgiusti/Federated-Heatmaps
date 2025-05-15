"""Unit tests for differential privacy utilities in federated heatmaps."""

import math

import pytest

from federated_heatmaps.config import Config
from federated_heatmaps.differential_privacy import (
    gamma_adjust_eps,
    phi_eps_from_std,
    sigma_tilde_target_std,
    stdgeo_tau_noise_std,
)
from federated_heatmaps.quad_tree import PrefixTree, TreeNode


@pytest.fixture
def config() -> Config:
    """Provides a default Config instance for tests."""
    return Config()


def test_stdgeo_tau_noise_std_regular_case() -> None:
    """Ensure tau noise standard deviation behaves for standard inputs."""
    sigma = stdgeo_tau_noise_std(epsilon=1.0, sensitivity_delta=1.0)
    assert sigma > 0.0
    assert math.isfinite(sigma)


def test_stdgeo_tau_noise_std_edge_cases() -> None:
    """Validate edge handling of epsilon=0 and extreme sensitivity deltas."""
    assert stdgeo_tau_noise_std(0.0, 1.0) == float("inf")

    # Large sensitivity_delta ⇒ sigma should diverge
    val = stdgeo_tau_noise_std(1.0, 1e6)
    assert val > 1e6, f"Expected large noise, got {val}"

    # Tiny sensitivity ⇒ small noise
    val_small = stdgeo_tau_noise_std(1.0, 1e-6)
    assert 0.0 <= val_small < 1.0, f"Expected small noise, got {val_small}"



def test_phi_eps_from_std_behavior() -> None:
    """Check phi(tilde{eps}) handles extremes and mid-range correctly."""
    assert phi_eps_from_std(float("inf")) == 0.0
    assert phi_eps_from_std(0.0) == float("inf")

    eps = phi_eps_from_std(1.0)
    assert eps > 0.0
    assert math.isfinite(eps)


def test_phi_eps_from_std_monotonicity() -> None:
    """Check that higher noise (σ) yields lower ε̃."""
    sigma_small, sigma_large = 0.1, 1.0
    eps_small = phi_eps_from_std(sigma_small)
    eps_large = phi_eps_from_std(sigma_large)
    assert eps_small > eps_large


def test_gamma_adjust_eps_cases() -> None:
    """Verify correct epsilon adjustment logic (Eq. 4)."""
    # tilde{eps} < b * ε_remaining → use tilde{eps}
    assert gamma_adjust_eps(1.0, 0.5, 1.5) == 0.5
    # tilde{eps} > b * ε_remaining → clamp to ε_remaining
    assert gamma_adjust_eps(0.4, 0.5, 1.5) == 0.4


def test_sigma_tilde_target_std_regular_case(config: Config) -> None:
    r"""Test \tilde{sigma} computation with non-trivial tree (2 reporting nodes)."""
    tree = PrefixTree(config)
    tree.root.children = {
        "00": TreeNode("00", 1, (0, 0, 128, 128), parent=tree.root),
        "01": TreeNode("01", 1, (128, 0, 256, 128), parent=tree.root),
    }
    tree.nodes.update(tree.root.children)

    sigma = sigma_tilde_target_std(config, tree, u=1000, s_max=200)
    assert sigma > 0.0
    assert math.isfinite(sigma)


def test_sigma_tilde_target_std_empty_tree(config: Config) -> None:
    r"""When the tree has only the root, \tilde{sigma} should still be finite (T=1)."""
    tree = PrefixTree(config)
    sigma = sigma_tilde_target_std(config, tree, u=1000, s_max=100)

    # Root is a reporting node → T=1
    assert sigma > 0.0
    assert math.isfinite(sigma)


def test_sigma_tilde_target_std_zero_users(config: Config) -> None:
    r"""If no users are sampled, \tilde{sigma} → ∞."""
    tree = PrefixTree(config)
    tree.split_node("")  # ensure T > 1

    sigma = sigma_tilde_target_std(config, tree, u=0, s_max=100)
    assert sigma == float("inf")

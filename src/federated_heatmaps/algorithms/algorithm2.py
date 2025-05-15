"""Adaptive hierarchical histogram via iterative tree refinement (Algorithm 2).

This module implements the adaptive histogram algorithm, which refines a quadtree
structure based on user data and differential privacy constraints.
It includes methods for initializing the tree, updating it based on noisy counts,
and running the adaptive histogram algorithm.
"""
from __future__ import annotations

import numpy as np

from federated_heatmaps.algorithms.algorithm1 import Histogram
from federated_heatmaps.config import Config
from federated_heatmaps.differential_privacy import (
    gamma_adjust_eps,
    phi_eps_from_std,
    sigma_tilde_target_std,
)
from federated_heatmaps.quad_tree import PrefixTree


class AdaptiveHist:
    """Adaptive hierarchical histogram via iterative tree refinement (Algorithm 2)."""

    def __init__(
        self,
        config: Config,
        all_user_data: list,
    ) -> None:
        """
        Initialize with global config and full user dataset.

        Args
        ------
            config (Config): Configuration object with parameters.
            all_user_data (list): List of all user data locations.
        """
        self.config = config
        self.all_user_data = all_user_data
        self.alg1 = Histogram(config)
        self.total_comm_cost = 0

    def _init_tree(self) -> PrefixTree:
        """Create a fresh prefix/quadtree rooted to the grid dims."""
        return PrefixTree(self.config)

    def _update_tree(
        self,
        tree: PrefixTree,
        counts: np.ndarray,
        idx_map: dict[int, str],
        threshold: float,
    ) -> PrefixTree:
        """
        Split or collapse nodes based on noisy counts.

        Args
        ------
            tree (PrefixTree): The prefix tree structure.
            counts (np.ndarray): Noisy counts for each node.
            idx_map (dict[int, str]): Mapping from indices to node IDs.
            threshold (float): Threshold for splitting/collapsing nodes.
        """
        if self.config.verbose:
            print(f"UpdateTree threshold={threshold:.2f}")

        for idx, node_id in idx_map.items():
            node = tree.get_node(node_id)
            if node is None:
                continue

            c = counts[idx] if idx < counts.size else 0.0
            if c > threshold and node.depth < self.config.tree.max_depth:
                if self.config.verbose:
                    print(
                        f"Splitting {node.id}: count={c:.2f} > threshold={threshold:.2f}"
                    )
                tree.split_node(node.id)
            elif c <= threshold / 4 and node.parent is not None:
                if self.config.verbose:
                    print(
                        f"Collapsing {node.id}: count={c:.2f} <= threshold/4={threshold/4:.2f}"
                    )
                tree.collapse_node(node.id)
        return tree

    def run(
        self,
        *,
        sensitivity_delta: float = Config.DEFAULT_SENSITIVITY,
        multi: bool = False,
    ) -> tuple[np.ndarray, dict[int, str], PrefixTree, int]:
        """
        Execute adaptive hierarchical histogram.

        Args
        -----
            sensitivity_delta (float): L1-sensitivity of encoding.
            multi (bool): Whether multi-location extension is enabled.
        
        Returns
        -------
                tuple[np.ndarray, dict[int, str], PrefixTree, int]: Tuple containing:
                    - Densities over reporting nodes.
                    - Mapping from indices to node IDs.
                    - Updated prefix tree.
                    - Total communication cost.
        """
        eps_rem = self.config.privacy.eps_total
        tree = self._init_tree()
        self.total_comm_cost = 0
        hist_vec: np.ndarray = np.array([])
        hist_map: dict[int, str] = {}

        if self.config.verbose:
            print(f"Starting Alg2 with eps_total={eps_rem:.2f}")

        iteration = 0
        while eps_rem > 1e-6:
            iteration += 1
            if self.config.verbose:
                print(f"Iteration {iteration}, eps_rem={eps_rem:.4f}")

            nodes = tree.get_reporting_nodes_ordered()
            t = len(nodes)
            self.total_comm_cost += t
            if t == 0:
                break

            sigma_tilde = sigma_tilde_target_std(
                self.config, tree, self.config.algorithm.U_alg1, self.config.secagg.S_max
            )
            eps_tilde = (
                0.0 if np.isinf(sigma_tilde)
                else float("inf") if sigma_tilde == 0
                else phi_eps_from_std(sigma_tilde, sensitivity_delta)
            )

            eps_q = gamma_adjust_eps(eps_rem, eps_tilde, self.config.algorithm.b_alg2)
            if eps_q <= 1e-6 and not np.isinf(eps_q):
                eps_q = eps_rem
            eps_rem = max(eps_rem - eps_q, 0.0)

            hist_vec, hist_map = self.alg1.run_histogram_query(
                eps_q,
                self.all_user_data,
                self.config.algorithm.U_alg1,
                tree,
                sensitivity_delta,
                multi=multi,
            )

            if hist_vec.size == 0:
                break

            threshold = float("inf") if np.isinf(sigma_tilde) else 2.0 * sigma_tilde
            tree = self._update_tree(tree, hist_vec, hist_map, threshold)

            # terminate if budget fully used
            if eps_rem <= 1e-12:
                break

        if self.config.verbose:
            print(f"Finished Alg2 after {iteration} iters, eps_rem={eps_rem:.4f}")

        return hist_vec, hist_map, tree, self.total_comm_cost

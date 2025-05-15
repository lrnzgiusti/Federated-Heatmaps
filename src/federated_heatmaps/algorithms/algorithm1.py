"""Algorithm 1: Differentially-private histogram via secure aggregation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
from numpy.random import default_rng

from federated_heatmaps.config import DEFAULT_SENSITIVITY, Config
from federated_heatmaps.differential_privacy import (
    add_noise_vectorized,
    modulo_clip,
    phi_eps_from_std,
    stdgeo_tau_noise_std,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from federated_heatmaps.quad_tree import PrefixTree

_rng = default_rng()
Location = tuple[int, int]
UserData =  list[Location]


class Histogram:
    """Algorithm 1: Differentially-private histogram via secure aggregation."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def _client_update(
        self,
        user_data: UserData,
        alpha: float,
        beta: float,
        tree: PrefixTree,
        modulus: int,
        delta: float,
        *,
        multi: bool = False,
    ) -> np.ndarray | None:
        """
        Encode client data, add discrete noise, and apply modulo clipping.

        Args:
            user_data (list[tuple[int, int]]): single Location or list of Locations.
            alpha (float): Pólya shape parameter.
            beta (float): noise mixing parameter.
            tree (PrefixTree): current prefix tree.
            modulus (int): secure aggregation modulus.
            delta (float): L1-sensitivity of encoding.
            multi (bool): whether multi-location extension is enabled.

        Returns
        -------
            Integer vector of length T or None if no reporting nodes.
        """
        reporting = tree.get_reporting_nodes_ordered()
        if not reporting:
            return None
        dim = len(reporting)

        # 1. Build raw frequency vector
        if not multi:
            vec = np.zeros(dim, dtype=np.int32)
            idx = tree.map_location_to_reporting_node_idx(user_data, reporting)
            if idx is not None:
                vec[idx] = 1

        else:
            # map each loc → an index (Python loop is tiny relative to rest)
            idxs = [
                tree.map_location_to_reporting_node_idx(loc, reporting)
                for loc in user_data
            ]
            # drop Nones and build counts in C
            idxs = [i for i in idxs if i is not None]
            counts = np.bincount(idxs, minlength=dim).astype(np.float64)

            total = counts.sum()
            if total > 0:
                freqs = counts / total
                scaled = freqs * self.config.multi_loc.gamma_scaling

                # floor + batched stochastic rounding
                floored = np.floor(scaled).astype(np.int32)
                frac_part = scaled - floored
                # one vectorized call to random()
                rnd = _rng.random(dim)
                vec = floored + (rnd < frac_part).astype(np.int32)
            else:
                vec = np.zeros(dim, dtype=np.int32)

        # 2. Add independent noise to each component
        noise = add_noise_vectorized(dim, alpha, beta)
        noisy = vec + noise

        # 3. Modulo clip
        return modulo_clip(noisy, modulus)

    def _aggregate_shard(
        self,
        shard_data: Iterable[UserData],
        alpha: float,
        beta: float,
        tree: PrefixTree,
        modulus: int,
        delta: float,
        *,
        multi: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """
        Blazing-fast secure-aggregation over a shard of clients.

        Fully vectorizes dropout and accumulation with NumPy, using a
        single C-level sum and avoiding Python loops for accumulation.
        
        Args
        ------
            shard_data (Iterable[UserData]): Shard of user data.
            alpha (float): Pólya shape parameter.
            beta (float): noise mixing parameter.
            tree (PrefixTree): current prefix tree.
            modulus (int): secure aggregation modulus.
            delta (float): L1-sensitivity of encoding.
            multi (bool): whether multi-location extension is enabled.
        
        Returns
        -------
            tuple[np.ndarray, int]: Tuple containing:
                - Integer vector of length T.
                - Count of non-zero vectors.
        """
        # Precompute constants
        reporting = tree.get_reporting_nodes_ordered()
        dim = len(reporting)
        half = modulus >> 1
        delta_drop = self.config.privacy.delta_drop

        # Realize shard and vectorize dropout
        users = tuple(shard_data)
        mask = _rng.random(len(users)) >= delta_drop

        # Gather client vectors in a single comprehension
        vecs = [
            vec for use_flag, user in zip(mask, users) if use_flag
            for vec in (
                self._client_update(user, alpha, beta, tree, modulus, delta, multi=multi),
            ) if vec is not None
        ]
        count = len(vecs)
        if count == 0:
            return np.zeros(dim, dtype=int), 0

        # Stack and sum in C
        arr = np.vstack(vecs).astype(np.int64)
        total = arr.sum(axis=0)

        # Center-lift via vectorized modular arithmetic
        modded = ((total + half) % modulus) - half
        return modded, count

    def run_histogram_query(
        self,
        eps: float,
        users: list[UserData],
        u: int,
        tree: PrefixTree,
        delta: float = DEFAULT_SENSITIVITY,
        *,
        multi: bool = False,
    ) -> tuple[np.ndarray, dict[int, str]]:
        """
        Execute Alg1: partition users into shards, add noise, aggregate, and return histogram.

        Args
        -----
            eps (float): privacy budget.
            users (list[UserData]): list of user data.
            u (int): number of users to sample.
            tree (PrefixTree): current prefix tree.
            delta (float): L1-sensitivity of encoding.
            multi (bool): whether multi-location extension is enabled.
        

        Returns
        -------
            tuple[np.ndarray, dict[int, str]]: Tuple containing:
                - Integer vector of length T.
                - Mapping from indices to node IDs.
        """
        reps = tree.get_reporting_nodes_ordered()
        t = len(reps)
        if eps <= 0:
            return np.zeros(t, int), {}

        # compute per-shard epsilon via discrete Gaussian calibration
        shards = int(np.ceil(u / self.config.secagg.S_max)) or 1
        sigma = stdgeo_tau_noise_std(eps, delta)
        eps_shard = (
            0.0
            if np.isinf(sigma)
            else phi_eps_from_std(sigma / np.sqrt(shards), delta)
        )

        # derive Pólya params
        s_eff = max((1 - self.config.privacy.delta_drop) * self.config.secagg.S_max, 1)
        if eps_shard <= 0 or np.isinf(eps_shard):
            alpha, beta = 1.0, 1e-9
        else:
            alpha = 1.0 / s_eff
            beta = max(min(np.exp(-eps_shard / delta), 1 - 1e-9), 1e-9)

        # sample users
        sel = (
            _rng.choice(len(users), size=u, replace=False)
            if u <= len(users)
            else list(range(len(users)))
        )
        data = [users[i] for i in sel]

        # aggregate across shards
        agg: np.ndarray | None = None
        for i in range(shards):
            chunk = data[i * self.config.secagg.S_max : (i + 1) * self.config.secagg.S_max]
            hist, _ = self._aggregate_shard(
                chunk, alpha, beta, tree, self.config.secagg.m, delta, multi=multi
            )
            agg = hist if agg is None else (agg + hist) % self.config.secagg.m

        hist_vec = agg if agg is not None else np.zeros(t, int)
        mapping = {i: node.id for i, node in enumerate(reps)}
        if multi and self.config.multi_loc.gamma_scaling > 0:
            hist_vec = hist_vec.astype(float) / self.config.multi_loc.gamma_scaling
        return hist_vec, mapping

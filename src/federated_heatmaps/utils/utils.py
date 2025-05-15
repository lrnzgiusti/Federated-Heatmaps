""""Utility functions for simulation, heatmap computation, reconstruction, and evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.random import default_rng

# Only import heavy types for type checking
if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from federated_heatmaps.quad_tree import PrefixTree

# Initialize a single random number generator for reproducible simulations
_rng = default_rng()

Location = tuple[int, int]
UserData = Location | list[Location]


def _sim_uniform(num_users: int, width: int, height: int) -> list[Location]:
    """
    Sample each user uniformly over the grid.

    Returns a list of (x,y) tuples.

    Args
    -----
        num_users (int): Number of users to simulate.
        width, height (int, int): Dimensions of the grid.
    
    Returns
    -------
        List of user locations as (x,y) tuples.
    """
    return [
        (int(_rng.integers(width)), int(_rng.integers(height)))
        for _ in range(num_users)
    ]


def _sim_clustered(
    num_users: int,
    width: int,
    height: int,
    num_clusters: int = 7,
) -> list[Location]:
    """
    Sample users from Gaussian clusters.

    Chooses cluster centers, then draws from a Normal distribution.
    
    Args
    -----
        num_users (int): Number of users to simulate.
        width, height (int, int): Dimensions of the grid.
        num_clusters (int): Number of clusters to sample from.
    
    Returns
    -------
        List of user locations as (x,y) tuples.
    """
    centers: list[tuple[float, float]] = [
        (width * 0.25, height * 0.25),
        (width * 0.75, height * 0.75),
    ][:num_clusters]
    std = width * 0.1

    samples: list[Location] = []
    for _ in range(num_users):
        cx, cy = _rng.choice(centers)
        x = int(_rng.normal(cx, std))
        y = int(_rng.normal(cy, std))
        # clamp to grid
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        samples.append((x, y))
    return samples


def _sim_multi_uniform(
    num_users: int,
    width: int,
    height: int,
    max_locs_per_user: int = 5,
) -> list[list[Location]]:
    """Each user reports 1-max_locs_per_user uniformly sampled locations.
    
    Args
    -----
        num_users (int): Number of users to simulate.
        width, height (int, int): Dimensions of the grid.
        max_locs_per_user (int): Maximum number of locations per user.
        
    Returns
    -------
        List of user data entries, each containing a list of locations.
    """
    # Each user reports 1-max_locs_per_user uniformly sampled locations
    all_data: list[list[Location]] = []
    for _ in range(num_users):
        k = int(_rng.integers(1, max_locs_per_user + 1))
        visits = [
            (int(_rng.integers(width)), int(_rng.integers(height)))
            for _ in range(k)
        ]
        all_data.append(visits)
    return all_data


def generate_simulated_user_data(
    num_users: int,
    width: int,
    height: int,
    distribution_type: str = "uniform",
) -> list[UserData]:
    """
    Generate simulated user-location data.

    Args
    -----
        num_users (int): Number of users to simulate.
        width, height (int, int): Dimensions of the grid.
        distribution_type: One of 'uniform', 'clustered', 'multi_uniform'.

    Returns
    -------
        List of user data entries, either single Location or list of Locations.
    """
    dispatch = {
        "uniform": _sim_uniform,
        "clustered": _sim_clustered,
        "multi_uniform": _sim_multi_uniform,
    }
    generator = dispatch.get(distribution_type)
    if generator is None:
        msg = f"Unknown distribution_type: {distribution_type!r}"
        raise ValueError(msg)

    return generator(num_users, width, height)


def compute_true_heatmap(
    user_data: Sequence[Location] | Sequence[list[Location]],
    tree: PrefixTree,
    *,
    is_multi_location: bool = False,
) -> tuple[NDArray[np.float64], dict[int, str]]:
    """
    Compute ground-truth densities over the tree's reporting nodes.

    Returns a density array summing to 1 and index→node-id map.
    
    Args
    -----
        user_data (Sequence[Location] | Sequence[list[Location]]): User data, either single or multiple locations.
        tree (PrefixTree): The prefix tree structure.
        is_multi_location (bool): Flag indicating if user_data contains multiple locations per user.
    
    Returns
    -------
        tuple[NDArray[np.float64], dict[int, str]]: A tuple containing:
            - Densities over reporting nodes.
            - Mapping from indices to node IDs.
    """
    reporting_nodes = tree.get_reporting_nodes_ordered()
    t = len(reporting_nodes)
    if t == 0:
        return np.zeros(0, dtype=float), {}

    idx_to_node: dict[int, str] = {i: node.id for i, node in enumerate(reporting_nodes)}
    counts = np.zeros(t, dtype=float)

    def _add(loc: Location, weight: float = 1.0) -> None:
        idx = tree.map_location_to_reporting_node_idx(loc, reporting_nodes)
        if idx is not None:
            counts[idx] += weight

    if not is_multi_location:
        for loc in user_data:  # type: ignore[assignment]
            _add(loc)
    else:
        for visits in user_data:  # type: ignore[assignment]
            for loc in visits:
                _add(loc)

    total = counts.sum()
    densities = counts / total if total > 0 else counts
    return densities, idx_to_node


def reconstruct_flat_heatmap_from_tree(
    hist_vector: NDArray[np.integer],
    hist_map: dict[int, str],
    tree: PrefixTree,
    grid_dims: tuple[int, int],
) -> NDArray[np.float64]:
    """
    Reconstruct full-resolution heatmap from a tree histogram.

    Skips stale node-ids not present in the tree.

    1. Build a *density table* for nodes that exist in ``tree``.
    2. Compute each node's *effective area* (cell count minus covered children).
    3. Traverse nodes **shallowest → deepest** and *slice-assign* the constant
       value ``density/area`` into ``flat_map``.  Deeper nodes overwrite their
       ancestral regions, so overlapping is handled implicitly.

    Complexity: **O(T)** slice assignments + bookkeeping, where *T* is the
    number of reporting nodes (≪ width x height).
    Args
    -----
        hist_vector (NDArray[np.integer]): Histogram vector.
        hist_map (dict[int, str]): Mapping from histogram indices to node IDs.
        tree (PrefixTree): The prefix tree structure.
        grid_dims (tuple[int, int]): Dimensions of the output grid.
    
    Returns
    -------
        NDArray[np.float64]: A 2D array representing the reconstructed heatmap.
    """
    width, height = grid_dims
    flat_map = np.zeros((height, width), dtype=float)
    if hist_vector is None or hist_vector.size == 0:
        return flat_map

    # Build density lookup for valid nodes
    total = hist_vector.sum()
    if total == 0:
        return flat_map

    density_by_node = {
        node_id: float(hist_vector[idx]) / total
        for idx, node_id in hist_map.items()
        if 0 <= idx < hist_vector.size and tree.get_node(node_id) is not None
    }

    if not density_by_node:
        return flat_map

    #  Compute effective (unique) area per node once
    effective_area: dict[str, int] = {}
    for nid, dens in density_by_node.items():
        node = tree.get_node(nid)
        xmin, ymin, xmax, ymax = map(int, node.bounds)
        total_cells = (xmax - xmin) * (ymax - ymin)
        child_area = sum(
            (int(c.bounds[2] - c.bounds[0]) * int(c.bounds[3] - c.bounds[1]))
            for c in node.children.values()
            if c.id in density_by_node
        )
        effective_area[nid] = max(total_cells - child_area, 1)

    # Slice‑assign per node (shallow→deep so children overwrite parents)
    for nid in sorted(density_by_node, key=len):
        node = tree.get_node(nid)
        if node is None:
            continue
        xmin, ymin, xmax, ymax = map(int, node.bounds)
        area = effective_area[nid]
        flat_map[ymin:ymax, xmin:xmax] = density_by_node[nid] / area

    return flat_map


def calculate_mse(true_flat_map: NDArray[np.float64], est_flat_map: NDArray[np.float64]) -> float:
    """Compute mean-squared error between two heatmaps of equal shape.
    
    Args
    -----
        true_flat_map (NDArray[np.float64]): The ground truth heatmap.
        est_flat_map (NDArray[np.float64]): The estimated heatmap.
        
    Returns
    -------
        float: The mean-squared error between the two heatmaps.
    """
    if true_flat_map.shape != est_flat_map.shape:
        msg = "Heatmaps must have the same dimensions for MSE."
        raise ValueError(msg)
    return float(np.mean((true_flat_map - est_flat_map) ** 2))


def calculate_l1_dist(true_flat_map: NDArray[np.float64], est_flat_map: NDArray[np.float64]) -> float:
    """Compute L1 distance between two heatmaps of equal shape.
    
     Args
    -----
        true_flat_map (NDArray[np.float64]): The ground truth heatmap.
        est_flat_map (NDArray[np.float64]): The estimated heatmap.
        
    Returns
    -------
        float: The mean-absolute error between the two heatmaps.
    """
    if true_flat_map.shape != est_flat_map.shape:
        msg = "Heatmaps must have the same dimensions for L1."
        raise ValueError(msg)
    return float(np.sum(np.abs(true_flat_map - est_flat_map)))

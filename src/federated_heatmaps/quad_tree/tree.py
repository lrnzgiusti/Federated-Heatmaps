"""Prefix Tree (QuadTree) implementation for spatial data partitioning."""

from collections import deque
from typing import Optional, List

from federated_heatmaps.config import Config


class TreeNode:
    """Node in a quadtree representing a spatial partition."""
    __slots__ = ("id", "depth", "bounds", "parent", "children")
    def __init__(self, node_id: str, depth: int, bounds: tuple[float, float, float, float], parent: Optional["TreeNode"] = None) -> None:
        """
        Initialize a TreeNode with an ID, depth, bounds, and optional parent.

        Args
        -----
        node_id (str): Unique identifier for the node.
        depth (int): Depth of the node in the tree.
        bounds (tuple): Spatial bounds of the node in the form (xmin, ymin, xmax, ymax).
        parent (TreeNode, optional): Parent node. Defaults to None.

        The children are stored in a dictionary mapping quadrant suffixes ("00" to "11") to TreeNode objects.
        """
        # String like "00", "0011", "" for root
        self.id = node_id
        self.depth = depth
        # (xmin, ymin, xmax, ymax)
        self.bounds = bounds
        self.parent = parent
        # Maps quadrant suffix ("00" to "11") to TreeNode
        self.children = {}
        # self.count_debug = 0 # For debugging counts from Algorithm 1's hist

    def __repr__(self):
        return f"TreeNode(id='{self.id}', depth={self.depth}, children={len(self.children)})"

    def get_child_bounds(self, quadrant_suffix: str) -> tuple[float, float, float, float]:
        """Calculate bounds for a child in a given quadrant."""
        xmin, ymin, xmax, ymax = self.bounds
        mid_x = (xmin + xmax) / 2
        mid_y = (ymin + ymax) / 2
        if quadrant_suffix == "00":  # NW
            return (xmin, ymin, mid_x, mid_y)
        if quadrant_suffix == "01":  # NE
            return (mid_x, ymin, xmax, mid_y)
        if quadrant_suffix == "10":  # SW
            return (xmin, mid_y, mid_x, ymax)
        if quadrant_suffix == "11":  # SE
            return (mid_x, mid_y, xmax, ymax)
        msg = "Invalid quadrant suffix"
        raise ValueError(msg)

class PrefixTree:
    """Prefix Tree (QuadTree) for spatial data partitioning."""

    def __init__(self, config: Config):
        """
        Initialize the Prefix Tree with a root node.

        The root node has an empty ID and a depth of 0.
        The bounds of the root node are set to the grid dimensions.
        """
        # Initialize the tree with a root node
        self.config = config
        self.root = TreeNode("", 0, (0, 0, config.tree.grid_width, config.tree.grid_height))
        # Stores all nodes by ID for quick access
        self.nodes = {"": self.root}
        self._mutation_counter: int = 0
        self._cached_reporting: List[TreeNode] = []
        self._cached_reporting_version: int = -1
        self._reporting_index_map: dict[str, int] = {}

    def _mark_mutated(self) -> None:
        """Increment the mutation counter whenever the tree structure changes."""
        self._mutation_counter += 1

    def get_node(self, node_id: str) -> TreeNode | None:
        """Retrieve a node by its ID."""
        return self.nodes.get(node_id)

    def _add_node(self, node_id: str, depth: int, bounds: tuple[float, float, float, float], parent_node: TreeNode | None) -> TreeNode:
        """Add a node to the tree, creating it if it doesn't exist.

        Args
        -----
        node_id (str): Unique identifier for the node.
        depth (int): Depth of the node in the tree.
        bounds (tuple): Spatial bounds of the node in the form (xmin, ymin, xmax, ymax).
        parent_node (TreeNode, optional): Parent node. Defaults to None.

        Returns
        -------
        TreeNode: The created or existing node.
        """
        # Check if the node already exists
        if node_id in self.nodes:
            return self.nodes[node_id]
        new_node = TreeNode(node_id, depth, bounds, parent_node)
        self.nodes[node_id] = new_node
        if parent_node:
            suffix = node_id[len(parent_node.id):]
            parent_node.children[suffix] = new_node
        self._mark_mutated()
        return new_node

    def split_node(self, node_id_to_split: str) -> bool:
        """Ensures a node has all 4 children created in the tree.
        
        Args
        -----
            node_id_to_split (str): ID of the node to split.

        Returns
        -------
                bool: True if any child was created, 
                    False if the node was already at max depth or no children were created.
        """
        node = self.get_node(node_id_to_split)
        if not node or node.depth >= self.config.tree.max_depth:
            return False
        created = False
        for suffix in ("00", "01", "10", "11"):
            child_id = node.id + suffix
            if child_id not in node.children:
                child_bounds = node.get_child_bounds(suffix)
                self._add_node(child_id, node.depth + 1, child_bounds, node)
                created = True
        if created:
            self._mark_mutated()
        return created


    def collapse_node(self, node_id_to_collapse: str) -> bool:
        """Deletes a node. Its parent will then have fewer children.
        
        Args
        -----
            node_id_to_collapse (str): ID of the node to collapse.
        
        Returns
        -------
            bool: True if the node was successfully collapsed, 
                  False if it was the root or not found.

        Raises
        ------
            ValueError: If the node to collapse is the root or not found.
        """
        node = self.get_node(node_id_to_collapse)
        if not node or not node.parent:
            return False
        parent = node.parent
        suffix = node.id[len(parent.id):]
        to_delete = deque([node])
        while to_delete:
            curr = to_delete.popleft()
            for child in curr.children.values():
                to_delete.append(child)
            self.nodes.pop(curr.id, None)
        parent.children.pop(suffix, None)
        self._mark_mutated()
        return True


    def get_reporting_nodes_ordered(self) -> list[TreeNode]:
        """Returns a list of nodes that can accumulate counts, ordered by ID.
        
        Reporting nodes are those with fewer than 4 children.
        This is based on the assumption that nodes with fewer than 4 children can accumulate counts.
        The root node is always included as a reporting node.

        Args
        -----
            None

        Returns
        -------
            list[TreeNode]: List of reporting nodes, ordered by ID.
        """
        # Recompute only if tree has changed since last cache
        if self._cached_reporting_version != self._mutation_counter:
            # Rebuild reporting list
            reporting = [n for n in self.nodes.values() if len(n.children) < 4]
            reporting.sort(key=lambda n: n.id)
            # Update cache and index map
            self._cached_reporting = reporting
            self._cached_reporting_version = self._mutation_counter
            self._reporting_index_map = {n.id: i for i, n in enumerate(reporting)}
        return self._cached_reporting

    def location_to_grid_cell_id(self, location_coords: tuple[int, int], depth: int) -> str:
        """Converts (x,y) location to a node ID string at a specific depth.

        The prefix uses *bit-interleaving* (x-bit first, y-bit second) of the
        **top** ``depth`` bits.  This matches the previous string-based method
        but is ~2x faster on CPython and far less alloc-heavy.

        Args
        -----
        location_coords (tuple): Coordinates of the location to convert.
        depth (int): Depth of the node in the tree.

        Returns
        -------
        str: Node ID string representing the location at the specified depth.

        Raises
        ------
        ValueError: If the location coordinates are out of bounds.
        """
        x, y = location_coords
        w, h = self.config.tree.grid_width, self.config.tree.grid_height
        max_depth = self.config.tree.max_depth

        if not (0 <= x < w and 0 <= y < h):  # bounds check
            raise ValueError(f"Location {location_coords} out of bounds.")
        if depth == 0:
            return ""
        if depth > max_depth:
            raise ValueError(f"Depth {depth} exceeds max_depth {max_depth}")

        # Interleave: take the *top* ``depth`` bits from MSB to LSB.
        id_int = 0
        for d in range(depth):
            shift = max_depth - 1 - d  # MSB first
            id_int = (id_int << 2) | (((x >> shift) & 1) << 1) | ((y >> shift) & 1)

        # Zero‑pad to 2*depth bits → string
        return format(id_int, f"0{2 * depth}b")

    def map_location_to_reporting_node_idx(self, location_coords: tuple[int, int],
                                           reporting_nodes_list: list[TreeNode]) -> int | None:
        """Finds the index of the reporting node for a location.

        Location is mapped to the longest prefix matching reporting node.

        Args
        -----
        location_coords (tuple): Coordinates of the location to map.
        reporting_nodes_list (list): List of reporting nodes. #maybeobsolete

        Returns
        -------
            int: Index of the reporting node in the list.
                 If the location does not map to any reporting node, returns None.
        """
        # Find the deepest node in the current tree containing the location
        # Default to root if no deeper match
        # Ensure reporting and index map are fresh
        # Refresh cache if needed
        self.get_reporting_nodes_ordered()
        # Walk from deepest level down to root
        for d in range(self.config.tree.max_depth, -1, -1):
            node_id = self.location_to_grid_cell_id(location_coords, d)
            idx = self._reporting_index_map.get(node_id)
            if idx is not None:
                return idx
        return None


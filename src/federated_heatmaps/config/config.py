"""Configuration module for Privacy Heatmaps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import yaml

DEFAULT_SENSITIVITY = 1.0


@dataclass(frozen=True)
class PrivacyConfig:
    """Privacy-budget parameters.
    
    Attributes
    ----------
        eps_total: float
            Total privacy budget for the entire run.
        delta_drop: float
            Dropout rate for the privacy budget.
    
    Raises
    ------
        ValueError: If eps_total is negative or if delta_drop is not in [0, 1).
    """

    eps_total: float = 1.0
    delta_drop: float = 0.05

    def __post_init__(self) -> None:
        """Validate values after initialization."""
        if self.eps_total < 0:
            msg = f"eps_total must be >= 0, got {self.eps_total}"
            raise ValueError(msg)
        if not (0.0 <= self.delta_drop < 1.0):
            msg = f"delta_drop must be in [0,1), got {self.delta_drop}"
            raise ValueError(msg)


@dataclass(frozen=True)
class SecAggConfig:
    """Secure-Aggregation parameters.
    
    Attributes
    ----------
        S_max: int
            Maximum number of users per shard.
        m: int
            Modulus for secure aggregation.
    Raises
    ------
        ValueError: If S_max or m is not positive.
    """

    S_max: int = 10_000
    m: int = 2**16

    def __post_init__(self) -> None:
        """Validate values after initialization."""
        if self.S_max <= 0:
            msg = f"S_max must be > 0, got {self.S_max}"
            raise ValueError(msg)
        if self.m <= 0:
            msg = f"m (modulus) must be > 0, got {self.m}"
            raise ValueError(msg)


@dataclass(frozen=True)
class AlgorithmConfig:
    """Algorithm-specific hyperparameters for Alg2.
    
    Attributes
    ----------
        U_alg1: int
            Number of users to sample for Alg1.
        c_alg2: float
            Constant for Alg2.
        b_alg2: float
            Constant for Alg2.

    Raises
    ------
        ValueError: If U_alg1, c_alg2, or b_alg2 is not positive.
    """

    U_alg1: int = 10_000
    c_alg2: float = 0.1
    b_alg2: float = 2.0

    def __post_init__(self) -> None:
        """Validate values after initialization."""
        if self.U_alg1 <= 0:
            msg = f"U_alg1 must be > 0, got {self.U_alg1}"
            raise ValueError(msg)
        if self.c_alg2 <= 0:
            msg = f"c_alg2 must be > 0, got {self.c_alg2}"
            raise ValueError(msg)
        if self.b_alg2 <= 0:
            msg = f"b_alg2 must be > 0, got {self.b_alg2}"
            raise ValueError(msg)


@dataclass(frozen=True)
class TreeConfig:
    """Tree and grid spatial parameters.
    
    Attributes
    ----------
        max_depth: int
            Maximum depth of the tree.
        grid_width: int
            Width of the grid.
        grid_height: int
            Height of the grid.
    Raises
    ------
        ValueError: If max_depth is negative or if grid dimensions are not positive.
    """

    max_depth: int = 10
    grid_width: int = 256
    grid_height: int = 256

    def __post_init__(self) -> None:
        """Validate values after initialization."""
        if self.max_depth < 0:
            msg = f"max_depth must be >= 0, got {self.max_depth}"
            raise ValueError(msg)
        if self.grid_width <= 0 or self.grid_height <= 0:
            msg = f"grid dimensions must be > 0, got ({self.grid_width}, {self.grid_height})"
            raise ValueError(msg)


@dataclass(frozen=True)
class MultiLocConfig:
    """Multi-location extension parameters.
    
    Attributes
    ----------
        gamma_scaling: float
            Scaling factor for gamma distribution.
    
    Raises
    ------
        ValueError: If gamma_scaling is not positive.
    """

    gamma_scaling: float = 100.0

    def __post_init__(self) -> None:
        """Validate values after initialization."""
        if self.gamma_scaling <= 0:
            msg = "gamma_scaling must be > 0"
            raise ValueError(msg)


@dataclass(frozen=True)
class Config:
    """
    Top-level configuration for Privacy Heatmaps.

    Groups
    ----------
        privacy: PrivacyConfig
            Parameters for differential privacy.
        secagg: SecAggConfig
            Parameters for secure aggregation.
        algorithm: AlgorithmConfig
            Hyperparameters for the algorithm.
        tree: TreeConfig
            Spatial parameters for the tree and grid.
        multi_loc: MultiLocConfig
            Parameters for multi-location extension.
        verbose: bool
            Flag to enable verbose output.
    
    Raises
    ------
        ValueError: If any of the sub-configs contain invalid values.
    """

    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    secagg: SecAggConfig = field(default_factory=SecAggConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    tree: TreeConfig = field(default_factory=TreeConfig)
    multi_loc: MultiLocConfig = field(default_factory=MultiLocConfig)
    verbose: bool = False

    DEFAULT_SENSITIVITY: ClassVar[float] = DEFAULT_SENSITIVITY

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert to plain dict (for logging, serialization)."""
        return asdict(self)

    def to_yaml(self) -> str:
        """Dump entire config as a YAML string."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Build Config by unpacking each sub-dict into its sub-config."""
        return cls(
            privacy=PrivacyConfig(**data.get("privacy", {})),
            secagg=SecAggConfig(**data.get("secagg", {})),
            algorithm=AlgorithmConfig(**data.get("algorithm", {})),
            tree=TreeConfig(**data.get("tree", {})),
            multi_loc=MultiLocConfig(**data.get("multi_loc", {})),
            verbose=data.get("verbose", False),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a YAML file and return a Config."""
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

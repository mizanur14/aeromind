"""Typed experiment configuration loaded from YAML.

Unknown keys raise :class:`ConfigError` so that typos never silently fall back to defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from aeromind.data.cmapss import check_dataset_name

ROLLING_STATS = ("mean", "std", "slope", "min", "max")
MODEL_NAMES = (
    "median",
    "mean_residual_life",
    "ridge",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
)


class ConfigError(ValueError):
    """Invalid configuration."""


def _build(cls: type, payload: dict[str, Any] | None, where: str) -> Any:
    payload = payload or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(payload).__name__}")
    names = {f.name for f in fields(cls)}
    unknown = set(payload) - names
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}; allowed {sorted(names)}")
    return cls(**payload)


@dataclass
class PathsConfig:
    """Filesystem locations (relative to the project root unless absolute)."""

    raw_dir: str = "data/raw/cmapss"
    processed_dir: str = "data/processed"
    manifest_dir: str = "data/manifests"
    artifacts_dir: str = "artifacts/runs"
    reports_dir: str = "reports"


@dataclass
class SplitConfig:
    """Engine-level split settings."""

    seed: int = 42
    fractions: list[float] = field(default_factory=lambda: [0.70, 0.15, 0.15])
    manifest: str | None = None  # default: <manifest_dir>/<dataset>_split_seed<seed>.json


@dataclass
class TargetConfig:
    """RUL target formulation. ``cap: null`` means uncapped true RUL."""

    cap: int | None = 125

    def __post_init__(self) -> None:
        if self.cap is not None and self.cap <= 0:
            raise ConfigError("target.cap must be positive or null")


@dataclass
class FeatureConfig:
    """Feature pipeline settings (see :mod:`aeromind.features.pipeline`)."""

    sensors: str | list[str] = "all"
    include_settings: bool = False
    include_cycle: bool = True
    windows: list[int] = field(default_factory=lambda: [10, 30])
    stats: list[str] = field(default_factory=lambda: list(ROLLING_STATS))
    include_diff: bool = False
    drop_constant: bool = True
    constant_tol: float = 1e-8
    scaler: str = "standard"
    condition_normalization: bool = False
    n_conditions: int = 1

    def __post_init__(self) -> None:
        if any(int(w) < 2 for w in self.windows):
            raise ConfigError("features.windows must all be >= 2")
        if len(set(self.windows)) != len(self.windows):
            raise ConfigError("features.windows must be unique")
        bad = set(self.stats) - set(ROLLING_STATS)
        if bad:
            raise ConfigError(f"features.stats: unknown {sorted(bad)}; allowed {ROLLING_STATS}")
        if self.scaler not in ("standard", "none"):
            raise ConfigError("features.scaler must be 'standard' or 'none'")
        if self.n_conditions < 1:
            raise ConfigError("features.n_conditions must be >= 1")
        if isinstance(self.sensors, str) and self.sensors != "all":
            raise ConfigError("features.sensors must be 'all' or a list of sensor names")


@dataclass
class EvaluationConfig:
    """Validation / test evaluation protocol."""

    snapshots_per_engine: int = 10
    snapshot_seed: int = 42
    capped_truth_cap: int = 125
    near_eol_threshold: int = 30
    rul_bins: list[int] = field(default_factory=lambda: [0, 25, 50, 75, 100, 125, 150, 200, 400, 1000])
    selection_metric: str = "snapshot_rmse_target"
    latency_repeats: int = 50

    def __post_init__(self) -> None:
        allowed = {"snapshot_rmse_target", "snapshot_rmse_true", "snapshot_rmse_capped"}
        if self.selection_metric not in allowed:
            raise ConfigError(f"evaluation.selection_metric must be one of {sorted(allowed)}")
        if self.snapshots_per_engine < 1:
            raise ConfigError("evaluation.snapshots_per_engine must be >= 1")


@dataclass
class TrainingConfig:
    """Training options."""

    sample_weight: str = "none"  # "none" | "engine_balanced"
    n_jobs: int = 4

    def __post_init__(self) -> None:
        if self.sample_weight not in ("none", "engine_balanced"):
            raise ConfigError("training.sample_weight must be 'none' or 'engine_balanced'")


@dataclass
class ModelSpec:
    """One model family: fixed ``params`` plus an optional hyperparameter ``grid``."""

    params: dict[str, Any] = field(default_factory=dict)
    grid: dict[str, list[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, values in self.grid.items():
            if not isinstance(values, list) or not values:
                raise ConfigError(f"grid entry {key!r} must be a non-empty list")


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str
    dataset: str = "FD001"
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    models: dict[str, ModelSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_dataset_name(self.dataset)
        bad = set(self.models) - set(MODEL_NAMES)
        if bad:
            raise ConfigError(f"models: unknown {sorted(bad)}; allowed {MODEL_NAMES}")
        if not self.models:
            raise ConfigError("models: at least one model is required")

    @property
    def split_manifest_path(self) -> Path:
        """Resolved split manifest path."""
        if self.split.manifest:
            return Path(self.split.manifest)
        return Path(self.paths.manifest_dir) / f"{self.dataset.lower()}_split_seed{self.split.seed}.json"

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict representation (for saving the resolved config)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentConfig:
        """Build and validate from a dict."""
        if not isinstance(payload, dict) or "name" not in payload:
            raise ConfigError("Config must be a mapping with a 'name' key")
        sections = {
            "paths": PathsConfig,
            "split": SplitConfig,
            "target": TargetConfig,
            "features": FeatureConfig,
            "evaluation": EvaluationConfig,
            "training": TrainingConfig,
        }
        top = {f.name for f in fields(cls)}
        unknown = set(payload) - top
        if unknown:
            raise ConfigError(f"Unknown top-level keys {sorted(unknown)}")
        kwargs: dict[str, Any] = {k: v for k, v in payload.items() if k not in sections and k != "models"}
        for key, section_cls in sections.items():
            kwargs[key] = _build(section_cls, payload.get(key), key)
        models_raw = payload.get("models") or {}
        kwargs["models"] = {name: _build(ModelSpec, spec, f"models.{name}") for name, spec in models_raw.items()}
        return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fh:
        payload = yaml.safe_load(fh)
    return ExperimentConfig.from_dict(payload)


def dump_config(config: ExperimentConfig, path: str | Path) -> Path:
    """Write the resolved config as YAML."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = config.to_dict() if is_dataclass(config) else dict(config)
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return out

"""Stage-2 experiment configuration and engine-level fold construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from aeromind.config import EvaluationConfig, FeatureConfig, PathsConfig, _build
from aeromind.data.cmapss import check_dataset_name
from aeromind.data.splits import SplitManifest

CLASSICAL_FAMILIES = ("median", "mean_residual_life", "ridge", "random_forest", "hist_gradient_boosting", "xgboost")


@dataclass
class ModelEntry:
    """One model in a Stage-2 experiment (classical or deep).

    ``target_cap`` overrides the experiment-wide cap ("default" = inherit; ``None`` = uncapped).
    """

    kind: str
    family: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    deep: dict[str, Any] = field(default_factory=dict)
    target_cap: int | None | str = "default"

    def __post_init__(self) -> None:
        if self.kind not in ("classical", "deep"):
            raise ValueError("model kind must be 'classical' or 'deep'")
        if self.kind == "classical" and self.family not in CLASSICAL_FAMILIES:
            raise ValueError(f"classical family must be one of {CLASSICAL_FAMILIES}")
        if self.kind == "deep" and "arch" not in self.deep:
            raise ValueError("deep models need deep.arch")

    def resolved_cap(self, default: int | None) -> int | None:
        """Training cap for this model."""
        return default if self.target_cap == "default" else self.target_cap  # type: ignore[return-value]

    def feature_config(self) -> FeatureConfig:
        """Feature config for classical models."""
        return FeatureConfig(**self.features)


@dataclass
class CVConfig:
    """Engine-level K-fold over development (train + validation) engines; calibration excluded."""

    folds: int = 5
    seed: int = 42
    include_canonical: bool = True
    run_folds: list[str] | None = None  # subset of fold names to execute (e.g. ["cv0", "cv1"])


@dataclass
class Stage2Config:
    """Top-level Stage-2 experiment configuration."""

    name: str
    dataset: str = "FD001"
    seed: int = 42
    target_cap: int | None = 125
    split_manifest: str | None = None
    paths: PathsConfig = field(default_factory=lambda: PathsConfig(artifacts_dir="artifacts/stage2"))
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    models: dict[str, ModelEntry] = field(default_factory=dict)
    latency_repeats: int = 30
    reference_model: str | None = None  # model used for paired differences in reports

    def __post_init__(self) -> None:
        check_dataset_name(self.dataset)
        if not self.models:
            raise ValueError("models must not be empty")
        if self.reference_model and self.reference_model not in self.models:
            raise ValueError("reference_model must be one of models")

    @property
    def split_manifest_path(self) -> Path:
        """Resolved split manifest path."""
        if self.split_manifest:
            return Path(self.split_manifest)
        return Path(self.paths.manifest_dir) / f"{self.dataset.lower()}_split_seed42.json"

    def to_dict(self) -> dict[str, Any]:
        """Plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Stage2Config:
        """Parse and validate."""
        allowed = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown Stage-2 keys {sorted(unknown)}")
        kw = {k: v for k, v in payload.items() if k not in ("paths", "evaluation", "cv", "models")}
        kw["paths"] = _build(PathsConfig, {"artifacts_dir": "artifacts/stage2", **(payload.get("paths") or {})}, "paths")
        kw["evaluation"] = _build(EvaluationConfig, payload.get("evaluation"), "evaluation")
        kw["cv"] = _build(CVConfig, payload.get("cv"), "cv")
        kw["models"] = {k: _build(ModelEntry, v, f"models.{k}") for k, v in (payload.get("models") or {}).items()}
        return cls(**kw)


def load_stage2_config(path: str | Path) -> Stage2Config:
    """Load a Stage-2 YAML file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    return Stage2Config.from_dict(yaml.safe_load(path.read_text()))


def dump_stage2_config(cfg: Stage2Config, path: str | Path) -> None:
    """Save the resolved config."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))


def make_folds(manifest: SplitManifest, k: int, seed: int, include_canonical: bool = True) -> dict[str, dict[str, list[int]]]:
    """Engine-level folds over train+validation engines. Calibration engines never appear.

    Returns ``{fold_name: {"train": ids, "validation": ids}}``; ``canonical`` is the manifest's
    own train/validation split, ``cv0..cv{k-1}`` are disjoint validation blocks.
    """
    dev = np.sort(np.array(manifest.train_engine_ids + manifest.validation_engine_ids))
    if k < 2 or k > len(dev):
        raise ValueError(f"folds must be in [2, {len(dev)}]")
    rng = np.random.default_rng(seed)
    shuffled = dev.copy()
    rng.shuffle(shuffled)
    folds: dict[str, dict[str, list[int]]] = {}
    if include_canonical:
        folds["canonical"] = {"train": list(manifest.train_engine_ids), "validation": list(manifest.validation_engine_ids)}
    for i, block in enumerate(np.array_split(shuffled, k)):
        val = sorted(int(x) for x in block)
        folds[f"cv{i}"] = {"train": sorted(int(x) for x in dev if x not in set(val)), "validation": val}
    cal = set(manifest.calibration_engine_ids)
    for name, f in folds.items():
        if cal & (set(f["train"]) | set(f["validation"])) or set(f["train"]) & set(f["validation"]):
            raise AssertionError(f"fold {name} violates engine separation")  # pragma: no cover
    return folds

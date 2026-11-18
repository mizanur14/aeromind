"""Deterministic engine-level train / validation / calibration splits.

The algorithm intentionally reproduces the Stage-0 ``scripts/create_splits.py``:
sorted engine IDs are shuffled with ``numpy.random.default_rng(seed).shuffle`` and
sliced into consecutive blocks. With FD001 and seed 42 this yields exactly the IDs in
``data/manifests/fd001_split_seed42.json``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aeromind.utils import read_json, write_json

logger = logging.getLogger(__name__)

PARTITIONS = ("train", "validation", "calibration")


class SplitError(ValueError):
    """Raised when a split is invalid or would silently change an existing manifest."""


@dataclass
class SplitManifest:
    """Engine-level partition of one subset's training pool."""

    dataset: str
    seed: int
    train_engine_ids: list[int]
    validation_engine_ids: list[int]
    calibration_engine_ids: list[int]
    split_method: str = "engine_level_random"
    fractions: dict[str, float] = field(
        default_factory=lambda: {"train": 0.70, "validation": 0.15, "calibration": 0.15}
    )
    source_sha256: str | None = None

    def ids(self, partition: str) -> list[int]:
        """Return the engine IDs of ``train``, ``validation`` or ``calibration``."""
        if partition not in PARTITIONS:
            raise SplitError(f"Unknown partition {partition!r}; expected {PARTITIONS}")
        return getattr(self, f"{partition}_engine_ids")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (backward compatible keys)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SplitManifest:
        """Build from a manifest dict, accepting the Stage-0 format (no fractions/hash)."""
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            logger.debug("Ignoring unknown split manifest keys: %s", sorted(unknown))
        kwargs = {k: v for k, v in payload.items() if k in known}
        for key in ("train_engine_ids", "validation_engine_ids", "calibration_engine_ids"):
            kwargs[key] = [int(i) for i in kwargs[key]]
        return cls(**kwargs)


def make_engine_split(
    engine_ids: np.ndarray | list[int],
    dataset: str,
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> SplitManifest:
    """Partition engine IDs deterministically.

    Args:
        engine_ids: unique engine IDs of the training pool.
        dataset: subset name recorded in the manifest.
        seed: RNG seed.
        fractions: train/validation/calibration fractions (must sum to 1). Block sizes are
            ``round(f * n)`` for train and validation; calibration gets the remainder.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise SplitError(f"Fractions must sum to 1, got {fractions}")
    if min(fractions) <= 0:
        raise SplitError("All partitions must be non-empty fractions")
    ids = np.sort(np.unique(np.asarray(engine_ids, dtype=np.int64)))
    if len(ids) != len(engine_ids):
        raise SplitError("Engine IDs must be unique")
    n = len(ids)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    if n_train < 1 or n_val < 1 or n - n_train - n_val < 1:
        raise SplitError(f"Too few engines ({n}) for fractions {fractions}")

    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    manifest = SplitManifest(
        dataset=dataset,
        seed=seed,
        train_engine_ids=sorted(ids[:n_train].tolist()),
        validation_engine_ids=sorted(ids[n_train : n_train + n_val].tolist()),
        calibration_engine_ids=sorted(ids[n_train + n_val :].tolist()),
        fractions=dict(zip(PARTITIONS, map(float, fractions), strict=True)),
    )
    verify_split(manifest, ids)
    return manifest


def verify_split(manifest: SplitManifest, all_engine_ids: np.ndarray | list[int]) -> None:
    """Raise :class:`SplitError` unless partitions are disjoint, non-empty and cover the pool."""
    parts = {p: set(manifest.ids(p)) for p in PARTITIONS}
    for p, ids in parts.items():
        if not ids:
            raise SplitError(f"Partition {p} is empty")
        if len(ids) != len(manifest.ids(p)):
            raise SplitError(f"Partition {p} contains duplicate IDs")
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1 :]:
            overlap = parts[a] & parts[b]
            if overlap:
                raise SplitError(f"Engines in both {a} and {b}: {sorted(overlap)[:10]}")
    union = set().union(*parts.values())
    pool = {int(i) for i in all_engine_ids}
    if union != pool:
        raise SplitError(
            f"Split does not cover the pool: missing={sorted(pool - union)[:10]}, extra={sorted(union - pool)[:10]}"
        )


def split_frame(df: pd.DataFrame, manifest: SplitManifest, partition: str) -> pd.DataFrame:
    """Return the rows of ``df`` belonging to one partition (copy)."""
    ids = manifest.ids(partition)
    return df[df["engine_id"].isin(ids)].copy()


def load_split(path: str | Path) -> SplitManifest:
    """Load a split manifest JSON."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}. Run `aeromind split` first.")
    return SplitManifest.from_dict(read_json(path))


def save_split(manifest: SplitManifest, path: str | Path, force: bool = False) -> Path:
    """Write a manifest, refusing to change the engine IDs of an existing one unless ``force``."""
    path = Path(path)
    if path.is_file() and not force:
        existing = load_split(path)
        for p in PARTITIONS:
            if sorted(existing.ids(p)) != sorted(manifest.ids(p)):
                raise SplitError(
                    f"{path} exists with different {p} IDs. Refusing to overwrite; "
                    "pass --force if you really intend to change the split."
                )
    return write_json(path, manifest.to_dict())


def default_split_path(manifest_dir: str | Path, dataset: str, seed: int) -> Path:
    """Conventional manifest path, e.g. ``data/manifests/fd001_split_seed42.json``."""
    return Path(manifest_dir) / f"{dataset.lower()}_split_seed{seed}.json"

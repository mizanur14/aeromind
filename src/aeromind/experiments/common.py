"""Shared helpers for experiment workflows: data preparation and run directories."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from aeromind.config import ExperimentConfig, load_config
from aeromind.data.cmapss import load_train
from aeromind.data.splits import SplitError, SplitManifest, load_split, split_frame, verify_split
from aeromind.evaluation.protocol import select_snapshots
from aeromind.features.targets import make_target
from aeromind.utils import read_json, sha256_file, write_json

logger = logging.getLogger(__name__)

REPORT_SUFFIXES = {".csv", ".json", ".png", ".md", ".yaml"}


@dataclass
class DevelopmentData:
    """Training-pool partitions with targets. Contains no official test data."""

    config: ExperimentConfig
    manifest: SplitManifest
    manifest_sha256: str
    train: pd.DataFrame
    validation: pd.DataFrame
    y_train: np.ndarray
    snapshot_index: pd.Index


def load_development_data(config: ExperimentConfig, pool: pd.DataFrame | None = None) -> DevelopmentData:
    """Load the training pool, apply the engine split, build targets and snapshots.

    Args:
        config: experiment configuration.
        pool: optional pre-loaded training pool (e.g. synthetic data in tests).
    """
    if pool is None:
        pool = load_train(config.paths.raw_dir, config.dataset)
    manifest_path = config.split_manifest_path
    manifest = load_split(manifest_path)
    if manifest.dataset != config.dataset:
        raise SplitError(f"Split manifest is for {manifest.dataset}, config is {config.dataset}")
    verify_split(manifest, pool["engine_id"].unique())
    train = split_frame(pool, manifest, "train").reset_index(drop=True)
    validation = split_frame(pool, manifest, "validation").reset_index(drop=True)
    train["target"] = make_target(train, config.target.cap)
    validation["target"] = make_target(validation, config.target.cap)
    snaps = select_snapshots(validation, config.evaluation.snapshots_per_engine, config.evaluation.snapshot_seed)
    logger.info(
        "Development data: %d train engines (%d rows), %d validation engines (%d rows, %d snapshots); "
        "%d calibration engines held out",
        train.engine_id.nunique(),
        len(train),
        validation.engine_id.nunique(),
        len(validation),
        len(snaps),
        len(manifest.calibration_engine_ids),
    )
    return DevelopmentData(
        config=config,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        train=train,
        validation=validation,
        y_train=train["target"].to_numpy(),
        snapshot_index=snaps,
    )


def new_run_dir(config: ExperimentConfig, run_dir: str | Path | None = None) -> Path:
    """Create ``<artifacts_dir>/<name>_<YYYYmmdd-HHMMSS>`` (or use ``run_dir``)."""
    if run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(config.paths.artifacts_dir) / f"{config.name}_{stamp}"
    path = Path(run_dir)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Run directory {path} already exists and is not empty")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_run(run_dir: str | Path) -> tuple[ExperimentConfig, dict]:
    """Load the resolved config and metadata of a finished run."""
    run_dir = Path(run_dir)
    if not (run_dir / "metadata.json").is_file():
        raise FileNotFoundError(f"{run_dir} is not a completed run (metadata.json missing)")
    return load_config(run_dir / "config.yaml"), read_json(run_dir / "metadata.json")


def publish_reports(src: Path, dest: Path) -> Path:
    """Copy small report files (csv/json/png/md/yaml) from ``src`` tree to ``dest``."""
    for path in src.rglob("*"):
        if path.is_file() and path.suffix in REPORT_SUFFIXES and path.stat().st_size < 5_000_000:
            target = dest / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return dest


def sync_published_metadata(run_dir: Path, config: ExperimentConfig, publish: bool = True) -> None:
    """Refresh ``report/metadata.json`` (and its published copy) from the authoritative run metadata.

    Engine-ID lists are omitted from the report copy for brevity; they stay in the run's
    ``metadata.json``.
    """
    meta = read_json(run_dir / "metadata.json")
    slim = {k: v for k, v in meta.items() if k != "engines"}
    write_json(run_dir / "report" / "metadata.json", slim)
    if publish:
        write_json(Path(config.paths.reports_dir) / config.dataset / "runs" / run_dir.name / "metadata.json", slim)

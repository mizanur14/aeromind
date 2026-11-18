"""Synthetic Stage-2 smoke configuration (tiny models, 2 epochs, seconds on CPU)."""

from __future__ import annotations

from pathlib import Path

from aeromind.data.cmapss import load_train
from aeromind.data.splits import make_engine_split, save_split
from aeromind.research.spec import Stage2Config
from aeromind.synthetic import write_raw_files

TINY_TRAIN = {"epochs": 2, "batch_size": 128, "patience": 5, "threads": 2}


def smoke_stage2_config(root: Path, archs: tuple[str, ...] = ("tcn", "gru", "transformer")) -> Stage2Config:
    """Every Stage-2 model family at toy size on synthetic data."""
    deep_nets = {
        "tcn": {"channels": 8, "levels": 2},
        "gru": {"hidden": 8},
        "transformer": {"d_model": 8, "heads": 2, "layers": 1, "ff": 16},
    }
    models = {
        "ridge": {"kind": "classical", "family": "ridge", "params": {"alpha": 1.0}, "features": {"windows": [5]}},
        "xgboost": {"kind": "classical", "family": "xgboost", "params": {"n_estimators": 20, "max_depth": 3}, "features": {"windows": [5]}},
    }
    for a in archs:
        models[a] = {"kind": "deep", "deep": {"arch": a, "window": 10, "net": deep_nets[a], "train": TINY_TRAIN}}
    return Stage2Config.from_dict(
        {
            "name": "s2_smoke",
            "dataset": "FD001",
            "target_cap": 60,
            "paths": {
                "raw_dir": str(root / "raw"),
                "manifest_dir": str(root / "manifests"),
                "artifacts_dir": str(root / "stage2"),
                "reports_dir": str(root / "reports"),
            },
            "evaluation": {"snapshots_per_engine": 3, "capped_truth_cap": 60, "near_eol_threshold": 10},
            "cv": {"folds": 3, "seed": 0},
            "models": models,
            "latency_repeats": 3,
            "reference_model": "xgboost",
        }
    )


def prepare_smoke_data(cfg: Stage2Config, n_train: int = 30) -> None:
    """Write synthetic raw files and a split manifest for ``cfg``."""
    write_raw_files(cfg.paths.raw_dir, n_train=n_train, n_test=8, seed=5)
    pool = load_train(cfg.paths.raw_dir, cfg.dataset)
    save_split(make_engine_split(pool.engine_id.unique(), cfg.dataset, 42), cfg.split_manifest_path)

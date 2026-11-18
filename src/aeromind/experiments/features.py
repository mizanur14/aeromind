"""Materialise model-ready features (``aeromind features``).

Fits the FeaturePipeline on training engines and writes Parquet files for the
train / validation / calibration partitions (with ``rul`` and ``target``) and for the
official test trajectories **without labels**. Feature columns are prefixed ``feat__``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from aeromind.config import ExperimentConfig
from aeromind.data.cmapss import load_test_features_only, load_train
from aeromind.data.splits import load_split, split_frame, verify_split
from aeromind.features.pipeline import FeaturePipeline
from aeromind.features.targets import make_target
from aeromind.utils import write_json

logger = logging.getLogger(__name__)

#: Prefix for (scaled) feature columns, keeping them distinct from raw ID columns such as ``cycle``.
FEATURE_PREFIX = "feat__"


def prepare_features(config: ExperimentConfig, output_dir: str | Path | None = None) -> Path:
    """Write ``{train,validation,calibration,test_unlabelled}.parquet`` + ``features.json``."""
    out = Path(output_dir or Path(config.paths.processed_dir) / config.name)
    out.mkdir(parents=True, exist_ok=True)
    pool = load_train(config.paths.raw_dir, config.dataset)
    manifest = load_split(config.split_manifest_path)
    verify_split(manifest, pool["engine_id"].unique())
    parts = {p: split_frame(pool, manifest, p).reset_index(drop=True) for p in ("train", "validation", "calibration")}
    pipeline = FeaturePipeline(config.features, config.seed).fit(parts["train"])
    for name, frame in parts.items():
        feats = pipeline.transform_frame(frame).add_prefix(FEATURE_PREFIX)
        meta = frame[["engine_id", "cycle", "rul"]].assign(target=make_target(frame, config.target.cap))
        pd.concat([meta, feats], axis=1).to_parquet(out / f"{name}.parquet", index=False)
        logger.info("%s: %d rows x %d features", name, len(frame), feats.shape[1])
    test = load_test_features_only(config.paths.raw_dir, config.dataset)
    pd.concat(
        [test[["engine_id", "cycle"]], pipeline.transform_frame(test).add_prefix(FEATURE_PREFIX)], axis=1
    ).to_parquet(out / "test_unlabelled.parquet", index=False)
    write_json(out / "features.json", {"config": config.name, "target_cap": config.target.cap, **pipeline.describe()})
    return out

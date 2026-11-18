"""Data access for the dashboard (no Streamlit imports; unit-testable).

Every loader returns ``None`` when its source is absent, so the dashboard degrades to
"not available" instead of inventing content. Paths come from environment variables:

- ``AEROMIND_RESULTS_INDEX`` (default ``configs/stage3/results_index.yaml``)
- ``AEROMIND_REGISTRY``      (default: the index's ``registry``)
- ``AEROMIND_ROOT``          (default ``.``; bundle paths are relative to it)
- ``AEROMIND_DATA_DIR``      (default ``data/raw/cmapss``)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aeromind.data.cmapss import SENSOR_COLUMNS, SENSOR_INFO, DatasetError, load_train
from aeromind.data.splits import load_split
from aeromind.utils import read_json

DATASETS = ("FD001", "FD002", "FD003", "FD004")

DATASET_INFO = {
    "FD001": {"conditions": 1, "fault_modes": "HPC degradation", "train_engines": 100},
    "FD002": {"conditions": 6, "fault_modes": "HPC degradation", "train_engines": 260},
    "FD003": {"conditions": 1, "fault_modes": "HPC + fan degradation", "train_engines": 100},
    "FD004": {"conditions": 6, "fault_modes": "HPC + fan degradation", "train_engines": 249},
}


@dataclass
class Paths:
    """Resolved locations of everything the dashboard reads."""

    index: dict[str, Any]
    registry: Path
    root: Path
    data_dir: Path

    @classmethod
    def from_env(cls) -> Paths:
        idx_path = Path(os.environ.get("AEROMIND_RESULTS_INDEX", "configs/stage3/results_index.yaml"))
        index = yaml.safe_load(idx_path.read_text()) if idx_path.is_file() else {}
        return cls(
            index=index,
            registry=Path(os.environ.get("AEROMIND_REGISTRY", index.get("registry", "reports/registry/model_registry.json"))),
            root=Path(os.environ.get("AEROMIND_ROOT", ".")),
            data_dir=Path(os.environ.get("AEROMIND_DATA_DIR", "data/raw/cmapss")),
        )

    def report(self, *keys: str) -> Path | None:
        """Path of a report directory in the index, or ``None``."""
        node: Any = self.index
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return None
            node = node[k]
        return Path(node) if isinstance(node, str) else None


def read_csv(path: Path | None, name: str) -> pd.DataFrame | None:
    """``path/name`` as a DataFrame, or ``None`` if missing."""
    if path is None or not (path / name).is_file():
        return None
    return pd.read_csv(path / name)


def read_json_opt(path: Path | None) -> dict[str, Any] | None:
    """JSON file or ``None``."""
    return read_json(path) if path is not None and path.is_file() else None


# ----------------------------------------------------------------------------- registry
def registry_table(registry: dict[str, Any], availability: dict[str, tuple[bool, str]]) -> pd.DataFrame:
    """One row per registered model with metrics by evidence type."""
    rows = []
    for m in registry["models"]:
        cv = (m["metrics"].get("cross_validation") or {}).get("values") or {}
        can = (m["metrics"].get("canonical_validation") or {}).get("values") or {}
        test = (m["metrics"].get("official_test") or {}).get("values") or {}
        res = m.get("resources") or {}
        ok, why = availability.get(m["id"], (False, "unknown"))
        rows.append(
            {
                "model_id": m["id"],
                "dataset": m["dataset"],
                "stage": m["stage"],
                "family": m["family"],
                "kind": m["kind"],
                "version": m["version"],
                "available_locally": ok,
                "availability": why,
                "cv_rmse_capped_mean": (cv.get("snapshot_rmse_capped") or {}).get("mean"),
                "cv_rmse_capped_sd": (cv.get("snapshot_rmse_capped") or {}).get("sd"),
                "cv_mae_capped_mean": (cv.get("snapshot_mae_capped") or {}).get("mean"),
                "cv_phm08_mean_capped": (cv.get("snapshot_phm08_mean_capped") or {}).get("mean"),
                "cv_near_eol_rmse_true": (cv.get("near_eol_rmse_true") or {}).get("mean"),
                "canonical_rmse_capped": can.get("snapshot_rmse_capped"),
                "canonical_mae_capped": can.get("snapshot_mae_capped"),
                "canonical_phm08_mean_capped": can.get("snapshot_phm08_mean_capped"),
                "official_test_rmse_true": test.get("last_cycle_rmse_true"),
                "single_input_ms": res.get("single_input_ms"),
                "fit_seconds": res.get("fit_seconds_mean"),
                "peak_rss_mb": res.get("peak_rss_mb"),
                "n_parameters": m.get("n_parameters"),
                "bundle_mb": m["bundle"]["bytes"] / 2**20,
                "intervals": m["intervals"] is not None,
                "source_run": m["source_run"],
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- raw data
def raw_available(data_dir: Path, dataset: str) -> bool:
    """True if the NASA training file for ``dataset`` exists locally."""
    return (data_dir / f"train_{dataset}.txt").is_file()


def load_partitioned(data_dir: Path, dataset: str, manifest: Path) -> pd.DataFrame | None:
    """Training-file trajectories with a ``partition`` column from the split manifest."""
    if not raw_available(data_dir, dataset) or not manifest.is_file():
        return None
    try:
        pool = load_train(data_dir, dataset)
    except DatasetError:
        return None
    split = load_split(manifest)
    part = {}
    for p in ("train", "validation", "calibration"):
        part.update(dict.fromkeys(split.ids(p), p))
    return pool.assign(partition=pool["engine_id"].map(part))


def sensor_label(col: str) -> str:
    """'sensor_11 (Ps30)' style label."""
    info = SENSOR_INFO.get(col)
    return f"{col} ({info[0]})" if info else col


def sensor_description(col: str) -> str:
    """Physical description and unit from Saxena et al. (2008), Table 2."""
    info = SENSOR_INFO.get(col)
    return f"{info[1]} [{info[2]}]" if info else ""


def informative_sensors(train: pd.DataFrame) -> list[str]:
    """Sensors with more than 3 distinct values on training engines (same rule as Stage 1/2)."""
    return [c for c in SENSOR_COLUMNS if train[c].nunique() > 3]


def correlation_with_rul(train: pd.DataFrame, sensors: list[str], cap: int = 125) -> pd.DataFrame:
    """Spearman correlation of each sensor with capped RUL on training engines."""
    y = np.minimum(train["rul"].to_numpy(float), cap)
    rows = [{"sensor": s, "spearman_rho_vs_capped_rul": pd.Series(train[s].to_numpy(float)).corr(pd.Series(y), method="spearman")} for s in sensors]
    return pd.DataFrame(rows).sort_values("spearman_rho_vs_capped_rul")


def xgboost_importance(predictor: Any) -> pd.DataFrame | None:
    """Gain importance of an XGBoost bundle mapped to its pipeline feature names."""
    est = getattr(predictor, "estimator", None)
    if est is None or not hasattr(est, "get_booster"):
        return None
    names = list(predictor.pipeline.feature_names_)
    score = est.get_booster().get_score(importance_type="gain")
    rows = []
    for k, v in score.items():
        idx = int(k[1:]) if k.startswith("f") and k[1:].isdigit() else None
        rows.append({"feature": names[idx] if idx is not None and idx < len(names) else k, "gain": float(v)})
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    df["share"] = df["gain"] / df["gain"].sum()
    return df.sort_values("gain", ascending=False).reset_index(drop=True)

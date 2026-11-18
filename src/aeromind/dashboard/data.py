"""Data access for the dashboard (no Streamlit imports; unit-testable).

Every loader returns ``None`` when its source is absent, so the dashboard degrades to
"not available" instead of inventing content.

Path resolution:

- Defaults and every path inside the results index are resolved against the **project
  root** (the directory holding ``configs/stage3/results_index.yaml``), not the current
  working directory. ``AEROMIND_PROJECT_ROOT`` overrides the detected root.
- Paths given explicitly through environment variables keep normal shell semantics:
  relative values are relative to the current working directory.

Environment variables:

- ``AEROMIND_DASHBOARD_MODE`` ``public`` (committed reports only; never loads model bundles
  or NASA files) or ``research`` (default for ``aeromind dashboard``; adds the local pages
  that need bundles and raw data)
- ``AEROMIND_RESULTS_INDEX`` (default ``configs/stage3/results_index.yaml``)
- ``AEROMIND_REGISTRY``      (default: the index's ``registry``)
- ``AEROMIND_ROOT``          (default: project root; bundle paths are relative to it)
- ``AEROMIND_DATA_DIR``      (default ``data/raw/cmapss`` under the project root)

Public mode imports only numpy, pandas and PyYAML from here; the model-loading stack
(joblib, scikit-learn, XGBoost, PyTorch) is imported lazily by research pages only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

DATASETS = ("FD001", "FD002", "FD003", "FD004")

# Measured from the NASA files; copied from docs/DATA.md ("Subsets (measured from the files)").
DATASET_INFO = {
    "FD001": {"conditions": 1, "fault_modes": "HPC degradation", "train_engines": 100, "test_engines": 100, "train_rows": 20631, "train_life_cycles": "128–362"},
    "FD002": {"conditions": 6, "fault_modes": "HPC degradation", "train_engines": 260, "test_engines": 259, "train_rows": 53759, "train_life_cycles": "128–378"},
    "FD003": {"conditions": 1, "fault_modes": "HPC + fan degradation", "train_engines": 100, "test_engines": 100, "train_rows": 24720, "train_life_cycles": "145–525"},
    "FD004": {"conditions": 6, "fault_modes": "HPC + fan degradation", "train_engines": 249, "test_engines": 248, "train_rows": 61249, "train_life_cycles": "128–543"},
}

MODES = ("public", "research")
INDEX_RELPATH = Path("configs") / "stage3" / "results_index.yaml"
DEFAULT_REGISTRY = "reports/registry/model_registry.json"

# Metric keys as stored in the registry → (label, description).
METRICS: dict[str, tuple[str, str]] = {
    "snapshot_rmse_capped": ("Snapshot RMSE · capped truth", "Root-mean-square error over 10 seeded cut cycles per validation engine, against min(RUL, 125). Cycles; lower is better."),
    "snapshot_mae_capped": ("Snapshot MAE · capped truth", "Mean absolute error over the same snapshots, against min(RUL, 125). Cycles; lower is better."),
    "snapshot_phm08_mean_capped": ("PHM08 score per snapshot · capped truth", "Mean asymmetric PHM08 score per snapshot; late (optimistic) predictions are penalised more. Lower is better."),
    "snapshot_rmse_true": ("Snapshot RMSE · true RUL", "Snapshot RMSE against uncapped true RUL. Dominated by early-life snapshots a capped model cannot represent."),
    "snapshot_mae_true": ("Snapshot MAE · true RUL", "Snapshot MAE against uncapped true RUL."),
    "window_ew_rmse_capped": ("All-window RMSE · engine-weighted · capped", "RMSE over every validation cycle, each engine weighted equally, against min(RUL, 125)."),
    "near_eol_rmse_true": ("Near-end-of-life RMSE · true RUL ≤ 30", "RMSE on windows whose true RUL is at most 30 cycles."),
}

EVIDENCE_KEYS = {
    "cross_validation": "Cross-validation",
    "canonical_validation": "Canonical validation",
    "official_test": "Official NASA test",
}


# ----------------------------------------------------------------------------- paths
def project_root() -> Path:
    """Directory containing ``configs/stage3/results_index.yaml`` (or ``AEROMIND_PROJECT_ROOT``)."""
    env = os.environ.get("AEROMIND_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / INDEX_RELPATH).is_file():
            return parent
    return Path.cwd().resolve()


def dashboard_mode() -> str:
    """``public`` or ``research``; anything unrecognised falls back to the safer ``public``."""
    mode = os.environ.get("AEROMIND_DASHBOARD_MODE", "research").strip().lower()
    return mode if mode in MODES else "public"


def _load_index(path: Path) -> dict[str, Any]:
    """Results index as a mapping; an absent, unreadable or malformed file counts as empty."""
    try:
        index = yaml.safe_load(path.read_text()) if path.is_file() else {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    return index if isinstance(index, dict) else {}


def _resolve(value: str | Path, base: Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else base / p


@dataclass
class Paths:
    """Resolved locations of everything the dashboard reads."""

    index: dict[str, Any]
    registry: Path
    root: Path
    data_dir: Path
    project_root: Path = field(default_factory=project_root)

    @classmethod
    def from_env(cls) -> Paths:
        proj = project_root()
        cwd = Path.cwd()
        env = os.environ.get
        idx_path = _resolve(env("AEROMIND_RESULTS_INDEX"), cwd) if env("AEROMIND_RESULTS_INDEX") else proj / INDEX_RELPATH
        index = _load_index(idx_path)
        registry = _resolve(env("AEROMIND_REGISTRY"), cwd) if env("AEROMIND_REGISTRY") else _resolve(index.get("registry", DEFAULT_REGISTRY), proj)
        root = _resolve(env("AEROMIND_ROOT"), cwd) if env("AEROMIND_ROOT") else proj
        data_dir = _resolve(env("AEROMIND_DATA_DIR"), cwd) if env("AEROMIND_DATA_DIR") else proj / "data" / "raw" / "cmapss"
        return cls(index=index, registry=registry, root=root, data_dir=data_dir, project_root=proj)

    def resolve(self, value: str | Path | None) -> Path | None:
        """A path from the index, resolved against the project root (``None`` stays ``None``)."""
        if value is None or value == "":
            return None
        return _resolve(value, self.project_root)

    def report(self, *keys: str) -> Path | None:
        """Path of an entry in the results index, or ``None``."""
        node: Any = self.index
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return None
            node = node[k]
        return self.resolve(node) if isinstance(node, str) else None

    def split_manifest(self, dataset: str) -> Path:
        """Split manifest for ``dataset`` (index entry, else the conventional seed-42 file)."""
        return self.report("data_manifests", "splits", dataset) or self.project_root / "data" / "manifests" / f"{dataset.lower()}_split_seed42.json"

    def display(self, path: str | Path | None) -> str:
        """Project-relative POSIX path for captions; never reveals absolute local paths."""
        if path is None:
            return "—"
        p = Path(path)
        if not p.is_absolute():
            return p.as_posix()
        try:
            return p.resolve().relative_to(self.project_root.resolve()).as_posix()
        except ValueError:
            return f"<external>/{p.name}"


# ----------------------------------------------------------------------------- generic readers
def read_csv(path: Path | None, name: str) -> pd.DataFrame | None:
    """``path/name`` as a DataFrame, or ``None`` if missing, unreadable or malformed."""
    if path is None or not (path / name).is_file():
        return None
    try:
        return pd.read_csv(path / name)
    except (OSError, UnicodeDecodeError, ValueError):  # pandas parse errors are ValueErrors
        return None


def read_json_opt(path: Path | None) -> Any | None:
    """Parsed JSON file, or ``None`` if missing, unreadable or malformed."""
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, ValueError):  # JSONDecodeError is a ValueError
        return None


def load_registry_json(path: Path | None) -> dict[str, Any] | None:
    """Registry JSON with a minimal structural check, without importing the model stack."""
    reg = read_json_opt(path)
    if not isinstance(reg, dict) or not isinstance(reg.get("models"), list):
        return None
    if not all(isinstance(m, dict) and "id" in m and "metrics" in m for m in reg["models"]):
        return None
    return reg


# ----------------------------------------------------------------------------- registry
def _values(entry: dict[str, Any], evidence: str) -> dict[str, Any]:
    return ((entry.get("metrics") or {}).get(evidence) or {}).get("values") or {}


def registry_table(registry: dict[str, Any], availability: dict[str, tuple[bool, str]] | None = None) -> pd.DataFrame:
    """One row per registered model with metrics by evidence type.

    ``availability`` (research mode) adds whether each bundle is present and hash-verified.
    """
    rows = []
    for m in registry["models"]:
        cv = _values(m, "cross_validation")
        can = _values(m, "canonical_validation")
        test = _values(m, "official_test")
        res = m.get("resources") or {}
        row = {
            "model_id": m["id"],
            "dataset": m["dataset"],
            "stage": m["stage"],
            "family": m["family"],
            "kind": m["kind"],
            "version": m["version"],
            "target_cap": m.get("target_cap"),
            "cv_folds": (m["metrics"].get("cross_validation") or {}).get("folds"),
            "cv_rmse_capped_mean": (cv.get("snapshot_rmse_capped") or {}).get("mean"),
            "cv_rmse_capped_sd": (cv.get("snapshot_rmse_capped") or {}).get("sd"),
            "cv_mae_capped_mean": (cv.get("snapshot_mae_capped") or {}).get("mean"),
            "cv_phm08_mean_capped": (cv.get("snapshot_phm08_mean_capped") or {}).get("mean"),
            "cv_near_eol_rmse_true": (cv.get("near_eol_rmse_true") or {}).get("mean"),
            "canonical_rmse_capped": can.get("snapshot_rmse_capped"),
            "canonical_mae_capped": can.get("snapshot_mae_capped"),
            "canonical_phm08_mean_capped": can.get("snapshot_phm08_mean_capped"),
            "canonical_rmse_true": can.get("snapshot_rmse_true"),
            "official_test_rmse_true": test.get("last_cycle_rmse_true"),
            "single_input_ms": res.get("single_input_ms"),
            "fit_seconds": res.get("fit_seconds_mean"),
            "peak_rss_mb": res.get("peak_rss_mb"),
            "n_parameters": m.get("n_parameters"),
            "n_training_engines": (m.get("training") or {}).get("n_engines"),
            "bundle_mb": m["bundle"]["bytes"] / 2**20,
            "bundle_sha256_short": m["bundle"]["sha256"][:12],
            "requires_torch": "torch" in (m.get("requires") or []),
            "intervals": m["intervals"] is not None,
            "interval_levels": ", ".join(sorted(k for k, v in (m["intervals"] or {}).get("thresholds", {}).items() if v is not None)) or "—",
            "source_run": m["source_run"],
            "report_dir": m.get("report_dir"),
        }
        if availability is not None:
            ok, why = availability.get(m["id"], (False, "unknown"))
            row["available_locally"], row["availability"] = ok, why
        rows.append(row)
    return pd.DataFrame(rows)


def metric_frame(registry: dict[str, Any], dataset: str, evidence: str) -> pd.DataFrame:
    """Long table ``model_id, family, stage, metric, value, sd`` for one evidence type.

    ``evidence`` is ``cross_validation`` (value = fold mean, sd = fold SD) or
    ``canonical_validation`` (single split, sd = NaN). Values are copied from the registry,
    which copies them from the run reports.
    """
    rows = []
    for m in registry["models"]:
        if m["dataset"] != dataset:
            continue
        for key, v in _values(m, evidence).items():
            if key not in METRICS:
                continue
            value, sd = (v.get("mean"), v.get("sd")) if isinstance(v, dict) else (v, None)
            if value is None:
                continue
            rows.append({"model_id": m["id"], "family": m["family"], "stage": m["stage"], "metric": key, "value": float(value), "sd": float(sd) if sd is not None else np.nan})
    return pd.DataFrame(rows, columns=["model_id", "family", "stage", "metric", "value", "sd"])


def official_test_table(registry: dict[str, Any], ledger: dict[str, Any] | None) -> pd.DataFrame:
    """Registry official-test metrics, each checked against the ledger entry for its run."""
    entries = {e["run"]: e for e in (ledger or {}).get("entries", [])}
    rows = []
    for m in registry["models"]:
        t = (m.get("metrics") or {}).get("official_test")
        if not t:
            continue
        v = t["values"]
        led = entries.get(m["source_run"])
        match = bool(led) and np.isclose(led.get("selected_rmse_true", np.nan), v["last_cycle_rmse_true"]) and np.isclose(led.get("selected_mae_true", np.nan), v["last_cycle_mae_true"])
        rows.append(
            {
                "model_id": m["id"],
                "dataset": m["dataset"],
                "family": m["family"],
                "test_engines": t.get("n_test_engines"),
                "rmse_true": v["last_cycle_rmse_true"],
                "mae_true": v["last_cycle_mae_true"],
                "phm08_sum_true": v["last_cycle_phm08_sum_true"],
                "rmse_capped": v.get("last_cycle_rmse_capped"),
                "evaluated_utc": t.get("evaluated_utc"),
                "ledger_entry_matches": bool(match),
            }
        )
    return pd.DataFrame(rows)


def split_summary(paths: Paths) -> pd.DataFrame:
    """Engine counts per partition from the committed split manifests."""
    rows = []
    for ds in DATASETS:
        j = read_json_opt(paths.split_manifest(ds))
        if not j:
            continue
        rows.append(
            {
                "subset": ds,
                "train": len(j.get("train_engine_ids", [])),
                "validation": len(j.get("validation_engine_ids", [])),
                "calibration": len(j.get("calibration_engine_ids", [])),
                "seed": j.get("seed"),
                "method": j.get("split_method"),
                "manifest": paths.display(paths.split_manifest(ds)),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- raw data (research mode)
def raw_available(data_dir: Path, dataset: str) -> bool:
    """True if the NASA training file for ``dataset`` exists locally."""
    return (data_dir / f"train_{dataset}.txt").is_file()


def load_partitioned(data_dir: Path, dataset: str, manifest: Path) -> pd.DataFrame | None:
    """Training-file trajectories with a ``partition`` column from the split manifest."""
    from aeromind.data.cmapss import DatasetError, load_train
    from aeromind.data.splits import load_split

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
    from aeromind.data.cmapss import SENSOR_INFO

    info = SENSOR_INFO.get(col)
    return f"{col} ({info[0]})" if info else col


def sensor_description(col: str) -> str:
    """Physical description and unit from Saxena et al. (2008), Table 2."""
    from aeromind.data.cmapss import SENSOR_INFO

    info = SENSOR_INFO.get(col)
    return f"{info[1]} [{info[2]}]" if info else ""


def informative_sensors(train: pd.DataFrame) -> list[str]:
    """Sensors with more than 3 distinct values on training engines (same rule as Stage 1/2)."""
    from aeromind.data.cmapss import SENSOR_COLUMNS

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

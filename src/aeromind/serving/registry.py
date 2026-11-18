"""Model registry: versioned metadata for every servable bundle, with provenance.

The registry JSON is small and committed (``reports/registry/model_registry.json``); the
bundles it references are **not** committed. Each entry pins its bundle by SHA-256, so a
service refuses to unpickle a file that differs from the one the metrics were measured on.

Sources:

- ``stage1_run``        a Stage-1 run directory (only the listed models, normally the
                        validation-selected one). Official test metrics come from the
                        test-set ledger, never recomputed here.
- ``stage2_comparison`` a Stage-2 comparison/benchmark run (canonical-fold bundles), with
                        CV and canonical-fold metrics and resource measurements.

Calibrators (optional): engine-level split conformal on the reserved **calibration** engines,
one seeded snapshot per engine (n = 15 on FD001), thresholds for the configured levels.
The empirical coverage of the same procedure on validation engines is copied from the
Stage-2 UQ run report.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aeromind.data.cmapss import SENSOR_COLUMNS, SETTING_COLUMNS, load_train
from aeromind.data.splits import load_split, verify_split
from aeromind.features.targets import cap_rul
from aeromind.models.bundle import BUNDLE_FORMAT_VERSION, load_bundle
from aeromind.research.spec import load_stage2_config
from aeromind.research.uq import conformal_quantile, one_per_engine
from aeromind.utils import environment_info, read_json, sha256_file, write_json

logger = logging.getLogger(__name__)

REGISTRY_SCHEMA_VERSION = 1
#: Raw C-MAPSS measurement columns every inference request must provide (per cycle).
REQUIRED_INPUT_COLUMNS: list[str] = ["cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]

#: Metrics copied into the registry (snapshot = engine-balanced validation snapshots).
METRIC_KEYS = (
    "snapshot_rmse_capped",
    "snapshot_mae_capped",
    "snapshot_phm08_mean_capped",
    "snapshot_rmse_true",
    "snapshot_mae_true",
    "window_ew_rmse_capped",
    "near_eol_rmse_true",
)

FAMILY_LABELS = {
    "hist_gradient_boosting": "HistGradientBoosting",
    "hgb_reference": "HistGradientBoosting",
    "xgboost": "XGBoost",
    "ridge": "Ridge regression",
    "random_forest": "Random forest",
    "tcn": "Temporal convolutional network",
    "gru": "Gated recurrent unit network",
    "transformer": "Transformer encoder",
}

COMMON_LIMITATIONS = [
    "Trained and evaluated on NASA C-MAPSS simulation data only; not validated on real engines.",
    "Not certified for operational or safety-critical maintenance decisions.",
    "Predicts capped RUL min(RUL, cap); early-life RUL above the cap is not identifiable.",
    "Expects the same operating regime and fault modes as its training subset.",
]


# ----------------------------------------------------------------------------- config
@dataclass
class SourceSpec:
    """One run directory contributing models to the registry."""

    kind: str
    run_dir: str
    report_dir: str
    models: list[str] | None = None


@dataclass
class CalibrationSpec:
    """Which registry models get conformal calibrators."""

    models: list[str] = field(default_factory=list)
    levels: list[float] = field(default_factory=lambda: [0.8, 0.9])
    seed: int = 42
    uq_report: str | None = None


@dataclass
class RegistryConfig:
    """Registry build configuration (``configs/stage3/registry.yaml``)."""

    name: str
    output: str
    sources: list[SourceSpec]
    calibration: CalibrationSpec = field(default_factory=CalibrationSpec)
    default_model: str | None = None
    default_model_rationale: str = ""
    ledger: str = "reports/test_set_ledger.json"

    @classmethod
    def load(cls, path: str | Path) -> RegistryConfig:
        """Load from YAML; unknown keys raise."""
        payload = yaml.safe_load(Path(path).read_text())
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown registry keys {sorted(unknown)}")
        payload["sources"] = [SourceSpec(**s) for s in payload["sources"]]
        payload["calibration"] = CalibrationSpec(**payload.get("calibration", {}))
        return cls(**payload)


# ----------------------------------------------------------------------------- helpers
def _num(v: Any) -> float | None:
    """JSON-safe float (NaN/inf → None)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def model_id(dataset: str, stage: str, name: str) -> str:
    """Stable registry identifier, e.g. ``fd001-s2-xgboost``."""
    return f"{dataset.lower()}-{stage}-{name}"


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _input_spec(predictor: Any) -> dict[str, Any]:
    pipe = predictor.pipeline
    spec: dict[str, Any] = {
        "required_columns": REQUIRED_INPUT_COLUMNS,
        "used_columns": list(pipe.kept_columns_),
        "condition_normalization": pipe.normalizer_ is not None,
        "rolling_windows": list(getattr(pipe.config, "windows", []) or []),
        "n_features": len(pipe.feature_names_),
    }
    if getattr(predictor, "input_kind", "") == "windows":
        spec["sequence_window"] = int(predictor.window)
    return spec


def _bundle_info(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": _rel(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "format_version": BUNDLE_FORMAT_VERSION,
    }


def _training_info(predictor: Any, manifest_path: Path, seed: int | None) -> dict[str, Any]:
    ids = [int(e) for e in predictor.pipeline.fitted_engine_ids_]
    return {
        "engine_ids": ids,
        "n_engines": len(ids),
        "split_manifest": manifest_path.as_posix(),
        "split_manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "seed": seed,
    }


# ----------------------------------------------------------------------------- sources
def _stage2_entries(src: SourceSpec, root: Path) -> list[dict[str, Any]]:
    run = Path(src.run_dir)
    s2 = load_stage2_config(run / "config.yaml")
    meta = read_json(run / "metadata.json")
    report = Path(src.report_dir)
    summary = pd.read_csv(report / "summary.csv").set_index("model")
    resources = pd.read_csv(report / "resources.csv").set_index("model")
    folds = pd.read_csv(report / "fold_metrics.csv")
    names = src.models or list(summary.index)
    stage = "s2"
    out = []
    for name in names:
        bundle = run / "models" / f"{name}.joblib"
        predictor, _ = load_bundle(bundle)
        row = summary.loc[name]
        n_cv = int(row.get("cv_folds", 0) or 0) if pd.notna(row.get("cv_folds", np.nan)) else 0
        cv = (
            {k: {"mean": _num(row[f"{k}_cv_mean"]), "sd": _num(row[f"{k}_cv_sd"])} for k in METRIC_KEYS}
            if n_cv > 0
            else None
        )
        canon = folds[(folds.model == name) & (folds.fold == "canonical")]
        canonical = {k: _num(canon.iloc[0][k]) for k in METRIC_KEYS} if len(canon) else None
        res = resources.loc[name]
        n_val = int(canon.iloc[0]["snapshot_n"]) // s2.evaluation.snapshots_per_engine if len(canon) else None
        kind = "deep" if getattr(predictor, "input_kind", "") == "windows" else "classical"
        out.append(
            {
                "id": model_id(s2.dataset, stage, name),
                "dataset": s2.dataset,
                "stage": "Stage 2",
                "name": name,
                "family": FAMILY_LABELS.get(name, name),
                "kind": kind,
                "framework": "pytorch" if kind == "deep" else type(predictor.estimator).__module__.split(".")[0],
                "target_cap": predictor.target_cap,
                "source_run": run.name,
                "run_dir": _rel(run, root),
                "report_dir": _rel(report, root),
                "run_finished_utc": meta.get("finished_utc"),
                "git_commit_at_training": meta.get("environment", {}).get("git_commit"),
                "bundle": _bundle_info(bundle, root),
                "input": _input_spec(predictor),
                "training": _training_info(predictor, Path(s2.split_manifest_path), s2.seed),
                "n_parameters": int(res["n_parameters"]) if pd.notna(res.get("n_parameters")) else None,
                "metrics": {
                    "cross_validation": (
                        {"folds": n_cv, "engines_per_fold": None, "values": cv} if cv is not None else None
                    ),
                    "canonical_validation": (
                        {"n_validation_engines": n_val, "values": canonical} if canonical is not None else None
                    ),
                    "official_test": None,
                },
                "resources": {
                    "fit_seconds_mean": _num(res["fit_seconds_mean"]),
                    "single_input_ms": _num(res["single_input_ms"]),
                    "batch_us_per_row": _num(res["batch_us_per_row"]),
                    "peak_rss_mb": _num(res["peak_rss_mb"]),
                    "epochs_canonical": _num(res.get("epochs_canonical")),
                },
                "requires": ["torch"] if kind == "deep" else [],
                "intervals": None,
            }
        )
    return out


def _stage1_entries(src: SourceSpec, root: Path, ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    run = Path(src.run_dir)
    meta = read_json(run / "metadata.json")
    report = Path(src.report_dir)
    vsum = read_json(report / "validation_summary.json")
    dataset = meta["dataset"]
    names = src.models or [meta["selected_model"]]
    out = []
    for name in names:
        bundle = run / "models" / f"{name}.joblib"
        predictor, _ = load_bundle(bundle)
        snap = vsum["validation"][name]["snapshot"]
        canonical = {
            "snapshot_rmse_capped": _num(snap.get("rmse_capped")),
            "snapshot_mae_capped": _num(snap.get("mae_capped")),
            "snapshot_phm08_mean_capped": _num(snap.get("phm08_mean_capped")),
            "snapshot_rmse_true": _num(snap.get("rmse_true")),
            "snapshot_mae_true": _num(snap.get("mae_true")),
        }
        test = None
        entries = [e for e in ledger if e.get("run") == run.name]
        if entries and name == entries[0].get("selected_model"):
            e = entries[0]
            test = {
                "n_test_engines": 100 if dataset == "FD001" else None,
                "evaluated_utc": e.get("evaluated_utc"),
                "protocol": "one prediction per official test engine at its last observed cycle (ledgered, once)",
                "values": {
                    "last_cycle_rmse_true": _num(e.get("selected_rmse_true")),
                    "last_cycle_mae_true": _num(e.get("selected_mae_true")),
                    "last_cycle_phm08_sum_true": _num(e.get("selected_phm08_sum_true")),
                    "last_cycle_rmse_capped": _num(e.get("selected_rmse_capped")),
                },
            }
        out.append(
            {
                "id": model_id(dataset, "s1", "hgb" if name == "hist_gradient_boosting" else name),
                "dataset": dataset,
                "stage": "Stage 1",
                "name": name,
                "family": FAMILY_LABELS.get(name, name),
                "kind": "classical",
                "framework": type(predictor.estimator).__module__.split(".")[0],
                "target_cap": predictor.target_cap,
                "source_run": run.name,
                "run_dir": _rel(run, root),
                "report_dir": _rel(report, root),
                "run_finished_utc": meta.get("finished_utc"),
                "git_commit_at_training": meta.get("environment", {}).get("git_commit"),
                "bundle": _bundle_info(bundle, root),
                "input": _input_spec(predictor),
                "training": _training_info(predictor, Path(meta["split_manifest"]), meta.get("seed")),
                "n_parameters": None,
                "metrics": {
                    "cross_validation": None,
                    "canonical_validation": {"n_validation_engines": len(meta["engines"]["validation"]), "values": canonical},
                    "official_test": test,
                },
                "resources": {},
                "requires": [],
                "intervals": None,
            }
        )
    return out


# ----------------------------------------------------------------------------- calibrators
def build_calibrator(entry: dict[str, Any], spec: CalibrationSpec, root: Path) -> dict[str, Any]:
    """Engine-level split conformal thresholds from the calibration engines of the entry's split."""
    run = root / entry["run_dir"]
    s2 = load_stage2_config(run / "config.yaml")
    predictor, _ = load_bundle(root / entry["bundle"]["path"])
    pool = load_train(s2.paths.raw_dir, s2.dataset)
    manifest = load_split(s2.split_manifest_path)
    verify_split(manifest, pool.engine_id.unique())
    cal = pool[pool.engine_id.isin(manifest.ids("calibration"))].reset_index(drop=True)
    cap_c = s2.evaluation.capped_truth_cap
    y = cap_rul(cal["rul"], cap_c)
    pred = predictor.predict(cal)
    rng = np.random.default_rng(spec.seed)
    idx = one_per_engine(cal["engine_id"].to_numpy(), rng)
    scores = np.abs(y[idx] - pred[idx])
    n = len(idx)
    thresholds = {f"{lv:.2f}": _num(conformal_quantile(scores, round(1 - lv, 10))) for lv in spec.levels}
    validated = _validated_coverage(spec.uq_report, entry["name"], spec.levels)
    return {
        "method": "split_conformal_engine_level",
        "description": (
            "Symmetric interval prediction ± q, clipped at 0. q is the ⌈(n+1)(1−α)⌉-th smallest absolute "
            "residual (vs capped truth) over one seeded random snapshot per calibration engine."
        ),
        "target": f"min(RUL, {cap_c})",
        "n_calibration_scores": n,
        "calibration_engine_ids": [int(e) for e in manifest.ids("calibration")],
        "seed": spec.seed,
        "scores": [round(float(s), 4) for s in np.sort(scores)],
        "thresholds": thresholds,
        "max_finite_level": round(n / (n + 1), 4),
        "validated_coverage": validated,
        "guarantee": (
            f"Marginal coverage ≥ level for a snapshot drawn uniformly from a new engine exchangeable with the "
            f"{n} calibration engines (averaged over calibration draws). Not conditional on RUL range; "
            "not valid under distribution shift (other subsets, sensor faults); levels above "
            f"{n}/{n + 1} are unattainable."
        ),
    }


def _validated_coverage(uq_report: str | None, name: str, levels: list[float]) -> dict[str, Any] | None:
    if not uq_report or not (Path(uq_report) / "intervals_main.csv").is_file():
        return None
    t = pd.read_csv(Path(uq_report) / "intervals_main.csv")
    sel = t[
        (t.method == "conformal_engine")
        & (t.model == name)
        & (t.calibrate_on == "calibration")
        & (t.evaluate_on == "validation")
    ]
    out: dict[str, Any] = {"source": Path(uq_report).as_posix(), "evaluated_on": "canonical validation engines", "levels": {}}
    for lv in levels:
        r = sel[np.isclose(sel.nominal, lv)]
        if len(r):
            r = r.iloc[0]
            out["levels"][f"{lv:.2f}"] = {
                "draws": int(r.draws),
                "coverage_capped_mean": _num(r.coverage_snapshot_capped_mean),
                "coverage_capped_p05": _num(r.coverage_snapshot_capped_p05),
                "coverage_capped_p95": _num(r.coverage_snapshot_capped_p95),
                "coverage_true_rul_mean": _num(r.coverage_snapshot_true_mean),
                "width_mean": _num(r.width_snapshot_mean),
            }
    return out if out["levels"] else None


# ----------------------------------------------------------------------------- build
def build_registry(cfg: RegistryConfig, root: str | Path = ".") -> dict[str, Any]:
    """Assemble the registry dictionary from the configured runs (bundles must be present)."""
    root = Path(root)
    ledger_path = Path(cfg.ledger)
    ledger = read_json(ledger_path).get("entries", []) if ledger_path.is_file() else []
    models: list[dict[str, Any]] = []
    for src in cfg.sources:
        if src.kind == "stage1_run":
            models += _stage1_entries(src, root, ledger)
        elif src.kind == "stage2_comparison":
            models += _stage2_entries(src, root)
        else:
            raise ValueError(f"Unknown source kind {src.kind!r}")
    by_id = {m["id"]: m for m in models}
    if len(by_id) != len(models):
        raise ValueError("Duplicate model ids in registry sources")
    for mid in cfg.calibration.models:
        if mid not in by_id:
            raise ValueError(f"Calibration model {mid!r} not in registry")
        by_id[mid]["intervals"] = build_calibrator(by_id[mid], cfg.calibration, root)
        logger.info("Calibrated %s: %s", mid, by_id[mid]["intervals"]["thresholds"])
    for m in models:
        m["version"] = m["bundle"]["sha256"][:12]
        m["limitations"] = _limitations(m)
    if cfg.default_model and cfg.default_model not in by_id:
        raise ValueError(f"default_model {cfg.default_model!r} not in registry")
    env = environment_info()
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "name": cfg.name,
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": env.get("git_commit"),
        "packages": env.get("packages"),
        "official_test_labels_read": False,
        "default_model": cfg.default_model,
        "default_model_rationale": cfg.default_model_rationale,
        "config": asdict(cfg),
        "models": models,
    }


def _limitations(m: dict[str, Any]) -> list[str]:
    lim = list(COMMON_LIMITATIONS)
    cv = (m["metrics"].get("cross_validation") or {}).get("values")
    if cv:
        r = cv["snapshot_rmse_capped"]
        lim.append(
            f"Cross-validated snapshot RMSE (capped) {r['mean']:.2f} ± {r['sd']:.2f} cycles over "
            f"{m['metrics']['cross_validation']['folds']} folds; differences between models below this spread "
            "are not meaningful."
        )
    else:
        lim.append("Evaluated on a single validation split (no cross-validation); no variability estimate.")
    if m["metrics"].get("official_test") is None:
        lim.append("Not evaluated on the official NASA test set.")
    if m["intervals"] is None:
        lim.append("No calibrated prediction intervals for this model.")
    if m["dataset"] in ("FD002", "FD004"):
        lim.append("Per-operating-condition normalisation assumes 6 settings clusters fitted on training engines.")
    return lim


def write_registry(registry: dict[str, Any], path: str | Path) -> Path:
    """Write the registry JSON."""
    return write_json(path, registry)


def load_registry(path: str | Path) -> dict[str, Any]:
    """Read a registry JSON and check its schema version."""
    reg = read_json(path)
    if reg.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ValueError(f"{path}: registry schema {reg.get('schema_version')} != {REGISTRY_SCHEMA_VERSION}")
    return reg

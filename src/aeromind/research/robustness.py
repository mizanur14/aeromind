"""Robustness of canonical-fold models to input corruption and subset shift.

Perturbations are applied to the **validation engines' raw inputs** only; models are not
retrained. Statistics used to build perturbations (sensor means/SDs) come from the canonical
training engines. Scenarios:

- ``noise``    additive Gaussian noise, σ = k × training SD of each sensor (k configurable).
- ``dropout``  one sensor frozen at its training mean for the whole trajectory.
- ``missing``  a fraction of individual readings removed, then imputed causally
               (per-engine forward fill; leading gaps → training mean).
- ``shift``    FD001-trained models applied, without adaptation, to the validation engines of
               another subset (e.g. FD003: additional fan fault mode). This is a distribution-
               shift stress test, **not** a transfer-learning or zero-shot protocol.

The official test set is never read.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aeromind.data.cmapss import SENSOR_COLUMNS, load_train
from aeromind.data.splits import load_split, verify_split
from aeromind.evaluation.protocol import select_snapshots
from aeromind.models.bundle import load_bundle
from aeromind.research.runner import evaluate_frame
from aeromind.research.spec import load_stage2_config
from aeromind.utils import peak_rss_mb, write_json

logger = logging.getLogger(__name__)


@dataclass
class RobustnessConfig:
    """Robustness experiment configuration."""

    name: str
    comparison_run: str
    models: list[str]
    noise_levels: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    missing_rates: list[float] = field(default_factory=lambda: [0.05, 0.2, 0.5])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    dropout_sensors: list[str] | str = "all_informative"
    shift_datasets: list[str] = field(default_factory=list)
    artifacts_dir: str = "artifacts/stage2"

    @classmethod
    def load(cls, path: str | Path) -> RobustnessConfig:
        """Load from YAML (unknown keys raise)."""
        payload = yaml.safe_load(Path(path).read_text())
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown robustness keys {sorted(unknown)}")
        return cls(**payload)


def add_noise(df: pd.DataFrame, cols: list[str], sd: pd.Series, k: float, seed: int) -> pd.DataFrame:
    """Additive Gaussian noise with per-sensor σ = k·sd."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out[cols] = df[cols].to_numpy(float) + rng.normal(size=(len(df), len(cols))) * (k * sd[cols].to_numpy())
    return out


def freeze_sensor(df: pd.DataFrame, col: str, value: float) -> pd.DataFrame:
    """Replace one sensor by a constant (failed / stuck sensor)."""
    out = df.copy()
    out[col] = value
    return out


def drop_readings(df: pd.DataFrame, cols: list[str], rate: float, seed: int) -> pd.DataFrame:
    """Set a random fraction of individual sensor readings to NaN."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    values = out[cols].to_numpy(dtype=float, copy=True)
    values[rng.random(values.shape) < rate] = np.nan
    out[cols] = values
    return out


def impute_causal(df: pd.DataFrame, cols: list[str], fill: pd.Series) -> pd.DataFrame:
    """Per-engine forward fill (past values only), then training means for leading gaps."""
    out = df.copy()
    out[cols] = out.groupby("engine_id", sort=False)[cols].ffill()
    out[cols] = out[cols].fillna(fill[cols])
    return out


def run_robustness(cfg: RobustnessConfig) -> Path:
    """Evaluate every configured scenario for every model; returns the output directory."""
    comp = Path(cfg.comparison_run)
    s2 = load_stage2_config(comp / "config.yaml")
    ev = s2.evaluation
    pool = load_train(s2.paths.raw_dir, s2.dataset)
    manifest = load_split(s2.split_manifest_path)
    verify_split(manifest, pool.engine_id.unique())
    train = pool[pool.engine_id.isin(manifest.train_engine_ids)]
    val = pool[pool.engine_id.isin(manifest.validation_engine_ids)].reset_index(drop=True)
    snaps = select_snapshots(val, ev.snapshots_per_engine, ev.snapshot_seed)
    mean, sd = train[SENSOR_COLUMNS].mean(), train[SENSOR_COLUMNS].std()
    informative = [c for c in SENSOR_COLUMNS if train[c].nunique() > 3]
    dropout = informative if cfg.dropout_sensors == "all_informative" else list(cfg.dropout_sensors)
    out = Path(cfg.artifacts_dir) / f"{cfg.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    (out / "report").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))

    scenarios: list[tuple[str, str, int | None, pd.DataFrame]] = [("clean", "-", None, val)]
    for k in cfg.noise_levels:
        for s in cfg.seeds:
            scenarios.append(("noise", f"{k}sd", s, add_noise(val, informative, sd, k, s)))
    for c in dropout:
        scenarios.append(("dropout", c, None, freeze_sensor(val, c, float(mean[c]))))
    for r in cfg.missing_rates:
        for s in cfg.seeds:
            scenarios.append(("missing", f"{r:.0%}", s, impute_causal(drop_readings(val, informative, r, s), informative, mean)))

    rows = []
    for name in cfg.models:
        predictor, _ = load_bundle(comp / "models" / f"{name}.joblib")
        cap = predictor.target_cap
        for kind, level, seed, df in scenarios:
            m, _ = evaluate_frame(predictor, df, snaps, cap, ev.capped_truth_cap, ev.near_eol_threshold)
            rows.append({"model": name, "scenario": kind, "level": level, "seed": seed, **m})
        logger.info("Robustness: %s done (%d scenarios)", name, len(scenarios))
        for ds in cfg.shift_datasets:
            other = load_train(s2.paths.raw_dir, ds)
            om = load_split(Path(s2.paths.manifest_dir) / f"{ds.lower()}_split_seed42.json")
            oval = other[other.engine_id.isin(om.validation_engine_ids)].reset_index(drop=True)
            osn = select_snapshots(oval, ev.snapshots_per_engine, ev.snapshot_seed)
            m, _ = evaluate_frame(predictor, oval, osn, cap, ev.capped_truth_cap, ev.near_eol_threshold)
            rows.append({"model": name, "scenario": "shift", "level": f"{s2.dataset}->{ds}", "seed": None, **m})
    table = pd.DataFrame(rows)
    table.to_csv(out / "report" / "robustness_raw.csv", index=False)
    summary = summarise_robustness(table)
    summary.to_csv(out / "report" / "robustness.csv", index=False)
    _write_md(summary, out / "report" / "robustness.md")
    _plot(summary, out / "report" / "robustness.png")
    write_json(out / "metadata.json", {"comparison_run": str(comp), "official_test_labels_read": False, "peak_rss_mb": peak_rss_mb(), "informative_sensors": informative})
    return out


def summarise_robustness(table: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds; Δ vs clean for each model."""
    agg = (
        table.groupby(["model", "scenario", "level"], sort=False)
        .agg(
            n=("snapshot_rmse_capped", "size"),
            rmse_capped=("snapshot_rmse_capped", "mean"),
            rmse_capped_sd=("snapshot_rmse_capped", "std"),
            rmse_true=("snapshot_rmse_true", "mean"),
            near_eol_rmse_true=("near_eol_rmse_true", "mean"),
        )
        .reset_index()
    )
    clean = agg[agg.scenario == "clean"].set_index("model")["rmse_capped"]
    agg["delta_rmse_capped"] = agg["rmse_capped"] - agg["model"].map(clean)
    agg["relative_increase"] = agg["delta_rmse_capped"] / agg["model"].map(clean)
    return agg


def _write_md(summary: pd.DataFrame, path: Path) -> None:
    lines = ["| Model | Scenario | Level | Runs | Snapshot RMSE capped | Δ vs clean | Snapshot RMSE true | Near-EOL RMSE true |", "|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        sd = f" ± {r.rmse_capped_sd:.2f}" if r.n > 1 and pd.notna(r.rmse_capped_sd) else ""
        lines.append(f"| {r.model} | {r.scenario} | {r.level} | {r.n} | {r.rmse_capped:.2f}{sd} | {r.delta_rmse_capped:+.2f} | {r.rmse_true:.2f} | {r.near_eol_rmse_true:.2f} |")
    path.write_text("\n".join(lines) + "\n")


def _plot(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    from aeromind.evaluation import plots as P

    sel = summary[summary.scenario.isin(["noise", "missing"])]
    models = list(dict.fromkeys(summary.model))[:3]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, scen in zip(axes, ["noise", "missing"], strict=True):
        d = sel[sel.scenario == scen]
        levels = list(dict.fromkeys(d.level))
        for i, m in enumerate(models):
            v = d[d.model == m].set_index("level").reindex(levels)["rmse_capped"]
            ax.plot(range(len(levels)), v.to_numpy(), marker="o", ms=7, color=P.PALETTE[i], label=m)
        ax.set_xticks(range(len(levels)), levels)
        ax.set(title=f"{scen}", xlabel="level")
    axes[0].set_ylabel("Validation snapshot RMSE, capped (cycles)")
    axes[0].legend()
    fig.suptitle("Robustness to input corruption (models not retrained)", fontsize=11)
    P.save(fig, path)


def summary_dict(out_dir: str | Path) -> dict[str, Any]:
    """Load the summary table as records."""
    return pd.read_csv(Path(out_dir) / "report" / "robustness.csv").to_dict(orient="records")

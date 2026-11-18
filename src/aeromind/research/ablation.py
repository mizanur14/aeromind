"""One-factor-at-a-time ablations compiled into a Stage-2 comparison run.

Each variant changes exactly one thing relative to its base model and is trained/evaluated on
the same engine-level folds, targets and snapshots. Results are paired per fold against the
base model. No exhaustive grid search.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aeromind.research.comparison import run_comparison
from aeromind.research.spec import Stage2Config, load_stage2_config

logger = logging.getLogger(__name__)

METRICS = ["snapshot_rmse_capped", "snapshot_rmse_true", "snapshot_mae_capped", "near_eol_rmse_true", "snapshot_phm08_mean_capped"]


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def compile_ablation(path: str | Path) -> tuple[Stage2Config, dict[str, tuple[str, str]]]:
    """Expand an ablation YAML into a Stage-2 config; return it with ``{model: (base, variant)}``."""
    spec = yaml.safe_load(Path(path).read_text())
    allowed = {"name", "base_config", "run_folds", "base_models", "variants"}
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"Unknown ablation keys {sorted(unknown)}")
    base_cfg = load_stage2_config(spec["base_config"]).to_dict()
    models: dict[str, Any] = {}
    lineage: dict[str, tuple[str, str]] = {}
    for base_name, base_entry in spec["base_models"].items():
        models[base_name] = base_entry
        lineage[base_name] = (base_name, "base")
        for v in spec["variants"]:
            if base_name not in v["applies_to"]:
                continue
            patch = {k: v[k] for k in ("deep", "params", "features", "target_cap", "family") if k in v}
            if base_entry["kind"] == "deep" and "features" in patch:
                patch.setdefault("deep", {})["features"] = patch.pop("features")
            name = f"{base_name}__{v['name']}"
            models[name] = _merge(base_entry, patch)
            lineage[name] = (base_name, v["name"])
    payload = {**base_cfg, "name": spec["name"], "models": models, "reference_model": None}
    payload["cv"] = {**base_cfg["cv"], "run_folds": spec.get("run_folds")}
    return Stage2Config.from_dict(payload), lineage


def run_ablation(path: str | Path, run_dir: str | Path | None = None, isolate: bool = True) -> Path:
    """Run all variants and write ``report/ablation.csv`` / ``ablation.md``."""
    cfg, lineage = compile_ablation(path)
    out = run_comparison(cfg, run_dir, isolate=isolate)
    (out / "report" / "lineage.yaml").write_text(yaml.safe_dump({k: list(v) for k, v in lineage.items()}))
    summarise_ablation(out, lineage)
    return out


def summarise_ablation(run_dir: str | Path, lineage: dict[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    """Per-variant mean ± SD over folds and paired difference vs its base model."""
    run_dir = Path(run_dir)
    if lineage is None:
        lineage = {k: tuple(v) for k, v in yaml.safe_load((run_dir / "report" / "lineage.yaml").read_text()).items()}
    folds = pd.read_csv(run_dir / "report" / "fold_metrics.csv")
    rows = []
    for model, (base, variant) in lineage.items():
        g = folds[folds.model == model].set_index("fold")
        b = folds[folds.model == base].set_index("fold")
        if g.empty:
            continue
        row: dict[str, Any] = {"base": base, "variant": variant, "model": model, "folds": len(g), "target_cap": g["target_cap"].iloc[0]}
        for m in METRICS:
            if m in g:
                row[f"{m}_mean"] = g[m].mean()
                row[f"{m}_sd"] = g[m].std(ddof=1) if len(g) > 1 else np.nan
        if variant != "base":
            d = (g["snapshot_rmse_capped"] - b["snapshot_rmse_capped"]).dropna()
            dt = (g["snapshot_rmse_true"] - b["snapshot_rmse_true"]).dropna()
            row.update(
                {
                    "diff_rmse_capped_mean": d.mean(),
                    "diff_rmse_capped_sd": d.std(ddof=1) if len(d) > 1 else np.nan,
                    "diff_rmse_true_mean": dt.mean(),
                    "folds_better_capped": f"{int((d < 0).sum())}/{len(d)}",
                }
            )
        row["fit_seconds_mean"] = g["fit_fit_seconds"].mean() if "fit_fit_seconds" in g else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "report" / "ablation.csv", index=False)
    lines = [
        "| Base | Variant | Cap | Snapshot RMSE capped | Δ vs base (capped) | Folds better | Snapshot RMSE true | Δ true | Near-EOL RMSE true | Fit s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        def ms(m: str, r: pd.Series = r) -> str:
            sd = r.get(f"{m}_sd", np.nan)
            return f"{r[f'{m}_mean']:.2f} ± {sd:.2f}" if pd.notna(sd) else f"{r[f'{m}_mean']:.2f}"

        diff = f"{r['diff_rmse_capped_mean']:+.2f} ± {r['diff_rmse_capped_sd']:.2f}" if r["variant"] != "base" else "—"
        dtrue = f"{r['diff_rmse_true_mean']:+.2f}" if r["variant"] != "base" else "—"
        cap = "none" if pd.isna(r["target_cap"]) else int(r["target_cap"])
        lines.append(
            f"| {r['base']} | {r['variant']} | {cap} | {ms('snapshot_rmse_capped')} | {diff} | {r.get('folds_better_capped', '—') if r['variant'] != 'base' else '—'} | {ms('snapshot_rmse_true')} | {dtrue} | {ms('near_eol_rmse_true')} | {r['fit_seconds_mean']:.1f} |"
        )
    (run_dir / "report" / "ablation.md").write_text("\n".join(lines) + "\n")
    return table

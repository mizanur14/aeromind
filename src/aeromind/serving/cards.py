"""Render model cards (Markdown) from the registry, so card numbers equal the saved runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aeromind.serving.registry import load_registry

INTENDED_USE = (
    "Research and education on remaining-useful-life estimation with the NASA C-MAPSS simulation "
    "benchmark: method comparison, uncertainty-quantification studies, demonstrations."
)
OUT_OF_SCOPE = (
    "Any operational, airworthiness, maintenance-scheduling or safety decision on real aircraft or "
    "engines; any data other than C-MAPSS-format simulation trajectories of the training subset."
)


def _f(v: Any, nd: int = 2) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def _metrics_table(m: dict[str, Any]) -> list[str]:
    cv = (m["metrics"].get("cross_validation") or {}).get("values")
    can = (m["metrics"].get("canonical_validation") or {}).get("values") or {}
    rows = [
        ("Snapshot RMSE vs capped RUL", "snapshot_rmse_capped"),
        ("Snapshot MAE vs capped RUL", "snapshot_mae_capped"),
        ("Snapshot PHM08 (mean per snapshot, capped)", "snapshot_phm08_mean_capped"),
        ("Snapshot RMSE vs true RUL", "snapshot_rmse_true"),
        ("Near-EOL RMSE vs true RUL (true RUL ≤ 30)", "near_eol_rmse_true"),
    ]
    head = "| Metric (cycles) | " + ("CV mean ± SD | " if cv else "") + "Canonical validation |"
    out = [head, "|---|" + ("---|" if cv else "") + "---|"]
    for label, k in rows:
        cvs = f"{_f(cv[k]['mean'])} ± {_f(cv[k]['sd'])} | " if cv and cv.get(k) and cv[k]["mean"] is not None else ("— | " if cv else "")
        out.append(f"| {label} | {cvs}{_f(can.get(k))} |")
    return out


def render_card(m: dict[str, Any]) -> str:
    """One model card."""
    lines = [f"### `{m['id']}`: {m['family']} ({m['dataset']}, {m['stage']})", ""]
    lines += [
        f"- **Version:** `{m['version']}` (first 12 hex of bundle SHA-256 `{m['bundle']['sha256']}`), "
        f"{m['bundle']['bytes'] / 2**20:.2f} MB, bundle format {m['bundle']['format_version']}",
        f"- **Source run:** `{m['source_run']}` (finished {m.get('run_finished_utc')}, git `{m.get('git_commit_at_training')}`); reports `{m['report_dir']}`",
        f"- **Type:** {m['kind']} ({m['framework']})" + (f", {m['n_parameters']:,} parameters" if m.get("n_parameters") else ""),
        f"- **Target:** `min(RUL, {m['target_cap']})` cycles; predictions clipped to [0, {m['target_cap']}]",
        f"- **Training data:** {m['training']['n_engines']} {m['dataset']} training-file engines "
        f"(split `{m['training']['split_manifest']}`, seed {m['training']['seed']}); validation and calibration engines excluded",
        f"- **Inputs:** {len(m['input']['used_columns'])} sensors/settings used of the 24 required raw columns; "
        + (f"causal window of {m['input']['sequence_window']} cycles" if m["input"].get("sequence_window") else f"causal rolling windows {m['input']['rolling_windows']}")
        + ("; per-operating-condition normalisation" if m["input"]["condition_normalization"] else ""),
        "",
        "**Validation metrics** (engine-level; official test set not used unless stated):",
        "",
        *_metrics_table(m),
        "",
    ]
    t = m["metrics"].get("official_test")
    if t:
        v = t["values"]
        lines += [
            f"**Official NASA test set** ({t['n_test_engines']} engines, {t['protocol']}, {t['evaluated_utc']}): "
            f"RMSE (true) {_f(v['last_cycle_rmse_true'])}, MAE (true) {_f(v['last_cycle_mae_true'])}, "
            f"PHM08 sum {_f(v['last_cycle_phm08_sum_true'], 1)}, RMSE (capped) {_f(v['last_cycle_rmse_capped'])}.",
            "",
        ]
    res = m.get("resources") or {}
    if res:
        lines += [
            f"**CPU resources** (Stage-2 measurement): fit {_f(res.get('fit_seconds_mean'), 1)} s, single-input latency "
            f"{_f(res.get('single_input_ms'))} ms, peak RSS {_f(res.get('peak_rss_mb'), 0)} MB.",
            "",
        ]
    iv = m.get("intervals")
    if iv:
        th = ", ".join(f"{k}: ±{_f(v)}" for k, v in iv["thresholds"].items())
        lines += [f"**Prediction intervals:** {iv['method']} on {iv['n_calibration_scores']} calibration engines (seed {iv['seed']}); half-widths {th} cycles."]
        val = (iv.get("validated_coverage") or {}).get("levels") or {}
        for k, v in val.items():
            lines.append(
                f"  - Level {k}, same procedure over {v['draws']} draws on validation engines: coverage of capped RUL "
                f"{_f(v['coverage_capped_mean'], 3)} (p05 {_f(v['coverage_capped_p05'], 3)}, p95 {_f(v['coverage_capped_p95'], 3)}), "
                f"of true RUL {_f(v['coverage_true_rul_mean'], 3)}; mean width {_f(v['width_mean'], 1)}."
            )
        lines += [f"  - {iv['guarantee']}", ""]
    lines += ["**Limitations:**", "", *[f"- {s}" for s in m["limitations"]], ""]
    return "\n".join(lines)


def render_cards(registry_path: str | Path) -> str:
    """All cards, grouped by subset."""
    reg = load_registry(registry_path)
    parts = [
        "# Model cards",
        "",
        f"Generated by `aeromind s3-model-cards` from `{Path(registry_path).as_posix()}` (built {reg['built_utc']}, "
        f"git `{reg.get('git_commit')}`). Every number is copied from a saved run report; do not edit by hand.",
        "",
        f"**Intended use.** {INTENDED_USE}",
        "",
        f"**Out of scope.** {OUT_OF_SCOPE}",
        "",
        "**Data.** NASA C-MAPSS turbofan simulation (Saxena et al., 2008). Simulation-generated; not flight data. "
        "Results are not certified for operational aviation safety. AeroMind is independent and not endorsed by NASA.",
        "",
        f"**Default demo model:** `{reg.get('default_model')}`. {reg.get('default_model_rationale', '')}",
        "",
    ]
    for ds in sorted({m["dataset"] for m in reg["models"]}):
        parts += [f"## {ds}", ""]
        parts += [render_card(m) for m in reg["models"] if m["dataset"] == ds]
    return "\n".join(parts)

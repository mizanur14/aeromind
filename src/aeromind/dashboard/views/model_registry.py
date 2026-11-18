"""E. Model Registry: the catalogue of registered models (metadata only; nothing is loaded)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from aeromind.dashboard import data as D
from aeromind.dashboard import state as S
from aeromind.dashboard import ui
from aeromind.dashboard.state import Ctx


def _evidence(m: dict[str, Any]) -> str:
    ms = m.get("metrics") or {}
    parts = [label for key, label in D.EVIDENCE_KEYS.items() if ms.get(key)]
    if m.get("intervals"):
        parts.append("Conformal intervals")
    return " · ".join(parts) or "—"


def _detail(ctx: Ctx, m: dict[str, Any]) -> None:
    st.markdown(f"#### {m['id']}  ·  {m['family']}")
    k = st.columns(4)
    k[0].metric("Subset", m["dataset"])
    k[1].metric("Stage", m["stage"])
    k[2].metric("Training engines", (m.get("training") or {}).get("n_engines", "—"))
    k[3].metric("Target", f"min(RUL, {m['target_cap']})" if m.get("target_cap") else "RUL")

    rows = []
    for key, label in D.EVIDENCE_KEYS.items():
        block = (m.get("metrics") or {}).get(key)
        if not block:
            continue
        for metric, v in (block.get("values") or {}).items():
            mean, sd = (v.get("mean"), v.get("sd")) if isinstance(v, dict) else (v, None)
            rows.append({"Evidence": label, "Metric": D.METRICS.get(metric, (metric, ""))[0], "Value": mean, "Fold SD": float("nan") if sd is None else sd})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={"Value": st.column_config.NumberColumn(format="%.3f"), "Fold SD": st.column_config.NumberColumn(format="%.3f")})
        if any(r["Evidence"] == "Official NASA test" for r in rows):
            st.caption("Official-test values use one prediction per engine at its last observed cycle; not comparable with validation snapshots.")

    cal = m.get("intervals")
    if cal:
        st.markdown("**Conformal calibrator** (fitted on calibration engines only)")
        lv = ((cal.get("validated_coverage") or {}).get("levels")) or {}
        cal_rows = [
            {"Level": k2, "Half-width (cycles)": q, "Mean validated coverage (capped)": (lv.get(k2) or {}).get("coverage_capped_mean"), "p05 over draws": (lv.get(k2) or {}).get("coverage_capped_p05"), "Coverage of true RUL": (lv.get(k2) or {}).get("coverage_true_rul_mean")}
            for k2, q in sorted(cal.get("thresholds", {}).items())
        ]
        st.dataframe(pd.DataFrame(cal_rows), hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ("Half-width (cycles)", "Mean validated coverage (capped)", "p05 over draws", "Coverage of true RUL")})
        st.caption(f"{cal.get('n_calibration_scores')} calibration engines · method `{cal.get('method')}` · {cal.get('guarantee', '')}")

    inp = m.get("input") or {}
    res = m.get("resources") or {}
    meta = {
        "version": m.get("version"),
        "framework": m.get("framework"),
        "kind": m.get("kind"),
        "requires": m.get("requires") or [],
        "source_run": m.get("source_run"),
        "report_dir": m.get("report_dir"),
        "bundle": {"path": m["bundle"]["path"], "sha256": m["bundle"]["sha256"], "bytes": m["bundle"]["bytes"], "format_version": m["bundle"].get("format_version")},
        "input": {"required_columns": len(inp.get("required_columns") or []), "used_columns": inp.get("used_columns"), "condition_normalization": inp.get("condition_normalization"), "rolling_windows": inp.get("rolling_windows"), "sequence_window": inp.get("sequence_window"), "n_features": inp.get("n_features")},
        "resources": res,
        "split_manifest": (m.get("training") or {}).get("split_manifest"),
        "split_manifest_sha256": (m.get("training") or {}).get("split_manifest_sha256"),
    }
    ui.provenance_box(meta, "Registry metadata (provenance, input specification, resources)")
    lim = m.get("limitations") or []
    if lim:
        with st.expander("Stated limitations"):
            st.markdown("\n".join(f"- {x}" for x in lim))


def render(ctx: Ctx) -> None:
    ui.hero("Stage 3 · catalogue", "Model Registry", "Every registered model with its evidence, calibration and provenance, exactly as recorded in the committed registry.")
    ui.badge("registry")
    if not ctx.registry:
        ui.unavailable("The model registry", "It is built locally with `uv run aeromind s3-registry`.")
        return
    if ctx.public:
        st.info(
            "This is a catalogue. The model files are not distributed with the repository and are **not loaded** by this website; "
            "no prediction is generated here. Each entry pins its model file by SHA-256 so a local copy can be verified before use."
        )
    avail = S.availability(ctx) if not ctx.public else None
    df = D.registry_table(ctx.registry, avail)
    by_id = {m["id"]: m for m in ctx.registry["models"]}
    df["evidence"] = df.model_id.map(lambda i: _evidence(by_id[i]))
    df["_rank"] = df.model_id.map(ui.family_rank)
    df = df.sort_values(["dataset", "stage", "_rank"]).drop(columns="_rank")

    f = st.columns(4)
    ds = f[0].multiselect("Subset", sorted(df.dataset.unique()), key="rg_ds", placeholder="All subsets")
    stage = f[1].multiselect("Stage", sorted(df.stage.unique()), key="rg_stage", placeholder="All stages")
    fam = f[2].multiselect("Family", sorted(df.family.unique()), key="rg_family", placeholder="All families")
    only_iv = f[3].toggle("Calibrated intervals only", key="rg_iv")
    view = df
    if ds:
        view = view[view.dataset.isin(ds)]
    if stage:
        view = view[view.stage.isin(stage)]
    if fam:
        view = view[view.family.isin(fam)]
    if only_iv:
        view = view[view.intervals]

    k = st.columns(4)
    k[0].metric("Models shown", f"{len(view)} of {len(df)}")
    k[1].metric("With cross-validation", int(view.cv_rmse_capped_mean.notna().sum()))
    k[2].metric("With calibrated intervals", int(view.intervals.sum()))
    k[3].metric("With an official test result", int(view.official_test_rmse_true.notna().sum()))

    cols = {
        "model_id": "Registry id",
        "dataset": "Subset",
        "stage": "Stage",
        "family": "Family",
        "kind": "Kind",
        "evidence": "Evidence available",
        "cv_rmse_capped_mean": "CV RMSE (capped)",
        "canonical_rmse_capped": "Canonical RMSE (capped)",
        "interval_levels": "Interval levels",
        "n_training_engines": "Training engines",
        "bundle_mb": "Bundle MB",
        "bundle_sha256_short": "Bundle SHA-256 (prefix)",
        "requires_torch": "Needs PyTorch",
        "source_run": "Source run",
    }
    if avail is not None:
        cols["available_locally"] = "Available locally"
    st.dataframe(
        view[list(cols)].rename(columns=cols),
        hide_index=True,
        width="stretch",
        column_config={"CV RMSE (capped)": st.column_config.NumberColumn(format="%.2f"), "Canonical RMSE (capped)": st.column_config.NumberColumn(format="%.2f"), "Bundle MB": st.column_config.NumberColumn(format="%.2f")},
    )
    reg = ctx.registry
    default = reg.get("default_model")
    if default:
        st.caption(f"Demo default model: **{default}**. {reg.get('default_model_rationale', '')}")
    ui.source(ctx.show(ctx.paths.registry), f"Built {reg.get('built_utc', '?')}; metrics copied from run reports, official-test metrics only from the ledger. Human-readable model cards: docs/MODEL_CARDS.md")

    st.subheader("Model details")
    ids = view.model_id.tolist()
    if not ids:
        st.caption("No model matches the filters.")
        return
    pick = st.selectbox("Registered model", ids, index=ids.index(default) if default in ids else 0, key="rg_pick")
    _detail(ctx, by_id[pick])
    st.markdown(f"[All model cards on GitHub]({ui.doc_url('docs/MODEL_CARDS.md')})")

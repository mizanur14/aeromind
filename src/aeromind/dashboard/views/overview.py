"""A. Research Overview: what AeroMind is, the data, objectives and validated outputs."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from aeromind.dashboard import data as D
from aeromind.dashboard import state as S
from aeromind.dashboard import ui
from aeromind.dashboard.state import Ctx


def _rul_definition_chart() -> alt.Chart:
    """Definition of the training target for a hypothetical 200-cycle engine (no data)."""
    life, cap = 200, 125
    t = np.arange(0, life + 1)
    df = pd.concat(
        [
            pd.DataFrame({"cycle": t, "value": life - t, "series": "True RUL = failure cycle − current cycle"}),
            pd.DataFrame({"cycle": t, "value": np.minimum(life - t, cap), "series": f"Training target min(RUL, {cap})"}),
        ]
    )
    dom = list(dict.fromkeys(df.series))
    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("cycle:Q", title="Operating cycle (hypothetical 200-cycle engine)"),
            y=alt.Y("value:Q", title="Remaining useful life (cycles)"),
            color=alt.Color("series:N", scale=alt.Scale(domain=dom, range=[ui.TEXT2, ui.FAMILY_COLOR["hgb_reference"]]), title=None),
            strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=dom, range=[[5, 4], [1, 0]]), title=None),
            tooltip=["series", "cycle", "value"],
        )
        .properties(height=330, title="Definition only: true RUL and the capped training target (not data)")
    )


def _headline_cv(ctx: Ctx) -> pd.DataFrame | None:
    reg = ctx.registry
    if not reg:
        return None
    cv = D.metric_frame(reg, "FD001", "cross_validation")
    if cv.empty:
        return None
    wide = cv.pivot_table(index="model_id", columns="metric", values=["value", "sd"])
    paired = S.csv_file(ctx.paths.report("comparison", "FD001"), "paired_vs_reference.csv")
    rows = []
    for mid in ui.order_models(list(wide.index)):
        r = {"Model": ui.pretty_family(mid), "Registry id": mid}
        r["CV snapshot RMSE, capped (mean ± SD)"] = f"{wide.loc[mid, ('value', 'snapshot_rmse_capped')]:.2f} ± {wide.loc[mid, ('sd', 'snapshot_rmse_capped')]:.2f}"
        if paired is not None:
            p = paired[paired.model == mid.split("-", 2)[-1]]
            r["Paired Δ RMSE vs HGB (folds better)"] = f"{p.mean_diff_rmse_capped.iloc[0]:+.2f} ({int(p.folds_better_than_reference.iloc[0])}/{int(p.folds.iloc[0])})" if not p.empty else "reference"
        if ("value", "near_eol_rmse_true") in wide.columns:
            r["Near-EOL RMSE, true RUL ≤ 30"] = f"{wide.loc[mid, ('value', 'near_eol_rmse_true')]:.2f}"
        rows.append(r)
    return pd.DataFrame(rows)


def _findings(ctx: Ctx) -> list[str]:
    """Key findings, each computed from a committed report at render time."""
    out = []
    s1 = S.csv_file(ctx.paths.report("stage1", "FD001"), "validation_model_comparison.csv")
    if s1 is not None and {"mean_residual_life", "hist_gradient_boosting"} <= set(s1.model):
        mrl = s1.loc[s1.model == "mean_residual_life", "snapshot_rmse_capped"].iloc[0]
        learned = s1.loc[~s1.model.isin(["mean_residual_life", "median"]), "snapshot_rmse_capped"].min()
        out.append(
            f"**Sensors carry information beyond engine age.** On the 15 canonical validation engines, the best learned model's "
            f"capped snapshot RMSE ({learned:.2f}) is {mrl / learned:.1f}× lower than an age-only baseline ({mrl:.2f}). "
            f"*Source: `{ctx.show(ctx.paths.report('stage1', 'FD001'))}`*"
        )
    paired = S.csv_file(ctx.paths.report("comparison", "FD001"), "paired_vs_reference.csv")
    ref = next((m for m in (ctx.registry or {}).get("models", []) if m["id"] == "fd001-s2-hgb_reference"), None)
    ref_sd = ((((ref or {}).get("metrics") or {}).get("cross_validation") or {}).get("values") or {}).get("snapshot_rmse_capped", {}).get("sd")
    if paired is not None and ref_sd:
        within = paired[paired.mean_diff_rmse_capped.abs() < ref_sd]
        if not within.empty:
            names = ", ".join(f"{ui.FAMILY_LABEL.get(m, m)} ({d:+.2f})" for m, d in zip(within.model, within.mean_diff_rmse_capped, strict=True))
            out.append(
                f"**No architecture is shown to be superior on FD001.** Mean paired differences against the HistGradientBoosting "
                f"reference ({names} cycles) are smaller than the reference's own fold-to-fold SD of {ref_sd:.2f} cycles. "
                f"*Source: `{ctx.show(ctx.paths.report('comparison', 'FD001'))}`*"
            )
    iv = S.csv_file(ctx.paths.report("uq"), "intervals_main.csv")
    if iv is not None:
        x = iv[(iv.method == "conformal_engine") & (iv.calibrate_on == "calibration") & np.isclose(iv.nominal, 0.9)]
        if not x.empty:
            out.append(
                f"**Calibrated 90 % intervals hold on average, with wide draw-to-draw spread.** Engine-level split conformal on 15 "
                f"calibration engines gave mean validation coverage {x.coverage_snapshot_capped_mean.min():.2f}–{x.coverage_snapshot_capped_mean.max():.2f} "
                f"(5th percentile over 500 draws {x.coverage_snapshot_capped_p05.min():.2f}–{x.coverage_snapshot_capped_p05.max():.2f}) for capped RUL. "
                f"*Source: `{ctx.show(ctx.paths.report('uq'))}`*"
            )
    rb = S.csv_file(ctx.paths.report("robustness"), "robustness.csv")
    if rb is not None:
        sh = rb[rb.scenario == "shift"]
        if not sh.empty:
            out.append(
                f"**Models do not transfer across fault modes.** Applied unchanged to FD003 engines (an extra fan fault mode), FD001 models' "
                f"capped RMSE rose by {sh.relative_increase.min():.0%} to {sh.relative_increase.max():.0%}. "
                f"*Source: `{ctx.show(ctx.paths.report('robustness'))}`*"
            )
    return out


def render(ctx: Ctx) -> None:
    ui.hero(
        "NASA C-MAPSS · Remaining Useful Life research",
        "AeroMind",
        "Uncertainty-aware prediction of how many operating cycles a turbofan engine has left, evaluated with "
        "leakage-safe, engine-level protocols and reported with full provenance.",
    )
    ui.disclaimer()

    reg = ctx.registry or {"models": []}
    models = reg["models"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Registered models", len(models), help="Entries in the committed model registry (catalogue only; no model is loaded by this website).")
    k2.metric("C-MAPSS subsets benchmarked", len({m["dataset"] for m in models}))
    k3.metric("Models with calibrated intervals", sum(m["intervals"] is not None for m in models), help="Split-conformal calibrators built on the 15 reserved FD001 calibration engines.")
    k4.metric("Official test evaluations", len((ctx.ledger or {}).get("entries", [])), help="Every evaluation ever run on the official NASA test labels is recorded in the ledger.")

    st.subheader("What is Remaining Useful Life?")
    c1, c2 = st.columns([5, 6], gap="large")
    with c1:
        st.markdown(
            "**Remaining Useful Life (RUL)** is the number of operating cycles an engine can still run before it reaches "
            "its failure threshold. AeroMind estimates RUL from an engine's history of sensor readings, using only the "
            "cycles observed so far.\n\n"
            "Early in life, engines look healthy and their true RUL cannot be identified from the sensors. Following the "
            "literature convention, models are therefore trained on a **capped target** `min(RUL, 125)`: a prediction of "
            "125 means *no detectable degradation yet*, not *125 cycles left*.\n\n"
            "Every result is reported against **both** capped truth (`_capped`) and true uncapped RUL (`_true`); the two "
            "are never compared with each other."
        )
    with c2:
        st.altair_chart(_rul_definition_chart(), width="stretch")

    st.subheader("The NASA C-MAPSS benchmark")
    ui.badge("truth")
    st.markdown(
        "C-MAPSS (Saxena et al., 2008) contains **simulated** run-to-failure trajectories generated by NASA's Commercial "
        "Modular Aero-Propulsion System Simulation: 21 sensors and 3 operating settings per cycle. It is **not** flight or "
        "test-cell data. Training engines run to failure; test engines are truncated at undocumented points."
    )
    info = pd.DataFrame([{"Subset": k, "Operating conditions": v["conditions"], "Fault modes": v["fault_modes"], "Train engines": v["train_engines"], "Test engines": v["test_engines"], "Train rows": v["train_rows"], "Train life (cycles)": v["train_life_cycles"]} for k, v in D.DATASET_INFO.items()])
    st.dataframe(info, hide_index=True, width="stretch")
    ui.source("docs/DATA.md", "Counts measured from the NASA files; the files themselves are not redistributed")

    st.subheader("Research objectives")
    o = st.columns(4)
    with o[0]:
        ui.card("Leakage-free baselines", "Engine-level splits, causal features and validation-only model selection for reproducible RUL baselines.")
    with o[1]:
        ui.card("Calibrated uncertainty", "Prediction intervals calibrated on reserved engines, with their finite-sample guarantee stated precisely.")
    with o[2]:
        ui.card("Transparent demonstration", "A hash-pinned model registry, API and this dashboard, never presented as operational or certified.")
    with o[3]:
        ui.card("Honest reporting", "Every number comes from a saved run; fold variability and evidence type are shown with every claim.")

    st.subheader("Research stages")
    s = st.columns(3)
    with s[0]:
        ui.card("Stage 1 · complete", "Validated data layer, engine-level splits, causal rolling features, six classical baselines, one ledgered official FD001 test evaluation.")
    with s[1]:
        ui.card("Stage 2 · complete", "TCN, GRU and Transformer; 5-fold engine-level CV; split-conformal and quantile UQ; ablations; robustness; FD002–FD004 single-split benchmarks.")
    with s[2]:
        ui.card("Stage 3 · this platform", "Model registry with hash-pinned bundles and calibrators, a local REST API and this dashboard. Not done: multi-seed deep runs, nested tuning, CV and UQ beyond FD001.")

    st.subheader("Validated outputs at a glance (FD001)")
    ui.badge("cv")
    head = _headline_cv(ctx)
    if head is None:
        ui.unavailable("The FD001 cross-validation summary")
    else:
        st.dataframe(head, hide_index=True, width="stretch", column_config={"Model": st.column_config.TextColumn(width="medium")})
        ui.source([ctx.show(ctx.paths.registry), ctx.show(ctx.paths.report("comparison", "FD001"))], "5 engine-level folds of 17 engines from the 85 development engines; capped truth min(RUL, 125); paired Δ is per fold, negative = lower error than the reference")

    test = D.official_test_table(reg, ctx.ledger) if models else pd.DataFrame()
    if not test.empty:
        ui.badge("test")
        for _, r in test.iterrows():
            st.markdown(
                f"**{ui.pretty_family(r.model_id)}**, registry id `{r.model_id}`, selected on validation before testing: RMSE **{r.rmse_true:.2f}**, "
                f"MAE {r.mae_true:.2f}, PHM08 sum {r.phm08_sum_true:.1f} against true RUL on {int(r.test_engines)} official test engines, "
                "one prediction per engine at its last observed cycle. A different protocol from the validation snapshots, so "
                "not comparable column-for-column. No Stage-2 model has been evaluated on the official test set."
            )
        ui.source(ctx.show(ctx.paths.report("test_set_ledger")))

    st.subheader("Key findings")
    findings = _findings(ctx)
    if findings:
        for f in findings:
            st.markdown(f"- {f}")
    else:
        ui.unavailable("The committed result reports")

    st.subheader("Limitations you should know")
    st.markdown(
        "- Simulation data only; behaviour on real engines is unknown.\n"
        "- Few engines: 15–17 per validation fold and 15 for calibration. Fold-to-fold spread (about ±2 cycles RMSE) exceeds most differences between models.\n"
        "- Intervals target **capped** RUL, have marginal (not per-engine) coverage, and are invalid under distribution shift.\n"
        "- AeroMind is not a physical digital twin and is not certified for any operational use.\n\n"
        f"Full list: [docs/LIMITATIONS.md]({ui.doc_url('docs/LIMITATIONS.md')})"
    )

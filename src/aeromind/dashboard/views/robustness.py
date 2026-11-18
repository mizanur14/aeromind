"""C. Robustness & Reliability: uncertainty calibration, input corruption, shift, ablations.

All results are FD001 validation/calibration-engine results copied from committed Stage-2
reports; nothing is recomputed here.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from aeromind.dashboard import state as S
from aeromind.dashboard import ui
from aeromind.dashboard.state import Ctx

METHOD_INFO = pd.DataFrame(
    [
        {"Method": "conformal_engine", "Calibration scores": "One random snapshot per calibration engine (n = 15), repeated over 500 draws", "Finite-sample guarantee": "Marginal coverage ≥ 1−α for a snapshot from a new engine exchangeable with the calibration engines"},
        {"Method": "conformal_pooled", "Calibration scores": "All ~3,000 calibration windows", "Finite-sample guarantee": "None: windows within an engine are dependent (heuristic)"},
        {"Method": "quantile_raw", "Calibration scores": "None (HGB or deep quantile regression)", "Finite-sample guarantee": "None"},
        {"Method": "cqr_engine / cqr_pooled", "Calibration scores": "Conformalised quantile regression scores, same two schemes", "Finite-sample guarantee": "As for the matching conformal scheme"},
    ]
)


def _fd001(model: str) -> str:
    return f"fd001-s2-{model}"


def _uq_tab(ctx: Ctx) -> None:
    ui.badge("calibration", "canonical")
    rep = ctx.paths.report("uq")
    iv = S.csv_file(rep, "intervals_main.csv")
    if iv is None:
        ui.unavailable("The uncertainty-quantification report")
        return
    st.markdown(
        "Intervals are calibrated on the **15 reserved calibration engines** and evaluated on the 15 canonical validation "
        "engines. They target **capped** RUL `min(RUL, 125)`, the quantity the models estimate. Because a threshold is one "
        "of only 15 scores, realised coverage depends on which snapshot is drawn per engine; the p05–p95 range over 500 "
        "draws shows that spread."
    )
    ui.table(METHOD_INFO)

    ce = iv[(iv.method == "conformal_engine") & (iv.calibrate_on == "calibration")].copy()
    if not ce.empty:
        levels = sorted(ce.nominal.unique(), reverse=True)
        lvl = st.radio("Nominal level", levels, horizontal=True, key="uq_level", format_func=lambda v: f"{v:.0%}")
        d = ce[np.isclose(ce.nominal, lvl)].assign(model_id=lambda x: x.model.map(_fd001))
        d["label"] = d.model_id.map(ui.pretty_family)
        ids = ui.order_models(d.model_id.tolist())
        lo = max(0.0, np.floor((min(d.coverage_snapshot_capped_p05.min(), lvl) - 0.05) * 20) / 20)
        y = alt.Y("model_id:N", sort=ids, title=None, axis=alt.Axis(labelExpr="split(datum.label, '-')[2]"))
        rule = alt.Chart(d).mark_rule(strokeWidth=2, color=ui.TEXT2).encode(y=y, x=alt.X("coverage_snapshot_capped_p05:Q", title="Coverage of capped RUL on validation engines", scale=alt.Scale(domain=[lo, 1.0])), x2="coverage_snapshot_capped_p95:Q")
        pts = alt.Chart(d).mark_point(size=110, filled=True, stroke=ui.SURFACE, strokeWidth=2).encode(
            y=y,
            x="coverage_snapshot_capped_mean:Q",
            color=alt.Color("model_id:N", scale=ui.color_scale(ids), legend=None),
            tooltip=[alt.Tooltip("label:N", title="Model"), alt.Tooltip("coverage_snapshot_capped_mean:Q", title="Mean coverage", format=".3f"), alt.Tooltip("coverage_snapshot_capped_p05:Q", title="p05", format=".3f"), alt.Tooltip("coverage_snapshot_capped_p95:Q", title="p95", format=".3f"), alt.Tooltip("coverage_snapshot_true_mean:Q", title="Coverage of true RUL", format=".3f"), alt.Tooltip("width_snapshot_mean:Q", title="Mean width (cycles)", format=".1f")],
        )
        nominal = alt.Chart(pd.DataFrame({"x": [lvl]})).mark_rule(strokeDash=[5, 4], color=ui.TEXT).encode(x="x:Q")
        st.altair_chart((rule + pts + nominal).properties(height=50 * len(ids) + 170, title=f"Engine-level split conformal at {lvl:.0%}: mean coverage (dot), p05–p95 over 500 draws (line), nominal (dashed)"), width="stretch")

    st.markdown("**All interval methods**")
    cal_on = st.radio("Calibrated on", sorted(iv.calibrate_on.unique()), horizontal=True, key="uq_calon", help="'validation' is the role-swap consistency check: calibrate on validation engines, evaluate on calibration engines.")
    show = iv[iv.calibrate_on == cal_on][["method", "model", "nominal", "evaluate_on", "draws", "coverage_snapshot_capped_mean", "coverage_snapshot_capped_p05", "coverage_snapshot_capped_p95", "coverage_snapshot_true_mean", "width_snapshot_mean"]]
    st.dataframe(show, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.3f") for c in show.columns[5:]})

    bins = S.csv_file(rep, "coverage_by_rul_bin.csv")
    if bins is not None:
        pick = st.selectbox("Coverage by true-RUL range for", sorted(bins.method.unique()), key="uq_bin")
        b = bins[bins.method == pick]
        chart = alt.Chart(b).mark_bar(cornerRadiusEnd=4, color=ui.SEQ_BLUE[2]).encode(
            x=alt.X("true_rul_bin:N", sort=list(b.true_rul_bin), title="True RUL range (cycles)"),
            y=alt.Y("coverage_capped:Q", title="Coverage of capped RUL", scale=alt.Scale(domain=[0, 1])),
            tooltip=["true_rul_bin", "n", alt.Tooltip("coverage_capped:Q", format=".3f"), alt.Tooltip("width_mean:Q", title="Mean width", format=".1f")],
        )
        st.altair_chart(chart.properties(height=360, title="Diagnostic only: the guarantee is marginal, not per RUL range"), width="stretch")
    ui.source(ctx.show(rep), "With 15 calibration engines, levels above 15/16 ≈ 0.94 are unattainable (no finite 95 % intervals). Intervals are invalid under distribution shift")


def _corruption_tab(ctx: Ctx) -> None:
    ui.badge("canonical")
    rep = ctx.paths.report("robustness")
    rb = S.csv_file(rep, "robustness.csv")
    if rb is None:
        ui.unavailable("The robustness report")
        return
    st.markdown("Canonical-fold FD001 models applied to **corrupted** validation inputs, **without retraining**. Noise and missing-data results are means over 3 seeds.")
    c1, c2 = st.columns(2, gap="large")
    for col, (scen, title) in zip((c1, c2), (("noise", "Gaussian noise (× training SD)"), ("missing", "Readings missing, causal forward-fill")), strict=True):
        d = pd.concat([rb[rb.scenario == "clean"].assign(level="none"), rb[rb.scenario == scen]])
        d = d.assign(model_id=d.model.map(_fd001))
        d["label"] = d.model_id.map(ui.pretty_family)
        order = ["none", *list(dict.fromkeys(rb[rb.scenario == scen].level))]
        ids = ui.order_models(d.model_id.tolist())
        chart = alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=60, filled=True)).encode(
            x=alt.X("level:N", sort=order, title=title),
            y=alt.Y("rmse_capped:Q", title="Snapshot RMSE, capped (cycles)"),
            color=alt.Color("model_id:N", scale=ui.color_scale(ids), title=None, legend=alt.Legend(labelExpr="split(datum.label, '-')[2]", columns=3)),
            tooltip=[alt.Tooltip("label:N", title="Model"), "level", alt.Tooltip("rmse_capped:Q", title="RMSE", format=".2f"), alt.Tooltip("relative_increase:Q", title="vs clean", format="+.0%")],
        )
        col.altair_chart(chart.properties(height=400), width="stretch")
    drop = rb[rb.scenario == "dropout"]
    if not drop.empty:
        st.markdown("**Single frozen sensor** (held at its training mean): Δ snapshot RMSE vs clean, cycles")
        piv = drop.pivot(index="level", columns="model", values="delta_rmse_capped")
        piv = piv[[c for c in ("hgb_reference", "xgboost", "gru", "transformer", "tcn", "ridge") if c in piv.columns]]
        st.dataframe(piv.style.format("{:+.2f}"), width="stretch")
    ui.source(ctx.show(rep))


def _shift_tab(ctx: Ctx) -> None:
    ui.badge("canonical")
    rep = ctx.paths.report("robustness")
    rb = S.csv_file(rep, "robustness.csv")
    sh = rb[rb.scenario == "shift"] if rb is not None else pd.DataFrame()
    if sh.empty:
        ui.unavailable("The distribution-shift stress test")
        return
    st.markdown(
        "FD001-trained models applied **unchanged** to FD003 validation engines, which add a fan-degradation fault mode. "
        "This is a stress test, **not** transfer learning or zero-shot generalisation."
    )
    d = sh.assign(model_id=sh.model.map(_fd001))
    d["label"] = d.model_id.map(ui.pretty_family)
    ids = ui.order_models(d.model_id.tolist())
    bars = alt.Chart(d).mark_bar(cornerRadiusEnd=4, height=16).encode(
        y=alt.Y("model_id:N", sort=ids, title=None, axis=alt.Axis(labelExpr="split(datum.label, '-')[2]")),
        x=alt.X("relative_increase:Q", title="Increase in capped snapshot RMSE vs FD001 clean", axis=alt.Axis(format="%")),
        color=alt.Color("model_id:N", scale=ui.color_scale(ids), legend=None),
        tooltip=[alt.Tooltip("label:N", title="Model"), alt.Tooltip("rmse_capped:Q", title="RMSE on FD003", format=".2f"), alt.Tooltip("delta_rmse_capped:Q", title="Δ cycles", format="+.2f"), alt.Tooltip("relative_increase:Q", title="Relative", format="+.0%")],
    )
    text = bars.mark_text(align="left", dx=6, fontSize=11).encode(text=alt.Text("relative_increase:Q", format="+.0%"), color=alt.value(ui.TEXT))
    st.altair_chart((bars + text).properties(height=40 * len(ids) + 150, title="FD001 → FD003 shift"), width="stretch")
    st.dataframe(d.set_index("model_id").loc[ids].reset_index()[["label", "rmse_capped", "delta_rmse_capped", "rmse_true", "near_eol_rmse_true"]].rename(columns={"label": "Model", "rmse_capped": "RMSE capped", "delta_rmse_capped": "Δ vs clean", "rmse_true": "RMSE true", "near_eol_rmse_true": "Near-EOL RMSE true"}), hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ("RMSE capped", "Δ vs clean", "RMSE true", "Near-EOL RMSE true")})
    ui.source(ctx.show(rep), "One canonical fold; conformal intervals are not valid under this shift")


def _ablation_tab(ctx: Ctx) -> None:
    ui.badge("cv")
    rep = ctx.paths.report("ablation")
    ab = S.csv_file(rep, "ablation.csv")
    if ab is None:
        ui.unavailable("The ablation report")
        return
    st.markdown("One factor changed at a time relative to a base model (GRU or XGBoost), on CV folds cv0–cv2 with one seed. Δ = paired difference in capped snapshot RMSE; **negative = lower error**.")
    d = ab[ab.variant != "base"].copy()
    if "diff_rmse_capped_mean" in d:
        d["label"] = d.base.str.upper().str.replace("XGBOOST", "XGBoost") + " · " + d.variant
        d["lo"] = d.diff_rmse_capped_mean - d.diff_rmse_capped_sd
        d["hi"] = d.diff_rmse_capped_mean + d.diff_rmse_capped_sd
        order = d.sort_values(["base", "diff_rmse_capped_mean"]).label.tolist()
        base_dom = ["xgboost", "gru"]
        bars = alt.Chart(d).mark_bar(cornerRadiusEnd=4, height=12).encode(
            y=alt.Y("label:N", sort=order, title=None),
            x=alt.X("diff_rmse_capped_mean:Q", title="Δ snapshot RMSE vs base (cycles; bar = mean, line = ± SD over 3 folds)"),
            color=alt.Color("base:N", scale=alt.Scale(domain=base_dom, range=[ui.FAMILY_COLOR["xgboost"], ui.FAMILY_COLOR["gru"]]), title="Base model"),
            tooltip=["label", alt.Tooltip("diff_rmse_capped_mean:Q", title="Mean Δ", format="+.2f"), alt.Tooltip("diff_rmse_capped_sd:Q", title="SD", format=".2f"), alt.Tooltip("folds_better_capped:N", title="Folds better")],
        )
        err = alt.Chart(d).mark_rule(color=ui.TEXT2).encode(y=alt.Y("label:N", sort=order), x="lo:Q", x2="hi:Q")
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=ui.MUTED, strokeDash=[4, 3]).encode(x="x:Q")
        st.altair_chart((bars + err + zero).properties(height=24 * len(d) + 170), width="stretch")
    keep = [c for c in ("base", "variant", "folds", "snapshot_rmse_capped_mean", "snapshot_rmse_capped_sd", "diff_rmse_capped_mean", "diff_rmse_capped_sd", "folds_better_capped", "near_eol_rmse_true_mean") if c in ab.columns]
    st.dataframe(ab[keep], hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.2f") for c in keep if ab[c].dtype.kind == "f"})
    ui.source(ctx.show(rep), "Effects smaller than their SD are not distinguishable from zero with 3 folds. The uncapped-target variants are scored on capped truth, which they were not trained for")


def render(ctx: Ctx) -> None:
    ui.hero("FD001 · Stage 2", "Robustness & Reliability", "How well calibrated the uncertainty is, how the models behave under corrupted inputs and shift, and which design choices matter.")
    st.markdown('<p class="am-note">All results are on FD001 validation or calibration engines and are copied from saved Stage-2 run reports. No model is run by this page.</p>', unsafe_allow_html=True)
    tabs = st.tabs(["Uncertainty calibration", "Input corruption", "Distribution shift", "Ablations"])
    with tabs[0]:
        _uq_tab(ctx)
    with tabs[1]:
        _corruption_tab(ctx)
    with tabs[2]:
        _shift_tab(ctx)
    with tabs[3]:
        _ablation_tab(ctx)

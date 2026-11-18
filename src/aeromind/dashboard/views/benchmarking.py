"""B. Model Benchmarking: validated metrics, separated by evidence type.

Every value is copied from the committed registry (which copies run reports) or read from a
committed report CSV. Cross-validation, canonical validation and the official test are shown
in separate tabs and never on a common axis.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from aeromind.dashboard import data as D
from aeromind.dashboard import state as S
from aeromind.dashboard import ui
from aeromind.dashboard.state import Ctx

FOLD_COLUMN = {  # registry metric key → fold_metrics.csv column
    "snapshot_rmse_capped": "snapshot_rmse_capped",
    "snapshot_mae_capped": "snapshot_mae_capped",
    "snapshot_phm08_mean_capped": "snapshot_phm08_mean_capped",
    "snapshot_rmse_true": "snapshot_rmse_true",
    "snapshot_mae_true": "snapshot_mae_true",
    "window_ew_rmse_capped": "window_ew_rmse_capped",
    "near_eol_rmse_true": "near_eol_rmse_true",
}


def _metric_picker(available: list[str], key: str) -> str:
    keys = [k for k in D.METRICS if k in available]
    choice = st.selectbox("Metric", keys, key=key, format_func=lambda k: D.METRICS[k][0])
    st.caption(D.METRICS[choice][1])
    return choice


def _report_dir(ctx: Ctx, dataset: str):
    return ctx.paths.report("comparison", dataset) or ctx.paths.report("benchmarks", dataset)


def _cv_tab(ctx: Ctx, dataset: str) -> None:
    ui.badge("cv")
    cv = D.metric_frame(ctx.registry, dataset, "cross_validation")
    if cv.empty:
        st.info(
            f"{dataset} has no cross-validation: it was benchmarked on a single engine-level split (see *Canonical "
            "validation*). Model rankings on FD002–FD004 are therefore not established."
        )
        return
    metric = _metric_picker(sorted(cv.metric.unique()), f"cv_metric_{dataset}")
    sub = cv[cv.metric == metric].copy()
    ids = ui.order_models(sub.model_id.tolist())
    sub["label"] = sub.model_id.map(ui.pretty_family)
    sub["lo"], sub["hi"] = sub.value - sub.sd, sub.value + sub.sd
    rep = _report_dir(ctx, dataset)
    folds = S.csv_file(rep, "fold_metrics.csv")
    col = FOLD_COLUMN.get(metric)

    c1, c2 = st.columns(2, gap="large")
    y = alt.Y("model_id:N", sort=ids, title=None, axis=alt.Axis(labelExpr="split(datum.label, '-')[2]"))
    rule = alt.Chart(sub).mark_rule(color=ui.TEXT2, strokeWidth=2).encode(y=y, x=alt.X("lo:Q", title=D.METRICS[metric][0] + " (mean ± SD over folds)", scale=alt.Scale(zero=False)), x2="hi:Q")
    pts = alt.Chart(sub).mark_point(size=110, filled=True, stroke=ui.SURFACE, strokeWidth=2).encode(
        y=y,
        x="value:Q",
        color=alt.Color("model_id:N", scale=ui.color_scale(ids), legend=None),
        tooltip=[alt.Tooltip("label:N", title="Model"), alt.Tooltip("value:Q", title="Mean", format=".2f"), alt.Tooltip("sd:Q", title="SD", format=".2f")],
    )
    c1.altair_chart((rule + pts).properties(height=40 * len(ids) + 150, title="Fold mean (dot) ± one SD (line)"), width="stretch")

    if folds is not None and col in folds.columns:
        f = folds[folds.fold.astype(str).str.startswith("cv")].copy()
        f["model_id"] = f.model.map(lambda n: f"{dataset.lower()}-s2-{n}")
        f = f[f.model_id.isin(ids)]
        f["label"] = f.model_id.map(ui.pretty_family)
        dots = alt.Chart(f).mark_point(size=70, filled=True, opacity=0.9, stroke=ui.SURFACE, strokeWidth=1.5).encode(
            y=y,
            x=alt.X(f"{col}:Q", title=D.METRICS[metric][0] + " per fold", scale=alt.Scale(zero=False)),
            color=alt.Color("model_id:N", scale=ui.color_scale(ids), legend=None),
            tooltip=[alt.Tooltip("label:N", title="Model"), "fold", alt.Tooltip(f"{col}:Q", title="Value", format=".2f")],
        )
        c2.altair_chart(dots.properties(height=40 * len(ids) + 150, title="Each engine-level fold (5 dots per model)"), width="stretch")

    paired = S.csv_file(rep, "paired_vs_reference.csv")
    if paired is not None and metric == "snapshot_rmse_capped":
        p = paired.assign(model_id=paired.model.map(lambda n: f"{dataset.lower()}-s2-{n}"))
        p = p[p.model_id.isin(ids)].assign(label=lambda d: d.model_id.map(ui.pretty_family))
        pids = ui.order_models(p.model_id.tolist())
        py = alt.Y("model_id:N", sort=pids, title=None, axis=alt.Axis(labelExpr="split(datum.label, '-')[2]"))
        bars = alt.Chart(p).mark_bar(cornerRadiusEnd=4, height=14).encode(
            y=py,
            x=alt.X("mean_diff_rmse_capped:Q", title="Paired Δ snapshot RMSE vs HGB reference (cycles; negative = lower error)"),
            color=alt.Color("model_id:N", scale=ui.color_scale(pids), legend=None),
            tooltip=[alt.Tooltip("label:N", title="Model"), alt.Tooltip("mean_diff_rmse_capped:Q", title="Mean Δ", format="+.2f"), alt.Tooltip("sd_diff:Q", title="SD of Δ", format=".2f"), alt.Tooltip("folds_better_than_reference:Q", title="Folds better")],
        )
        rng = alt.Chart(p).mark_rule(color=ui.TEXT2).encode(y=py, x="min_diff:Q", x2="max_diff:Q")
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=ui.MUTED, strokeDash=[4, 3]).encode(x="x:Q")
        st.altair_chart((rng + bars + zero).properties(height=40 * len(pids) + 150, title="Paired per-fold difference against the reference (bar = mean, line = min–max over folds)"), width="stretch")

    table = sub[["label", "model_id", "value", "sd"]].rename(columns={"label": "Model", "model_id": "Registry id", "value": "Fold mean", "sd": "Fold SD"})
    table = table.set_index("Registry id").loc[ids].reset_index()
    st.dataframe(table, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ("Fold mean", "Fold SD")})
    ui.source([ctx.show(ctx.paths.registry), ctx.show(rep)], "5 engine-level folds of 17 engines over the 85 development engines. Folds share training engines, so the SD understates uncertainty; no significance test is claimed")


def _canonical_tab(ctx: Ctx, dataset: str) -> None:
    ui.badge("canonical")
    can = D.metric_frame(ctx.registry, dataset, "canonical_validation")
    if can.empty:
        ui.unavailable(f"Canonical validation for {dataset}")
        return
    metric = _metric_picker(sorted(can.metric.unique()), f"can_metric_{dataset}")
    sub = can[can.metric == metric].copy()
    ids = ui.order_models(sub.model_id.tolist())
    sub["label"] = sub.model_id.map(ui.pretty_family)
    n_val = next((m["metrics"]["canonical_validation"].get("n_validation_engines") for m in ctx.registry["models"] if m["dataset"] == dataset and m["metrics"].get("canonical_validation")), None)
    bars = alt.Chart(sub).mark_bar(cornerRadiusEnd=4, height=16).encode(
        y=alt.Y("model_id:N", sort=ids, title=None),
        x=alt.X("value:Q", title=D.METRICS[metric][0]),
        color=alt.Color("model_id:N", scale=ui.color_scale(ids), legend=None),
        tooltip=[alt.Tooltip("label:N", title="Model"), "model_id", alt.Tooltip("value:Q", title="Value", format=".2f")],
    )
    text = bars.mark_text(align="left", dx=6, color=ui.TEXT, fontSize=11).encode(text=alt.Text("value:Q", format=".2f"), color=alt.value(ui.TEXT))
    st.altair_chart((bars + text).properties(height=40 * len(ids) + 150, title=f"Single canonical split: {n_val or '?'} held-out validation engines"), width="stretch")
    st.dataframe(
        sub.set_index("model_id").loc[ids].reset_index()[["label", "model_id", "stage", "value"]].rename(columns={"label": "Model", "model_id": "Registry id", "stage": "Stage", "value": D.METRICS[metric][0]}),
        hide_index=True,
        width="stretch",
        column_config={D.METRICS[metric][0]: st.column_config.NumberColumn(format="%.3f")},
    )
    note = "One split only, with no fold variability: small differences are not evidence of a better model"
    if dataset == "FD001":
        note += ". The Stage-1 entry comes from the Stage-1 run on the same 15 validation engines"
    ui.source([ctx.show(ctx.paths.registry), ctx.show(_report_dir(ctx, dataset))], note)


def _stage1_tab(ctx: Ctx) -> None:
    ui.badge("canonical")
    rep = ctx.paths.report("stage1", "FD001")
    s1 = S.csv_file(rep, "validation_model_comparison.csv")
    if s1 is None:
        ui.unavailable("The Stage-1 validation comparison")
        return
    st.markdown(
        "Stage 1 compared six model families on the FD001 canonical validation engines (best grid point per family), "
        "including two **sensor-free baselines**: a constant median and the mean residual life by engine age. "
        "HistGradientBoosting was selected on validation **before** the single official test evaluation."
    )
    names = {"hist_gradient_boosting": "HistGradientBoosting", "xgboost": "XGBoost", "random_forest": "Random Forest", "ridge": "Ridge", "mean_residual_life": "Mean residual life (age only)", "median": "Constant median"}
    t = s1.assign(Model=s1.model.map(lambda m: names.get(m, m)))
    order = t.sort_values("snapshot_rmse_capped").Model.tolist()
    bars = alt.Chart(t).mark_bar(cornerRadiusEnd=4, height=16, color=ui.SEQ_BLUE[2]).encode(
        y=alt.Y("Model:N", sort=order, title=None),
        x=alt.X("snapshot_rmse_capped:Q", title="Snapshot RMSE · capped truth (cycles)"),
        tooltip=["Model", alt.Tooltip("snapshot_rmse_capped:Q", title="RMSE capped", format=".2f"), alt.Tooltip("snapshot_rmse_true:Q", title="RMSE true", format=".2f")],
    )
    text = bars.mark_text(align="left", dx=6, color=ui.TEXT, fontSize=11).encode(text=alt.Text("snapshot_rmse_capped:Q", format=".2f"))
    st.altair_chart((bars + text).properties(height=40 * len(t) + 150, title="Stage-1 canonical validation, 15 engines × 10 snapshots"), width="stretch")
    cols = {"Model": "Model", "snapshot_rmse_capped": "Snapshot RMSE (capped)", "snapshot_mae_capped": "Snapshot MAE (capped)", "snapshot_rmse_true": "Snapshot RMSE (true)", "window_engine_weighted_rmse_capped": "All-window RMSE, engine-weighted (capped)", "near_eol_rmse_true": "Near-EOL RMSE (true RUL ≤ 30)", "fit_seconds": "Fit (s)"}
    show = t[[c for c in cols if c in t.columns]].rename(columns=cols).sort_values("Snapshot RMSE (capped)")
    st.dataframe(show, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.2f") for c in show.columns if c != "Model"})
    ui.source(ctx.show(rep), "The three tree ensembles are statistically indistinguishable on 15 validation engines")


def _test_tab(ctx: Ctx) -> None:
    ui.badge("test")
    st.markdown(
        "The official NASA test labels are used **once**, for the model selected on validation, through a ledgered command "
        "that refuses to re-evaluate. Protocol: one prediction per test engine at its last observed cycle, compared with the "
        "official RUL. This differs from the validation snapshot protocol, so the numbers are **not comparable** with the "
        "validation tabs. No Stage-2 model has been evaluated on the test set."
    )
    test = D.official_test_table(ctx.registry, ctx.ledger)
    if test.empty:
        st.info("No model in the registry has an official test evaluation.")
        return
    k = st.columns(4)
    r = test.iloc[0]
    k[0].metric("RMSE · true RUL", f"{r.rmse_true:.2f}")
    k[1].metric("MAE · true RUL", f"{r.mae_true:.2f}")
    k[2].metric("PHM08 score (sum)", f"{r.phm08_sum_true:.1f}", help="Sum over test engines, the challenge convention.")
    k[3].metric("RMSE · capped truth", f"{r.rmse_capped:.2f}" if pd.notna(r.rmse_capped) else "—")
    st.caption(f"{ui.pretty_family(r.model_id)} (`{r.model_id}`) on {int(r.test_engines)} FD001 test engines, evaluated {r.evaluated_utc}.")
    if not test.ledger_entry_matches.all():
        st.warning("A registry test metric does not match its ledger entry; treat it as unverified.")
    if ctx.ledger:
        led = pd.DataFrame(ctx.ledger.get("entries", []))
        keep = [c for c in ("run", "dataset", "selected_model", "selection_finished_utc", "evaluated_utc", "repeat", "reason") if c in led.columns]
        st.markdown("**Test-set ledger**: every official test evaluation ever run")
        st.dataframe(led[keep], hide_index=True, width="stretch")
        st.caption("Ledger policy: " + str(ctx.ledger.get("policy", "")))
    ui.source([ctx.show(ctx.paths.report("test_set_ledger")), ctx.show(ctx.paths.registry)], "Registry test metrics are cross-checked against the ledger")


def _cost_tab(ctx: Ctx, dataset: str) -> None:
    ui.badge("registry")
    reg_df = D.registry_table(ctx.registry)
    sub = reg_df[reg_df.dataset == dataset]
    has_cv = sub.cv_rmse_capped_mean.notna().any()
    xcol = "cv_rmse_capped_mean" if has_cv else "canonical_rmse_capped"
    sc = sub.dropna(subset=[xcol, "single_input_ms"]).copy()
    if sc.empty:
        ui.unavailable(f"Model latency for {dataset}")
    else:
        ids = ui.order_models(sc.model_id.tolist())
        sc["label"] = sc.model_id.map(ui.pretty_family)
        base = alt.Chart(sc).encode(
            x=alt.X(f"{xcol}:Q", title=("CV" if has_cv else "Canonical") + " snapshot RMSE, capped (cycles)", scale=alt.Scale(zero=False)),
            y=alt.Y("single_input_ms:Q", title="Model latency per input (ms, log scale)", scale=alt.Scale(type="log")),
        )
        pts = base.mark_point(size=120, filled=True, stroke=ui.SURFACE, strokeWidth=2).encode(
            color=alt.Color("model_id:N", scale=ui.color_scale(ids), legend=None),
            tooltip=[alt.Tooltip("label:N", title="Model"), alt.Tooltip(f"{xcol}:Q", title="RMSE", format=".2f"), alt.Tooltip("single_input_ms:Q", title="ms / input", format=".3f")],
        )
        lbl = base.mark_text(align="left", dx=9, color=ui.TEXT, fontSize=11).encode(text="label:N")
        st.altair_chart((pts + lbl).properties(height=420, title="Accuracy vs model-only CPU latency (median over repeats, Stage-2 measurement)"), width="stretch")
        cols = {"model_id": "Registry id", "family": "Family", "single_input_ms": "Model ms / input", "fit_seconds": "Fit s (mean)", "peak_rss_mb": "Peak RSS MB", "n_parameters": "Parameters", "bundle_mb": "Bundle MB"}
        st.dataframe(sub.set_index("model_id").loc[ui.order_models(sub.model_id.tolist())].reset_index()[list(cols)].rename(columns=cols), hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ("Model ms / input", "Fit s (mean)", "Peak RSS MB", "Bundle MB")})
    bench_dir = ctx.paths.report("api_benchmark")
    bench = S.json_file(bench_dir / "benchmark.json") if bench_dir else None
    st.markdown("**Serving latency**: a complete request for one engine's full history (feature pipeline at every cycle plus the model)")
    if bench:
        b = pd.DataFrame(bench.get("results", []))
        b = b[b.model_id.str.startswith(dataset.lower())] if not b.empty else b
        if b.empty:
            st.caption("No serving benchmark for this subset.")
        else:
            st.dataframe(b.drop(columns=[c for c in ("n",) if c in b.columns]), hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ("load_s", "p50_ms", "p95_ms", "min_ms") if c in b.columns})
            st.caption(f"Measured {bench.get('measured_utc', '?')} · {bench.get('protocol', '')}")
    else:
        ui.unavailable("The serving benchmark")
    ui.source([ctx.show(ctx.paths.registry), ctx.show(bench_dir) if bench_dir else "—"], "Laptop CPU, CPU only; the laptop throttles, so treat latencies as indicative. See docs/CPU_BENCHMARKING.md")


def render(ctx: Ctx) -> None:
    ui.hero("Validated results", "Model Benchmarking", "Accuracy, variability and CPU cost of every registered model, with each kind of evidence kept separate.")
    if not ctx.registry:
        ui.unavailable("The model registry", "Benchmark tables are built from `reports/registry/model_registry.json`.")
        return
    datasets = [d for d in D.DATASETS if any(m["dataset"] == d for m in ctx.registry["models"])]
    c1, c2 = st.columns([1, 3])
    dataset = c1.selectbox("C-MAPSS subset", datasets, key="bm_ds", help="FD001 has 5-fold cross-validation; FD002–FD004 have a single canonical split each.")
    with c2:
        st.markdown(
            '<p class="am-note">Snapshot metrics use 10 seeded cut cycles per validation engine, identical for every model in a fold. '
            "<b>Capped truth</b> is min(RUL, 125), the models' target; <b>true RUL</b> is uncapped. Never compare a capped number with a true one.</p>",
            unsafe_allow_html=True,
        )
    tabs = st.tabs(["Cross-validation", "Canonical validation", "Stage-1 baselines", "Official NASA test", "CPU cost"])
    with tabs[0]:
        _cv_tab(ctx, dataset)
    with tabs[1]:
        _canonical_tab(ctx, dataset)
    with tabs[2]:
        if dataset == "FD001":
            _stage1_tab(ctx)
        else:
            st.info("Stage-1 classical baselines were run on FD001 only.")
    with tabs[3]:
        _test_tab(ctx)
    with tabs[4]:
        _cost_tab(ctx, dataset)

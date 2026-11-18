"""AeroMind research dashboard.

Run: ``uv run aeromind dashboard`` (or ``streamlit run src/aeromind/dashboard/app.py``).

Evidence labels used throughout:

- **Model result (validation)**: computed by a registered, hash-verified bundle, or copied
  from a saved validation report.
- **Dataset ground truth**: values from the NASA files (run-to-failure training file).
- **Official test (ledgered)**: the single Stage-1 official test evaluation.
- **Hypothetical illustration**: user-assumed inputs; not a measurement.

Engine-level predictions are shown only for *validation* engines (never used to fit the
model or its calibrator) and for user-uploaded trajectories (no ground truth). The official
NASA test files are never read here.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from aeromind.dashboard import data as D
from aeromind.data.cmapss import COLUMNS, SENSOR_COLUMNS
from aeromind.serving import maintenance as M
from aeromind.serving.service import ModelService, ServiceError

# ----------------------------------------------------------------------------- theme
SURFACE = "#0e1726"
TEXT, TEXT2, GRID = "#e6edf7", "#a9b4c6", "#24324a"
# Validated categorical palette (dark steps; all checks pass on SURFACE). Colour follows the
# model family, never its rank.
FAMILY_COLOR = {
    "hgb_reference": "#3987e5",
    "hist_gradient_boosting": "#9085e9",
    "xgboost": "#d95926",
    "gru": "#199e70",
    "transformer": "#c98500",
    "tcn": "#d55181",
    "ridge": "#0ca30c",
}
TRUTH = "#a9b4c6"
SEQ_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

EVIDENCE = {
    "model": ("Model result · validation", "#3987e5"),
    "truth": ("Dataset ground truth · NASA C-MAPSS simulation", "#0ca30c"),
    "test": ("Official test · ledgered, evaluated once", "#c98500"),
    "hypo": ("Hypothetical illustration · assumed inputs", "#d55181"),
}

st.set_page_config(page_title="AeroMind", page_icon="✈️", layout="wide")
st.markdown(
    f"""
<style>
.am-badge {{display:inline-block;padding:2px 10px;border-radius:999px;font-size:0.78rem;
  font-weight:600;letter-spacing:.02em;border:1px solid;margin:0 6px 6px 0}}
.am-hero h1 {{margin-bottom:0}}
.am-hero p {{color:{TEXT2};margin-top:4px}}
.am-note {{color:{TEXT2};font-size:0.85rem}}
div[data-testid="stMetricValue"] {{font-variant-numeric: tabular-nums}}
</style>""",
    unsafe_allow_html=True,
)


@alt.theme.register("aeromind", enable=True)
def _altair_theme() -> alt.theme.ThemeConfig:
    return alt.theme.ThemeConfig({
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent"},
            "axis": {"labelColor": TEXT2, "titleColor": TEXT2, "gridColor": GRID, "domainColor": GRID, "tickColor": GRID, "labelFontSize": 11, "titleFontSize": 12, "titleFontWeight": 500},
            "legend": {"labelColor": TEXT, "titleColor": TEXT2, "orient": "top", "labelFontSize": 11},
            "title": {"color": TEXT, "fontSize": 13, "anchor": "start", "fontWeight": 600},
            "header": {"labelColor": TEXT, "titleColor": TEXT2},
            "line": {"strokeWidth": 2},
            "point": {"size": 70, "filled": True},
        }
    })


def badge(kind: str) -> None:
    """Render an evidence-type badge."""
    label, color = EVIDENCE[kind]
    st.markdown(f'<span class="am-badge" style="color:{color};border-color:{color}">{label}</span>', unsafe_allow_html=True)


def provenance_box(prov: dict[str, Any], title: str = "Provenance") -> None:
    """Expandable provenance for a displayed result."""
    with st.expander(title):
        st.json(prov, expanded=False)


def family_of(model_id: str) -> str:
    """Registry id → family key (``fd001-s2-xgboost`` → ``xgboost``)."""
    return model_id.split("-", 2)[-1].replace("hgb", "hist_gradient_boosting") if "-s1-" in model_id else model_id.split("-", 2)[-1]


def color_scale(ids: list[str]) -> alt.Scale:
    """Colour scale keyed by model id, coloured by family."""
    return alt.Scale(domain=ids, range=[FAMILY_COLOR.get(family_of(i), TEXT2) for i in ids])


# ----------------------------------------------------------------------------- cached resources
def get_paths() -> D.Paths:
    """Re-read each run (cheap) so caches below are keyed by the active paths."""
    return D.Paths.from_env()


@st.cache_resource(show_spinner=False)
def get_service(registry: str, root: str) -> ModelService | None:
    try:
        return ModelService(registry, root)
    except (FileNotFoundError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def get_partitioned(data_dir: str, dataset: str, manifest: str) -> pd.DataFrame | None:
    return D.load_partitioned(Path(data_dir), dataset, Path(manifest))


@st.cache_data(show_spinner="Running the registered model…", max_entries=64)
def predict_cached(registry: str, root: str, mid: str, frame: pd.DataFrame, levels: tuple[float, ...]) -> dict[str, Any]:
    svc = get_service(registry, root)
    return svc.predict_engine(mid, frame, "engine", list(levels), include_trajectory=True)


@st.cache_data(show_spinner="Predicting validation trajectories…", max_entries=16)
def fleet_predictions(registry: str, root: str, mid: str, data_dir: str, dataset: str, manifest: str, level: float | None) -> pd.DataFrame:
    svc = get_service(registry, root)
    df = get_partitioned(data_dir, dataset, manifest)
    val = df[df.partition == "validation"]
    parts = []
    for eid, g in val.groupby("engine_id"):
        res = svc.predict_engine(mid, g.drop(columns=["engine_id", "rul", "partition"]), int(eid), [level] if level else [], True)
        t = pd.DataFrame(res["trajectory"]).assign(engine_id=int(eid))
        parts.append(t.merge(g[["cycle", "rul"]], on="cycle"))
    out = pd.concat(parts, ignore_index=True).rename(columns={"predicted_rul": "pred"})
    if level:
        out = out.rename(columns={f"lower_{level:.2f}": "lower"})
    return out


paths = get_paths()
service = get_service(str(paths.registry), str(paths.root))
registry = service.registry if service else None
availability = {m: service.availability(m) for m in service.model_ids} if service else {}
reg_df = D.registry_table(registry, availability) if registry else pd.DataFrame()


def manifest_for(dataset: str) -> Path:
    return Path((paths.index.get("data_manifests", {}).get("splits", {}) or {}).get(dataset, f"data/manifests/{dataset.lower()}_split_seed42.json"))


def available_models(dataset: str) -> list[str]:
    if reg_df.empty:
        return []
    return reg_df[(reg_df.dataset == dataset) & reg_df.available_locally].model_id.tolist()


def model_picker(dataset: str, key: str, need_intervals: bool = False) -> str | None:
    ids = available_models(dataset)
    if need_intervals:
        ids = [i for i in ids if service.entry(i)["intervals"] is not None]
    if not ids:
        st.info(f"No locally available model bundles for {dataset}. See *Getting model artifacts* in the README.")
        return None
    default = registry.get("default_model")
    idx = ids.index(default) if default in ids else 0
    return st.selectbox("Model (registry id)", ids, index=idx, key=key, format_func=lambda i: f"{i}  ·  {service.entry(i)['family']}")


# ----------------------------------------------------------------------------- sidebar
SECTIONS = [
    "Overview",
    "Engine Health Intelligence",
    "Model Benchmarking",
    "Sensor Intelligence",
    "Robustness & Experiments",
    "Maintenance Simulation",
]
with st.sidebar:
    st.markdown("## ✈️ AeroMind")
    st.caption("Remaining-useful-life research on NASA C-MAPSS")
    section = st.radio("Section", SECTIONS, key="section", label_visibility="collapsed")
    st.divider()
    st.caption(
        "Research demonstration. Simulation data. **Not** certified for operational aviation "
        "maintenance. Independent project, not endorsed by NASA."
    )
    if registry:
        st.caption(f"Registry built {registry['built_utc']} · commit `{registry.get('git_commit')}`")


# ============================================================================= A. Overview
def page_overview() -> None:
    st.markdown(
        '<div class="am-hero"><h1>AeroMind</h1><p>Uncertainty-aware Remaining Useful Life (RUL) '
        "prediction for turbofan engines, built on NASA's C-MAPSS simulation benchmark.</p></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registered models", len(reg_df) if not reg_df.empty else 0)
    c2.metric("Available locally", int(reg_df.available_locally.sum()) if not reg_df.empty else 0)
    c3.metric("With calibrated intervals", int(reg_df.intervals.sum()) if not reg_df.empty else 0)
    c4.metric("Official test evaluations", len((D.read_json_opt(Path(paths.index.get("test_set_ledger", ""))) or {}).get("entries", [])))

    st.subheader("Objective")
    st.markdown(
        "- Estimate how many operating cycles an engine has left before failure from its sensor history.\n"
        "- Quantify uncertainty with intervals calibrated on held-out engines.\n"
        "- Report every result with leakage-safe, engine-level protocols and full provenance."
    )
    st.subheader("Dataset")
    badge("truth")
    st.markdown(
        "C-MAPSS (Saxena et al., 2008) contains **simulated** run-to-failure trajectories of a "
        "turbofan model: 21 sensors and 3 operating settings per cycle. It was generated by "
        "NASA's Commercial Modular Aero-Propulsion System Simulation; it is not flight data."
    )
    st.dataframe(
        pd.DataFrame([{"subset": k, **v} for k, v in D.DATASET_INFO.items()]),
        hide_index=True,
        width="stretch",
    )
    st.subheader("Research status")
    s1, s2, s3 = st.columns(3)
    s1.markdown("**Stage 1 · complete**\n\nValidated data layer, engine-level splits, causal features, six classical baselines, one ledgered official FD001 test evaluation.")
    s2.markdown("**Stage 2 · complete**\n\nTCN / GRU / Transformer, 5-fold engine-level CV, split-conformal and quantile UQ, ablations, robustness, FD002–FD004 benchmarks.")
    s3.markdown("**Stage 3 · this dashboard**\n\nModel registry with hash-pinned bundles, REST API, this dashboard. Not done: multi-seed deep runs, nested tuning, CV/UQ beyond FD001.")
    st.subheader("Model availability")
    if reg_df.empty:
        st.warning("No model registry found. Build it with `uv run aeromind s3-registry` (see README).")
    else:
        st.dataframe(
            reg_df[["model_id", "dataset", "stage", "family", "version", "available_locally", "intervals", "availability"]],
            hide_index=True,
            width="stretch",
        )
        if registry.get("default_model"):
            st.caption(f"Default model **{registry['default_model']}**: {registry.get('default_model_rationale', '')}")
    st.subheader("Limitations you should know")
    st.markdown(
        "- Simulation data only; behaviour on real engines is unknown.\n"
        "- Validation folds have 17 engines and calibration has 15: fold-to-fold spread (≈ ±2 cycles RMSE) exceeds most differences between models.\n"
        "- Prediction intervals target **capped** RUL `min(RUL, 125)`, have marginal (not per-engine) coverage, and assume new engines resemble the calibration engines.\n"
        "- AeroMind is not a physical digital twin and is not certified for any operational use."
    )


# ============================================================================= B. Engine health
def page_engine() -> None:
    st.header("Engine Health Intelligence")
    left, right = st.columns([1, 3])
    with left:
        source = st.radio("Trajectory source", ["Validation engine (held out)", "Upload CSV"], key="eng_source")
        dataset = st.selectbox("Subset", D.DATASETS, key="eng_ds")
        mid = model_picker(dataset, "eng_model")
    if mid is None:
        return
    entry = service.entry(mid)
    truth = None
    if source.startswith("Validation"):
        df = get_partitioned(str(paths.data_dir), dataset, str(manifest_for(dataset)))
        if df is None:
            st.warning(f"NASA file `train_{dataset}.txt` not found in `{paths.data_dir}`. See docs/DATA.md.")
            return
        val_ids = sorted(df[df.partition == "validation"].engine_id.unique())
        with left:
            eid = st.selectbox("Validation engine", val_ids, key="eng_id")
            g = df[df.engine_id == eid].reset_index(drop=True)
            cut = st.slider("Observe up to cycle", 1, int(g.cycle.max()), int(g.cycle.max()), key="eng_cut")
        g = g[g.cycle <= cut]
        frame = g.drop(columns=["engine_id", "rul", "partition"])
        truth = g[["cycle", "rul"]]
        fitted = set(entry["training"].get("engine_ids", []))
        if int(eid) in fitted:  # pragma: no cover - guarded by construction
            st.error("This engine was used to fit the model; in-sample predictions are not shown.")
            return
    else:
        with left:
            up = st.file_uploader("C-MAPSS rows for ONE engine (CSV with headers, or raw whitespace .txt)", type=["csv", "txt"], key="eng_upload")
        if up is None:
            st.info("Upload one engine's cycles: columns `cycle, setting_1..3, sensor_1..21` (an `engine_id` column is ignored). No ground truth is shown for uploads.")
            return
        raw = up.getvalue().decode("utf-8", errors="replace")
        try:
            if up.name.endswith(".txt"):
                frame = pd.read_csv(io.StringIO(raw), sep=r"\s+", header=None, names=COLUMNS)
            else:
                frame = pd.read_csv(io.StringIO(raw))
        except (ValueError, pd.errors.ParserError) as exc:
            st.error(f"Could not parse the file: {exc}")
            return
        if "engine_id" in frame and frame["engine_id"].nunique() > 1:
            st.error("The file contains several engines; upload one engine at a time.")
            return
    levels: tuple[float, ...] = ()
    if entry["intervals"]:
        with left:
            lv = st.selectbox("Prediction interval", ["none", *sorted(entry["intervals"]["thresholds"], reverse=True)], index=1, key="eng_level")
        levels = () if lv == "none" else (float(lv),)
    try:
        res = predict_cached(str(paths.registry), str(paths.root), mid, frame, levels)
    except ServiceError as exc:
        st.error(f"{exc.code}: {exc.message}")
        return

    with right:
        badge("model")
        if truth is not None:
            badge("truth")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Predicted RUL (cycles)", f"{res['predicted_rul']:.1f}")
        if res["intervals"]:
            iv = res["intervals"][0]
            k2.metric(f"{iv['level']:.0%} interval", f"{iv['lower']:.0f} – {iv['upper']:.0f}")
        else:
            k2.metric("Interval", "not calibrated")
        if truth is not None:
            true_rul = int(truth.rul.iloc[-1])
            k3.metric("Actual RUL (ground truth)", f"{true_rul}", help="Known because training-file engines run to failure.")
            k4.metric("Actual capped RUL", f"{min(true_rul, entry['target_cap'] or true_rul)}", help=f"The model's target: min(RUL, {entry['target_cap']}).")
        else:
            k3.metric("Cycles observed", res["n_cycles"])
        for w in res["warnings"]:
            st.warning(w)

        traj = pd.DataFrame(res["trajectory"])
        base = alt.Chart(traj).encode(x=alt.X("cycle:Q", title="Cycle"))
        layers = []
        if res["intervals"]:
            lvl = f"{res['intervals'][0]['level']:.2f}"
            layers.append(base.mark_area(opacity=0.22, color=FAMILY_COLOR.get(family_of(mid), TEXT2)).encode(y=alt.Y(f"lower_{lvl}:Q", title="RUL (cycles)"), y2=f"upper_{lvl}:Q"))
        long = traj[["cycle", "predicted_rul"]].rename(columns={"predicted_rul": "value"}).assign(series=f"Predicted ({mid})")
        if truth is not None:
            cap = entry["target_cap"] or 10**9
            long = pd.concat([long, truth.assign(value=np.minimum(truth.rul, cap), series="Ground truth, capped")[["cycle", "value", "series"]], truth.assign(value=truth.rul, series="Ground truth, true RUL")[["cycle", "value", "series"]]])
        dom = list(dict.fromkeys(long.series))
        rng = [FAMILY_COLOR.get(family_of(mid), TEXT2), TRUTH, "#5d6b82"][: len(dom)]
        dash = alt.condition(alt.datum.series == f"Predicted ({mid})", alt.value([1, 0]), alt.value([5, 4]))
        layers.append(
            alt.Chart(long).mark_line().encode(
                x="cycle:Q",
                y=alt.Y("value:Q", title="RUL (cycles)"),
                color=alt.Color("series:N", scale=alt.Scale(domain=dom, range=rng), title=None),
                strokeDash=dash,
                tooltip=["series", "cycle", alt.Tooltip("value:Q", format=".1f")],
            )
        )
        st.altair_chart(alt.layer(*layers).properties(height=300, title="RUL history: prediction at each cycle uses only cycles up to it"), width="stretch")
        if len(traj) > 10:
            recent = traj.predicted_rul.iloc[-1] - traj.predicted_rul.iloc[-11]
            st.caption(f"Change in predicted RUL over the last 10 cycles: {recent:+.1f} cycles (−10 means degrading at the expected rate).")
        provenance_box({**res["provenance"], "intervals": [{k: v for k, v in i.items() if k != "guarantee"} for i in res["intervals"]], "inference_ms": round(res["inference_ms"], 2)})
        if res["intervals"]:
            st.caption("Interval guarantee: " + res["intervals"][0]["guarantee"])

        st.subheader("Sensor trajectories")
        badge("truth")
        sensors = [c for c in SENSOR_COLUMNS if frame[c].nunique() > 3]
        pick = st.multiselect("Sensors", sensors, default=sensors[:4], key="eng_sensors", format_func=D.sensor_label, max_selections=6)
        if pick:
            s = frame[["cycle", *pick]].melt("cycle", var_name="sensor", value_name="reading")
            s["sensor"] = s["sensor"].map(D.sensor_label)
            chart = alt.Chart(s).mark_line(color=SEQ_BLUE[2], strokeWidth=1.5).encode(
                x=alt.X("cycle:Q", title="Cycle"), y=alt.Y("reading:Q", title=None, scale=alt.Scale(zero=False)), tooltip=["sensor", "cycle", "reading"]
            ).properties(height=130, width=260).facet(facet=alt.Facet("sensor:N", title=None), columns=3).resolve_scale(y="independent")
            st.altair_chart(chart)


# ============================================================================= C. Benchmarking
def page_benchmark() -> None:
    st.header("Model Benchmarking")
    if reg_df.empty:
        st.warning("No registry found.")
        return
    dataset = st.selectbox("Subset", D.DATASETS, key="bm_ds")
    sub = reg_df[reg_df.dataset == dataset]
    picked = st.multiselect("Compare models", sub.model_id.tolist(), default=sub.model_id.tolist(), key="bm_models")
    sub = sub[sub.model_id.isin(picked)]
    if sub.empty:
        return
    badge("model")
    has_cv = sub.cv_rmse_capped_mean.notna().any()
    st.markdown(
        "**Cross-validation** = mean ± SD over 5 engine-level folds of the 85 development engines. "
        "**Canonical** = the single 15-engine validation split. Snapshot metrics, capped truth `min(RUL, 125)`; "
        "PHM08 is the mean asymmetric score per snapshot (late predictions penalised more)."
        if has_cv
        else "Single canonical validation split only (no cross-validation) for this subset: rankings are not established."
    )
    cols = {
        "model_id": "Model",
        "family": "Family",
        "cv_rmse_capped_mean": "CV RMSE",
        "cv_rmse_capped_sd": "CV RMSE SD",
        "cv_mae_capped_mean": "CV MAE",
        "cv_phm08_mean_capped": "CV PHM08/snapshot",
        "cv_near_eol_rmse_true": "CV near-EOL RMSE",
        "canonical_rmse_capped": "Canonical RMSE",
        "canonical_mae_capped": "Canonical MAE",
        "canonical_phm08_mean_capped": "Canonical PHM08/snapshot",
        "single_input_ms": "CPU ms / input",
        "fit_seconds": "Fit s",
        "n_parameters": "Parameters",
        "bundle_mb": "Bundle MB",
        "peak_rss_mb": "Peak RSS MB",
    }
    table = sub[list(cols)].rename(columns=cols).dropna(axis=1, how="all")
    st.dataframe(table, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.2f") for c in table.columns if c not in ("Model", "Family", "Parameters")})
    rep = paths.report("comparison", dataset) or paths.report("benchmarks", dataset)
    st.caption(f"Source: `{rep}` (summary.csv, resources.csv, fold_metrics.csv) via the registry.")

    ids = sub.model_id.tolist()
    folds = D.read_csv(rep, "fold_metrics.csv")
    c1, c2 = st.columns(2)
    if folds is not None and has_cv:
        f = folds[folds.fold.str.startswith("cv")].copy()
        f["model_id"] = f.model.map(lambda n: f"{dataset.lower()}-s2-{n}")
        f = f[f.model_id.isin(ids)]
        dots = alt.Chart(f).mark_point(opacity=0.85).encode(
            y=alt.Y("model_id:N", title=None, sort=ids),
            x=alt.X("snapshot_rmse_capped:Q", title="Snapshot RMSE, capped (cycles)", scale=alt.Scale(zero=False)),
            color=alt.Color("model_id:N", scale=color_scale(ids), legend=None),
            tooltip=["model_id", "fold", alt.Tooltip("snapshot_rmse_capped:Q", format=".2f")],
        )
        means = alt.Chart(f).mark_tick(thickness=3, size=22, color=TEXT).encode(y=alt.Y("model_id:N", sort=ids), x="mean(snapshot_rmse_capped):Q")
        c1.altair_chart((dots + means).properties(height=260, title="Per-fold RMSE (dots) and mean (tick)"), width="stretch")
    paired = D.read_csv(rep, "paired_vs_reference.csv")
    if paired is not None and has_cv:
        p = paired.assign(model_id=paired.model.map(lambda n: f"{dataset.lower()}-s2-{n}"))
        p = p[p.model_id.isin(ids)]
        bars = alt.Chart(p).mark_bar(cornerRadiusEnd=4, height=14).encode(
            y=alt.Y("model_id:N", title=None, sort=ids),
            x=alt.X("mean_diff_rmse_capped:Q", title="Paired Δ RMSE vs HGB reference (negative = better)"),
            color=alt.Color("model_id:N", scale=color_scale(ids), legend=None),
            tooltip=["model_id", alt.Tooltip("mean_diff_rmse_capped:Q", format="+.2f"), alt.Tooltip("sd_diff:Q", format=".2f"), "folds_better_than_reference"],
        )
        rule = alt.Chart(p).mark_rule(color=TEXT2).encode(y=alt.Y("model_id:N", sort=ids), x="min_diff:Q", x2="max_diff:Q")
        c2.altair_chart((rule + bars).properties(height=260, title="Paired per-fold difference (bar = mean, line = range)"), width="stretch")
    xcol = "cv_rmse_capped_mean" if has_cv else "canonical_rmse_capped"
    sc = sub.dropna(subset=[xcol, "single_input_ms"])
    if not sc.empty:
        base = alt.Chart(sc).encode(
            x=alt.X(f"{xcol}:Q", title=("CV" if has_cv else "Canonical") + " snapshot RMSE, capped (cycles)", scale=alt.Scale(zero=False)),
            y=alt.Y("single_input_ms:Q", title="CPU latency per input (ms, log)", scale=alt.Scale(type="log")),
        )
        pts = base.mark_point(size=110).encode(color=alt.Color("model_id:N", scale=color_scale(ids), legend=None), tooltip=["model_id", alt.Tooltip(f"{xcol}:Q", format=".2f"), alt.Tooltip("single_input_ms:Q", format=".2f")])
        lbl = base.mark_text(align="left", dx=8, color=TEXT, fontSize=11).encode(text="family:N")
        st.altair_chart((pts + lbl).properties(height=280, title="Accuracy vs CPU latency (median single-input latency, Stage-2 measurement)"), width="stretch")

    bench = D.read_json_opt((paths.report("api_benchmark") or Path("_")) / "benchmark.json")
    st.subheader("Serving latency (API and in-process)")
    if bench:
        b = pd.DataFrame(bench["results"])
        b = b[b.model_id.str.startswith(dataset.lower())]
        if not b.empty:
            st.dataframe(b, hide_index=True, width="stretch")
            st.caption(f"Measured {bench['measured_utc']} on {bench['environment'].get('processor')} · {bench['protocol']}")
        else:
            st.caption("No serving benchmark for this subset.")
    else:
        st.caption("Run `uv run aeromind s3-benchmark` to measure API latency.")

    test_rows = sub[sub.official_test_rmse_true.notna()]
    st.subheader("Official NASA test set")
    if test_rows.empty:
        st.caption("None of these models has been evaluated on the official test set (by design; see the test-set ledger).")
    else:
        badge("test")
        for _, r in test_rows.iterrows():
            t = service.entry(r.model_id)["metrics"]["official_test"]
            st.markdown(f"**{r.model_id}**: RMSE (true RUL) {t['values']['last_cycle_rmse_true']:.2f}, MAE {t['values']['last_cycle_mae_true']:.2f}, PHM08 sum {t['values']['last_cycle_phm08_sum_true']:.1f} on {t['n_test_engines']} test engines, evaluated once ({t['evaluated_utc']}). Different protocol from the validation snapshots above; not comparable column-for-column.")


# ============================================================================= D. Sensors
def page_sensors() -> None:
    st.header("Sensor Intelligence")
    dataset = st.selectbox("Subset", D.DATASETS, key="sn_ds")
    df = get_partitioned(str(paths.data_dir), dataset, str(manifest_for(dataset)))
    if df is None:
        st.warning(f"NASA file `train_{dataset}.txt` not found in `{paths.data_dir}`.")
        return
    train = df[df.partition == "train"]
    badge("truth")
    st.caption(f"Training-partition engines only ({train.engine_id.nunique()} engines), the scope used for model development.")
    sensors = D.informative_sensors(train)
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Correlations", "Feature importance", "Trajectory comparison"])
    with tab1:
        s = st.selectbox("Sensor", sensors, key="sn_dist", format_func=D.sensor_label)
        st.caption(D.sensor_description(s))
        h = train.assign(phase=np.where(train.rul <= 30, "last 30 cycles before failure", np.where(train.cycle <= 30, "first 30 cycles", "mid-life")))
        h = h[h.phase != "mid-life"]
        chart = alt.Chart(h).mark_bar(opacity=0.75, cornerRadiusEnd=2).encode(
            x=alt.X(f"{s}:Q", bin=alt.Bin(maxbins=40), title=D.sensor_label(s)),
            y=alt.Y("count():Q", stack=None, title="Readings"),
            color=alt.Color("phase:N", scale=alt.Scale(domain=["first 30 cycles", "last 30 cycles before failure"], range=[SEQ_BLUE[2], "#d95926"]), title=None),
        ).properties(height=280, title="Early-life vs near-failure readings")
        st.altair_chart(chart, width="stretch")
        if D.DATASET_INFO[dataset]["conditions"] > 1:
            st.info("Six operating conditions: raw readings are dominated by the operating regime; the models normalise per condition.")
    with tab2:
        cor = D.correlation_with_rul(train, sensors)
        cor["sensor"] = cor.sensor.map(D.sensor_label)
        bar = alt.Chart(cor).mark_bar(cornerRadiusEnd=4, height=12).encode(
            y=alt.Y("sensor:N", sort=cor.sensor.tolist(), title=None),
            x=alt.X("spearman_rho_vs_capped_rul:Q", title="Spearman ρ with capped RUL", scale=alt.Scale(domain=[-1, 1])),
            color=alt.Color("spearman_rho_vs_capped_rul:Q", scale=alt.Scale(domain=[-1, 0, 1], range=["#e66767", "#383835", "#3987e5"]), legend=None),
            tooltip=["sensor", alt.Tooltip("spearman_rho_vs_capped_rul:Q", format=".2f")],
        ).properties(height=26 * len(cor), title="Monotone association with capped RUL (pooled rows; not causal)")
        st.altair_chart(bar, width="stretch")
        m = train[sensors].corr(method="spearman").reset_index().melt("index", var_name="other", value_name="rho")
        heat = alt.Chart(m).mark_rect().encode(
            x=alt.X("index:N", title=None, sort=sensors), y=alt.Y("other:N", title=None, sort=sensors),
            color=alt.Color("rho:Q", scale=alt.Scale(domain=[-1, 0, 1], range=["#e66767", "#383835", "#3987e5"]), title="ρ"),
            tooltip=["index", "other", alt.Tooltip("rho:Q", format=".2f")],
        ).properties(height=420, title="Sensor–sensor Spearman correlation")
        st.altair_chart(heat, width="stretch")
    with tab3:
        xid = f"{dataset.lower()}-s2-xgboost"
        if xid in service.model_ids and availability.get(xid, (False,))[0]:
            imp = D.xgboost_importance(service.load(xid))
            if imp is not None:
                badge("model")
                top = imp.head(20)
                chart = alt.Chart(top).mark_bar(cornerRadiusEnd=4, height=12, color=FAMILY_COLOR["xgboost"]).encode(
                    y=alt.Y("feature:N", sort=top.feature.tolist(), title=None), x=alt.X("share:Q", title="Share of total gain", axis=alt.Axis(format="%")),
                    tooltip=["feature", alt.Tooltip("share:Q", format=".1%")],
                ).properties(height=22 * len(top), title=f"XGBoost gain importance ({xid}, top 20)")
                st.altair_chart(chart, width="stretch")
                st.caption("Gain importance of the trained XGBoost bundle: how much each engineered feature reduced the training loss. Correlated features share credit; this is not a causal attribution.")
                provenance_box(service.provenance(xid))
        else:
            st.info(f"XGBoost bundle {xid} not available locally.")
    with tab4:
        s = st.selectbox("Sensor", sensors, key="sn_traj", index=min(3, len(sensors) - 1), format_func=D.sensor_label)
        engines = sorted(train.engine_id.unique())
        pick = st.multiselect("Engines (training partition)", engines, default=engines[:5], key="sn_engines", max_selections=8)
        if pick:
            t = train[train.engine_id.isin(pick)].assign(cycles_to_failure=lambda d: -d.rul, engine=lambda d: d.engine_id.astype(str))
            dom = [str(e) for e in pick]
            pal = list(FAMILY_COLOR.values()) + ["#e66767"]
            chart = alt.Chart(t).mark_line(strokeWidth=1.5).encode(
                x=alt.X("cycles_to_failure:Q", title="Cycles relative to failure (0 = failure)"),
                y=alt.Y(f"{s}:Q", title=D.sensor_label(s), scale=alt.Scale(zero=False)),
                color=alt.Color("engine:N", scale=alt.Scale(domain=dom, range=pal[: len(dom)]), title="Engine"),
                tooltip=["engine", "cycle", "rul", s],
            ).properties(height=320, title="Trajectories aligned at failure")
            st.altair_chart(chart, width="stretch")


# ============================================================================= E. Robustness & experiments
def page_experiments() -> None:
    st.header("Robustness & Experiments")
    badge("model")
    st.caption("All results below are on validation or calibration engines of FD001 and copied from saved run reports.")
    t1, t2, t3, t4 = st.tabs(["Ablations", "Noise · dropout · missing", "Uncertainty calibration", "Reproducibility"])
    with t1:
        rep = paths.report("ablation")
        ab = D.read_csv(rep, "ablation.csv")
        if ab is None:
            st.info("Ablation report not found.")
        else:
            st.markdown("One factor changed at a time relative to a base model, folds cv0–cv2 (3 folds, one seed). Δ = paired difference in snapshot RMSE (capped); negative = better.")
            st.dataframe(ab, hide_index=True, width="stretch")
            d = ab[ab.variant != "base"].copy() if "variant" in ab else None
            if d is not None and "diff_rmse_capped_mean" in d:
                d["label"] = d.base + " · " + d.variant
                chart = alt.Chart(d).mark_bar(cornerRadiusEnd=4, height=12).encode(
                    y=alt.Y("label:N", sort=d.sort_values("diff_rmse_capped_mean").label.tolist(), title=None),
                    x=alt.X("diff_rmse_capped_mean:Q", title="Δ snapshot RMSE vs base (cycles)"),
                    color=alt.Color("base:N", scale=alt.Scale(domain=["gru", "xgboost"], range=[FAMILY_COLOR["gru"], FAMILY_COLOR["xgboost"]]), title="Base"),
                    tooltip=["label", alt.Tooltip("diff_rmse_capped_mean:Q", format="+.2f"), alt.Tooltip("diff_rmse_capped_sd:Q", format=".2f")],
                ).properties(height=22 * len(d))
                st.altair_chart(chart, width="stretch")
            st.caption(f"Source: `{rep}`. Effects smaller than their SD are not distinguishable from zero with 3 folds.")
    with t2:
        rep = paths.report("robustness")
        rb = D.read_csv(rep, "robustness.csv")
        if rb is None:
            st.info("Robustness report not found.")
        else:
            st.markdown("Canonical-fold models applied to corrupted validation inputs **without retraining** (noise/missing: mean of 3 seeds).")
            for scen, title in (("noise", "Gaussian noise (× training SD)"), ("missing", "Missing readings, causal forward-fill")):
                d = pd.concat([rb[rb.scenario == "clean"].assign(level="0"), rb[rb.scenario == scen]])
                d = d.assign(model_id=d.model.map(lambda n: f"fd001-s2-{n}"))
                order = ["0", *list(dict.fromkeys(rb[rb.scenario == scen].level))]
                ids = list(dict.fromkeys(d.model_id))
                chart = alt.Chart(d).mark_line(point=True).encode(
                    x=alt.X("level:N", sort=order, title=title), y=alt.Y("rmse_capped:Q", title="Snapshot RMSE, capped"),
                    color=alt.Color("model_id:N", scale=color_scale(ids), title=None),
                    tooltip=["model_id", "level", alt.Tooltip("rmse_capped:Q", format=".2f")],
                ).properties(height=260)
                st.altair_chart(chart, width="stretch")
            drop = rb[rb.scenario == "dropout"].pivot(index="level", columns="model", values="delta_rmse_capped")
            st.markdown("**Single frozen sensor**: Δ RMSE vs clean")
            st.dataframe(drop.style.format("{:+.2f}"), width="stretch")
            st.markdown("**Distribution shift**: FD001-trained models on FD003 validation engines (stress test, not transfer learning)")
            st.dataframe(rb[rb.scenario == "shift"][["model", "rmse_capped", "delta_rmse_capped", "rmse_true", "near_eol_rmse_true"]], hide_index=True, width="stretch")
            st.caption(f"Source: `{rep}`.")
    with t3:
        rep = paths.report("uq")
        iv = D.read_csv(rep, "intervals_main.csv")
        if iv is None:
            st.info("UQ report not found.")
        else:
            st.markdown(
                "Intervals calibrated on the **15 calibration engines**, evaluated on the 15 validation engines. "
                "`conformal_engine`: one random snapshot per calibration engine (n = 15), repeated over 500 draws; "
                "p05–p95 shows how much realised coverage depends on the draw. `pooled` variants have **no** finite-sample guarantee."
            )
            show = iv[(iv.calibrate_on == "calibration")][["method", "model", "nominal", "coverage_snapshot_capped_mean", "coverage_snapshot_capped_p05", "coverage_snapshot_capped_p95", "coverage_snapshot_true_mean", "width_snapshot_mean"]]
            st.dataframe(show, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.3f") for c in show.columns[2:]})
            bins = D.read_csv(rep, "coverage_by_rul_bin.csv")
            if bins is not None:
                pick = st.selectbox("Coverage by true-RUL range for", sorted(bins.method.unique()), key="uq_bin")
                b = bins[bins.method == pick]
                chart = alt.Chart(b).mark_bar(cornerRadiusEnd=4, color=SEQ_BLUE[2]).encode(
                    x=alt.X("true_rul_bin:N", sort=list(b.true_rul_bin), title="True RUL range (cycles)"), y=alt.Y("coverage_capped:Q", title="Coverage of capped RUL", scale=alt.Scale(domain=[0, 1])),
                    tooltip=["true_rul_bin", "n", alt.Tooltip("coverage_capped:Q", format=".3f"), alt.Tooltip("width_mean:Q", format=".1f")],
                )
                rule = alt.Chart(pd.DataFrame({"y": [0.9]})).mark_rule(strokeDash=[5, 4], color=TEXT2).encode(y="y:Q")
                st.altair_chart((chart + rule).properties(height=260, title="Diagnostic only: the guarantee is marginal, not per RUL range (dashed = 90 %)"), width="stretch")
            st.caption(f"Source: `{rep}`. With 15 calibration engines, levels above 15/16 ≈ 0.94 are unattainable.")
    with t4:
        if registry:
            st.markdown(f"**Model registry** `{paths.registry}` · schema {registry['schema_version']} · built {registry['built_utc']} · git `{registry.get('git_commit')}` · official test labels read during build: **{registry['official_test_labels_read']}**")
            st.json({"packages": registry.get("packages")}, expanded=False)
        ledger = D.read_json_opt(Path(paths.index.get("test_set_ledger", "")))
        if ledger:
            badge("test")
            st.markdown("**Official test-set ledger** (every evaluation ever run)")
            st.dataframe(pd.DataFrame(ledger["entries"])[["run", "dataset", "selected_model", "evaluated_utc", "reason"]], hide_index=True, width="stretch")
        prov = D.read_json_opt(Path((paths.index.get("data_manifests") or {}).get("provenance", "")))
        if prov:
            st.markdown("**Data provenance manifest** (SHA-256 of the NASA files)")
            files = prov.get("files", prov)
            if isinstance(files, dict):
                st.dataframe(pd.DataFrame([{"file": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in files.items()]), hide_index=True, width="stretch")
        for ds, mpath in ((paths.index.get("data_manifests") or {}).get("splits") or {}).items():
            j = D.read_json_opt(Path(mpath))
            if j:
                st.caption(f"{ds} split `{mpath}`: seed {j.get('seed')}, train/validation/calibration = {len(j['train_engine_ids'])}/{len(j['validation_engine_ids'])}/{len(j['calibration_engine_ids'])} engines")


# ============================================================================= F. Maintenance simulation
def page_maintenance() -> None:
    st.header("Maintenance Simulation")
    badge("hypo")
    st.warning(
        "Hypothetical illustration. Policies are applied to the 15 FD001 validation engines (simulated, run to failure). "
        "Failure and unused-cycle **counts** come from real model predictions vs ground truth; **costs are your assumptions** "
        "in arbitrary units. Nothing here estimates real maintenance savings or safety outcomes."
    )
    mid = model_picker("FD001", "mt_model")
    if mid is None:
        return
    if get_partitioned(str(paths.data_dir), "FD001", str(manifest_for("FD001"))) is None:
        st.warning("FD001 data not found.")
        return
    entry = service.entry(mid)
    c1, c2, c3 = st.columns(3)
    lead = c1.number_input("Lead time to schedule maintenance (cycles)", 0, 100, 10, key="mt_lead")
    cf = c2.number_input("Assumed cost: unscheduled failure (units)", 0.0, 1e6, 10.0, key="mt_cf")
    cp = c3.number_input("Assumed cost: planned maintenance (units)", 0.0, 1e6, 1.0, key="mt_cp")
    cu = c1.number_input("Assumed cost per unused cycle (units)", 0.0, 1e3, 0.0, key="mt_cu")
    th = c2.multiselect("Trigger thresholds T (predicted RUL, cycles)", [10, 15, 20, 25, 30, 40, 50], default=[15, 20, 30], key="mt_th")
    top = max((float(k) for k, v in (entry["intervals"] or {}).get("thresholds", {}).items() if v is not None), default=None)
    use_iv = top is not None and c3.checkbox(f"Also trigger on interval lower bound ({top:.0%})", value=True, key="mt_iv")
    traj = fleet_predictions(str(paths.registry), str(paths.root), mid, str(paths.data_dir), "FD001", str(manifest_for("FD001")), top if use_iv else None)
    df = get_partitioned(str(paths.data_dir), "FD001", str(manifest_for("FD001")))
    lifetimes = df[df.partition == "train"].groupby("engine_id").cycle.max().to_numpy()
    ages = M.fixed_age_candidates(lifetimes)
    table = M.compare_policies(traj, sorted(th), int(lead), M.CostAssumptions(cf, cp, cu), ages, "lower" if use_iv else None)
    st.dataframe(table, hide_index=True, width="stretch", column_config={"mean_unused_cycles_when_planned": st.column_config.NumberColumn(format="%.1f"), "assumed_cost_per_engine": st.column_config.NumberColumn("assumed cost / engine (hypothetical)", format="%.2f")})
    pts = table.assign(label=table.policy + " · " + table.parameter.astype(str))
    pol = list(dict.fromkeys(pts.policy))
    chart = alt.Chart(pts).mark_point(size=120).encode(
        x=alt.X("mean_unused_cycles_when_planned:Q", title="Mean unused cycles when maintained early"),
        y=alt.Y("unscheduled_failures:Q", title="Unscheduled failures (of 15 engines)"),
        color=alt.Color("policy:N", scale=alt.Scale(domain=pol, range=["#3987e5", "#d95926", "#199e70", "#a9b4c6"][: len(pol)]), title=None),
        shape=alt.Shape("policy:N", legend=None),
        tooltip=["label", "unscheduled_failures", alt.Tooltip("mean_unused_cycles_when_planned:Q", format=".1f")],
    ).properties(height=300, title="Trade-off: failures vs wasted life (counts from the simulation)")
    st.altair_chart(chart, width="stretch")
    st.caption(
        f"Fixed-age policies replace at training-engine lifetime quantiles {ages} (no model, no validation data). "
        "The oracle uses ground-truth RUL and is unattainable; it only bounds what any policy could achieve. "
        "Model: " + mid + f" (version {entry['version']})."
    )


PAGES = {
    "Overview": page_overview,
    "Engine Health Intelligence": page_engine,
    "Model Benchmarking": page_benchmark,
    "Sensor Intelligence": page_sensors,
    "Robustness & Experiments": page_experiments,
    "Maintenance Simulation": page_maintenance,
}
if service is None and section != "Overview":
    st.error(f"Model registry not found at `{paths.registry}`. Build it with `uv run aeromind s3-registry` or set AEROMIND_REGISTRY.")
else:
    PAGES[section]()

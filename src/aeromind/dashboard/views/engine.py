"""Research mode · Engine Health Intelligence (needs local bundles; validation engines need NASA files).

Engine-level predictions only for validation engines (never used to fit the model or its
calibrator) and for uploaded trajectories (no ground truth).
"""

from __future__ import annotations

import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from aeromind.dashboard import data as D
from aeromind.dashboard import state as S
from aeromind.dashboard import ui
from aeromind.dashboard.state import Ctx
from aeromind.dashboard.views import _research as R

MAX_UPLOAD_BYTES = 2 * 2**20  # mirrors server.maxUploadSize; checked before decoding


def render(ctx: Ctx) -> None:
    from aeromind.data.cmapss import COLUMNS, SENSOR_COLUMNS
    from aeromind.serving.service import ServiceError

    st.header("Engine Health Intelligence")
    left, right = st.columns([1, 3])
    with left:
        source = st.radio("Trajectory source", ["Validation engine (held out)", "Upload CSV"], key="eng_source")
        dataset = st.selectbox("Subset", D.DATASETS, key="eng_ds")
        mid = R.model_picker(ctx, dataset, "eng_model")
    if mid is None:
        return
    svc = R.service(ctx)
    entry = svc.entry(mid)
    truth = None
    if source.startswith("Validation"):
        df = R.partitioned(ctx, dataset)
        if df is None:
            R.missing_data(ctx, dataset)
            return
        val_ids = sorted(df[df.partition == "validation"].engine_id.unique())
        with left:
            eid = st.selectbox("Validation engine", val_ids, key="eng_id")
            g = df[df.engine_id == eid].reset_index(drop=True)
            cut = st.slider("Observe up to cycle", 1, int(g.cycle.max()), int(g.cycle.max()), key="eng_cut")
        g = g[g.cycle <= cut]
        frame = g.drop(columns=["engine_id", "rul", "partition"])
        truth = g[["cycle", "rul"]]
        if int(eid) in set(entry["training"].get("engine_ids", [])):  # pragma: no cover - guarded by construction
            st.error("This engine was used to fit the model; in-sample predictions are not shown.")
            return
    else:
        with left:
            up = st.file_uploader("C-MAPSS rows for ONE engine (CSV with headers, or raw whitespace .txt)", type=["csv", "txt"], key="eng_upload")
        if up is None:
            st.info("Upload one engine's cycles: columns `cycle, setting_1..3, sensor_1..21` (an `engine_id` column is ignored). No ground truth is shown for uploads.")
            return
        if up.size > MAX_UPLOAD_BYTES:
            st.error(f"The file is larger than {MAX_UPLOAD_BYTES // 2**20} MB; upload a single engine's cycles.")
            return
        raw = up.getvalue().decode("utf-8", errors="replace")
        try:
            frame = pd.read_csv(io.StringIO(raw), sep=r"\s+", header=None, names=COLUMNS) if up.name.endswith(".txt") else pd.read_csv(io.StringIO(raw))
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
        res = S.predict_cached(str(ctx.paths.registry), str(ctx.paths.root), mid, frame, levels)
    except ServiceError as exc:
        st.error(f"{exc.code}: {exc.message}")
        return

    color = ui.FAMILY_COLOR.get(ui.family_of(mid), ui.TEXT2)
    with right:
        ui.badge("model", *(["truth"] if truth is not None else []))
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
            layers.append(base.mark_area(opacity=0.22, color=color).encode(y=alt.Y(f"lower_{lvl}:Q", title="RUL (cycles)"), y2=f"upper_{lvl}:Q"))
        long = traj[["cycle", "predicted_rul"]].rename(columns={"predicted_rul": "value"}).assign(series=f"Predicted ({mid})")
        if truth is not None:
            cap = entry["target_cap"] or 10**9
            long = pd.concat([long, truth.assign(value=np.minimum(truth.rul, cap), series="Ground truth, capped")[["cycle", "value", "series"]], truth.assign(value=truth.rul, series="Ground truth, true RUL")[["cycle", "value", "series"]]])
        dom = list(dict.fromkeys(long.series))
        rng = [color, ui.TEXT2, ui.MUTED][: len(dom)]
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
        st.altair_chart(alt.layer(*layers).properties(height=380, title="RUL history: prediction at each cycle uses only cycles up to it"), width="stretch")
        if len(traj) > 10:
            recent = traj.predicted_rul.iloc[-1] - traj.predicted_rul.iloc[-11]
            st.caption(f"Change in predicted RUL over the last 10 cycles: {recent:+.1f} cycles (−10 means degrading at the expected rate).")
        ui.provenance_box({**res["provenance"], "intervals": [{k: v for k, v in i.items() if k != "guarantee"} for i in res["intervals"]], "inference_ms": round(res["inference_ms"], 2)})
        if res["intervals"]:
            st.caption("Interval guarantee: " + res["intervals"][0]["guarantee"])

        st.subheader("Sensor trajectories")
        ui.badge("truth")
        sensors = [c for c in SENSOR_COLUMNS if c in frame and frame[c].nunique() > 3]
        pick = st.multiselect("Sensors", sensors, default=sensors[:4], key="eng_sensors", format_func=D.sensor_label, max_selections=6)
        if pick:
            s = frame[["cycle", *pick]].melt("cycle", var_name="sensor", value_name="reading")
            s["sensor"] = s["sensor"].map(D.sensor_label)
            chart = (
                alt.Chart(s)
                .mark_line(color=ui.SEQ_BLUE[2], strokeWidth=1.5)
                .encode(x=alt.X("cycle:Q", title="Cycle"), y=alt.Y("reading:Q", title=None, scale=alt.Scale(zero=False)), tooltip=["sensor", "cycle", "reading"])
                .properties(height=130, width=260)
                .facet(facet=alt.Facet("sensor:N", title=None), columns=3)
                .resolve_scale(y="independent")
            )
            st.altair_chart(chart)

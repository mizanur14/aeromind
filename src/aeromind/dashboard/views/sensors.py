"""Research mode · Sensor Intelligence (needs the NASA training files locally)."""

from __future__ import annotations

import altair as alt
import numpy as np
import streamlit as st

from aeromind.dashboard import data as D
from aeromind.dashboard import ui
from aeromind.dashboard.state import Ctx
from aeromind.dashboard.views import _research as R


def render(ctx: Ctx) -> None:
    st.header("Sensor Intelligence")
    dataset = st.selectbox("Subset", D.DATASETS, key="sn_ds")
    df = R.partitioned(ctx, dataset)
    if df is None:
        R.missing_data(ctx, dataset)
        return
    train = df[df.partition == "train"]
    ui.badge("truth")
    st.caption(f"Training-partition engines only ({train.engine_id.nunique()} engines), the scope used for model development.")
    sensors = D.informative_sensors(train)
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Correlations", "Feature importance", "Trajectory comparison"])
    with tab1:
        s = st.selectbox("Sensor", sensors, key="sn_dist", format_func=D.sensor_label)
        st.caption(D.sensor_description(s))
        h = train.assign(phase=np.where(train.rul <= 30, "last 30 cycles before failure", np.where(train.cycle <= 30, "first 30 cycles", "mid-life")))
        h = h[h.phase != "mid-life"]
        chart = (
            alt.Chart(h)
            .mark_bar(opacity=0.75, cornerRadiusEnd=2)
            .encode(
                x=alt.X(f"{s}:Q", bin=alt.Bin(maxbins=40), title=D.sensor_label(s)),
                y=alt.Y("count():Q", stack=None, title="Readings"),
                color=alt.Color("phase:N", scale=alt.Scale(domain=["first 30 cycles", "last 30 cycles before failure"], range=[ui.FAMILY_COLOR["hgb_reference"], ui.FAMILY_COLOR["xgboost"]]), title=None),
            )
            .properties(height=360, title="Early-life vs near-failure readings")
        )
        st.altair_chart(chart, width="stretch")
        if D.DATASET_INFO[dataset]["conditions"] > 1:
            st.info("Six operating conditions: raw readings are dominated by the operating regime; the models normalise per condition.")
    with tab2:
        cor = D.correlation_with_rul(train, sensors)
        cor["sensor"] = cor.sensor.map(D.sensor_label)
        bar = (
            alt.Chart(cor)
            .mark_bar(cornerRadiusEnd=4, height=12)
            .encode(
                y=alt.Y("sensor:N", sort=cor.sensor.tolist(), title=None),
                x=alt.X("spearman_rho_vs_capped_rul:Q", title="Spearman ρ with capped RUL", scale=alt.Scale(domain=[-1, 1])),
                color=alt.Color("spearman_rho_vs_capped_rul:Q", scale=alt.Scale(domain=[-1, 0, 1], range=ui.DIVERGING), legend=None),
                tooltip=["sensor", alt.Tooltip("spearman_rho_vs_capped_rul:Q", format=".2f")],
            )
            .properties(height=26 * len(cor) + 150, title="Monotone association with capped RUL (pooled rows; not causal)")
        )
        st.altair_chart(bar, width="stretch")
        m = train[sensors].corr(method="spearman").reset_index().melt("index", var_name="other", value_name="rho")
        heat = (
            alt.Chart(m)
            .mark_rect()
            .encode(
                x=alt.X("index:N", title=None, sort=sensors),
                y=alt.Y("other:N", title=None, sort=sensors),
                color=alt.Color("rho:Q", scale=alt.Scale(domain=[-1, 0, 1], range=ui.DIVERGING), title="ρ"),
                tooltip=["index", "other", alt.Tooltip("rho:Q", format=".2f")],
            )
            .properties(height=540, title="Sensor–sensor Spearman correlation")
        )
        st.altair_chart(heat, width="stretch")
    with tab3:
        xid = f"{dataset.lower()}-s2-xgboost"
        svc = R.service(ctx)
        if svc is not None and xid in svc.model_ids and svc.availability(xid)[0]:
            imp = D.xgboost_importance(svc.load(xid))
            if imp is not None:
                ui.badge("model")
                top = imp.head(20)
                chart = (
                    alt.Chart(top)
                    .mark_bar(cornerRadiusEnd=4, height=12, color=ui.FAMILY_COLOR["xgboost"])
                    .encode(
                        y=alt.Y("feature:N", sort=top.feature.tolist(), title=None),
                        x=alt.X("share:Q", title="Share of total gain", axis=alt.Axis(format="%")),
                        tooltip=["feature", alt.Tooltip("share:Q", format=".1%")],
                    )
                    .properties(height=22 * len(top) + 150, title=f"XGBoost gain importance ({xid}, top 20)")
                )
                st.altair_chart(chart, width="stretch")
                st.caption("Gain importance of the trained XGBoost bundle: how much each engineered feature reduced the training loss. Correlated features share credit; this is not a causal attribution.")
                ui.provenance_box(svc.provenance(xid))
        else:
            st.info(f"XGBoost bundle {xid} not available locally.")
    with tab4:
        s = st.selectbox("Sensor", sensors, key="sn_traj", index=min(3, len(sensors) - 1), format_func=D.sensor_label)
        engines = sorted(train.engine_id.unique())
        pick = st.multiselect("Engines (training partition)", engines, default=engines[:5], key="sn_engines", max_selections=7)
        if pick:
            t = train[train.engine_id.isin(pick)].assign(cycles_to_failure=lambda d: -d.rul, engine=lambda d: d.engine_id.astype(str))
            dom = [str(e) for e in pick]
            pal = [ui.FAMILY_COLOR[f] for f in ui.FAMILY_ORDER]
            chart = (
                alt.Chart(t)
                .mark_line(strokeWidth=1.5)
                .encode(
                    x=alt.X("cycles_to_failure:Q", title="Cycles relative to failure (0 = failure)"),
                    y=alt.Y(f"{s}:Q", title=D.sensor_label(s), scale=alt.Scale(zero=False)),
                    color=alt.Color("engine:N", scale=alt.Scale(domain=dom, range=pal[: len(dom)]), title="Engine"),
                    tooltip=["engine", "cycle", "rul", s],
                )
                .properties(height=400, title="Trajectories aligned at failure")
            )
            st.altair_chart(chart, width="stretch")

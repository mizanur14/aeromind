"""Research mode · Maintenance Simulation (hypothetical; needs local bundles and FD001 files)."""

from __future__ import annotations

import altair as alt
import streamlit as st

from aeromind.dashboard import state as S
from aeromind.dashboard import ui
from aeromind.dashboard.state import Ctx
from aeromind.dashboard.views import _research as R


def render(ctx: Ctx) -> None:
    from aeromind.serving import maintenance as M

    st.header("Maintenance Simulation")
    ui.badge("hypo")
    st.warning(
        "Hypothetical illustration. Policies are applied to the 15 FD001 validation engines (simulated, run to failure). "
        "Failure and unused-cycle **counts** come from real model predictions vs ground truth; **costs are your assumptions** "
        "in arbitrary units. Nothing here estimates real maintenance savings or safety outcomes."
    )
    mid = R.model_picker(ctx, "FD001", "mt_model")
    if mid is None:
        return
    df = R.partitioned(ctx, "FD001")
    if df is None:
        R.missing_data(ctx, "FD001")
        return
    entry = R.service(ctx).entry(mid)
    c1, c2, c3 = st.columns(3)
    lead = c1.number_input("Lead time to schedule maintenance (cycles)", 0, 100, 10, key="mt_lead")
    cf = c2.number_input("Assumed cost: unscheduled failure (units)", 0.0, 1e6, 10.0, key="mt_cf")
    cp = c3.number_input("Assumed cost: planned maintenance (units)", 0.0, 1e6, 1.0, key="mt_cp")
    cu = c1.number_input("Assumed cost per unused cycle (units)", 0.0, 1e3, 0.0, key="mt_cu")
    th = c2.multiselect("Trigger thresholds T (predicted RUL, cycles)", [10, 15, 20, 25, 30, 40, 50], default=[15, 20, 30], key="mt_th")
    top = max((float(k) for k, v in (entry["intervals"] or {}).get("thresholds", {}).items() if v is not None), default=None)
    use_iv = top is not None and c3.checkbox(f"Also trigger on interval lower bound ({top:.0%})", value=True, key="mt_iv")
    traj = S.fleet_predictions(str(ctx.paths.registry), str(ctx.paths.root), mid, str(ctx.paths.data_dir), "FD001", str(ctx.paths.split_manifest("FD001")), top if use_iv else None)
    lifetimes = df[df.partition == "train"].groupby("engine_id").cycle.max().to_numpy()
    ages = M.fixed_age_candidates(lifetimes)
    table = M.compare_policies(traj, sorted(th), int(lead), M.CostAssumptions(cf, cp, cu), ages, "lower" if use_iv else None)
    st.dataframe(table, hide_index=True, width="stretch", column_config={"mean_unused_cycles_when_planned": st.column_config.NumberColumn(format="%.1f"), "assumed_cost_per_engine": st.column_config.NumberColumn("assumed cost / engine (hypothetical)", format="%.2f")})
    pts = table.assign(label=table.policy + " · " + table.parameter.astype(str))
    pol = list(dict.fromkeys(pts.policy))
    chart = (
        alt.Chart(pts)
        .mark_point(size=120, filled=True)
        .encode(
            x=alt.X("mean_unused_cycles_when_planned:Q", title="Mean unused cycles when maintained early"),
            y=alt.Y("unscheduled_failures:Q", title="Unscheduled failures (of 15 engines)"),
            color=alt.Color("policy:N", scale=alt.Scale(domain=pol, range=["#3987e5", "#d95926", "#199e70", ui.TEXT2][: len(pol)]), title=None),
            shape=alt.Shape("policy:N", legend=None),
            tooltip=["label", "unscheduled_failures", alt.Tooltip("mean_unused_cycles_when_planned:Q", format=".1f")],
        )
        .properties(height=380, title="Trade-off: failures vs wasted life (counts from the simulation)")
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        f"Fixed-age policies replace at training-engine lifetime quantiles {ages} (no model, no validation data). "
        "The oracle uses ground-truth RUL and is unattainable; it only bounds what any policy could achieve. "
        f"Model: {mid} (version {entry['version']})."
    )

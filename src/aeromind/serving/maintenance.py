"""Hypothetical maintenance-policy simulation on run-to-failure validation engines.

This is an *illustration*, not a savings estimate. Outcome counts (unscheduled failures,
unused cycles) are computed from real model predictions on simulated C-MAPSS validation
engines whose failure cycle is known; any monetary figure is the user's assumed cost
multiplied by those counts.

Policy model (one decision per cycle, every cycle inspected):

- a policy *triggers* at the first cycle where its signal ≤ threshold;
- maintenance happens ``lead_time`` cycles after the trigger;
- if the engine fails first (true RUL at trigger < lead_time) or the policy never triggers,
  the outcome is an **unscheduled failure**; otherwise the remaining true RUL at maintenance
  is counted as **unused life**.

Policies: point prediction, conformal lower bound (when calibrated), fixed age (no model),
and an *oracle* that uses ground-truth RUL (unattainable; a reference bound only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostAssumptions:
    """User-assumed costs in arbitrary units. No default represents a real-world value."""

    unscheduled_failure: float
    planned_maintenance: float
    per_unused_cycle: float = 0.0


def simulate_policy(
    traj: pd.DataFrame,
    signal: str | None,
    threshold: float,
    lead_time: int,
    fixed_age: int | None = None,
) -> pd.DataFrame:
    """Per-engine outcome for one policy.

    ``traj`` has columns ``engine_id, cycle, rul`` (true, uncapped) and the ``signal`` column.
    ``signal=None`` with ``fixed_age`` simulates age-based maintenance at that cycle.
    """
    rows = []
    for eid, g in traj.sort_values(["engine_id", "cycle"]).groupby("engine_id", sort=True):
        if signal is None:
            if fixed_age is None:
                raise ValueError("fixed_age is required when signal is None")
            hit = g[g["cycle"] >= fixed_age]
            trigger_rul = float(hit["rul"].iloc[0]) + 0.0 if len(hit) else None
            lead = 0
        else:
            hit = g[g[signal] <= threshold]
            trigger_rul = float(hit["rul"].iloc[0]) if len(hit) else None
            lead = lead_time
        if trigger_rul is None or trigger_rul < lead:
            rows.append({"engine_id": eid, "failure": True, "unused_cycles": 0.0, "trigger_rul": trigger_rul})
        else:
            rows.append({"engine_id": eid, "failure": False, "unused_cycles": trigger_rul - lead, "trigger_rul": trigger_rul})
    return pd.DataFrame(rows)


def summarise(outcomes: pd.DataFrame, costs: CostAssumptions) -> dict[str, float]:
    """Counts from the simulation plus the cost implied by the *assumed* unit costs."""
    n = len(outcomes)
    fails = int(outcomes["failure"].sum())
    unused = float(outcomes["unused_cycles"].sum())
    cost = fails * costs.unscheduled_failure + (n - fails) * costs.planned_maintenance + unused * costs.per_unused_cycle
    return {
        "engines": n,
        "unscheduled_failures": fails,
        "planned_maintenance": n - fails,
        "mean_unused_cycles_when_planned": float(unused / (n - fails)) if n > fails else float("nan"),
        "assumed_cost_total": float(cost),
        "assumed_cost_per_engine": float(cost / n) if n else float("nan"),
    }


def compare_policies(
    traj: pd.DataFrame,
    thresholds: list[float],
    lead_time: int,
    costs: CostAssumptions,
    fixed_ages: list[int] | None = None,
    lower_col: str | None = None,
) -> pd.DataFrame:
    """Table of policies × thresholds. ``traj`` needs ``pred`` (and optionally ``lower_col``)."""
    out = []
    for t in thresholds:
        out.append({"policy": "point prediction ≤ T", "parameter": t, **summarise(simulate_policy(traj, "pred", t, lead_time), costs)})
        if lower_col:
            out.append({"policy": "interval lower bound ≤ T", "parameter": t, **summarise(simulate_policy(traj, lower_col, t, lead_time), costs)})
    for age in fixed_ages or []:
        out.append({"policy": "fixed age (no model)", "parameter": age, **summarise(simulate_policy(traj, None, 0, lead_time, age), costs)})
    oracle = traj.assign(oracle=traj["rul"].astype(float))
    out.append({"policy": "oracle: true RUL (unattainable)", "parameter": lead_time, **summarise(simulate_policy(oracle, "oracle", lead_time, lead_time), costs)})
    return pd.DataFrame(out)


def fixed_age_candidates(lifetimes: np.ndarray, quantiles: tuple[float, ...] = (0.05, 0.1, 0.25)) -> list[int]:
    """Age-replacement cycles from *training-engine* lifetime quantiles (no validation data)."""
    return sorted({int(np.quantile(lifetimes, q)) for q in quantiles})

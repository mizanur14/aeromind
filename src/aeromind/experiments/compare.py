"""Compare finished runs on validation (and test, where evaluated)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aeromind.experiments.common import load_run
from aeromind.utils import read_json


def compare_runs(run_dirs: list[str | Path], output: str | Path | None = None) -> pd.DataFrame:
    """One row per run with its selected model's validation (and last test) metrics.

    All error columns are labelled with their truth: ``_true`` (uncapped) or ``_capped``.
    """
    rows = []
    for rd in map(Path, run_dirs):
        config, meta = load_run(rd)
        summary = read_json(rd / "report" / "validation_summary.json")
        sel = meta["selected_model"]
        snap = summary["validation"][sel]["snapshot"]
        row = {
            "run": rd.name,
            "dataset": config.dataset,
            "target_cap": config.target.cap,
            "selected_model": sel,
            "val_snapshot_rmse_true": snap["rmse_true"],
            "val_snapshot_mae_true": snap["mae_true"],
            "val_snapshot_rmse_capped": snap["rmse_capped"],
            "val_snapshot_phm08_sum_true": snap["phm08_sum_true"],
            "n_features": meta.get("n_features"),
            "total_seconds": meta.get("total_seconds"),
        }
        tests = meta.get("test_evaluations") or []
        if tests:
            m = tests[-1]["selected_metrics"]
            row.update(
                {
                    "test_rmse_true": m["rmse_true"],
                    "test_mae_true": m["mae_true"],
                    "test_rmse_capped": m["rmse_capped"],
                    "test_phm08_sum_true": m["phm08_sum_true"],
                }
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output, index=False)
    return table

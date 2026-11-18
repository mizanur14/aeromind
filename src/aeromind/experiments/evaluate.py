"""Re-evaluate a finished run from its saved bundles (proves save/load consistency)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from aeromind.experiments.common import load_development_data, load_run
from aeromind.experiments.train import evaluate_on_validation
from aeromind.models.bundle import load_bundle
from aeromind.utils import write_json

logger = logging.getLogger(__name__)


def evaluate_run(
    run_dir: str | Path, pool: pd.DataFrame | None = None, atol: float = 1e-9, publish: bool = True
) -> dict:
    """Reload every bundle, recompute validation predictions and metrics, compare to saved.

    Raises:
        AssertionError: if reloaded predictions differ from the saved ones by more than ``atol``.
    """
    run_dir = Path(run_dir)
    config, metadata = load_run(run_dir)
    data = load_development_data(config, pool)
    saved = pd.read_csv(run_dir / "predictions" / "validation_windows.csv")
    results: dict[str, dict] = {}
    for family, rel in metadata["models"].items():
        predictor, _ = load_bundle(run_dir / rel)
        metrics, pred = evaluate_on_validation(predictor, predictor.inputs(data.validation), data)
        ref = saved.loc[saved["model"] == family, "pred"].to_numpy()
        max_diff = float(np.max(np.abs(ref - pred))) if len(ref) == len(pred) else float("inf")
        results[family] = {
            "max_abs_diff_vs_saved": max_diff,
            "snapshot_rmse_target": metrics["snapshot"]["rmse_target"],
            "snapshot_rmse_true": metrics["snapshot"]["rmse_true"],
            "snapshot_rmse_capped": metrics["snapshot"]["rmse_capped"],
        }
        logger.info("%-24s reloaded: max |Δpred| = %.3g", family, max_diff)
    write_json(run_dir / "report" / "reload_check.json", results)
    if publish:
        write_json(
            Path(config.paths.reports_dir) / config.dataset / "runs" / run_dir.name / "reload_check.json", results
        )
    bad = {k: v["max_abs_diff_vs_saved"] for k, v in results.items() if not v["max_abs_diff_vs_saved"] <= atol}
    if bad:
        raise AssertionError(f"Reloaded predictions differ from saved ones: {bad}")
    return results

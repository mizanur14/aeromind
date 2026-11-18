"""Fast end-to-end smoke test on synthetic data (seconds, no NASA files)."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from aeromind.config import ExperimentConfig
from aeromind.data.cmapss import load_train
from aeromind.data.splits import make_engine_split, save_split
from aeromind.experiments.evaluate import evaluate_run
from aeromind.experiments.test_eval import run_test
from aeromind.experiments.train import run_training
from aeromind.synthetic import write_raw_files

logger = logging.getLogger(__name__)


def smoke_config(root: Path) -> ExperimentConfig:
    """Tiny configuration exercising every model family."""
    return ExperimentConfig.from_dict(
        {
            "name": "smoke",
            "dataset": "FD001",
            "paths": {
                "raw_dir": str(root / "raw"),
                "manifest_dir": str(root / "manifests"),
                "artifacts_dir": str(root / "runs"),
                "reports_dir": str(root / "reports"),
                "processed_dir": str(root / "processed"),
            },
            "target": {"cap": 60},
            "features": {"windows": [5, 10]},
            "evaluation": {"snapshots_per_engine": 4, "capped_truth_cap": 60, "latency_repeats": 5},
            "training": {"n_jobs": 2},
            "models": {
                "median": {},
                "mean_residual_life": {"params": {"min_engines": 2}},
                "ridge": {"grid": {"alpha": [1.0, 10.0]}},
                "random_forest": {"params": {"n_estimators": 20, "max_depth": 6}},
                "hist_gradient_boosting": {"params": {"max_iter": 30}},
                "xgboost": {"params": {"n_estimators": 30, "max_depth": 3}},
            },
        }
    )


def run_smoke(workdir: str | Path | None = None) -> dict:
    """Synthetic data → split → train → reload check → test evaluation."""
    ctx = tempfile.TemporaryDirectory() if workdir is None else None
    root = Path(ctx.name if ctx else workdir)
    try:
        cfg = smoke_config(root)
        write_raw_files(cfg.paths.raw_dir, n_train=30, n_test=10)
        pool = load_train(cfg.paths.raw_dir, cfg.dataset)
        save_split(make_engine_split(pool["engine_id"].unique(), cfg.dataset, cfg.split.seed), cfg.split_manifest_path)
        run_dir = run_training(cfg)
        reload = evaluate_run(run_dir)  # smoke paths live in a temp dir
        test = run_test(run_dir)
        result = {"run_dir": str(run_dir), "reload": reload, "test_selected": test["selected_metrics"]}
        logger.info(
            "Smoke test passed: selected %s, synthetic test RMSE(true)=%.2f",
            test["models"][0],
            test["selected_metrics"]["rmse_true"],
        )
        return result
    finally:
        if ctx:
            ctx.cleanup()

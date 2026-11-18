"""Final evaluation on the official NASA test set.

One prediction per test engine at its last observed cycle, compared with
``RUL_FD00x.txt``. This is the only workflow that reads official test labels. By default
only the model selected on validation is evaluated; ``all_models=True`` adds a post-hoc
table that is explicitly marked as *not used for selection*.

Test-set policy (enforced here and in the CLI):

- A run can be evaluated on the official test set **once**; repeats need ``allow_repeat``.
- Every evaluation is appended to ``<reports_dir>/test_set_ledger.json``. Earlier entries
  for the same subset are logged as warnings, because every additional configuration
  evaluated on the test set weakens it as an unbiased estimate.
- Test results must never feed back into feature, model, hyperparameter, target or
  calibration choices; those are made on validation engines only.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from aeromind.data.cmapss import load_test
from aeromind.data.provenance import dataset_hashes
from aeromind.evaluation import plots
from aeromind.evaluation.protocol import final_cycle_index, metrics_against_truths
from aeromind.experiments.common import load_run, publish_reports, sync_published_metadata
from aeromind.models.bundle import load_bundle
from aeromind.utils import read_json, write_json

logger = logging.getLogger(__name__)

LEDGER_NAME = "test_set_ledger.json"
LEDGER_POLICY = (
    "Official test labels are for final reporting only. Entries here must not be used to "
    "choose features, models, hyperparameters, targets or calibration. Model development "
    "selects on validation engines."
)


class TestSetPolicyError(ValueError):
    """Raised when an official test evaluation would violate the test-set policy."""

    __test__ = False  # not a pytest test class


def ledger_path(reports_dir: str | Path) -> Path:
    """Location of the official-test ledger."""
    return Path(reports_dir) / LEDGER_NAME


def read_ledger(reports_dir: str | Path) -> dict:
    """Return the ledger (empty structure if absent)."""
    path = ledger_path(reports_dir)
    return read_json(path) if path.is_file() else {"policy": LEDGER_POLICY, "entries": []}


def append_ledger(reports_dir: str | Path, entry: dict) -> Path:
    """Append one evaluation record to the ledger."""
    ledger = read_ledger(reports_dir)
    ledger["policy"] = LEDGER_POLICY
    ledger["entries"].append(entry)
    return write_json(ledger_path(reports_dir), ledger)


def run_test(
    run_dir: str | Path,
    all_models: bool = False,
    test_df: pd.DataFrame | None = None,
    publish: bool = True,
    allow_repeat: bool = False,
    reason: str = "",
) -> dict:
    """Evaluate the selected (or every) model of a run on the official test set.

    Raises:
        TestSetPolicyError: the run was already evaluated and ``allow_repeat`` is False.
    """
    run_dir = Path(run_dir)
    config, metadata = load_run(run_dir)
    previous = metadata.get("test_evaluations") or []
    if previous and not allow_repeat:
        raise TestSetPolicyError(
            f"{run_dir.name} was already evaluated on the official test set at "
            f"{previous[-1]['evaluated_utc']}. Re-evaluation is refused (results are already recorded); "
            "pass allow_repeat/--allow-repeat with a --reason only if genuinely necessary."
        )
    prior = [e for e in read_ledger(config.paths.reports_dir)["entries"] if e.get("dataset") == config.dataset]
    if prior:
        logger.warning(
            "%s official test set already used by %d earlier evaluation(s): %s",
            config.dataset,
            len(prior),
            sorted({e["run"] for e in prior}),
        )
    test = test_df if test_df is not None else load_test(config.paths.raw_dir, config.dataset)
    test = test.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    last = final_cycle_index(test)

    families = list(metadata["models"]) if all_models else [metadata["selected_model"]]
    out_dir = run_dir / "report" / "test"
    rows, per_engine = [], []
    for family in families:
        predictor, _ = load_bundle(run_dir / metadata["models"][family])
        # Features use each engine's full observed history; we keep the last row only.
        pred = predictor.predict(test)[test.index.get_indexer(last)]
        truth = test.loc[last, "rul"].to_numpy(dtype=float)
        m = metrics_against_truths(truth, pred, config.target.cap, config.evaluation.capped_truth_cap)
        rows.append({"model": family, "selected_on_validation": family == metadata["selected_model"], **m})
        frame = test.loc[last, ["engine_id", "cycle", "rul"]].copy()
        frame["pred"] = pred
        frame["model"] = family
        per_engine.append(frame)
        logger.info(
            "TEST %-24s RMSE(true)=%.2f MAE(true)=%.2f PHM08=%.1f",
            family,
            m["rmse_true"],
            m["mae_true"],
            m["phm08_sum_true"],
        )

    table = pd.DataFrame(rows).sort_values("rmse_true").reset_index(drop=True)
    preds = pd.concat(per_engine, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "test_metrics.csv", index=False)
    preds.to_csv(out_dir / "test_final_cycle_predictions.csv", index=False)
    sel = preds[preds["model"] == metadata["selected_model"]]
    plots.plot_pred_vs_true(
        sel["rul"],
        sel["pred"],
        f"{config.dataset} official test — {metadata['selected_model']} (last cycle)",
        out_dir / "test_pred_vs_true.png",
        config.target.cap,
    )
    plots.plot_residuals(
        sel["rul"], sel["pred"], f"{config.dataset} official test residuals", out_dir / "test_residuals.png"
    )

    record = {
        "evaluated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "models": families,
        "all_models_post_hoc": all_models,
        "note": "Only the validation-selected model is the reported result; others are post-hoc.",
        "data_sha256": dataset_hashes(config.paths.raw_dir, config.dataset, include_test_labels=True)
        if test_df is None
        else {"synthetic": True},
        "selected_metrics": table[table["selected_on_validation"]].iloc[0].to_dict(),
        "reason": reason,
        "repeat": bool(previous),
    }
    write_json(out_dir / "test_summary.json", record)
    meta_path = run_dir / "metadata.json"
    meta = read_json(meta_path)
    meta["official_test_labels_read"] = True
    meta.setdefault("test_evaluations", []).append(record)
    write_json(meta_path, meta)
    if len(meta["test_evaluations"]) > 1:
        logger.warning("This run has now been evaluated on the test set %d times", len(meta["test_evaluations"]))
    sync_published_metadata(run_dir, config, publish)
    append_ledger(
        config.paths.reports_dir,
        {
            "run": run_dir.name,
            "dataset": config.dataset,
            "config_name": config.name,
            "target_cap": config.target.cap,
            "selected_model": metadata["selected_model"],
            "selection_finished_utc": metadata.get("finished_utc"),
            **{k: record[k] for k in ("evaluated_utc", "models", "all_models_post_hoc", "reason", "repeat")},
            "selected_rmse_true": record["selected_metrics"]["rmse_true"],
            "selected_mae_true": record["selected_metrics"]["mae_true"],
            "selected_phm08_sum_true": record["selected_metrics"]["phm08_sum_true"],
            "selected_rmse_capped": record["selected_metrics"]["rmse_capped"],
        },
    )
    if publish:
        publish_reports(out_dir, Path(config.paths.reports_dir) / config.dataset / "runs" / run_dir.name / "test")
    return record

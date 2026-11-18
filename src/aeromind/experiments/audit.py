"""Research-integrity audit of finished runs (``aeromind audit``).

The audit never reads official test labels (``RUL_FD00x.txt``) and never trains. It checks:

1. **Static label access** — which source modules call the label readers.
2. **Engine independence** — manifest hash/IDs vs run metadata, pipeline fit engines,
   validation prediction engines, calibration engines absent everywhere.
3. **Selection integrity** — selected model is the argmin of the recorded selection metric,
   the per-family winners match the search log, and selection finished before any official
   test evaluation.
4. **Protocol consistency** — identical snapshot sets across models and across runs.
5. **Reproducibility** — reloaded bundles reproduce saved validation predictions; saved test
   predictions are reproduced from *unlabelled* test inputs; recorded test metrics are
   recomputed from the saved per-engine file.
6. **Resource claims** — recorded runtime/memory, plus measured load/predict time and peak
   RSS of this inference-only audit process.
"""

from __future__ import annotations

import ast
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aeromind.data.cmapss import load_test_features_only
from aeromind.evaluation.protocol import final_cycle_index, metrics_against_truths
from aeromind.experiments.common import load_development_data, load_run
from aeromind.experiments.test_eval import read_ledger
from aeromind.experiments.train import SELECTION_KEYS, evaluate_on_validation
from aeromind.models.bundle import load_bundle
from aeromind.utils import peak_rss_mb, read_json, sha256_file, write_json

logger = logging.getLogger(__name__)

LABEL_READERS = frozenset({"load_test", "read_test_rul", "load_cmapss", "add_test_rul"})
ALLOWED_LABEL_MODULES = frozenset({"data/cmapss.py", "experiments/test_eval.py"})


def scan_label_access(package_root: str | Path) -> dict[str, list[str]]:
    """Map ``module path -> [called label readers]`` for every call site in the package (AST based)."""
    root = Path(package_root)
    found: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
                if name in LABEL_READERS:
                    calls.add(name)
        if calls:
            found[path.relative_to(root).as_posix()] = sorted(calls)
    return found


def _check(results: list[dict[str, Any]], name: str, ok: bool, detail: Any = None) -> bool:
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    (logger.info if ok else logger.error)("%s %s %s", "PASS" if ok else "FAIL", name, "" if ok else detail)
    return bool(ok)


def audit_run(run_dir: str | Path, manifest_dir: str | Path | None = None) -> dict[str, Any]:
    """Audit one run directory; returns checks plus measured numbers."""
    run_dir = Path(run_dir)
    config, meta = load_run(run_dir)
    checks: list[dict[str, Any]] = []
    measured: dict[str, Any] = {}

    # ---- engine independence
    manifest_path = config.split_manifest_path
    _check(checks, "split manifest hash matches run", sha256_file(manifest_path) == meta["split_manifest_sha256"])
    manifest = read_json(manifest_path)
    eng = meta["engines"]
    _check(
        checks,
        "run engine IDs match manifest",
        eng["train"] == manifest["train_engine_ids"]
        and eng["validation"] == manifest["validation_engine_ids"]
        and eng["calibration_untouched"] == manifest["calibration_engine_ids"],
    )
    sets = [set(eng["train"]), set(eng["validation"]), set(eng["calibration_untouched"])]
    _check(checks, "partitions disjoint", not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]))
    fp = read_json(run_dir / "report" / "feature_pipeline.json")
    _check(checks, "feature pipeline fitted on train engines only", fp["fitted_engine_ids"] == eng["train"])
    windows = pd.read_csv(run_dir / "predictions" / "validation_windows.csv")
    snaps = pd.read_csv(run_dir / "predictions" / "validation_snapshots.csv")
    _check(checks, "validation predictions cover exactly validation engines", set(windows.engine_id) == sets[1])
    _check(
        checks,
        "no calibration/train engine in any prediction file",
        not (set(windows.engine_id) | set(snaps.engine_id)) & (sets[0] | sets[2]),
    )
    prov = read_json(Path(manifest_dir or config.paths.manifest_dir) / "cmapss_provenance.json")["files"]
    _check(
        checks,
        "run data hashes match provenance manifest",
        all(prov[name]["sha256"] == h for name, h in meta["data_sha256"].items()),
        meta["data_sha256"],
    )
    _check(
        checks,
        "training run did not hash/read test labels",
        f"RUL_{config.dataset}.txt" not in meta["data_sha256"],
    )

    # ---- protocol consistency within the run
    n_expected = len(eng["validation"]) * config.evaluation.snapshots_per_engine
    keys = {m: frozenset(map(tuple, g[["engine_id", "cycle"]].to_numpy())) for m, g in snaps.groupby("model")}
    _check(checks, "same snapshot set for every model", len(set(keys.values())) == 1)
    _check(
        checks,
        "snapshot count = engines x snapshots_per_engine",
        all(len(k) == n_expected for k in keys.values()),
        n_expected,
    )
    per_engine = snaps[snaps.model == meta["selected_model"]].groupby("engine_id").size()
    _check(checks, "snapshots engine-balanced", per_engine.nunique() == 1, per_engine.unique().tolist())

    # ---- selection integrity
    sel_col = f"snapshot_{SELECTION_KEYS[config.evaluation.selection_metric]}"
    comp = pd.read_csv(run_dir / "report" / "validation_model_comparison.csv")
    search = pd.read_csv(run_dir / "report" / "search_results.csv")
    _check(
        checks,
        "selected model is argmin of validation selection metric",
        comp.loc[comp[sel_col].idxmin(), "model"] == meta["selected_model"],
    )
    best = search.loc[search.groupby("model")[sel_col].idxmin()].set_index("model")[sel_col]
    _check(
        checks,
        "per-family winners match search log",
        np.allclose(best.loc[comp.model].to_numpy(), comp[sel_col].to_numpy()),
    )
    measured["n_fits"] = len(search)
    tests = meta.get("test_evaluations") or []
    if tests:
        _check(
            checks,
            "selection finished before (or same second as) first official test evaluation",
            meta["finished_utc"] <= tests[0]["evaluated_utc"],  # 1-s resolution; metadata exists before any test run
            {"selection_finished": meta["finished_utc"], "first_test": tests[0]["evaluated_utc"]},
        )
    measured["official_test_evaluations"] = len(tests)

    # ---- reproducibility (validation)
    data = load_development_data(config)
    reload: dict[str, Any] = {}
    for family, rel in meta["models"].items():
        t0 = time.perf_counter()
        predictor, _ = load_bundle(run_dir / rel)
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        X = predictor.inputs(data.validation)
        metrics, pred = evaluate_on_validation(predictor, X, data)
        pred_s = time.perf_counter() - t0
        ref = windows.loc[windows.model == family, "pred"].to_numpy()
        reload[family] = {
            "max_abs_diff": float(np.max(np.abs(ref - pred))),
            "snapshot_rmse_target_recomputed": metrics["snapshot"]["rmse_target"],
            "snapshot_rmse_target_recorded": float(comp.set_index("model").loc[family, "snapshot_rmse_target"]),
            "bundle_load_ms": load_s * 1e3,
            "features_plus_predict_validation_ms": pred_s * 1e3,
        }
    _check(
        checks,
        "reloaded bundles reproduce saved validation predictions (<=1e-9)",
        all(r["max_abs_diff"] <= 1e-9 for r in reload.values()),
    )
    _check(
        checks,
        "recomputed validation metrics equal recorded",
        all(
            np.isclose(r["snapshot_rmse_target_recomputed"], r["snapshot_rmse_target_recorded"])
            for r in reload.values()
        ),
    )
    measured["reload"] = reload

    # ---- reproducibility (test predictions, WITHOUT labels)
    test_csv = run_dir / "report" / "test" / "test_final_cycle_predictions.csv"
    if tests and test_csv.is_file():
        saved = pd.read_csv(test_csv)
        test_inputs = load_test_features_only(config.paths.raw_dir, config.dataset)
        last = final_cycle_index(test_inputs)
        diffs = {}
        for family in saved.model.unique():
            predictor, _ = load_bundle(run_dir / meta["models"][family])
            pred = predictor.predict(test_inputs)[test_inputs.index.get_indexer(last)]
            ref = saved[saved.model == family].sort_values("engine_id")
            same_ids = np.array_equal(ref.engine_id.to_numpy(), test_inputs.loc[last, "engine_id"].to_numpy())
            diffs[family] = float(np.max(np.abs(ref.pred.to_numpy() - pred))) if same_ids else float("inf")
        _check(
            checks,
            "saved test predictions reproduced from unlabelled test inputs",
            all(d <= 1e-9 for d in diffs.values()),
            diffs,
        )
        # Arithmetic check of the recorded numbers from the saved per-engine file (no new evaluation).
        recorded = pd.read_csv(run_dir / "report" / "test" / "test_metrics.csv").set_index("model")
        ok = True
        for family, g in saved.groupby("model"):
            m = metrics_against_truths(
                g.rul.to_numpy(float), g.pred.to_numpy(float), config.target.cap, config.evaluation.capped_truth_cap
            )
            ok &= all(
                np.isclose(m[k], recorded.loc[family, k])
                for k in ("rmse_true", "mae_true", "phm08_sum_true", "rmse_capped")
            )
        _check(checks, "recorded test metrics recomputed from saved per-engine predictions", ok)
        _check(
            checks,
            "test-label hash recorded at test time matches provenance",
            all(prov[n]["sha256"] == h for n, h in tests[0]["data_sha256"].items()),
        )
    # ---- published metadata in sync
    pub = Path(config.paths.reports_dir) / config.dataset / "runs" / run_dir.name / "metadata.json"
    if pub.is_file():
        p = read_json(pub)
        _check(
            checks,
            "published metadata reflects test evaluations",
            p.get("official_test_labels_read") == meta.get("official_test_labels_read")
            and len(p.get("test_evaluations", [])) == len(tests),
        )

    measured["recorded"] = {
        "total_seconds": meta["total_seconds"],
        "sum_fit_seconds": float(search.fit_seconds.sum()),
        "peak_rss_mb_training_process": meta["peak_rss_mb"],
        "model_file_mb": {f: (run_dir / r).stat().st_size / 2**20 for f, r in meta["models"].items()},
    }
    return {"run": run_dir.name, "target_cap": config.target.cap, "checks": checks, "measured": measured}


def audit_runs(
    run_dirs: list[str | Path], output: str | Path, package_root: str | Path | None = None
) -> dict[str, Any]:
    """Audit several runs plus cross-run and static checks; write a JSON report."""
    import aeromind

    package_root = Path(package_root or Path(aeromind.__file__).parent)
    global_checks: list[dict[str, Any]] = []
    access = scan_label_access(package_root)
    _check(
        global_checks,
        "label readers only called in allowed modules",
        set(access) <= ALLOWED_LABEL_MODULES,
        access,
    )
    runs = [audit_run(rd) for rd in run_dirs]

    if len(run_dirs) > 1:
        snapsets, manifests, feats = [], [], []
        for rd in map(Path, run_dirs):
            s = pd.read_csv(rd / "predictions" / "validation_snapshots.csv")
            snapsets.append(frozenset(map(tuple, s[["engine_id", "cycle"]].drop_duplicates().to_numpy())))
            manifests.append(read_json(rd / "metadata.json")["split_manifest_sha256"])
            feats.append(tuple(read_json(rd / "report" / "feature_pipeline.json")["feature_names"]))
        _check(global_checks, "identical validation snapshots across runs", len(set(snapsets)) == 1)
        _check(global_checks, "identical split manifest across runs", len(set(manifests)) == 1)
        _check(global_checks, "identical feature set across runs", len(set(feats)) == 1)

    configs_meta = [(load_run(rd)[0], read_json(Path(rd) / "metadata.json")) for rd in run_dirs]
    reports_dir = configs_meta[0][0].paths.reports_dir
    ledger = read_ledger(reports_dir)["entries"]
    recorded = sorted((m["run"], t["evaluated_utc"]) for _, m in configs_meta for t in m.get("test_evaluations") or [])
    in_ledger = sorted(
        (e["run"], e["evaluated_utc"]) for e in ledger if e["run"] in {m["run"] for _, m in configs_meta}
    )
    _check(
        global_checks,
        "test ledger matches run metadata",
        recorded == in_ledger,
        {"metadata": recorded, "ledger": in_ledger},
    )

    report = {
        "global_checks": global_checks,
        "runs": runs,
        "audit_process_peak_rss_mb": peak_rss_mb(),
        "all_passed": all(c["ok"] for c in global_checks) and all(c["ok"] for r in runs for c in r["checks"]),
        "notes": [
            "The audit reads no official test labels and trains nothing.",
            "Peak RSS covers this inference-only process (data loading, all bundles, predictions).",
        ],
    }
    write_json(output, report)
    return report

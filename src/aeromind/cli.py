"""AeroMind command-line interface.

Usage: ``uv run aeromind <command> ...`` or ``uv run python -m aeromind <command> ...``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from aeromind.utils import setup_logging

logger = logging.getLogger("aeromind")

DEFAULT_RAW = "data/raw/cmapss"
DEFAULT_MANIFESTS = "data/manifests"
PROVENANCE = "cmapss_provenance.json"


def _cmd_validate(args: argparse.Namespace) -> int:
    from aeromind.data.cmapss import DATASETS, validate_dataset
    from aeromind.data.provenance import verify_provenance, write_provenance

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    ok = True
    for ds in datasets:
        report = validate_dataset(args.data_dir, ds)
        ok &= report.ok
        status = "OK" if report.ok else "FAILED"
        print(f"[{status}] {ds}: {json.dumps(report.stats.get('train', {}))}")
        print(f"        test: {json.dumps(report.stats.get('test', {}))}")
        for w in report.warnings:
            print(f"        warning: {w}")
        for e in report.errors:
            print(f"        error: {e}")
    manifest = Path(args.manifest_dir) / PROVENANCE
    if args.write_manifest:
        write_provenance(args.data_dir, manifest)
        print(f"Provenance manifest written: {manifest}")
    elif manifest.is_file():
        problems = verify_provenance(args.data_dir, manifest, tuple(datasets))
        print("Provenance: " + ("all hashes match" if not problems else "; ".join(problems)))
        ok &= not problems
    else:
        print(f"Provenance manifest {manifest} not found (create with --write-manifest)")
    return 0 if ok else 1


def _cmd_split(args: argparse.Namespace) -> int:
    from aeromind.data.cmapss import dataset_paths, load_train
    from aeromind.data.splits import default_split_path, make_engine_split, save_split
    from aeromind.utils import sha256_file

    pool = load_train(args.data_dir, args.dataset)
    manifest = make_engine_split(pool["engine_id"].unique(), args.dataset, args.seed, tuple(args.fractions))
    manifest.source_sha256 = sha256_file(dataset_paths(args.data_dir, args.dataset)["train"])
    path = Path(args.output) if args.output else default_split_path(args.manifest_dir, args.dataset, args.seed)
    save_split(manifest, path, force=args.force)
    for p in ("train", "validation", "calibration"):
        print(f"{p:12s} {len(manifest.ids(p)):4d} engines")
    print(f"Manifest: {path}")
    return 0


def _cmd_eda(args: argparse.Namespace) -> int:
    from aeromind.data.splits import default_split_path
    from aeromind.eda.report import run_eda

    manifest = args.manifest or default_split_path(args.manifest_dir, args.dataset, args.seed)
    for scope in ("pool", "train") if args.scope == "both" else (args.scope,):
        out = run_eda(args.data_dir, args.dataset, scope, manifest, args.reports_dir, args.cap)
        print(f"EDA ({scope}) written to {out}")
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    from aeromind.config import load_config
    from aeromind.experiments.features import prepare_features

    out = prepare_features(load_config(args.config), args.output)
    print(f"Features written to {out}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from aeromind.config import load_config
    from aeromind.experiments.train import run_training

    run_dir = run_training(load_config(args.config), args.run_dir, publish=not args.no_publish)
    print(f"Run directory: {run_dir}")
    print((run_dir / "report" / "validation_model_comparison.csv").read_text().splitlines()[0])
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from aeromind.experiments.evaluate import evaluate_run

    results = evaluate_run(args.run_dir)
    for fam, r in results.items():
        print(f"{fam:24s} max|Δpred|={r['max_abs_diff_vs_saved']:.2e} snapshot RMSE true={r['snapshot_rmse_true']:.2f}")
    return 0


def _cmd_test(args: argparse.Namespace) -> int:
    from aeromind.experiments.test_eval import run_test

    if not args.confirm_final:
        logger.error(
            "Official test evaluation is final reporting, not development. Re-run with --confirm-final "
            "once model selection on validation is complete (see docs/METHODOLOGY.md)."
        )
        return 2
    record = run_test(args.run_dir, all_models=args.all_models, allow_repeat=args.allow_repeat, reason=args.reason)
    print(json.dumps(record["selected_metrics"], indent=2, default=str))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    import pandas as pd

    from aeromind.experiments.compare import compare_runs

    table = compare_runs(args.runs, args.output)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.to_string(index=False))
    if any(c.startswith("test_") for c in table.columns):
        print(
            "\nNOTE: test_* columns are final-reporting values from the official test ledger. "
            "They must not be used to choose between runs; compare runs on val_* columns."
        )
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    from aeromind.experiments.audit import audit_runs

    report = audit_runs(args.runs, args.output)
    for c in report["global_checks"] + [c for r in report["runs"] for c in r["checks"]]:
        print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['check']}")
    print(f"\nAudit {'PASSED' if report['all_passed'] else 'FAILED'} -> {args.output}")
    return 0 if report["all_passed"] else 1


def _publish_stage2(run_dir: Path, dataset: str, reports_dir: str = "reports") -> Path:
    from aeromind.experiments.common import publish_reports

    dest = Path(reports_dir) / dataset / "stage2" / run_dir.name
    publish_reports(run_dir / "report", dest)
    for extra in ("config.yaml", "metadata.json"):
        if (run_dir / extra).is_file():
            (dest / extra).write_bytes((run_dir / extra).read_bytes())
    return dest


def _cmd_s2_compare(args: argparse.Namespace) -> int:
    from aeromind.research.comparison import run_comparison
    from aeromind.research.spec import load_stage2_config

    cfg = load_stage2_config(args.config)
    run_dir = run_comparison(cfg, args.run_dir, isolate=not args.no_isolate, log_level=args.log_level, models=args.models)
    dest = _publish_stage2(run_dir, cfg.dataset, cfg.paths.reports_dir)
    print((run_dir / "report" / "summary.md").read_text())
    print(f"Run: {run_dir}\nReport: {dest}")
    return 0


def _cmd_s2_uq(args: argparse.Namespace) -> int:
    from aeromind.research.uq import UQConfig, run_uq, summarise_uq

    cfg = UQConfig.load(args.config)
    out = run_uq(cfg)
    print(summarise_uq(out))
    from aeromind.research.spec import load_stage2_config

    dataset = load_stage2_config(Path(cfg.comparison_run) / "config.yaml").dataset
    print(f"Run: {out}\nReport: {_publish_stage2(out, dataset)}")
    return 0


def _cmd_s2_ablation(args: argparse.Namespace) -> int:
    from aeromind.research.ablation import run_ablation

    out = run_ablation(args.config, args.run_dir, isolate=not args.no_isolate)
    print((out / "report" / "ablation.md").read_text())
    from aeromind.research.spec import load_stage2_config

    print(f"Run: {out}\nReport: {_publish_stage2(out, load_stage2_config(out / 'config.yaml').dataset)}")
    return 0


def _cmd_s2_robustness(args: argparse.Namespace) -> int:
    from aeromind.research.robustness import RobustnessConfig, run_robustness
    from aeromind.research.spec import load_stage2_config

    cfg = RobustnessConfig.load(args.config)
    out = run_robustness(cfg)
    print((out / "report" / "robustness.md").read_text())
    ds = load_stage2_config(Path(cfg.comparison_run) / "config.yaml").dataset
    print(f"Run: {out}\nReport: {_publish_stage2(out, ds)}")
    return 0


def _cmd_s2_smoke(args: argparse.Namespace) -> int:
    import tempfile

    from aeromind.research.comparison import run_comparison
    from aeromind.research.smoke import prepare_smoke_data, smoke_stage2_config
    from aeromind.research.uq import UQConfig, run_uq

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(args.workdir or tmp)
        cfg = smoke_stage2_config(root)
        prepare_smoke_data(cfg)
        run_dir = run_comparison(cfg, isolate=not args.no_isolate, log_level=args.log_level)
        uq = run_uq(
            UQConfig(
                name="s2_smoke_uq",
                comparison_run=str(run_dir),
                point_models=["xgboost", "gru"],
                quantile_models={
                    "hgb_q": {"kind": "classical", "family": "hist_gradient_boosting", "params": {"max_iter": 30}, "features": {"windows": [5]}},
                    "gru_q": {"kind": "deep", "deep": {"arch": "gru", "window": 10, "net": {"hidden": 8}, "train": {"epochs": 2, "threads": 2}}},
                },
                draws=20,
                artifacts_dir=str(root / "stage2"),
            )
        )
        print((run_dir / "report" / "summary.md").read_text())
        print(f"Stage-2 smoke OK (comparison + UQ): {run_dir.name}, {uq.name}")
    return 0


def _cmd_s3_registry(args: argparse.Namespace) -> int:
    from aeromind.serving.registry import RegistryConfig, build_registry, write_registry

    cfg = RegistryConfig.load(args.config)
    reg = build_registry(cfg)
    out = write_registry(reg, args.output or cfg.output)
    for m in reg["models"]:
        cal = "intervals " + ",".join(m["intervals"]["thresholds"]) if m["intervals"] else ""
        print(f"{m['id']:28s} v{m['version']}  {m['kind']:9s} {cal}")
    print(f"Registry with {len(reg['models'])} models -> {out}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import os

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("The API needs the optional extra: uv sync --extra api") from exc
    os.environ.setdefault("AEROMIND_REGISTRY", args.registry)
    uvicorn.run("aeromind.api.app:app", host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    import subprocess

    if importlib.util.find_spec("streamlit") is None:
        raise RuntimeError("The dashboard needs the optional extra: uv sync --extra dashboard")
    app = Path(__file__).resolve().parent / "dashboard" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port), "--server.address", args.host]
    if args.headless:
        cmd += ["--server.headless", "true"]
    return subprocess.call(cmd)


def _cmd_s3_benchmark(args: argparse.Namespace) -> int:
    from aeromind.serving.benchmark import run_benchmark, summary

    out = run_benchmark(args.registry, models=args.models, repeats=args.repeats, http=not args.no_http)
    print(summary(out))
    print(f"Report: {out} (also copied to reports/benchmarks/latest)")
    return 0


def _cmd_s3_cards(args: argparse.Namespace) -> int:
    from aeromind.serving.cards import render_cards

    Path(args.output).write_text(render_cards(args.registry))
    print(f"Model cards -> {args.output}")
    return 0


def _cmd_s3_export(args: argparse.Namespace) -> int:
    from aeromind.serving.artifacts import export_bundles

    res = export_bundles(args.registry, args.output)
    print(json.dumps(res, indent=2))
    return 0 if not res["skipped"] else 1


def _cmd_s3_import(args: argparse.Namespace) -> int:
    from aeromind.serving.artifacts import import_bundles

    res = import_bundles(args.archive, args.registry)
    print(f"Installed {len(res['installed'])} bundles; rejected {len(res['rejected'])}; missing {len(res['missing'])}")
    for r in res["rejected"]:
        print(f"REJECTED {r}")
    return 0 if not res["rejected"] else 1


def _cmd_s3_smoke(args: argparse.Namespace) -> int:
    from aeromind.serving.smoke import run_s3_smoke

    result = run_s3_smoke(args.workdir)
    print(f"Stage-3 smoke OK: {len(result['models'])} synthetic models registered; API {result['api']}; dashboard {result['dashboard']}")
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    from aeromind.experiments.smoke import run_smoke

    result = run_smoke(args.workdir)
    print(f"Smoke test OK. Selected-model synthetic test RMSE(true) = {result['test_selected']['rmse_true']:.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(prog="aeromind", description=__doc__.splitlines()[0])
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    def data_args(p: argparse.ArgumentParser, allow_all: bool = False) -> None:
        choices = ["FD001", "FD002", "FD003", "FD004"] + (["all"] if allow_all else [])
        p.add_argument("--dataset", default="FD001", choices=choices)
        p.add_argument("--data-dir", default=DEFAULT_RAW)
        p.add_argument("--manifest-dir", default=DEFAULT_MANIFESTS)

    p = sub.add_parser("validate", help="validate raw files and provenance hashes")
    data_args(p, allow_all=True)
    p.add_argument("--write-manifest", action="store_true", help="(re)write the provenance manifest")
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("split", help="create a deterministic engine-level split manifest")
    data_args(p)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fractions", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    p.add_argument("--output")
    p.add_argument("--force", action="store_true", help="allow changing an existing manifest's IDs")
    p.set_defaults(func=_cmd_split)

    p = sub.add_parser("eda", help="generate EDA reports")
    data_args(p)
    p.add_argument("--scope", choices=["pool", "train", "both"], default="both")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--manifest")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--cap", type=int, default=125)
    p.set_defaults(func=_cmd_eda)

    p = sub.add_parser("features", help="materialise features to Parquet")
    p.add_argument("--config", required=True)
    p.add_argument("--output")
    p.set_defaults(func=_cmd_features)

    p = sub.add_parser("train", help="train baselines and select on validation")
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir")
    p.add_argument("--no-publish", action="store_true", help="do not copy report files to reports/")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("evaluate", help="reload saved models and re-check validation metrics")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("test", help="FINAL evaluation on the official NASA test set")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--all-models", action="store_true", help="also report non-selected models (post hoc)")
    p.add_argument("--confirm-final", action="store_true", help="required: confirm this is a final evaluation")
    p.add_argument("--allow-repeat", action="store_true", help="permit re-evaluating an already evaluated run")
    p.add_argument("--reason", default="", help="why this evaluation is run (stored in the ledger)")
    p.set_defaults(func=_cmd_test)

    p = sub.add_parser("compare", help="compare finished runs")
    p.add_argument("runs", nargs="+")
    p.add_argument("--output")
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("audit", help="research-integrity audit of finished runs (no training, no test labels)")
    p.add_argument("runs", nargs="+")
    p.add_argument("--output", default="reports/FD001/audit/stage1_audit.json")
    p.set_defaults(func=_cmd_audit)

    p = sub.add_parser("s2-compare", help="Stage 2: classical vs deep comparison with engine-level CV")
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", help="existing run dir to resume (models with results are skipped)")
    p.add_argument("--models", nargs="+", help="subset of models to run")
    p.add_argument("--no-isolate", action="store_true", help="run models in-process (no per-model memory)")
    p.set_defaults(func=_cmd_s2_compare)

    p = sub.add_parser("s2-uq", help="Stage 2: quantile regression + split conformal on calibration engines")
    p.add_argument("--config", required=True)
    p.set_defaults(func=_cmd_s2_uq)

    p = sub.add_parser("s2-ablation", help="Stage 2: one-factor-at-a-time ablations")
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir")
    p.add_argument("--no-isolate", action="store_true")
    p.set_defaults(func=_cmd_s2_ablation)

    p = sub.add_parser("s2-robustness", help="Stage 2: input corruption and subset-shift evaluation")
    p.add_argument("--config", required=True)
    p.set_defaults(func=_cmd_s2_robustness)

    p = sub.add_parser("s2-smoke", help="Stage 2 synthetic smoke test (all architectures + UQ)")
    p.add_argument("--workdir")
    p.add_argument("--no-isolate", action="store_true")
    p.set_defaults(func=_cmd_s2_smoke)

    p = sub.add_parser("s3-registry", help="Stage 3: build the model registry (hash-pinned bundles + conformal calibrators)")
    p.add_argument("--config", default="configs/stage3/registry.yaml")
    p.add_argument("--output", help="override the registry output path")
    p.set_defaults(func=_cmd_s3_registry)

    p = sub.add_parser("serve", help="Stage 3: run the REST API (needs --extra api)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--registry", default="reports/registry/model_registry.json")
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("dashboard", help="Stage 3: run the Streamlit dashboard (needs --extra dashboard)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--headless", action="store_true")
    p.set_defaults(func=_cmd_dashboard)

    p = sub.add_parser("s3-benchmark", help="Stage 3: CPU latency of in-process and HTTP inference")
    p.add_argument("--registry", default="reports/registry/model_registry.json")
    p.add_argument("--models", nargs="+")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--no-http", action="store_true")
    p.set_defaults(func=_cmd_s3_benchmark)

    p = sub.add_parser("s3-model-cards", help="Stage 3: render docs/MODEL_CARDS.md from the registry")
    p.add_argument("--registry", default="reports/registry/model_registry.json")
    p.add_argument("--output", default="docs/MODEL_CARDS.md")
    p.set_defaults(func=_cmd_s3_cards)

    p = sub.add_parser("s3-export",help="Stage 3: pack registry bundles into a .tar.gz (for a release asset)")
    p.add_argument("--registry", default="reports/registry/model_registry.json")
    p.add_argument("--output", default="dist/aeromind-models.tar.gz")
    p.set_defaults(func=_cmd_s3_export)

    p = sub.add_parser("s3-import", help="Stage 3: install bundles whose path and SHA-256 match the registry")
    p.add_argument("archive")
    p.add_argument("--registry", default="reports/registry/model_registry.json")
    p.set_defaults(func=_cmd_s3_import)

    p = sub.add_parser("s3-smoke", help="Stage 3 synthetic smoke test (registry, API, dashboard)")
    p.add_argument("--workdir")
    p.set_defaults(func=_cmd_s3_smoke)

    p = sub.add_parser("smoke", help="fast synthetic end-to-end check (no NASA data)")
    p.add_argument("--workdir")
    p.set_defaults(func=_cmd_smoke)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, FileExistsError, RuntimeError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())

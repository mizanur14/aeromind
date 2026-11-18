"""End-to-end workflow tests on synthetic data (fast)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aeromind.data.cmapss as cmapss
from aeromind.cli import main as cli_main
from aeromind.data.splits import make_engine_split, save_split
from aeromind.eda.report import run_eda
from aeromind.experiments.compare import compare_runs
from aeromind.experiments.evaluate import evaluate_run
from aeromind.experiments.smoke import smoke_config
from aeromind.experiments.test_eval import run_test
from aeromind.experiments.train import run_training
from aeromind.models.bundle import load_bundle
from aeromind.synthetic import write_raw_files
from aeromind.utils import read_json


@pytest.fixture
def smoke_setup(tmp_path):
    cfg = smoke_config(tmp_path)
    write_raw_files(cfg.paths.raw_dir, n_train=30, n_test=10, seed=11)
    pool = cmapss.load_train(cfg.paths.raw_dir, "FD001")
    save_split(make_engine_split(pool.engine_id.unique(), "FD001", 42), cfg.split_manifest_path)
    return cfg


def test_training_never_reads_test_labels(smoke_setup, monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("official test labels were read during training")

    # Belt and braces: remove the label file and forbid the readers.
    (Path(smoke_setup.paths.raw_dir) / "RUL_FD001.txt").unlink()
    monkeypatch.setattr(cmapss, "read_test_rul", forbidden)
    monkeypatch.setattr(cmapss, "load_test", forbidden)
    run_dir = run_training(smoke_setup)
    meta = read_json(run_dir / "metadata.json")
    assert meta["official_test_labels_read"] is False
    assert set(meta["engines"]["train"]).isdisjoint(meta["engines"]["validation"])
    assert set(meta["engines"]["calibration_untouched"]).isdisjoint(meta["engines"]["train"])


def test_full_workflow_save_load_test(smoke_setup):
    run_dir = run_training(smoke_setup)
    meta = read_json(run_dir / "metadata.json")
    assert set(meta["models"]) == set(smoke_setup.models)
    comparison = pd.read_csv(run_dir / "report" / "validation_model_comparison.csv")
    assert comparison.loc[0, "model"] == meta["selected_model"]
    # Learned models should beat the constant baseline on synthetic degradation data.
    rmse = comparison.set_index("model")["snapshot_rmse_target"]
    assert rmse.drop(["median"]).min() < rmse["median"]

    reload = evaluate_run(run_dir)
    assert all(r["max_abs_diff_vs_saved"] <= 1e-9 for r in reload.values())

    predictor, _ = load_bundle(run_dir / meta["models"]["ridge"])
    preds = predictor.predict(cmapss.load_train(smoke_setup.paths.raw_dir, "FD001"))
    assert preds.min() >= 0 and preds.max() <= smoke_setup.target.cap

    record = run_test(run_dir, all_models=True)
    assert record["selected_metrics"]["n"] == 10
    meta = read_json(run_dir / "metadata.json")
    assert meta["official_test_labels_read"] is True and len(meta["test_evaluations"]) == 1
    preds = pd.read_csv(run_dir / "report" / "test" / "test_final_cycle_predictions.csv")
    assert len(preds) == 10 * len(smoke_setup.models)

    table = compare_runs([run_dir])
    assert table.loc[0, "selected_model"] == meta["selected_model"]
    assert "test_rmse_true" in table.columns


def test_run_dir_must_be_new(smoke_setup):
    run_dir = run_training(smoke_setup)
    with pytest.raises(FileExistsError):
        run_training(smoke_setup, run_dir)


def test_test_prediction_uses_last_cycle_only(smoke_setup):
    run_dir = run_training(smoke_setup)
    meta = read_json(run_dir / "metadata.json")
    predictor, _ = load_bundle(run_dir / meta["models"][meta["selected_model"]])
    test = cmapss.load_test(smoke_setup.paths.raw_dir, "FD001")
    full = predictor.predict(test)
    for eid, g in test.groupby("engine_id"):
        single = predictor.predict(g)
        assert single[-1] == pytest.approx(full[test.index.get_indexer([g.index[-1]])][0])
        assert eid > 0


def test_eda_scopes(smoke_setup):
    out_pool = run_eda(smoke_setup.paths.raw_dir, "FD001", "pool", None, smoke_setup.paths.reports_dir, cap=60)
    out_train = run_eda(
        smoke_setup.paths.raw_dir,
        "FD001",
        "train",
        smoke_setup.split_manifest_path,
        smoke_setup.paths.reports_dir,
        cap=60,
    )
    s_pool, s_train = read_json(out_pool / "summary.json"), read_json(out_train / "summary.json")
    assert s_pool["engines"] == 30 and s_train["engines"] == 21
    assert "EXPLORATORY" in s_pool["scope_description"]
    assert set(s_train["constant_columns"]) >= {"sensor_1", "sensor_5"}
    for name in ("lifetimes.png", "vif.csv", "within_engine_trends.csv", "early_vs_late_ks.csv"):
        assert (out_train / name).is_file()
    with pytest.raises(ValueError):
        run_eda(smoke_setup.paths.raw_dir, "FD001", "train", None, smoke_setup.paths.reports_dir)


def test_cli_validate_and_split(synthetic_raw, tmp_path, capsys):
    manifests = tmp_path / "m"
    assert (
        cli_main(["validate", "--data-dir", str(synthetic_raw), "--manifest-dir", str(manifests), "--write-manifest"])
        == 0
    )
    assert cli_main(["validate", "--data-dir", str(synthetic_raw), "--manifest-dir", str(manifests)]) == 0
    assert "all hashes match" in capsys.readouterr().out
    # Tamper with the label file -> provenance check fails.
    np.savetxt(synthetic_raw / "RUL_FD001.txt", np.arange(10) + 1, fmt="%d")
    assert cli_main(["validate", "--data-dir", str(synthetic_raw), "--manifest-dir", str(manifests)]) == 1
    assert cli_main(["split", "--data-dir", str(synthetic_raw), "--manifest-dir", str(manifests)]) == 0
    assert (manifests / "fd001_split_seed42.json").is_file()
    # Different seed to same path is refused (exit code 2, no traceback).
    out = str(manifests / "fd001_split_seed42.json")
    assert cli_main(["split", "--data-dir", str(synthetic_raw), "--seed", "1", "--output", out]) == 2


def test_cli_smoke(tmp_path):
    assert cli_main(["--log-level", "WARNING", "smoke", "--workdir", str(tmp_path)]) == 0


def test_prepare_features(smoke_setup, tmp_path):
    from aeromind.experiments.features import FEATURE_PREFIX, prepare_features

    out = prepare_features(smoke_setup, tmp_path / "feats")
    train = pd.read_parquet(out / "train.parquet")
    test = pd.read_parquet(out / "test_unlabelled.parquet")
    assert {"engine_id", "cycle", "rul", "target"} <= set(train.columns)
    assert "rul" not in test.columns and "target" not in test.columns
    feats = [c for c in train.columns if c.startswith(FEATURE_PREFIX)]
    assert feats == [c for c in test.columns if c.startswith(FEATURE_PREFIX)]
    assert train["target"].max() <= smoke_setup.target.cap


# --------------------------------------------------------------------------- Stage-1 audit
def _with_provenance(cfg):
    from aeromind.data.provenance import write_provenance

    write_provenance(cfg.paths.raw_dir, Path(cfg.paths.manifest_dir) / "cmapss_provenance.json")
    return cfg


def test_test_set_is_evaluated_once_and_ledgered(smoke_setup):
    from aeromind.experiments.test_eval import TestSetPolicyError, read_ledger

    run_dir = run_training(smoke_setup)
    run_test(run_dir)
    with pytest.raises(TestSetPolicyError, match="already evaluated"):
        run_test(run_dir)
    run_test(run_dir, allow_repeat=True, reason="unit test")
    entries = read_ledger(smoke_setup.paths.reports_dir)["entries"]
    assert [e["repeat"] for e in entries] == [False, True]
    assert entries[1]["reason"] == "unit test"
    published = read_json(Path(smoke_setup.paths.reports_dir) / "FD001" / "runs" / run_dir.name / "metadata.json")
    assert published["official_test_labels_read"] is True and len(published["test_evaluations"]) == 2


def test_cli_test_requires_confirmation(smoke_setup):
    run_dir = run_training(smoke_setup)
    assert cli_main(["test", "--run-dir", str(run_dir)]) == 2
    assert not read_json(run_dir / "metadata.json")["test_evaluations"]


def test_audit_passes_and_detects_tampering(smoke_setup, tmp_path):
    from aeromind.experiments.audit import audit_runs

    _with_provenance(smoke_setup)
    run_dir = run_training(smoke_setup)
    run_test(run_dir)
    report = audit_runs([run_dir], tmp_path / "audit.json")
    assert report["all_passed"], [c for r in report["runs"] for c in r["checks"] if not c["ok"]]

    # Tamper 1: altered saved validation predictions -> reload check fails.
    preds = run_dir / "predictions" / "validation_windows.csv"
    df = pd.read_csv(preds)
    df.loc[0, "pred"] += 1.0
    df.to_csv(preds, index=False)
    # Tamper 2: split manifest changed after training -> hash check fails.
    manifest = smoke_setup.split_manifest_path
    manifest.write_text(manifest.read_text() + "\n")
    report = audit_runs([run_dir], tmp_path / "audit2.json")
    failed = {c["check"] for r in report["runs"] for c in r["checks"] if not c["ok"]}
    assert not report["all_passed"]
    assert "split manifest hash matches run" in failed
    assert "reloaded bundles reproduce saved validation predictions (<=1e-9)" in failed


def test_static_label_scan(tmp_path):
    import aeromind
    from aeromind.experiments.audit import ALLOWED_LABEL_MODULES, scan_label_access

    assert set(scan_label_access(Path(aeromind.__file__).parent)) <= ALLOWED_LABEL_MODULES
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "sneaky.py").write_text("from x import load_test\nload_test('d', 'FD001')\n")
    assert scan_label_access(pkg) == {"sneaky.py": ["load_test"]}


def test_validate_does_not_expose_test_label_distribution(synthetic_raw):
    report = cmapss.validate_dataset(synthetic_raw, "FD001")
    assert report.stats["test_labels_valid"] is True
    assert not any("rul" in k for k in report.stats)


def test_legacy_create_splits_is_safe(synthetic_raw, tmp_path):
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts" / "create_splits.py"
    out = tmp_path / "split.json"
    base = [sys.executable, str(script), "--data-dir", str(synthetic_raw), "--output", str(out)]
    assert subprocess.run(base, capture_output=True).returncode == 0
    first = out.read_bytes()
    assert subprocess.run(base, capture_output=True).returncode == 0
    assert out.read_bytes() == first  # idempotent, byte-identical
    changed = subprocess.run([*base, "--seed", "7"], capture_output=True, text=True)
    assert changed.returncode == 2 and "Refusing" in changed.stderr
    assert out.read_bytes() == first

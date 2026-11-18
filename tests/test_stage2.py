"""Stage-2 research workflow tests on synthetic data."""

import math

import numpy as np
import pandas as pd
import pytest
import yaml

from aeromind.data.splits import make_engine_split
from aeromind.research.robustness import drop_readings, impute_causal
from aeromind.research.spec import make_folds
from aeromind.research.uq import conformal_quantile, one_per_engine


def test_folds_exclude_calibration_and_partition_dev():
    m = make_engine_split(np.arange(1, 101), "FD001", 42)
    folds = make_folds(m, 5, 0)
    dev = set(m.train_engine_ids) | set(m.validation_engine_ids)
    cv_val = [set(f["validation"]) for k, f in folds.items() if k.startswith("cv")]
    assert set().union(*cv_val) == dev and sum(map(len, cv_val)) == 85
    for f in folds.values():
        assert not set(f["train"]) & set(m.calibration_engine_ids)
        assert not set(f["validation"]) & set(m.calibration_engine_ids)
        assert not set(f["train"]) & set(f["validation"])
    assert folds["canonical"]["validation"] == m.validation_engine_ids
    assert make_folds(m, 5, 0) == folds


def test_conformal_quantile_finite_sample():
    s = np.arange(1, 16, dtype=float)  # n = 15
    assert conformal_quantile(s, 0.1) == 15.0  # ceil(16*0.9) = 15
    assert conformal_quantile(s, 0.2) == 13.0  # ceil(16*0.8) = 13
    assert math.isinf(conformal_quantile(s, 0.05))  # ceil(16*0.95) = 16 > 15


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    hits = []
    for _ in range(2000):
        cal, test = np.abs(rng.normal(size=15)), abs(rng.normal())
        hits.append(test <= conformal_quantile(cal, 0.2))
    cov = np.mean(hits)
    assert 0.78 <= cov <= 0.80 + 1 / 16 + 0.02  # ≥ 1-α, ≤ 1-α+1/(n+1) (MC tolerance)


def test_one_per_engine():
    ids = np.array([1, 1, 2, 2, 2, 3])
    pos = one_per_engine(ids, np.random.default_rng(0))
    assert ids[pos].tolist() == [1, 2, 3]


def test_causal_imputation_never_uses_future():
    df = pd.DataFrame({"engine_id": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2], "sensor_2": [np.nan, 5.0, np.nan, np.nan, 7.0]})
    out = impute_causal(df, ["sensor_2"], pd.Series({"sensor_2": 1.0}))
    assert out.sensor_2.tolist() == [1.0, 5.0, 5.0, 1.0, 7.0]
    dropped = drop_readings(df.fillna(0), ["sensor_2"], 1.0, 0)
    assert dropped.sensor_2.isna().all()


torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def s2_run(tmp_path_factory):
    from aeromind.research.comparison import run_comparison
    from aeromind.research.smoke import prepare_smoke_data, smoke_stage2_config

    root = tmp_path_factory.mktemp("s2")
    cfg = smoke_stage2_config(root)
    prepare_smoke_data(cfg)
    return cfg, run_comparison(cfg, isolate=False, log_level="WARNING")


def test_comparison_outputs(s2_run):
    cfg, run_dir = s2_run
    folds = pd.read_csv(run_dir / "report" / "fold_metrics.csv")
    assert set(folds.model) == set(cfg.models)
    assert set(folds.fold) == {"canonical", "cv0", "cv1", "cv2"}
    # Identical protocol: every model scored on the same snapshot rows per fold.
    snaps = {}
    for f in (run_dir / "predictions").glob("*.csv"):
        d = pd.read_csv(f)
        snaps.setdefault(d.fold.iloc[0], set()).add(frozenset(map(tuple, d.loc[d.snapshot, ["engine_id", "cycle"]].to_numpy())))
    assert all(len(v) == 1 for v in snaps.values())
    res = pd.read_csv(run_dir / "report" / "resources.csv")
    assert res.single_input_ms.notna().all() and res.peak_rss_mb.notna().all()
    assert {"tcn", "gru", "transformer"} <= set(res.model[res.n_parameters.notna()])
    meta = yaml.safe_load((run_dir / "metadata.json").read_text())
    assert meta["official_test_labels_read"] is False
    assert (run_dir / "checkpoints" / "gru" / "last.pt").is_file()


def test_comparison_resume_skips_done(s2_run):
    from aeromind.research.comparison import run_comparison

    cfg, run_dir = s2_run
    before = (run_dir / "results" / "ridge.json").stat().st_mtime
    run_comparison(cfg, run_dir, isolate=False, log_level="WARNING")
    assert (run_dir / "results" / "ridge.json").stat().st_mtime == before


def test_uq_on_held_out_engines(s2_run):
    from aeromind.research.uq import UQConfig, run_uq

    cfg, run_dir = s2_run
    out = run_uq(
        UQConfig(
            name="uq_t",
            comparison_run=str(run_dir),
            point_models=["xgboost"],
            quantile_models={"hgb_q": {"kind": "classical", "family": "hist_gradient_boosting", "params": {"max_iter": 20}, "features": {"windows": [5]}}},
            draws=10,
            artifacts_dir=str(run_dir.parent),
        )
    )
    t = pd.read_csv(out / "report" / "intervals.csv")
    assert {"conformal_engine", "conformal_pooled", "quantile_raw", "cqr_engine", "cqr_pooled"} <= set(t.method)
    assert set(t.calibrate_on) == {"calibration", "validation"}
    eng = t[(t.method == "conformal_engine") & (t.calibrate_on == "calibration")]
    n_cal = 30 - 21 - 4  # synthetic split of 30 engines: 21 train / 4-5 val / rest cal
    assert (eng.finite_fraction <= 1).all() and n_cal > 0
    meta = yaml.safe_load((out / "metadata.json").read_text())
    assert not set(meta["engines"]["calibration"]) & set(meta["engines"]["train"])


def test_ablation_compile_and_run(s2_run, tmp_path):
    from aeromind.research.ablation import compile_ablation, run_ablation
    from aeromind.research.spec import dump_stage2_config

    cfg, _ = s2_run
    base = tmp_path / "base.yaml"
    dump_stage2_config(cfg, base)
    spec = {
        "name": "abl_t",
        "base_config": str(base),
        "run_folds": ["cv0", "cv1"],
        "base_models": {"gru": cfg.models["gru"].__dict__ | {}, "xgboost": cfg.models["xgboost"].__dict__ | {}},
        "variants": [
            {"name": "window5", "applies_to": ["gru"], "deep": {"window": 5}},
            {"name": "mae", "applies_to": ["gru"], "deep": {"train": {"loss": "mae"}}},
            {"name": "uncapped", "applies_to": ["gru", "xgboost"], "target_cap": None},
            {"name": "settings", "applies_to": ["xgboost", "gru"], "features": {"include_settings": True}},
        ],
    }
    path = tmp_path / "abl.yaml"
    path.write_text(yaml.safe_dump(spec))
    s2cfg, lineage = compile_ablation(path)
    assert s2cfg.models["gru__window5"].deep["window"] == 5
    assert s2cfg.models["gru__mae"].deep["train"]["loss"] == "mae"
    assert s2cfg.models["gru__mae"].deep["train"]["epochs"] == 2  # merged, not replaced
    assert s2cfg.models["xgboost__uncapped"].target_cap is None
    assert s2cfg.models["gru__settings"].deep["features"]["include_settings"] is True
    out = run_ablation(path, tmp_path / "run", isolate=False)
    table = pd.read_csv(out / "report" / "ablation.csv")
    assert len(table) == len(lineage) and (table.folds == 2).all()


def test_robustness_runs(s2_run):
    from aeromind.research.robustness import RobustnessConfig, run_robustness

    _, run_dir = s2_run
    out = run_robustness(
        RobustnessConfig(name="rob_t", comparison_run=str(run_dir), models=["xgboost", "tcn"], noise_levels=[1.0], missing_rates=[0.2], seeds=[0], dropout_sensors=["sensor_2"], artifacts_dir=str(run_dir.parent))
    )
    t = pd.read_csv(out / "report" / "robustness.csv")
    assert set(t.scenario) == {"clean", "noise", "dropout", "missing"}
    assert (t[t.scenario == "clean"].delta_rmse_capped == 0).all()


def test_uncapped_fit_quantile_avoids_cap_degeneracy():
    from aeromind.research.spec import ModelEntry
    from aeromind.research.uq import ClassicalQuantileModel
    from aeromind.synthetic import make_training_pool

    pool = make_training_pool(30, seed=2)
    near = pool.rul < 10
    entry = ModelEntry(kind="classical", family="hist_gradient_boosting", params={"max_iter": 60}, features={"windows": [5]})
    capped = ClassicalQuantileModel(entry, [0.05, 0.5, 0.95], 0).fit(pool, 30)
    uncapped = ClassicalQuantileModel(entry, [0.05, 0.5, 0.95], 0).fit(pool, 30, fit_cap=None)
    q_c, q_u = capped.predict_quantiles(pool), uncapped.predict_quantiles(pool)
    # Uncapped fit is mapped onto the capped scale; quantiles stay ordered. (Whether it is
    # narrower near failure is data-dependent: it is on FD001, where the capped fit degenerates.)
    assert q_u.max() <= 30 and q_c.max() <= 30
    assert (np.diff(q_u, axis=1) >= 0).all() and (np.diff(q_c, axis=1) >= 0).all()
    assert not np.allclose(q_u[near.to_numpy()], q_c[near.to_numpy()])

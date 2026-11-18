import numpy as np
import pandas as pd
import pytest

from aeromind.config import FeatureConfig
from aeromind.features.pipeline import FORBIDDEN_FEATURES, FeaturePipeline
from aeromind.features.rolling import check_trajectory_order, trailing_features
from aeromind.features.targets import cap_rul, make_target
from aeromind.synthetic import make_trajectories

COLS = ["sensor_2", "sensor_11"]


def test_rolling_values_partial_windows():
    df = pd.DataFrame(
        {"engine_id": [1, 1, 1, 1, 2, 2], "cycle": [1, 2, 3, 4, 1, 2], "sensor_2": [1.0, 3.0, 5.0, 10.0, 100.0, 200.0]}
    )
    f = trailing_features(df, ["sensor_2"], [3], include_diff=True)
    assert f["sensor_2__mean_w3"].tolist() == pytest.approx([1, 2, 3, 6, 100, 150])
    assert f["sensor_2__min_w3"].tolist() == [1, 1, 1, 3, 100, 100]
    assert f["sensor_2__max_w3"].tolist() == [1, 3, 5, 10, 100, 200]
    assert f["sensor_2__std_w3"].iloc[0] == 0.0 and f["sensor_2__std_w3"].iloc[4] == 0.0
    assert f["sensor_2__slope_w3"].tolist() == pytest.approx([0, 2, 2, 3.5, 0, 100])
    assert f["sensor_2__diff1"].tolist() == [0, 2, 2, 5, 0, 100]


def test_slope_matches_polyfit():
    df = make_trajectories(3, seed=1)
    f = trailing_features(df, COLS, [10], ["slope"])
    e = df[df.engine_id == 2]
    for i in e.index[[0, 3, 9, 25, -1]]:
        seg = e.loc[max(e.index[0], i - 9) : i]
        expected = 0.0 if len(seg) == 1 else np.polyfit(seg.cycle, seg.sensor_11, 1)[0]
        assert f.loc[i, "sensor_11__slope_w10"] == pytest.approx(expected, abs=1e-9)


def test_no_future_leakage_truncation_invariance():
    """Features at cycle t must not change when later cycles are removed."""
    df = make_trajectories(8, seed=2)
    full = trailing_features(df, COLS, [5, 15], include_diff=True)
    rng = np.random.default_rng(0)
    keep = []
    for _, g in df.groupby("engine_id"):
        keep.append(g.iloc[: int(rng.integers(1, len(g)))])
    trunc = pd.concat(keep)
    part = trailing_features(trunc, COLS, [5, 15], include_diff=True)
    pd.testing.assert_frame_equal(part, full.loc[trunc.index], rtol=1e-10, atol=1e-10)


def test_no_cross_engine_leakage_order_invariance():
    df = make_trajectories(6, seed=4)
    full = trailing_features(df, COLS, [7])
    order = [4, 1, 6, 2, 5, 3]
    shuffled = pd.concat([df[df.engine_id == e] for e in order])
    part = trailing_features(shuffled, COLS, [7])
    pd.testing.assert_frame_equal(part.sort_index(), full.sort_index(), rtol=1e-10, atol=1e-10)
    # Engine alone == engine within the fleet.
    alone = trailing_features(df[df.engine_id == 3], COLS, [7])
    pd.testing.assert_frame_equal(alone, full.loc[alone.index], rtol=1e-10, atol=1e-10)


def test_order_check():
    df = pd.DataFrame({"engine_id": [1, 2, 1], "cycle": [1, 1, 2]})
    with pytest.raises(ValueError, match="contiguous"):
        check_trajectory_order(df)
    with pytest.raises(ValueError, match="increasing"):
        check_trajectory_order(pd.DataFrame({"engine_id": [1, 1], "cycle": [2, 1]}))


def test_targets():
    assert cap_rul([0, 50, 200], 125).tolist() == [0, 50, 125]
    assert cap_rul([0, 50, 200], None).tolist() == [0, 50, 200]
    with pytest.raises(ValueError):
        cap_rul([-1], 125)
    df = pd.DataFrame({"rul": [300, 10]})
    assert make_target(df, 130).tolist() == [130, 10]
    assert df["rul"].tolist() == [300, 10]  # uncapped truth untouched


def test_pipeline_fits_only_on_given_engines(synthetic_pool):
    train = synthetic_pool[synthetic_pool.engine_id <= 12]
    other = synthetic_pool[synthetic_pool.engine_id > 12].copy()
    pipe = FeaturePipeline(FeatureConfig(windows=[5]), seed=0).fit(train)
    assert pipe.fitted_engine_ids_ == list(range(1, 13))
    X_train = pipe.transform(train)
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-8)
    # Scaler statistics must not change when non-training engines change wildly.
    mean_before = pipe.scaler_.mean_.copy()
    other.loc[:, "sensor_2"] += 1e6
    pipe.transform(other)
    assert np.array_equal(pipe.scaler_.mean_, mean_before)
    refit = FeaturePipeline(FeatureConfig(windows=[5]), seed=0).fit(train)
    assert np.array_equal(refit.scaler_.mean_, mean_before)


def test_pipeline_constant_removal_and_forbidden(synthetic_pool):
    pipe = FeaturePipeline(FeatureConfig(windows=[5], include_settings=True), 0).fit(synthetic_pool)
    assert set(pipe.dropped_constant_) == {
        "setting_3",
        "sensor_1",
        "sensor_5",
        "sensor_10",
        "sensor_16",
        "sensor_18",
        "sensor_19",
    }
    assert not FORBIDDEN_FEATURES & set(pipe.feature_names_)
    assert not any(n.startswith(("engine_id", "rul")) for n in pipe.feature_names_)
    keep = FeaturePipeline(FeatureConfig(windows=[5], drop_constant=False), 0).fit(synthetic_pool)
    assert keep.dropped_constant_ == [] and len(keep.kept_columns_) == 21


def test_pipeline_inference_identical_and_label_free(synthetic_pool):
    pipe = FeaturePipeline(FeatureConfig(windows=[5, 10], include_diff=True), 0).fit(synthetic_pool)
    a = pipe.transform(synthetic_pool)
    b = pipe.transform(synthetic_pool.drop(columns=["rul"]))
    assert np.array_equal(a, b)


def test_pipeline_sensor_subset_and_errors(synthetic_pool):
    pipe = FeaturePipeline(FeatureConfig(sensors=["sensor_2", "sensor_11"], windows=[4], stats=["mean"]), 0)
    pipe.fit(synthetic_pool)
    assert pipe.feature_names_ == ["cycle", "sensor_2", "sensor_11", "sensor_2__mean_w4", "sensor_11__mean_w4"]
    with pytest.raises(ValueError, match="Unknown sensors"):
        FeaturePipeline(FeatureConfig(sensors=["sensor_99"]), 0).fit(synthetic_pool)
    with pytest.raises(RuntimeError, match="not fitted"):
        FeaturePipeline(FeatureConfig(), 0).transform(synthetic_pool)
    with pytest.raises(KeyError):
        pipe.transform(synthetic_pool.drop(columns=["sensor_2"]))


def test_condition_normalization_multi_regime():
    df = make_trajectories(12, seed=5)
    rng = np.random.default_rng(0)
    regime = rng.integers(0, 3, len(df))
    df["setting_1"] = regime * 10.0 + rng.normal(0, 0.01, len(df))
    df["sensor_2"] = df["sensor_2"] + regime * 50.0
    cfg = FeatureConfig(windows=[3], stats=["mean"], condition_normalization=True, n_conditions=3, scaler="none")
    pipe = FeaturePipeline(cfg, 0).fit(df)
    raw = pipe.transform_frame(df)["sensor_2"]
    assert abs(raw.groupby(regime).mean()).max() < 1e-6

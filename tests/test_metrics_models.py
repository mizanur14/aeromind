import math

import numpy as np
import pytest

from aeromind.evaluation.metrics import mae, phm08_score, regression_metrics, rmse
from aeromind.evaluation.protocol import (
    engine_balanced_weights,
    error_by_rul_bin,
    final_cycle_index,
    metrics_against_truths,
    select_snapshots,
)
from aeromind.models.baselines import MeanResidualLifeRegressor, MedianRULRegressor
from aeromind.models.factory import build_estimator, expand_grid


def test_basic_metrics():
    t, p = [10, 20, 30], [12, 18, 30]
    assert mae(t, p) == pytest.approx(4 / 3)
    assert rmse(t, p) == pytest.approx(math.sqrt(8 / 3))
    assert mae([0, 10], [1, 10], weights=[1, 3]) == pytest.approx(0.25)


def test_phm08_asymmetry():
    # late (over-prediction) penalised more than early
    assert phm08_score([50], [60]) == pytest.approx(math.e - 1)
    assert phm08_score([50], [37]) == pytest.approx(math.e - 1)
    assert phm08_score([50], [40]) < phm08_score([50], [60])
    assert phm08_score([5, 5], [5, 5]) == 0.0


def test_metric_input_validation():
    with pytest.raises(ValueError):
        rmse([1, 2], [1])
    with pytest.raises(ValueError):
        rmse([], [])
    with pytest.raises(ValueError):
        rmse([1.0], [np.nan])


def test_truth_variants():
    m = metrics_against_truths(np.array([200.0, 10.0]), np.array([125.0, 10.0]), target_cap=None, capped_truth_cap=125)
    assert m["rmse_capped"] == 0.0
    assert m["rmse_true"] == pytest.approx(math.sqrt(75**2 / 2))
    assert m["rmse_target"] == m["rmse_true"]


def test_snapshots_balanced_and_deterministic(synthetic_pool):
    a = select_snapshots(synthetic_pool, 5, seed=1)
    b = select_snapshots(synthetic_pool.sample(frac=1, random_state=0), 5, seed=1)
    assert sorted(a) == sorted(b)
    counts = synthetic_pool.loc[a, "engine_id"].value_counts()
    assert (counts == 5).all() and len(counts) == synthetic_pool.engine_id.nunique()


def test_engine_weights_and_final_cycle(synthetic_pool):
    w = engine_balanced_weights(synthetic_pool["engine_id"])
    totals = synthetic_pool.assign(w=w).groupby("engine_id")["w"].sum()
    assert np.allclose(totals, 1.0)
    last = final_cycle_index(synthetic_pool)
    assert (synthetic_pool.loc[last, "rul"] == 0).all()
    assert len(last) == synthetic_pool.engine_id.nunique()


def test_error_by_bin():
    tab = error_by_rul_bin(np.array([5, 15, 30]), np.array([6, 13, 30]), [0, 10, 20, 50])
    assert tab["n"].tolist() == [1, 1, 1]
    assert tab["bias"].tolist() == [1, -2, 0]


def test_median_baseline():
    m = MedianRULRegressor().fit(np.zeros((5, 2)), [1, 2, 3, 4, 100])
    assert m.predict(np.zeros((3, 2))).tolist() == [3, 3, 3]
    w = MedianRULRegressor().fit(np.zeros((3, 1)), [1, 2, 3], sample_weight=[0.1, 0.1, 10])
    assert w.value_ == 3


def test_mean_residual_life_baseline():
    # Two engines with lifetimes 3 and 5 (cycles 1..L, RUL L - cycle).
    cycles = np.array([1, 2, 3, 1, 2, 3, 4, 5])[:, None]
    rul = np.array([2, 1, 0, 4, 3, 2, 1, 0])
    m = MeanResidualLifeRegressor(min_engines=2).fit(cycles, rul)
    assert m.predict(np.array([[1], [2], [3]])).tolist() == [3, 2, 1]
    # Beyond the last cycle with >=2 survivors: decrement, then 0 beyond max life.
    assert m.predict(np.array([[4], [5], [9]])).tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        MeanResidualLifeRegressor().fit(np.zeros((4, 2)), np.zeros(4))


def test_expand_grid():
    grid = expand_grid({"a": 1}, {"b": [1, 2], "c": [3]})
    assert grid == [{"a": 1, "b": 1, "c": 3}, {"a": 1, "b": 2, "c": 3}]
    assert expand_grid({"a": 1}, {}) == [{"a": 1}]


@pytest.mark.parametrize("name", ["ridge", "random_forest", "hist_gradient_boosting", "xgboost"])
def test_estimators_fit_and_are_deterministic(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = X[:, 0] * 10 + rng.normal(size=200)
    params = {"n_estimators": 10} if name in ("random_forest", "xgboost") else {}
    p1 = build_estimator(name, params, seed=1, n_jobs=1).fit(X, y).predict(X)
    p2 = build_estimator(name, params, seed=1, n_jobs=1).fit(X, y).predict(X)
    assert np.allclose(p1, p2)
    assert rmse(y, p1) < np.std(y)


def test_hgb_early_stopping_rejected():
    with pytest.raises(ValueError, match="early_stopping"):
        build_estimator("hist_gradient_boosting", {"early_stopping": True})


def test_regression_metrics_keys():
    m = regression_metrics([1, 2], [1, 3])
    assert set(m) == {"n", "mae", "rmse", "bias", "phm08_sum", "phm08_mean"}

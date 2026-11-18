"""Deep-model infrastructure tests (skipped without the optional PyTorch extra)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from aeromind.deep.nets import build_network, count_parameters  # noqa: E402
from aeromind.deep.trainer import DeepSpec, TrainSettings, fit_deep, inner_split  # noqa: E402
from aeromind.deep.windows import build_windows  # noqa: E402
from aeromind.features.targets import make_target  # noqa: E402
from aeromind.models.bundle import load_bundle, save_bundle  # noqa: E402
from aeromind.synthetic import make_training_pool, make_trajectories  # noqa: E402

TINY = {"tcn": {"channels": 8, "levels": 2}, "gru": {"hidden": 8}, "transformer": {"d_model": 8, "heads": 2, "layers": 1, "ff": 16}}


def test_windows_causal_right_padded():
    values = np.arange(10, dtype=float).reshape(-1, 1)  # engine 1: 0..5, engine 2: 6..9
    ids = np.array([1] * 6 + [2] * 4)
    w = build_windows(values, ids, 3)
    assert w.lengths.tolist() == [1, 2, 3, 3, 3, 3, 1, 2, 3, 3]
    assert w.x[0, :, 0].tolist() == [0, 0, 0]
    assert w.x[1, :, 0].tolist() == [0, 1, 0]  # right padding
    assert w.x[5, :, 0].tolist() == [3, 4, 5]  # newest last
    assert w.x[6, :, 0].tolist() == [6, 0, 0]  # never crosses engines
    assert w.x[9, :, 0].tolist() == [7, 8, 9]


def test_windows_truncation_invariance():
    df = make_trajectories(5, seed=3)
    v = df[["sensor_2", "sensor_11"]].to_numpy()
    full = build_windows(v, df.engine_id.to_numpy(), 7)
    keep = df.groupby("engine_id").head(20).index
    part = build_windows(v[keep], df.engine_id.to_numpy()[keep], 7)
    assert np.array_equal(part.x, full.x[keep]) and np.array_equal(part.lengths, full.lengths[keep])


def test_windows_reject_interleaved():
    with pytest.raises(ValueError):
        build_windows(np.zeros((3, 1)), np.array([1, 2, 1]), 2)


@pytest.mark.parametrize("arch", ["tcn", "gru", "transformer"])
def test_padding_does_not_change_output(arch):
    torch.manual_seed(0)
    net = build_network(arch, 3, 2, TINY[arch]).eval()
    x = torch.randn(4, 6, 3)
    lengths = torch.tensor([6, 4, 1, 3])
    for b, n in enumerate(lengths.tolist()):
        x[b, n:] = 0.0
    y0 = net(x, lengths)
    x2 = x.clone()
    for b, n in enumerate(lengths.tolist()):
        x2[b, n:] = 99.0  # garbage in padded positions
    assert torch.allclose(y0, net(x2, lengths), atol=1e-5)
    assert y0.shape == (4, 2) and count_parameters(net) > 0


def test_tcn_receptive_field():
    net = build_network("tcn", 2, 1, {"channels": 4, "levels": 3, "kernel": 3})
    assert net.receptive_field == 1 + 2 * 2 * (1 + 2 + 4)


def _train_df(n=16, seed=1):
    df = make_training_pool(n, seed=seed)
    df["target"] = make_target(df, 60)
    return df


def _spec(arch="gru", epochs=3, **train):
    return DeepSpec.from_dict({"arch": arch, "window": 8, "net": TINY[arch], "train": {"epochs": epochs, "batch_size": 64, "threads": 1, "patience": 10, **train}})


def test_inner_split_engine_level():
    fit, es = inner_split(np.repeat(np.arange(1, 21), 3), 0.2, 0)
    assert len(es) == 4 and not set(fit) & set(es) and len(fit) + len(es) == 20


@pytest.mark.parametrize("arch", ["tcn", "gru", "transformer"])
def test_training_is_deterministic(arch):
    df = _train_df()
    a, ha = fit_deep(_spec(arch), df, 60, seed=3)
    b, hb = fit_deep(_spec(arch), df, 60, seed=3)
    for k in a.state_dict:
        assert torch.equal(a.state_dict[k], b.state_dict[k])
    assert ha["best_inner_val_rmse_target"] == hb["best_inner_val_rmse_target"]
    assert set(ha["early_stopping_engines"]).isdisjoint(set(range(0)))


def test_resume_matches_uninterrupted(tmp_path):
    df = _train_df()
    full, hf = fit_deep(_spec(epochs=4), df, 60, seed=5, checkpoint_dir=tmp_path / "a")
    fit_deep(_spec(epochs=4), df, 60, seed=5, checkpoint_dir=tmp_path / "b", max_epochs=2)  # "interrupted"
    resumed, hr = fit_deep(_spec(epochs=4), df, 60, seed=5, checkpoint_dir=tmp_path / "b", resume=True)
    assert hr["epochs_run"] == 4
    for k in full.state_dict:
        assert torch.allclose(full.state_dict[k], resumed.state_dict[k], atol=1e-6)
    assert (tmp_path / "b" / "best.pt").is_file() and (tmp_path / "b" / "last.pt").is_file()


@pytest.mark.parametrize("loss", ["huber", "mae", "mse", "quantile"])
def test_losses_and_save_load(loss, tmp_path):
    df = _train_df()
    pred, _ = fit_deep(_spec("tcn", epochs=2, loss=loss, quantiles=[0.1, 0.5, 0.9]), df, 60, seed=0)
    p1 = pred.predict(df)
    assert p1.min() >= 0 and p1.max() <= 60
    path = save_bundle(pred, tmp_path / "m.joblib")
    loaded, _ = load_bundle(path)
    assert np.array_equal(loaded.predict(df), p1)
    if loss == "quantile":
        q = loaded.predict_quantiles(loaded.inputs(df))
        assert q.shape[1] == 3 and (np.diff(q, axis=1) >= 0).all()
    else:
        with pytest.raises(ValueError):
            loaded.predict_quantiles(loaded.inputs(df))


def test_prediction_uses_only_past(tmp_path):
    df = _train_df()
    pred, _ = fit_deep(_spec("transformer", epochs=1), df, 60, seed=0)
    full = pred.predict(df)
    trunc = df.groupby("engine_id").head(15)
    assert np.allclose(pred.predict(trunc), full[trunc.index.to_numpy()], atol=1e-5)


def test_settings_validation():
    with pytest.raises(ValueError):
        TrainSettings(loss="hinge")
    with pytest.raises(ValueError):
        TrainSettings(loss="quantile", quantiles=[0.1, 0.9])
    with pytest.raises(ValueError):
        DeepSpec.from_dict({"arch": "gru", "bogus": 1})
    with pytest.raises(ValueError):
        build_network("lstm", 2, 1, {})
    assert isinstance(pd.DataFrame(), pd.DataFrame)

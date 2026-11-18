"""Training loop for temporal RUL networks.

- Normalisation: the Stage-1 :class:`FeaturePipeline` fitted on the fold's training engines
  (constant removal + scaler), giving per-time-step channels.
- Early stopping: on an **inner engine-level holdout** drawn from the training engines, so the
  outer validation engines remain untouched for reporting.
- AdamW, ReduceLROnPlateau or cosine schedule, gradient-norm clipping.
- Determinism: seeded NumPy/PyTorch, seeded shuffling generator, deterministic algorithms,
  fixed thread count. ``last.pt`` stores model, optimiser, scheduler, RNG and early-stopping
  state, so an interrupted run resumes to the same result (tested).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from aeromind.config import FeatureConfig
from aeromind.deep.nets import build_network, count_parameters
from aeromind.deep.predictor import DeepRULPredictor
from aeromind.deep.windows import WindowInputs
from aeromind.evaluation.metrics import rmse
from aeromind.evaluation.protocol import engine_balanced_weights
from aeromind.features.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)

LOSSES = ("huber", "mae", "mse", "quantile")


@dataclass
class TrainSettings:
    """Optimisation settings (all overridable from YAML)."""

    epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    loss: str = "huber"
    huber_delta: float = 10.0  # cycles
    quantiles: list[float] = field(default_factory=lambda: [0.05, 0.5, 0.95])
    patience: int = 8
    scheduler: str = "plateau"  # plateau | cosine | none
    grad_clip: float = 1.0
    inner_val_fraction: float = 0.15
    target_scale: float = 125.0  # targets are divided by this inside the network
    threads: int = 4

    def __post_init__(self) -> None:
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}")
        if self.scheduler not in ("plateau", "cosine", "none"):
            raise ValueError("scheduler must be plateau, cosine or none")
        if self.loss == "quantile" and 0.5 not in self.quantiles:
            raise ValueError("quantile loss needs 0.5 among quantiles (point prediction)")
        if not 0 < self.inner_val_fraction < 0.5:
            raise ValueError("inner_val_fraction must be in (0, 0.5)")


@dataclass
class DeepSpec:
    """Architecture + input + training specification for one deep model."""

    arch: str
    window: int = 30
    net: dict[str, Any] = field(default_factory=dict)
    features: FeatureConfig = field(default_factory=lambda: FeatureConfig(windows=[], stats=[]))
    train: TrainSettings = field(default_factory=TrainSettings)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DeepSpec:
        """Build from YAML mapping (unknown keys raise)."""
        allowed = {"arch", "window", "net", "features", "train"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"deep spec: unknown keys {sorted(unknown)}")
        feats = {"windows": [], "stats": [], **(payload.get("features") or {})}
        return cls(
            arch=payload["arch"],
            window=int(payload.get("window", 30)),
            net=dict(payload.get("net") or {}),
            features=FeatureConfig(**feats),
            train=TrainSettings(**(payload.get("train") or {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain dict."""
        return asdict(self)


def set_determinism(seed: int, threads: int) -> None:
    """Seed PyTorch and force deterministic CPU kernels (NumPy uses explicit Generators)."""
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)


def inner_split(engine_ids: np.ndarray, fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic engine-level split of training engines into (fit, early-stopping) sets."""
    ids = np.sort(np.unique(engine_ids))
    rng = np.random.default_rng(seed + 1009)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    k = max(1, int(round(fraction * len(ids))))
    return sorted(shuffled[k:].tolist()), sorted(shuffled[:k].tolist())


def _loss(pred: torch.Tensor, y: torch.Tensor, settings: TrainSettings) -> torch.Tensor:
    if settings.loss == "quantile":
        q = torch.tensor(settings.quantiles, dtype=pred.dtype)
        diff = y.unsqueeze(1) - pred
        return torch.maximum(q * diff, (q - 1) * diff).mean()
    p = pred[:, 0]
    if settings.loss == "mae":
        return F.l1_loss(p, y)
    if settings.loss == "mse":
        return F.mse_loss(p, y)
    return F.huber_loss(p, y, delta=settings.huber_delta / settings.target_scale)


def _tensors(w: WindowInputs) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(w.x), torch.from_numpy(w.lengths)


def fit_deep(
    spec: DeepSpec,
    train_df: pd.DataFrame,
    target_cap: int | None,
    seed: int = 42,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    name: str | None = None,
    max_epochs: int | None = None,
) -> tuple[DeepRULPredictor, dict[str, Any]]:
    """Train one network on ``train_df`` (training engines with a ``target`` column).

    Args:
        spec: architecture/input/training specification.
        train_df: training-engine rows sorted by engine, cycle; must contain ``target``.
        target_cap: training cap (for clipping predictions); ``None`` = uncapped.
        seed: seed for initialisation, shuffling and the inner split.
        checkpoint_dir: if given, ``best.pt`` and ``last.pt`` are written each epoch.
        resume: continue from ``checkpoint_dir/last.pt`` if present.
        max_epochs: stop after this many total epochs (used to simulate interruption).

    Returns:
        The best (inner-validation) predictor and a training history dict.
    """
    s = spec.train
    set_determinism(seed, s.threads)
    fit_ids, es_ids = inner_split(train_df["engine_id"].to_numpy(), s.inner_val_fraction, seed)
    fit_df = train_df[train_df.engine_id.isin(fit_ids)]
    es_df = train_df[train_df.engine_id.isin(es_ids)]

    pipeline = FeaturePipeline(spec.features, seed).fit(train_df)
    n_out = len(s.quantiles) if s.loss == "quantile" else 1
    probe = DeepRULPredictor(
        name=name or spec.arch,
        pipeline=pipeline,
        arch=spec.arch,
        net_params=spec.net,
        n_channels=len(pipeline.feature_names_),
        n_outputs=n_out,
        window=spec.window,
        state_dict={},
        target_cap=target_cap,
        target_scale=s.target_scale,
        quantiles=list(s.quantiles) if s.loss == "quantile" else None,
        threads=s.threads,
    )
    t0 = time.perf_counter()
    X_fit, X_es = probe.inputs(fit_df), probe.inputs(es_df)
    window_seconds = time.perf_counter() - t0
    y_fit = torch.from_numpy((fit_df["target"].to_numpy() / s.target_scale).astype(np.float32))
    es_truth = es_df["target"].to_numpy(dtype=float)
    es_weights = engine_balanced_weights(es_df["engine_id"])
    xf, lf = _tensors(X_fit)

    net = build_network(spec.arch, probe.n_channels, n_out, spec.net)
    opt = torch.optim.AdamW(net.parameters(), lr=s.lr, weight_decay=s.weight_decay)
    if s.scheduler == "plateau":
        sched: Any = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, s.patience // 3))
    elif s.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=s.epochs)
    else:
        sched = None
    gen = torch.Generator().manual_seed(seed)

    state = {"epoch": 0, "best_score": math.inf, "best_epoch": -1, "bad_epochs": 0, "history": [], "best_state": None}
    ckpt = Path(checkpoint_dir) if checkpoint_dir else None
    if ckpt and resume and (ckpt / "last.pt").is_file():
        saved = torch.load(ckpt / "last.pt", weights_only=False)
        net.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        if sched is not None and saved["scheduler"] is not None:
            sched.load_state_dict(saved["scheduler"])
        gen.set_state(saved["generator"])
        torch.set_rng_state(saved["torch_rng"])
        state = saved["state"]
        logger.info("Resumed %s from epoch %d", probe.name, state["epoch"])

    train_seconds = 0.0
    n = len(X_fit)
    stop_at = s.epochs if max_epochs is None else min(s.epochs, max_epochs)
    while state["epoch"] < stop_at and state["bad_epochs"] < s.patience:
        t_ep = time.perf_counter()
        net.train()
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for start in range(0, n, s.batch_size):
            idx = perm[start : start + s.batch_size]
            opt.zero_grad()
            loss = _loss(net(xf[idx], lf[idx]), y_fit[idx], s)
            loss.backward()
            if s.grad_clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), s.grad_clip)
            opt.step()
            total += loss.item() * len(idx)
        probe.state_dict = net.state_dict()
        probe._net = net
        es_pred = probe.predict_inputs(X_es)
        score = rmse(np.minimum(es_truth, target_cap) if target_cap else es_truth, es_pred, es_weights)
        if sched is not None:
            sched.step(score) if s.scheduler == "plateau" else sched.step()
        state["epoch"] += 1
        improved = score < state["best_score"] - 1e-6
        if improved:
            state.update(best_score=score, best_epoch=state["epoch"], bad_epochs=0)
            state["best_state"] = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            state["bad_epochs"] += 1
        epoch_s = time.perf_counter() - t_ep
        train_seconds += epoch_s
        state["history"].append(
            {"epoch": state["epoch"], "train_loss": total / n, "inner_val_rmse": score, "lr": opt.param_groups[0]["lr"], "seconds": epoch_s}
        )
        logger.debug("%s epoch %d loss %.4f inner-val RMSE %.2f", probe.name, state["epoch"], total / n, score)
        if ckpt:
            ckpt.mkdir(parents=True, exist_ok=True)
            if improved:
                torch.save(state["best_state"], ckpt / "best.pt")
            torch.save(
                {
                    "model": net.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict() if sched is not None else None,
                    "generator": gen.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "state": state,
                },
                ckpt / "last.pt",
            )

    net.load_state_dict(state["best_state"])
    probe.state_dict = {k: v.detach().clone() for k, v in net.state_dict().items()}
    probe._net = None
    history = {
        "epochs_run": state["epoch"],
        "best_epoch": state["best_epoch"],
        "best_inner_val_rmse_target": state["best_score"],
        "stopped_early": state["bad_epochs"] >= s.patience,
        "train_seconds_this_session": train_seconds,
        "window_build_seconds": window_seconds,
        "n_parameters": count_parameters(net),
        "n_train_windows": n,
        "fit_engines": len(fit_ids),
        "early_stopping_engines": es_ids,
        "train_window_mb": X_fit.nbytes / 2**20,
        "curve": state["history"],
    }
    return probe, history

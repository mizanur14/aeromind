"""Deep predictor bundle with the same interface as :class:`aeromind.models.bundle.RULPredictor`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch

from aeromind.deep.nets import build_network
from aeromind.deep.windows import WindowInputs, build_windows
from aeromind.features.pipeline import FeaturePipeline


@dataclass
class DeepRULPredictor:
    """Fitted feature pipeline + network weights. Pickles without the live ``nn.Module``."""

    name: str
    pipeline: FeaturePipeline
    arch: str
    net_params: dict[str, Any]
    n_channels: int
    n_outputs: int
    window: int
    state_dict: dict[str, torch.Tensor]
    target_cap: int | None
    target_scale: float
    quantiles: list[float] | None = None
    threads: int = 4
    input_kind: str = "windows"
    _net: Any = field(default=None, repr=False, compare=False)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_net"] = None
        return state

    @property
    def net(self) -> torch.nn.Module:
        """Lazily rebuilt network in eval mode."""
        if self._net is None:
            net = build_network(self.arch, self.n_channels, self.n_outputs, self.net_params)
            net.load_state_dict(self.state_dict)
            self._net = net
        return self._net

    def inputs(self, df: pd.DataFrame) -> WindowInputs:
        """Causal windows of pipeline-normalised channels, one per row of ``df``."""
        values = self.pipeline.transform(df).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite network inputs (values overflow float32)")
        return build_windows(values, df["engine_id"].to_numpy(), self.window)

    def raw_outputs(self, X: WindowInputs, batch_size: int = 4096) -> np.ndarray:
        """Network outputs in cycles, shape (N, n_outputs)."""
        torch.set_num_threads(self.threads)
        net = self.net
        was_training = net.training
        net.eval()
        outs = []
        with torch.no_grad():
            for s in range(0, len(X), batch_size):
                xb = torch.from_numpy(X.x[s : s + batch_size])
                lb = torch.from_numpy(X.lengths[s : s + batch_size])
                outs.append(net(xb, lb).numpy())
        if was_training:
            net.train()
        return np.concatenate(outs, axis=0).astype(float) * self.target_scale

    def _clip(self, values: np.ndarray) -> np.ndarray:
        upper = float(self.target_cap) if self.target_cap is not None else None
        return np.clip(values, 0.0, upper)

    def predict_inputs(self, X: WindowInputs) -> np.ndarray:
        """Point predictions (median output for quantile models), clipped to ``[0, cap]``."""
        out = self.raw_outputs(X)
        col = self.quantiles.index(0.5) if self.quantiles else 0
        return self._clip(out[:, col])

    def predict_quantiles(self, X: WindowInputs) -> np.ndarray:
        """Quantile predictions (N, n_quantiles), sorted per row to avoid crossing, clipped."""
        if not self.quantiles:
            raise ValueError(f"{self.name} was not trained with a quantile loss")
        return self._clip(np.sort(self.raw_outputs(X), axis=1))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict RUL for every row of ``df`` using only that row's engine history."""
        return self.predict_inputs(self.inputs(df))

    @property
    def n_parameters(self) -> int:
        """Trainable parameter count."""
        return int(sum(v.numel() for v in self.state_dict.values()))

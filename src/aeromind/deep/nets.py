"""Compact temporal regressors: TCN, GRU and Transformer encoder.

All take right-padded windows ``x`` (B, L, C) and ``lengths`` (B,) and return (B, n_outputs).
Each reads the representation at (or pooled up to) the newest valid step, so padding after
the newest step never influences the output.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _last_valid(h: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Gather h[b, lengths[b]-1, :] from (B, L, H)."""
    idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, h.size(-1))
    return h.gather(1, idx).squeeze(1)


class _CausalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(F.relu(self.conv1(F.pad(x, (self.pad, 0)))))
        y = self.drop(F.relu(self.conv2(F.pad(y, (self.pad, 0)))))
        return F.relu(y + self.down(x))


class TCNRegressor(nn.Module):
    """Causal dilated temporal convolutional network (Bai et al., 2018 style).

    Receptive field ``1 + 2 (k−1) Σ 2^i``. Left zero-padding keeps convolutions causal.
    """

    def __init__(
        self, n_channels: int, n_outputs: int = 1, channels: int = 32, levels: int = 3, kernel: int = 3, dropout: float = 0.1
    ) -> None:
        super().__init__()
        blocks, c_in = [], n_channels
        for i in range(levels):
            blocks.append(_CausalBlock(c_in, channels, kernel, 2**i, dropout))
            c_in = channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, n_outputs)
        self.receptive_field = 1 + 2 * (kernel - 1) * sum(2**i for i in range(levels))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x.transpose(1, 2)).transpose(1, 2)
        return self.head(_last_valid(h, lengths))


class GRURegressor(nn.Module):
    """GRU over packed variable-length windows; final hidden state → linear head."""

    def __init__(self, n_channels: int, n_outputs: int = 1, hidden: int = 32, layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.gru = nn.GRU(n_channels, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_outputs)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.head(self.drop(h_n[-1]))


class TransformerRegressor(nn.Module):
    """Small pre-norm Transformer encoder with learned positions and key-padding mask.

    Output head sees [newest-step token, mean of valid tokens]. All tokens in a window are
    observations at or before the prediction cycle, so bidirectional attention is causal
    with respect to time.
    """

    def __init__(
        self,
        n_channels: int,
        n_outputs: int = 1,
        d_model: int = 32,
        heads: int = 4,
        layers: int = 2,
        ff: int = 64,
        dropout: float = 0.1,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(n_channels, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(2 * d_model, n_outputs)
        self.max_len = max_len

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        if length > self.max_len:
            raise ValueError(f"window {length} exceeds max_len {self.max_len}")
        pos = torch.arange(length, device=x.device)
        pad = pos.unsqueeze(0) >= lengths.unsqueeze(1)  # True = padding
        h = self.encoder(self.proj(x) + self.pos(pos), src_key_padding_mask=pad)
        h = self.norm(h).masked_fill(pad.unsqueeze(-1), 0.0)
        mean = h.sum(1) / lengths.clamp(min=1).unsqueeze(1).to(h.dtype)
        return self.head(torch.cat([_last_valid(h, lengths), mean], dim=1))


ARCHITECTURES = {"tcn": TCNRegressor, "gru": GRURegressor, "transformer": TransformerRegressor}


def build_network(arch: str, n_channels: int, n_outputs: int, params: dict[str, Any]) -> nn.Module:
    """Instantiate an architecture by name."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {arch!r}; expected {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[arch](n_channels=n_channels, n_outputs=n_outputs, **params)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

"""Shared MLP backbone: 3 x 128, SiLU, LayerNorm (row-wise).

Sources: protocol section 4.2 (P3-17: MLP 3 x 128, SiLU, **LayerNorm**, no BatchNorm); proofs
Proposition 3.4 scope (no batch-coupled layer upstream of Phi, so rows stay independent);
values from params ``model.backbone``.

Each hidden block is ``Linear -> LayerNorm -> SiLU``. LayerNorm normalizes each row over its own
features, so the trunk acts row-wise in training and inference alike (no running statistics).
Parameters are created in float64 on CPU (protocol 4.2, O-8); initialization uses PyTorch's
default scheme under the global seed set by the trainer (P6-D5, Batch D).

Torch-only, float64, CPU. Owning phase: 6.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.config import BackboneParams

__all__ = ["Backbone", "check_features"]


def check_features(x: Tensor, n_features: int, name: str = "x") -> None:
    """Assert ``x`` is a finite float64 CPU matrix ``(B, n_features)``; raise otherwise."""
    if x.dtype != torch.float64 or x.device.type != "cpu":
        raise TypeError(f"{name} must be float64 on CPU (protocol 4.2, O-8)")
    if x.dim() != 2 or int(x.shape[1]) != n_features:
        raise ValueError(f"{name} must have shape (B, {n_features}), got {tuple(x.shape)}")
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"non-finite values in {name}")


class Backbone(nn.Module):
    """Row-wise MLP trunk ``R^F -> R^H`` (protocol 4.2; params ``model.backbone``).

    Parameters
    ----------
    in_features
        Number of input features ``F`` (``> 0``).
    params
        ``hidden_layers`` blocks of width ``hidden_width``; ``activation`` = SiLU and
        ``norm`` = LayerNorm are the only values the config admits.
    """

    def __init__(self, in_features: int, params: BackboneParams) -> None:
        super().__init__()
        if in_features < 1:
            raise ValueError("in_features must be >= 1")
        self.in_features = in_features
        self.out_features = params.hidden_width
        layers: list[nn.Module] = []
        width = in_features
        for _ in range(params.hidden_layers):
            layers += [
                nn.Linear(width, params.hidden_width, dtype=torch.float64),
                nn.LayerNorm(params.hidden_width, dtype=torch.float64),
                nn.SiLU(),
            ]
            width = params.hidden_width
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Map ``x`` (B, F) to the hidden representation (B, H); raises on bad input."""
        check_features(x, self.in_features)
        out: Tensor = self.net(x)
        return out

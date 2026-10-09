"""M2-MLP: pure ML with independent heteroscedastic heads for ln W, ln dbar, ln d_max, ln A.

Sources: protocol section 4.1 (M2 row: **independent** log targets, closure not enforced; shared
backbone with a heteroscedastic Gaussian head per target) and section 4.2 (float64 for parity;
beta-NLL ``beta_N = 0.5`` after a 20-epoch MSE warm start; ``sigma_min = 0.01``); proofs section
9.1 (``sigma = sigma_min + softplus``; NLL per target), section 8.5 (beta-NLL); P6-Q3 (a)
(beta-NLL on the 4 heads, all weights 1).

The native targets are ``y' = [ln W, ln dbar, ln d_max, ln A]`` with ``ln d_max = y_1 + y_2``
from the batch contract. :class:`~src.models.heads.ModelOutput` uses the ladder's canonical
columns ``[ln W, ln dbar, t, ln A]``; here ``t = mu_dmax - mu_dbar`` (it may be ``<= 0``: M2
does not enforce closure) and **assumption (Batch C):** ``sigma_t = sqrt(sigma_dmax^2 +
sigma_dbar^2)`` as if the two heads were independent. The native moments stay in
``extras["mu_native"]`` and ``extras["sigma_native"]``; the d_max interval uses those.

Torch-only, float64, CPU. Owning phase: 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.config import BackboneParams, ProtocolParams
from src.models import losses as ls
from src.models.heads import (
    Batch,
    LadderNet,
    ModelOutput,
    Objective,
    RoutedHeads,
    check_batch,
    sigma_from_raw,
)

__all__ = ["M2MLP", "M2Config", "native_targets"]

_MU = ("m_w", "m_dbar", "m_dmax", "m_a")
_SIGMA = ("s_w", "s_dbar", "s_dmax", "s_a")


@dataclass(frozen=True)
class M2Config:
    """M2-MLP settings (protocol 4.2; P6-Q3 (a): all four weights 1)."""

    sigma_min: float
    beta_n: float
    warm_start_epochs: int

    @classmethod
    def from_protocol(cls, protocol: ProtocolParams) -> M2Config:
        """Read ``uq``; there are no M2-specific params."""
        return cls(
            sigma_min=protocol.uq.sigma_min,
            beta_n=protocol.uq.beta_nll,
            warm_start_epochs=protocol.uq.warm_start_epochs,
        )


def native_targets(y: Tensor) -> Tensor:
    """``[ln W, ln dbar, ln d_max, ln A]`` from the batch ``y`` (``ln d_max = ln dbar + t``)."""
    if y.dim() != 2 or int(y.shape[1]) != 4:
        raise ValueError("y must be (B, 4) (batch contract, ADL-077)")
    return torch.stack([y[:, 0], y[:, 1], y[:, 1] + y[:, 2], y[:, 3]], dim=1)


class M2MLP(LadderNet):
    """Full trunk -> 4 means and 4 sigma heads; no blind trunk (protocol 4.1)."""

    def __init__(self, in_features: int, cfg: M2Config, backbone: BackboneParams) -> None:
        super().__init__()
        self.in_features = in_features
        self.cfg = cfg
        self.heads = RoutedHeads(in_features, [*_MU, *_SIGMA], [], None, backbone)
        self.register_buffer("weights", torch.ones(4, dtype=torch.float64))

    def forward(self, batch: Batch) -> ModelOutput:
        """Native moments in ``extras``; canonical ``mu``/``sigma`` as described above."""
        check_batch(batch, self.in_features)
        raw = self.heads(batch["x"])
        mu_n = torch.stack([raw[k] for k in _MU], dim=1)
        sig_n = torch.stack([sigma_from_raw(raw[k], self.cfg.sigma_min) for k in _SIGMA], dim=1)
        mu = torch.stack([mu_n[:, 0], mu_n[:, 1], mu_n[:, 2] - mu_n[:, 1], mu_n[:, 3]], dim=1)
        sigma_t = torch.sqrt(sig_n[:, 2] ** 2 + sig_n[:, 1] ** 2)
        sigma = torch.stack([sig_n[:, 0], sig_n[:, 1], sigma_t, sig_n[:, 3]], dim=1)
        return ModelOutput(mu=mu, sigma=sigma, extras={"mu_native": mu_n, "sigma_native": sig_n})

    def objective(
        self, batch: Batch, out: ModelOutput, epoch: int, *, monitor: bool = False
    ) -> Objective:
        """MSE warm start, then beta-NLL on the native targets, weights 1 (P6-Q3 (a))."""
        cfg = self.cfg
        mode = ls.loss_mode(epoch, cfg.warm_start_epochs, cfg.beta_n)
        weights: Tensor = self.get_buffer("weights")
        data = ls.data_term(
            mode,
            out.extras["mu_native"],
            native_targets(batch["y"]),
            batch["omega"],
            weights,
            sigma=out.extras["sigma_native"],
            beta_n=cfg.beta_n,
        )
        return Objective(total=data, parts={"data": data})

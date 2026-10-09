"""Output heads (w, r, nu, ell) with bounded logistic transforms and heteroscedastic outputs.

Sources: proofs section 7.6 (bounded transforms ``w = z_W``, ``nu = nu_min + Delta_n sigma(z_n)``,
``ln r = ln r_min + Delta_r sigma(z_r)``; logistic preferred over hard clipping), section 7.5
(A3-SR logit-domain residual ``z_n = logit pi_S + Delta``), section 8.5 and section 9.1
(``sigma = sigma_min + softplus(z_sigma)``); protocol section 4.1 (routing ``I_blind``, P3-11, O-4:
the n-head sees every feature except ``Q_E``, ``S``, ``F_S``; ``V_E`` is never a feature) and
section 4.2 (P3-13 ``sigma_min``); P6-D4 (a): two independent trunks.

:class:`RoutedHeads` holds a *full* trunk on ``x`` and, when blind outputs are requested, a
separate *blind* trunk on ``x[:, x_blind_idx]``. The blind outputs (``z_n`` for M4/M5,
``g_nu`` for M6-NN) depend on the blind features only, so ``d z_n / d x_j == 0`` exactly for
every routed-out feature ``j``. Models assemble their own head names (Batch C).

Torch-only, float64, CPU. Owning phase: 6.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as fn

from src.config import BackboneParams, DomainParams
from src.models.backbone import Backbone, check_features

__all__ = ["BoundedTransforms", "RoutedHeads", "sigma_from_raw", "strickler_logit"]


@dataclass(frozen=True)
class BoundedTransforms:
    """Logistic maps of proofs 7.6 with bounds from params ``domain`` (protocol 2.2)."""

    nu_min: float
    delta_n: float
    ln_r_min: float
    delta_r: float

    def __post_init__(self) -> None:
        """Reject empty bands (``Delta_n``, ``Delta_r`` must be positive and finite)."""
        values = (self.nu_min, self.delta_n, self.ln_r_min, self.delta_r)
        if not all(math.isfinite(v) for v in values) or self.delta_n <= 0 or self.delta_r <= 0:
            raise ValueError("bounds must be finite with Delta_n > 0 and Delta_r > 0")

    @classmethod
    def from_domain(cls, domain: DomainParams) -> BoundedTransforms:
        """``nu_min = ln n_min``, ``Delta_n = ln(n_max/n_min)``, likewise for ``r``."""
        return cls(
            nu_min=domain.nu_bounds[0],
            delta_n=domain.delta_n,
            ln_r_min=math.log(domain.r_bounds[0]),
            delta_r=domain.delta_r,
        )

    def nu(self, z_n: Tensor) -> Tensor:
        """``nu = nu_min + Delta_n sigma(z_n)`` in (nu_min, nu_max) (proofs 7.6, invariant 7)."""
        return self.nu_min + self.delta_n * torch.sigmoid(z_n)

    def log_r(self, z_r: Tensor) -> Tensor:
        """``ln r = ln r_min + Delta_r sigma(z_r)`` in (ln r_min, ln r_max) (proofs 7.6)."""
        return self.ln_r_min + self.delta_r * torch.sigmoid(z_r)

    def r(self, z_r: Tensor) -> Tensor:
        """``r = exp(ln r)``; ``dr/dz_r = r Delta_r sigma (1 - sigma)`` (proofs 7.6)."""
        return torch.exp(self.log_r(z_r))


def sigma_from_raw(z_sigma: Tensor, sigma_min: float) -> Tensor:
    """``sigma = sigma_min + softplus(z_sigma)`` (proofs 8.5 (ii), 9.1); ``sigma >= sigma_min``."""
    if not sigma_min > 0:
        raise ValueError(
            "sigma_min must be > 0 (Proposition 8.5: the NLL is degenerate without it)"
        )
    return sigma_min + fn.softplus(z_sigma)


def strickler_logit(nu_s: Tensor, transforms: BoundedTransforms, varpi: float) -> Tensor:
    """Return ``logit pi_S``, ``pi_S = clip((nu_S - nu_min)/Delta_n, varpi, 1 - varpi)`` (7.5 b).

    The clip is part of the pre-registered A3-SR parameterization (protocol 4.1,
    ``varpi = 0.01``); it acts on the Strickler offset, not on a prediction.
    """
    if not 0 < varpi < 0.5:
        raise ValueError("varpi must lie in (0, 1/2)")
    if not bool(torch.isfinite(nu_s).all()):
        raise ValueError("non-finite nu_S")
    pi_s = ((nu_s - transforms.nu_min) / transforms.delta_n).clamp(varpi, 1.0 - varpi)
    return torch.logit(pi_s)


def _linear(width: int, n_out: int) -> nn.Linear:
    return nn.Linear(width, n_out, dtype=torch.float64)


class RoutedHeads(nn.Module):
    """Full trunk -> ``full_outputs``; blind trunk on ``x[:, blind_idx]`` -> ``blind_outputs``.

    Parameters
    ----------
    in_features
        ``F``, the width of ``x`` (batch contract, ADL-077).
    full_outputs, blind_outputs
        Names of the raw scalar outputs. ``blind_outputs`` may be empty (M2-MLP, M3), in which
        case no blind trunk is built and ``blind_idx`` must be ``None``.
    blind_idx
        ``x_blind_idx`` from the dataset (``I_blind``): unique, in range, a proper subset.
    params
        ``model.backbone``; both trunks use it (P6-D4 (a)).
    """

    def __init__(
        self,
        in_features: int,
        full_outputs: Sequence[str],
        blind_outputs: Sequence[str],
        blind_idx: Sequence[int] | Tensor | None,
        params: BackboneParams,
    ) -> None:
        super().__init__()
        names = [*full_outputs, *blind_outputs]
        if not full_outputs or len(set(names)) != len(names):
            raise ValueError("need at least one full output and unique output names")
        self.in_features = in_features
        self.full_outputs = tuple(full_outputs)
        self.blind_outputs = tuple(blind_outputs)
        self.full_trunk = Backbone(in_features, params)
        self.full_head = _linear(params.hidden_width, len(self.full_outputs))
        self.blind_trunk: Backbone | None = None
        self.blind_head: nn.Linear | None = None
        if self.blind_outputs:
            if blind_idx is None:
                raise ValueError("blind outputs need blind_idx (I_blind, protocol 4.1)")
            idx = torch.as_tensor(blind_idx, dtype=torch.int64).flatten()
            n_blind = int(idx.numel())
            if n_blind == 0 or n_blind >= in_features:
                raise ValueError("blind_idx must be a non-empty proper subset of the features")
            if len(set(idx.tolist())) != n_blind:
                raise ValueError("blind_idx has duplicates")
            if int(idx.min()) < 0 or int(idx.max()) >= in_features:
                raise ValueError("blind_idx out of range")
            self.register_buffer("blind_idx", idx)
            self.blind_trunk = Backbone(n_blind, params)
            self.blind_head = _linear(params.hidden_width, len(self.blind_outputs))
        elif blind_idx is not None:
            raise ValueError("blind_idx given but no blind outputs requested")

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Return ``{name: (B,)}`` raw outputs; blind outputs see ``x[:, blind_idx]`` only."""
        check_features(x, self.in_features)
        full = self.full_head(self.full_trunk(x))
        out = {name: full[:, j] for j, name in enumerate(self.full_outputs)}
        if self.blind_trunk is not None and self.blind_head is not None:
            idx: Tensor = self.get_buffer("blind_idx")
            blind = self.blind_head(self.blind_trunk(x.index_select(1, idx)))
            out.update({name: blind[:, j] for j, name in enumerate(self.blind_outputs)})
        return out

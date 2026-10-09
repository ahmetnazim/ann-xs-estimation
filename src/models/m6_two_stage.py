"""M6 (two-stage): back-calculated labels, regression of ``(nu, w, t)``, deterministic Phi.

Sources: proofs section 7.8 (stage 1 labels ``nu_obs`` by Proposition 6.2 with ``Q_E``, variant
M6-E; ``w_obs`` and ``t_obs`` taken directly; stage 2 ``L^M6 = sum_i omega_i [(g_nu - nu_obs)^2 +
w_W (g_w - w_obs)^2 + w_t (g_t - t_obs)^2]``; stage 3 ``nu_hat = Pi(g_nu)``,
``ell_hat = ell*(g_w, r(g_t), nu_hat; q_E, s)``), Proposition 6.1 (``r = 1/(e^t - 1)``),
Proposition 6.2 (explicit ``Phi^-1``); protocol section 4.1 (P3-10: M6-E headline; M6-NN primary
with heads ``g_w, g_t, g_nu`` and ``g_nu`` under I_blind; M6-XGB secondary; labels outside
``[nu_min, nu_max]`` are **retained** and projected by ``Pi`` at stage 3, share reported; labels at
``F_tout`` rows use ``r_obs`` clipped to ``[r_min, r_max]`` for kappa); P6-Q3 (a) (squared loss,
no sigma heads).

Pre-registered projections (the only clamps in this module):

* ``Pi``: ``nu`` onto ``[nu_min, nu_max]`` at stage 3 (proofs 7.8; protocol 4.1);
* ``r_obs`` clipped to ``[r_min, r_max]`` for the label's kappa (protocol 4.1). Since
  ``r = 1/(e^t - 1)`` is decreasing, this equals clamping ``t_obs`` to
  ``[t(r_max), t(r_min)] = [ln 1.1, ln 3]`` before inverting, which also covers ``t_obs = 0``
  (SA3 ``F_t`` rows, ``r = inf``);
* ``g_t`` onto the same t band at stage 3 (Q-C1 (a); protocol 11.2) before ``r(g_t)``, so
  ``r_hat`` is defined (``r(t)`` needs ``t > 0``) and lies in the transform band of M4/M5.

Torch-only, float64, CPU. M6-XGB fits its boosters in a torch-free worker
(:mod:`src.models.baselines`) and calls :func:`m6_stage3` here. Owning phase: 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from src.config import BackboneParams, ProtocolParams
from src.models import losses as ls
from src.models.heads import (
    Batch,
    BoundedTransforms,
    LadderNet,
    ModelOutput,
    Objective,
    RoutedHeads,
    check_batch,
    project_nu,
)
from src.physics.manning_layer import PhiConfig, check_phys, phi, phi_inverse_nu
from src.physics.shape import check_rows, r_of_t

__all__ = ["M6NN", "M6Config", "M6Labels", "m6_labels", "m6_stage3", "project_t"]


@dataclass(frozen=True)
class M6Config:
    """M6 settings (P3-10); ``weights`` = ``(1, w_W, w_t)`` for ``(nu, w, t)`` (proofs 7.8)."""

    phi: PhiConfig
    transforms: BoundedTransforms
    t_band: tuple[float, float]
    weights: tuple[float, float, float]
    s_floor: float

    def __post_init__(self) -> None:
        """Require a non-empty positive t band ``0 < t(r_max) < t(r_min)``."""
        lo, hi = self.t_band
        if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo < hi):
            raise ValueError("t_band must satisfy 0 < t(r_max) < t(r_min)")

    @classmethod
    def from_protocol(cls, protocol: ProtocolParams) -> M6Config:
        """``t_band = [ln(1 + 1/r_max), ln(1 + 1/r_min)]`` from ``domain.r_bounds`` (P3-02)."""
        r_min, r_max = protocol.domain.r_bounds
        lw = protocol.model.loss_weights
        return cls(
            phi=PhiConfig.from_solver(protocol.solver),
            transforms=BoundedTransforms.from_domain(protocol.domain),
            t_band=(math.log1p(1.0 / r_max), math.log1p(1.0 / r_min)),
            weights=(1.0, lw.w_w, lw.w_t),
            s_floor=protocol.domain.s_floor,
        )


def project_t(t: Tensor, t_band: tuple[float, float]) -> Tensor:
    """Clamp ``t`` to ``[t(r_max), t(r_min)]``, i.e. ``r`` to ``[r_min, r_max]`` (protocol 4.1).

    Used for the label kappa at ``F_tout`` rows (pre-registered) and for ``g_t`` at stage 3
    (Q-C1). Proposition 6.1: ``r(t)`` is decreasing, so the two clamps coincide.
    """
    check_rows(t=t)
    return t.clamp(t_band[0], t_band[1])


@dataclass(frozen=True)
class M6Labels:
    """Stage-1 labels (proofs 7.8): ``nu`` (Prop 6.2), ``w``, ``t`` and diagnostics.

    ``r`` is the clipped ``r_obs`` used for kappa; ``nu_outside`` marks labels outside the n band
    (retained, protocol 4.1); ``r_clipped`` marks rows whose ``r_obs`` was clipped.
    """

    nu: Tensor
    w: Tensor
    t: Tensor
    r: Tensor
    nu_outside: Tensor
    r_clipped: Tensor

    def stacked(self) -> Tensor:
        """``(B, 3)`` labels ``[nu, w, t]`` in the column order of :class:`M6Config` weights."""
        return torch.stack([self.nu, self.w, self.t], dim=1)


def m6_labels(y: Tensor, phys: Tensor, cfg: M6Config) -> M6Labels:
    """Back-calculate ``nu_obs = Phi^-1(ell_obs; w_obs, r_obs, q, s)`` (Proposition 6.2).

    ``y`` is the batch target ``[ln W, ln dbar, t_obs, ln A]`` and ``phys = [ln Q, ln S]``
    (``Q_E``; ``Q_C`` under SA1). No gradient: labels are data.
    """
    if y.dim() != 2 or int(y.shape[1]) != 4 or phys.dim() != 2 or int(phys.shape[1]) != 2:
        raise ValueError("y must be (B, 4) and phys (B, 2) (batch contract, ADL-077)")
    w_obs, ell_obs, t_obs = y[:, 0], y[:, 1], y[:, 2]
    q, s = phys[:, 0], phys[:, 1]
    check_phys(q, s, s_floor=cfg.s_floor)
    if not bool((t_obs >= 0).all()):
        raise ValueError("t_obs must be >= 0 (protocol 2.3)")
    with torch.no_grad():
        t_k = project_t(t_obs, cfg.t_band)
        r = r_of_t(t_k)
        nu = phi_inverse_nu(ell_obs, w_obs, r, q, s, cfg.phi)
    tr = cfg.transforms
    outside = (nu < tr.nu_min) | (nu > tr.nu_min + tr.delta_n)
    return M6Labels(nu=nu, w=w_obs, t=t_obs, r=r, nu_outside=outside, r_clipped=t_k != t_obs)


def m6_stage3(g_w: Tensor, g_t: Tensor, g_nu: Tensor, phys: Tensor, cfg: M6Config) -> ModelOutput:
    """Stage 3 (proofs 7.8): ``nu = Pi(g_nu)``, ``t = Pi_t(g_t)`` (Q-C1), ``ell = Phi``.

    Deterministic and without gradient. Returns ``mu = [g_w, ell, t, g_w + ell]``, no sigma
    (P6-Q3 (a)); ``extras`` carry the projected ``nu``, ``r`` and the projection masks.
    """
    q, s = phys[:, 0], phys[:, 1]
    check_phys(q, s, s_floor=cfg.s_floor)
    check_rows(g_w=g_w, g_t=g_t, g_nu=g_nu, q=q, s=s)
    with torch.no_grad():
        nu = project_nu(g_nu, cfg.transforms)
        t = project_t(g_t, cfg.t_band)
        r = r_of_t(t)
        res = phi(g_w, r, nu, q, s, cfg.phi)
        ell = res.ell
        mu = torch.stack([g_w, ell, t, g_w + ell], dim=1)
    extras = {
        "nu": nu,
        "r": r,
        "iterations": res.iterations,
        "nu_projected": nu != g_nu,
        "t_projected": t != g_t,
    }
    return ModelOutput(mu=mu, sigma=None, extras=extras)


class M6NN(LadderNet):
    """M6-NN (primary M6): full trunk -> ``(g_w, g_t)``, blind trunk -> ``g_nu`` (P3-10)."""

    def __init__(
        self, in_features: int, blind_idx: Tensor, cfg: M6Config, backbone: BackboneParams
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.cfg = cfg
        self.heads = RoutedHeads(in_features, ["g_w", "g_t"], ["g_nu"], blind_idx, backbone)
        self.register_buffer("weights", torch.tensor(cfg.weights, dtype=torch.float64))

    def forward(self, batch: Batch) -> ModelOutput:
        """Stage-2 regressors (with gradient, in ``extras``) and the stage-3 prediction."""
        check_batch(batch, self.in_features)
        raw = self.heads(batch["x"])
        g_w, g_t, g_nu = raw["g_w"], raw["g_t"], raw["g_nu"]
        out = m6_stage3(g_w.detach(), g_t.detach(), g_nu.detach(), batch["phys"], self.cfg)
        extras = {**out.extras, "g_w": g_w, "g_t": g_t, "g_nu": g_nu}
        return ModelOutput(mu=out.mu, sigma=None, extras=extras)

    def objective(
        self, batch: Batch, out: ModelOutput, epoch: int, *, monitor: bool = False
    ) -> Objective:
        """Stage-2 squared loss on ``[nu, w, t]`` with weights ``(1, w_W, w_t)`` (proofs 7.8).

        ``epoch`` and ``monitor`` do not change it (no warm start, no penalty).
        """
        labels = m6_labels(batch["y"], batch["phys"], self.cfg)
        pred = torch.stack([out.extras["g_nu"], out.extras["g_w"], out.extras["g_t"]], dim=1)
        weights: Tensor = self.get_buffer("weights")
        data = ls.squared_loss(pred, labels.stacked(), batch["omega"], weights)
        return Objective(total=data, parts={"data": data})

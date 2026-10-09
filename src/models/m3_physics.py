"""M3 (physics only): learned W, fixed ``r_bar``, empirical n (Strickler / Jarrett), depth via Phi.

Sources: proofs section 7.9 (``n_emp = n_S`` for ``S < S_J``, ``n_J`` for ``S >= S_J``; only the
W regressor is trained, ``L^M3 = sum_i omega_i e_W^2``; **no n is learned**; no gradient flows
through n), Proposition 7.5 (unique root of ``rho_J`` for ``b_J < 5/3``), section 7.5
(``nu_S = ln c_S + (1/6) ln D50``); protocol section 4.1 (M3 row: ``r = r_bar`` = training-fold
median of ``r_obs`` computed as ``1/(e^median(t_obs) - 1)``; Jarrett branch via ``rho_J``; n
projected onto ``[n_min, n_max]``; P3-07 Strickler, P3-08 Jarrett; Jarrett validity rows flagged,
never clipped); P6-Q2 (a) (``ln D50`` from ``NormStats``), P6-Q3 (a) (squared loss), P6-Q4 (a)
(inner grid then ``M = 10`` members; the ensemble point prediction is Phi at the mean member
``w``, :meth:`M3Physics.physics`).

Projection ``Pi`` onto the n band (pre-registered, protocol 4.1): Strickler rows use
``Pi(nu_S)``; Jarrett rows keep the ``rho_J`` root when ``nu_J`` lies in the band, otherwise they
are re-solved by Phi at ``Pi(nu_J)``. Phi and ``rho_J`` run without gradient.

Torch-only, float64, CPU. Owning phase: 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from src.config import BackboneParams, JarrettParams, ProtocolParams, StricklerParams
from src.models import losses as ls
from src.models.heads import (
    Batch,
    BoundedTransforms,
    LadderNet,
    LogFeature,
    ModelOutput,
    Objective,
    RoutedHeads,
    check_batch,
    project_nu,
)
from src.physics.manning_layer import PhiConfig, check_phys, phi, solve_jarrett
from src.physics.roughness_priors import jarrett_branch, jarrett_valid, strickler_nu
from src.physics.shape import check_rows, log_kappa, t_of_r

__all__ = ["M3Config", "M3Physics", "m3_physics", "m3_r_bar"]


def m3_r_bar(t_obs_train: np.ndarray) -> float:
    """``r_bar = 1/(e^median(t_obs) - 1)`` over the training rows (protocol 4.1, P3-08).

    ``numpy.median`` (mean of the two central values for an even count); requires a positive,
    finite median (Proposition 6.1).
    """
    t = np.asarray(t_obs_train, dtype=np.float64)
    if t.ndim != 1 or t.size == 0 or not bool(np.isfinite(t).all()):
        raise ValueError("t_obs_train must be a non-empty finite 1-D array")
    med = float(np.median(t))
    if not med > 0.0:
        raise ValueError("median(t_obs) must be > 0 (Proposition 6.1)")
    return 1.0 / math.expm1(med)


@dataclass(frozen=True)
class M3Config:
    """M3 settings: Phi, n band, Strickler / Jarrett constants and the fitted ``r_bar``."""

    phi: PhiConfig
    transforms: BoundedTransforms
    strickler: StricklerParams
    jarrett: JarrettParams
    s_floor: float
    r_bar: float

    def __post_init__(self) -> None:
        """``r_bar`` must be positive and finite (Proposition 3.6: Phi needs ``r > 0``)."""
        if not (math.isfinite(self.r_bar) and self.r_bar > 0.0):
            raise ValueError("r_bar must be finite and > 0")
        if not self.phi.kappa:
            raise ValueError("M3 uses full kappa (protocol 4.1)")

    @classmethod
    def from_protocol(cls, protocol: ProtocolParams, *, r_bar: float) -> M3Config:
        """Read params; ``r_bar`` comes from :func:`m3_r_bar` on the fit's training rows."""
        return cls(
            phi=PhiConfig.from_solver(protocol.solver),
            transforms=BoundedTransforms.from_domain(protocol.domain),
            strickler=protocol.priors.strickler,
            jarrett=protocol.priors.jarrett,
            s_floor=protocol.domain.s_floor,
            r_bar=r_bar,
        )


def m3_physics(w: Tensor, ln_d50: Tensor, phys: Tensor, cfg: M3Config) -> ModelOutput:
    """Physics part of M3 for given ``w`` (proofs 7.9): ``mu = [w, ell, t(r_bar), w + ell]``.

    No gradient. ``extras``: ``nu`` (after ``Pi``), ``jarrett`` (branch mask), ``nu_projected``,
    ``jarrett_invalid`` (Jarrett rows outside eq. 6 validity; reported, not clipped) and
    ``iterations``.
    """
    q, s = phys[:, 0], phys[:, 1]
    check_phys(q, s, s_floor=cfg.s_floor)
    check_rows(w=w, ln_d50=ln_d50, q=q, s=s)
    tr = cfg.transforms
    with torch.no_grad():
        w = w.detach()
        r = torch.full_like(w, cfg.r_bar)
        steep = jarrett_branch(s, cfg.jarrett)
        nu_s = strickler_nu(ln_d50, cfg.strickler)
        nu = project_nu(nu_s, tr)
        projected = nu != nu_s
        ell = torch.empty_like(w)
        iters = torch.zeros_like(w, dtype=torch.int64)
        invalid = torch.zeros_like(steep)
        j = steep.nonzero().flatten()
        if j.numel() > 0:
            ell_j, nu_j, it_j = solve_jarrett(w[j], r[j], q[j], s[j], cfg.jarrett, cfg.phi)
            nu_jp = project_nu(nu_j, tr)
            nu[j], projected[j], ell[j], iters[j] = nu_jp, nu_jp != nu_j, ell_j, it_j
            ln_r_hyd = ell_j + log_kappa(
                ell_j, w[j], r[j], level=cfg.phi.quad_level, trunc=cfg.phi.quad_trunc
            )
            invalid[j] = ~jarrett_valid(s[j], ln_r_hyd, cfg.jarrett)
        resolve = (~steep) | projected
        k = resolve.nonzero().flatten()
        if k.numel() > 0:
            res = phi(w[k], r[k], nu[k], q[k], s[k], cfg.phi)
            ell[k], iters[k] = res.ell, res.iterations
        mu = torch.stack([w, ell, t_of_r(r), w + ell], dim=1)
    extras = {
        "nu": nu,
        "jarrett": steep,
        "nu_projected": projected,
        "jarrett_invalid": invalid,
        "iterations": iters,
    }
    return ModelOutput(mu=mu, sigma=None, extras=extras)


class M3Physics(LadderNet):
    """M3: the W regressor (full trunk, one head) plus :func:`m3_physics` (proofs 7.9)."""

    def __init__(
        self, in_features: int, cfg: M3Config, backbone: BackboneParams, ln_d50: LogFeature
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.cfg = cfg
        self.ln_d50 = ln_d50
        self.heads = RoutedHeads(in_features, ["w"], [], None, backbone)
        self.register_buffer("weights", torch.ones(1, dtype=torch.float64))

    def physics(self, w: Tensor, batch: Batch) -> ModelOutput:
        """Phi part at a given ``w``; the M3 ensemble calls it with the mean member ``w``."""
        check_batch(batch, self.in_features)
        return m3_physics(w, self.ln_d50(batch["x"]), batch["phys"], self.cfg)

    def forward(self, batch: Batch) -> ModelOutput:
        """``w`` keeps its gradient in ``mu[:, 0]``; the other columns come from Phi (no grad)."""
        check_batch(batch, self.in_features)
        w = self.heads(batch["x"])["w"]
        out = self.physics(w, batch)
        mu = torch.cat([w[:, None], out.mu[:, 1:]], dim=1)
        return ModelOutput(mu=mu, sigma=None, extras=out.extras)

    def objective(
        self, batch: Batch, out: ModelOutput, epoch: int, *, monitor: bool = False
    ) -> Objective:
        """``L^M3 = sum_i omega_i (w_i - ln W_i)^2`` (proofs 7.9); independent of the epoch."""
        weights: Tensor = self.get_buffer("weights")
        data = ls.squared_loss(out.mu[:, :1], batch["y"][:, :1], batch["omega"], weights)
        return Objective(total=data, parts={"data": data})

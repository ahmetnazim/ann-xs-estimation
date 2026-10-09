"""M5 (primary): hard hybrid through Phi with full kappa; L_phys identically zero.

Sources: proofs section 7.4 (``L^M5 = L_data``, replaced by the NLL of 9.1; ``L_phys = 0`` by
Theorem 4.1), section 7.5 (A3-SP prior, A3-SR logit residual), section 7.6 (bounded
transforms), section 7.10 (ablations A1 ``kappa = 1``, A2 ``w_t = 0``, A3), section 9.1
(physics-exact means ``(w, ell, t)``, log-space sigma heads, derived ``ln A``); protocol
section 4.1 (heads ``(w, r, nu)``, n-head under ``I_blind``, P3-11; A3-SP ``lambda_p = 2``,
A3-SR ``varpi = 0.01``, ``lambda_Delta = 2.865``; ``w_A = 1`` sensitivity) and section 4.2
(beta-NLL ``beta_N = 0.5`` after a 20-epoch MSE warm start; ``sigma_min = 0.01``);
P6-D4 (a), P6-D12 (a), P6-Q2 (a), P6-Q3 (a), Batch B Q-B1 (a).

Ablation A4 (no D50, AI) is a feature-set switch applied by the caller (``in_features``).

Torch-only, float64, CPU. Owning phase: 6.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import Tensor

from src.config import BackboneParams, ProtocolParams, StricklerParams
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
    physics_sigmas,
    strickler_logit,
)
from src.physics.manning_layer import PhiConfig, check_phys, phi
from src.physics.roughness_priors import strickler_nu
from src.physics.shape import t_of_r

__all__ = ["M5Config", "M5Hard", "Prior"]

#: Strickler ablation of proofs 7.5: none (primary), A3-SP (prior) or A3-SR (logit residual).
Prior = Literal["none", "SP", "SR"]


@dataclass(frozen=True)
class M5Config:
    """M5 settings; :meth:`from_protocol` reads params, flags select the ablations (P6-D12)."""

    phi: PhiConfig
    transforms: BoundedTransforms
    weights: tuple[float, float, float, float]
    sigma_min: float
    beta_n: float
    warm_start_epochs: int
    s_floor: float
    prior: Prior = "none"
    c_s: float = 0.047
    lambda_p: float = 0.0
    lambda_delta: float = 0.0
    varpi: float = 0.01

    @classmethod
    def from_protocol(
        cls,
        protocol: ProtocolParams,
        *,
        kappa: bool = True,
        a2: bool = False,
        w_a: float | None = None,
        prior: Prior = "none",
    ) -> M5Config:
        """Primary M5 by default; ``kappa=False`` A1, ``a2`` A2, ``prior`` A3, ``w_a=1`` SA."""
        weights = protocol.model.loss_weights
        if w_a is not None:
            weights = weights.model_copy(update={"w_a": w_a})
        w = ls.target_weights(weights, a2=a2).tolist()
        return cls(
            phi=PhiConfig.from_solver(protocol.solver, kappa=kappa),
            transforms=BoundedTransforms.from_domain(protocol.domain),
            weights=(w[0], w[1], w[2], w[3]),
            sigma_min=protocol.uq.sigma_min,
            beta_n=protocol.uq.beta_nll,
            warm_start_epochs=protocol.uq.warm_start_epochs,
            s_floor=protocol.domain.s_floor,
            prior=prior,
            c_s=protocol.priors.strickler.c_s,
            lambda_p=protocol.priors.strickler_prior.lambda_p,
            lambda_delta=protocol.priors.strickler_residual.lambda_delta,
            varpi=protocol.priors.strickler_residual.varpi,
        )

    def with_flags(self, **changes: object) -> M5Config:
        """Return a copy with dataclass fields replaced (used by tests and sweeps)."""
        return replace(self, **changes)  # type: ignore[arg-type]


class M5Hard(LadderNet):
    """``x -> (w, r, nu) -> Phi -> (w, ell, t)`` with sigma heads (proofs 7.4, 9.1).

    Parameters
    ----------
    in_features, blind_idx
        Width of ``x`` and ``x_blind_idx`` (``I_blind``) from the dataset.
    cfg
        :class:`M5Config`.
    backbone
        ``params.model.backbone`` (both trunks).
    ln_d50
        Recovery of ``ln D50`` from ``x`` (P6-Q2 (a)); required for A3-SP and A3-SR.
    """

    def __init__(
        self,
        in_features: int,
        blind_idx: Tensor,
        cfg: M5Config,
        backbone: BackboneParams,
        ln_d50: LogFeature | None = None,
    ) -> None:
        super().__init__()
        if cfg.prior != "none" and ln_d50 is None:
            raise ValueError("A3-SP / A3-SR need ln D50 (LogFeature of D50_m)")
        self.in_features = in_features
        self.cfg = cfg
        self.ln_d50 = ln_d50
        self.strickler = StricklerParams(c_s=cfg.c_s)
        self.heads = RoutedHeads(
            in_features, ["z_w", "z_r", "s_w", "s_dbar", "s_t"], ["z_n"], blind_idx, backbone
        )
        self.register_buffer("weights", torch.tensor(cfg.weights, dtype=torch.float64))

    def forward(self, batch: Batch) -> ModelOutput:
        """Heads, bounded transforms, Phi, sigma heads; ``mu = [w, ell, t, w + ell]``."""
        check_batch(batch, self.in_features)
        x, phys = batch["x"], batch["phys"]
        q, s = phys[:, 0], phys[:, 1]
        check_phys(q, s, s_floor=self.cfg.s_floor)
        raw = self.heads(x)
        tr = self.cfg.transforms
        extras: dict[str, Tensor] = {}
        z_n = raw["z_n"]
        if self.cfg.prior != "none" and self.ln_d50 is not None:
            nu_s = strickler_nu(self.ln_d50(x), self.strickler)
            extras["nu_s"] = nu_s
            if self.cfg.prior == "SR":
                extras["delta"] = z_n
                z_n = strickler_logit(nu_s, tr, self.cfg.varpi) + z_n
        w, r, nu = raw["z_w"], tr.r(raw["z_r"]), tr.nu(z_n)
        res = phi(w, r, nu, q, s, self.cfg.phi)
        ell = res.ell
        mu = torch.stack([w, ell, t_of_r(r), w + ell], dim=1)
        sigma = physics_sigmas(raw, self.cfg.sigma_min)
        extras.update({"nu": nu, "r": r, "iterations": res.iterations})
        return ModelOutput(mu=mu, sigma=sigma, extras=extras)

    def objective(
        self, batch: Batch, out: ModelOutput, epoch: int, *, monitor: bool = False
    ) -> Objective:
        """``L_data`` (MSE warm start, then beta-NLL) ``[+ prior | + ridge]`` (proofs 7.4, 7.5).

        ``monitor`` does not change the M5 objective (P6-D5 (ii) concerns M4 only).
        """
        cfg = self.cfg
        mode = ls.loss_mode(epoch, cfg.warm_start_epochs, cfg.beta_n)
        weights: Tensor = self.get_buffer("weights")
        omega = batch["omega"]
        parts = {
            "data": ls.data_term(
                mode, out.mu, batch["y"], omega, weights, sigma=out.sigma, beta_n=cfg.beta_n
            )
        }
        if cfg.prior == "SP":
            parts["prior"] = ls.strickler_prior_penalty(
                out.extras["nu"], out.extras["nu_s"], omega, cfg.lambda_p
            )
        elif cfg.prior == "SR":
            parts["ridge"] = ls.ridge_penalty(out.extras["delta"], omega, cfg.lambda_delta)
        total = torch.stack(list(parts.values())).sum()
        return Objective(total=total, parts=parts)

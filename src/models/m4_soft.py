"""M4 (soft hybrid): heads ``(w, ell, r, nu)`` with the capped Manning penalty, under I_blind.

Sources: proofs section 7.7 (direct ``ell`` head; hard shape and area closure; explicit residual
``R_M = (5/3) ell + (2/3) k(e^(ell - w), r) + w + s/2 - nu - q_E``; ``L^M4 = L_data +
lambda_M sum_i omega_i R_M,i^2``), Proposition 7.4 (a free n-head degenerates the penalty under
I_full, hence I_blind), section 7.7.1 and Proposition 7.9 (capped schedule
``lambda_M(e) = min(lambda0 gamma^e, lambda_max)``), (A7.3) (the ``Xi_e`` diagnostic), section
9.1 (sigma heads, derived ``ln A``); protocol section 4.1 (P3-09: base M4 under I_blind,
``lambda0 = 1``, ``gamma = 1.1``, ``lambda_max`` in {10, 100, 1000} by inner CV; no M4+, so no
``lambda_A``); P6-D5 (ii) (the early-stopping monitor uses ``lambda_max``); P6-Q3 (a) (same data
term as M5).

``Objective.parts`` holds ``data`` and ``penalty`` (= ``lambda_M(e) P``) separately, so the
trainer can form ``Xi_e = ||grad penalty|| / ||grad data||`` (A7.3) from the two parts.

Torch-only, float64, CPU. Owning phase: 6.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

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
    physics_sigmas,
)
from src.physics.manning_layer import PhiConfig, check_phys, manning_residual
from src.physics.shape import t_of_r

__all__ = ["M4Config", "M4Soft"]


@dataclass(frozen=True)
class M4Config:
    """M4 settings; ``lambda_max`` is the inner-CV hyperparameter (P3-09, P6-D12)."""

    phi: PhiConfig
    transforms: BoundedTransforms
    weights: tuple[float, float, float, float]
    sigma_min: float
    beta_n: float
    warm_start_epochs: int
    s_floor: float
    lambda0: float
    gamma: float
    lambda_max: float

    def __post_init__(self) -> None:
        """Validate the schedule constants (Proposition 7.9 needs ``lambda_max >= lambda0``)."""
        ls.m4_cap_epoch(self.lambda0, self.gamma, self.lambda_max)

    @classmethod
    def from_protocol(
        cls, protocol: ProtocolParams, *, lambda_max: float, w_a: float | None = None
    ) -> M4Config:
        """Build base M4 with ``lambda_max`` from ``model.inner_grid.m4_lambda_max``."""
        if lambda_max not in protocol.model.inner_grid.m4_lambda_max:
            raise ValueError(
                f"lambda_max {lambda_max} is not in the pre-registered grid "
                f"{protocol.model.inner_grid.m4_lambda_max} (P3-09)"
            )
        weights = protocol.model.loss_weights
        if w_a is not None:
            weights = weights.model_copy(update={"w_a": w_a})
        w = ls.target_weights(weights).tolist()
        return cls(
            phi=PhiConfig.from_solver(protocol.solver),
            transforms=BoundedTransforms.from_domain(protocol.domain),
            weights=(w[0], w[1], w[2], w[3]),
            sigma_min=protocol.uq.sigma_min,
            beta_n=protocol.uq.beta_nll,
            warm_start_epochs=protocol.uq.warm_start_epochs,
            s_floor=protocol.domain.s_floor,
            lambda0=protocol.model.m4_schedule.lambda0,
            gamma=protocol.model.m4_schedule.gamma,
            lambda_max=lambda_max,
        )

    def with_flags(self, **changes: object) -> M4Config:
        """Return a copy with dataclass fields replaced (used by tests and sweeps)."""
        return replace(self, **changes)  # type: ignore[arg-type]

    def penalty_weight(self, epoch: int, *, monitor: bool = False) -> float:
        """``lambda_M(e)``; the early-stopping monitor uses ``lambda_max`` (P6-D5 (ii))."""
        if monitor:
            return self.lambda_max
        return ls.m4_lambda(epoch, self.lambda0, self.gamma, self.lambda_max)


class M4Soft(LadderNet):
    """``x -> (w, ell, r, nu)``; ``mu = [w, ell, t(r), w + ell]``; Manning as a penalty (7.7).

    The ``ell`` head is direct (not solved); ``nu`` comes from the blind trunk (I_blind, O-4).
    """

    def __init__(
        self, in_features: int, blind_idx: Tensor, cfg: M4Config, backbone: BackboneParams
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.cfg = cfg
        self.heads = RoutedHeads(
            in_features,
            ["z_w", "ell", "z_r", "s_w", "s_dbar", "s_t"],
            ["z_n"],
            blind_idx,
            backbone,
        )
        self.register_buffer("weights", torch.tensor(cfg.weights, dtype=torch.float64))

    def forward(self, batch: Batch) -> ModelOutput:
        """Heads, bounded transforms, explicit residual ``R_M`` (eq. 3.1 with a free ``ell``)."""
        check_batch(batch, self.in_features)
        x, phys = batch["x"], batch["phys"]
        q, s = phys[:, 0], phys[:, 1]
        check_phys(q, s, s_floor=self.cfg.s_floor)
        raw = self.heads(x)
        tr = self.cfg.transforms
        w, ell, r, nu = raw["z_w"], raw["ell"], tr.r(raw["z_r"]), tr.nu(raw["z_n"])
        r_m = manning_residual(ell, w, r, nu, q, s, self.cfg.phi)
        mu = torch.stack([w, ell, t_of_r(r), w + ell], dim=1)
        sigma = physics_sigmas(raw, self.cfg.sigma_min)
        return ModelOutput(mu=mu, sigma=sigma, extras={"nu": nu, "r": r, "R_M": r_m})

    def objective(
        self, batch: Batch, out: ModelOutput, epoch: int, *, monitor: bool = False
    ) -> Objective:
        """``L_data + lambda_M(e) sum omega R_M^2``; ``monitor`` uses ``lambda_max`` (7.7.1)."""
        cfg = self.cfg
        mode = ls.loss_mode(epoch, cfg.warm_start_epochs, cfg.beta_n)
        weights: Tensor = self.get_buffer("weights")
        omega = batch["omega"]
        parts = {
            "data": ls.data_term(
                mode, out.mu, batch["y"], omega, weights, sigma=out.sigma, beta_n=cfg.beta_n
            ),
            "penalty": ls.residual_penalty(
                out.extras["R_M"], omega, cfg.penalty_weight(epoch, monitor=monitor)
            ),
        }
        total = parts["data"] + parts["penalty"]
        return Objective(total=total, parts=parts)

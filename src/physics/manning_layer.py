"""Physics layer Phi: depth solve by Picard iteration, implicit-function gradients (proofs 3).

Sources: proofs section 3.1 (Phi and the Manning residual rho, eq. 3.1), section 3.2 (ablation
A1, kappa = 1), Propositions 3.1-3.2 (unique root; contraction L <= 2/5 from the wide-channel
start; stopping rule ``|ell_m - ell_(m-1)| <= (3/2) tau_l``), Proposition 3.3 and Algorithm Phi
(closed-form partials, no unrolling), Proposition 3.4 (row-wise action; per-row freezing),
Proposition 3.6 (Phi is C^1 on D_nat), section 1.3 and D9 (data domain), Theorem 4.1 (outputs),
Proposition 6.2 (explicit inverse ``nu = Phi^-1``), Proposition 7.5 (Jarrett residual rho_J, M3);
protocol section 4.2 (P3-12: tau_l = 1e-10, K_max = 40, per-row stopping); P6-D2 (a).

Inputs are log coordinates (proofs 1.1.1) in SI: ``w = ln W``, ``r`` (> 0), ``nu = ln n``,
``q = ln Q`` (``Q_E``; ``Q_C`` for SA1), ``s = ln S``. Outputs ``(B, 4)`` =
``[ell, ln d_max, ln A, ln V]`` with ``ln d_max = ell + t(r)``, ``ln A = w + ell``,
``ln V = q - w - ell`` (section 3.1).

Gradient mechanism (P6-D2 (a)): :class:`ManningPhi` runs the Picard loop under ``no_grad`` with
per-row freezing; its backward applies Algorithm Phi at the converged ``ell``::

    D = 5/3 - (2/3) eta;   d ell/dq = d ell/dnu = 1/D;   d ell/ds = -1/(2D);
    d ell/dw = -(1 + 2 eta/3)/D;   d ell/dr = -(2/3) k_r / D.

``D`` lies in (1, 5/3] (Proposition 3.1), so no guard on the division is needed. Non-convergence
within ``K_max`` raises :class:`PhiConvergenceError`; it is never returned silently.
:func:`phi_one_step` is the P6-D2 (b) surrogate, kept as a cross-check.

The backward is first order only (P7-Q1, P7-B4): ``eta`` and ``k_r`` are saved as constants, so
a double backward would omit ``d(1/D)/dtheta``. Any backward run with ``create_graph=True`` (grad
mode on inside the backward) raises ``RuntimeError`` before computing anything, on every autograd
route (``.backward()`` and ``torch.autograd.grad(inputs=...)``). No model or loss in this project
uses second derivatives.

Torch-only, float64, CPU. Owning phase: 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

from src.config import JarrettParams, SolverParams
from src.physics.quadrature import perimeter_fm1
from src.physics.shape import check_rows, forbid_create_graph, log_kappa, section_terms, t_of_r

__all__ = [
    "ManningPhi",
    "PhiConfig",
    "PhiConvergenceError",
    "PhiOutput",
    "check_phys",
    "manning_residual",
    "phi",
    "phi_inverse_nu",
    "phi_one_step",
    "solve_jarrett",
    "wide_channel_depth",
]


class PhiConvergenceError(RuntimeError):
    """Raised when rows fail the stopping rule within ``K_max`` iterations (never silenced)."""


@dataclass(frozen=True)
class PhiConfig:
    """Solver and quadrature settings for Phi (params ``solver``; protocol 4.2, P3-12).

    ``kappa=False`` is ablation A1 (kappa = 1, proofs 3.2): the closed form, no iteration.
    """

    tau_l: float
    k_max: int
    quad_level: int
    quad_trunc: float
    kappa: bool = True

    def __post_init__(self) -> None:
        """Reject a non-positive tolerance or an empty iteration cap."""
        if not (self.tau_l > 0 and self.k_max >= 1):
            raise ValueError("tau_l must be > 0 and k_max >= 1")

    @classmethod
    def from_solver(cls, solver: SolverParams, *, kappa: bool = True) -> PhiConfig:
        """Build from the frozen ``params.solver`` group."""
        return cls(
            tau_l=solver.tau_l,
            k_max=solver.k_max,
            quad_level=solver.quad_level,
            quad_trunc=solver.quad_trunc,
            kappa=kappa,
        )


@dataclass(frozen=True)
class PhiOutput:
    """Phi outputs: ``out`` (B, 4) = [ell, ln d_max, ln A, ln V]; Picard ``iterations`` (B,)."""

    out: Tensor
    iterations: Tensor

    @property
    def ell(self) -> Tensor:
        """``ell = ln dbar`` (B,)."""
        return self.out[:, 0]


def check_phys(q: Tensor, s: Tensor, *, s_floor: float) -> None:
    """Data-domain check for ``(q, s)`` (proofs 1.3, D9): finite, ``S >= S_floor``.

    ``q`` finite is equivalent to ``0 < Q_E < inf``. The floor is compared in log space with the
    same NumPy ``log`` the loader uses (``src.data.dataset``), so a floored reach
    (``S = S_floor``, flag ``F_S``) passes exactly. Violations raise; nothing is clipped.
    """
    check_rows(q=q, s=s)
    ln_floor = float(np.log(s_floor))
    if not bool((s >= ln_floor).all()):
        bad = int((s < ln_floor).sum())
        raise ValueError(f"{bad} rows with S < S_floor = {s_floor} (outside the domain D)")


def wide_channel_depth(w: Tensor, nu: Tensor, q: Tensor, s: Tensor) -> Tensor:
    """``ell_0 = (3/5)(q + nu - w - s/2)``: the kappa = 1 depth (proofs 3.2, ablation A1)."""
    return 0.6 * (q + nu - w - 0.5 * s)


def manning_residual(
    ell: Tensor, w: Tensor, r: Tensor, nu: Tensor, q: Tensor, s: Tensor, cfg: PhiConfig
) -> Tensor:
    """``rho = (5/3) ell + (2/3) k(e^(ell-w), r) + w + s/2 - nu - q`` (eq. 3.1), differentiable.

    ``k`` uses the analytic partials (:func:`src.physics.shape.log_kappa`); ``k = 0`` when
    ``cfg.kappa`` is False. This is also M4's explicit residual ``R_M`` (proofs 7.7).
    """
    check_rows(ell=ell, w=w, r=r, nu=nu, q=q, s=s)
    k = (
        log_kappa(ell, w, r, level=cfg.quad_level, trunc=cfg.quad_trunc)
        if cfg.kappa
        else torch.zeros_like(ell)
    )
    return (5.0 / 3.0) * ell + (2.0 / 3.0) * k + w + 0.5 * s - nu - q


def _check_phi_inputs(w: Tensor, r: Tensor, nu: Tensor, q: Tensor, s: Tensor) -> int:
    n = check_rows(w=w, r=r, nu=nu, q=q, s=s)
    if not bool((r > 0).all()):
        raise ValueError("r must be > 0 (Proposition 3.6: Phi is defined on r > 0)")
    return n


def _picard(
    ell0: Tensor,
    w: Tensor,
    r: Tensor,
    gain: float,
    stop: float,
    cfg: PhiConfig,
) -> tuple[Tensor, Tensor]:
    """Per-row Picard ``ell_m = ell0 + gain * ln F(beta(ell_(m-1)), r)`` (Props 3.2, 3.4c, 7.5).

    Each row is frozen at the first ``m`` with ``|ell_m - ell_(m-1)| <= stop``; only active rows
    are evaluated, so a row's iterates depend on that row alone.
    """
    ell = ell0.clone()
    iters = torch.zeros_like(ell0, dtype=torch.int64)
    active = torch.arange(ell0.shape[0])
    for m in range(1, cfg.k_max + 1):
        if active.numel() == 0:
            break
        prev = ell[active]
        a = torch.exp(prev - w[active])
        beta = 2.0 * (r[active] + 1.0) * a
        fm1 = perimeter_fm1(beta, r[active], level=cfg.quad_level, trunc=cfg.quad_trunc)
        new = ell0[active] + gain * torch.log1p(fm1)
        ell[active] = new
        iters[active] = m
        active = active[(new - prev).abs() > stop]
    if active.numel() > 0:
        raise PhiConvergenceError(
            f"{active.numel()} of {ell0.shape[0]} rows did not meet the stopping rule within "
            f"K_max = {cfg.k_max} iterations (tau_l = {cfg.tau_l})"
        )
    return ell, iters


class ManningPhi(torch.autograd.Function):
    """``ell* = ell*(w, r, nu, q, s)``: no-grad Picard forward, IFT backward (P6-D2 (a)).

    ``apply(w, r, nu, q, s, cfg)`` returns ``(ell, iterations)``; ``iterations`` is int64 and
    not differentiable.
    """

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        w: Tensor,
        r: Tensor,
        nu: Tensor,
        q: Tensor,
        s: Tensor,
        cfg: PhiConfig,
    ) -> tuple[Tensor, Tensor]:
        """Algorithm Phi, forward: wide-channel start, per-row Picard, save ``eta, k_r``."""
        _check_phi_inputs(w, r, nu, q, s)
        with torch.no_grad():
            ell0 = wide_channel_depth(w, nu, q, s)
            if cfg.kappa:
                # Proposition 3.2: gain 2/5; stop at (3/2) tau_l so that |ell - ell*| <= tau_l.
                ell, iters = _picard(ell0, w, r, 0.4, 1.5 * cfg.tau_l, cfg)
                terms = section_terms(ell, w, r, level=cfg.quad_level, trunc=cfg.quad_trunc)
                eta, k_r = terms.eta, terms.k_r
            else:
                ell, iters = ell0.clone(), torch.zeros_like(ell0, dtype=torch.int64)
                eta, k_r = torch.zeros_like(ell0), torch.zeros_like(ell0)
        ctx.save_for_backward(eta, k_r)
        ctx.mark_non_differentiable(iters)
        return ell, iters

    @staticmethod
    def backward(
        ctx: FunctionCtx, g: Tensor, _g_iters: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, None]:
        """Algorithm Phi, backward: Proposition 3.3 at the converged root (no unrolling)."""
        forbid_create_graph("ManningPhi")
        eta, k_r = ctx.saved_tensors  # type: ignore[attr-defined]
        inv_d = 1.0 / (5.0 / 3.0 - (2.0 / 3.0) * eta)  # 1/D, D in (1, 5/3] (Prop 3.1)
        gd = g * inv_d
        return (
            -gd * (1.0 + (2.0 / 3.0) * eta),  # d ell / dw
            -gd * (2.0 / 3.0) * k_r,  # d ell / dr
            gd,  # d ell / dnu
            gd,  # d ell / dq
            -0.5 * gd,  # d ell / ds
            None,
        )


def phi(w: Tensor, r: Tensor, nu: Tensor, q: Tensor, s: Tensor, cfg: PhiConfig) -> PhiOutput:
    """Batch Phi (proofs 3.1, 3.5): ``(B,)`` inputs -> ``(B, 4)`` ``[ell, ln d_max, ln A, ln V]``.

    Theorem 4.1: ``ln A = w + ell`` and ``ln V = q - ln A`` hold by construction;
    ``|rho(ell)| <= (5/3) tau_l``; ``d_max / dbar = 1 + 1/r``.
    """
    n = _check_phi_inputs(w, r, nu, q, s)
    ell, iters = ManningPhi.apply(w, r, nu, q, s, cfg)
    out = torch.stack([ell, ell + t_of_r(r), w + ell, q - w - ell], dim=1)
    if tuple(out.shape) != (n, 4):
        raise RuntimeError(f"Phi output shape {tuple(out.shape)} != {(n, 4)}")
    return PhiOutput(out=out, iterations=iters)


def phi_one_step(w: Tensor, r: Tensor, nu: Tensor, q: Tensor, s: Tensor, cfg: PhiConfig) -> Tensor:
    """P6-D2 (b) surrogate ``ell = sg(ell*) - rho(sg(ell*); theta) / sg(D)`` (cross-check only).

    ``rho`` is differentiated by plain autograd **through the quadrature integrand**, so its
    gradient is independent of the analytic partials used by :class:`ManningPhi`. Value equals
    ``ell*`` up to ``|rho| / D <= (5/3) tau_l``; first-order gradient equals Proposition 3.3.
    """
    with torch.no_grad():
        ell_star, _ = ManningPhi.apply(w, r, nu, q, s, cfg)
        if cfg.kappa:
            eta = section_terms(ell_star, w, r, level=cfg.quad_level, trunc=cfg.quad_trunc).eta
        else:
            eta = torch.zeros_like(ell_star)
        d = 5.0 / 3.0 - (2.0 / 3.0) * eta
    if cfg.kappa:
        beta = 2.0 * (r + 1.0) * torch.exp(ell_star - w)
        fm1 = perimeter_fm1(beta, r, level=cfg.quad_level, trunc=cfg.quad_trunc)
        k = -torch.log1p(fm1)
    else:
        k = torch.zeros_like(ell_star)
    rho = (5.0 / 3.0) * ell_star + (2.0 / 3.0) * k + w + 0.5 * s - nu - q
    out: Tensor = ell_star - rho / d
    return out


def phi_inverse_nu(
    ell: Tensor, w: Tensor, r: Tensor, q: Tensor, s: Tensor, cfg: PhiConfig
) -> Tensor:
    """Explicit ``nu = (5/3) ell + w + (2/3) k(a, r) + s/2 - q`` (Proposition 6.2).

    The M6 label ``nu_obs`` (observed ``ell, w, r``; ``q = ln Q_E``) and the ensemble roughness
    ``n* = Phi^-1(mu*)`` (proofs 9.3). No fixed point is needed because ``ell`` is given.
    """
    check_rows(ell=ell, w=w, r=r, q=q, s=s)
    if not bool((r > 0).all()):
        raise ValueError("r must be > 0")
    k = (
        log_kappa(ell, w, r, level=cfg.quad_level, trunc=cfg.quad_trunc)
        if cfg.kappa
        else torch.zeros_like(ell)
    )
    return (5.0 / 3.0) * ell + w + (2.0 / 3.0) * k + 0.5 * s - q


def solve_jarrett(
    w: Tensor, r: Tensor, q: Tensor, s: Tensor, jarrett: JarrettParams, cfg: PhiConfig
) -> tuple[Tensor, Tensor, Tensor]:
    """Root of ``rho_J`` (Proposition 7.5) for M3's steep branch; no gradient (M3 learns no n).

    ``rho_J(ell) = (5/3 - b) ell + (2/3 - b) k + w + (1/2 - a) s - ln c_J' - q`` with
    ``n_J = c_J' S^a R^b`` (``R = kappa dbar`` in m; protocol 4.1 SI form). In fixed-point form
    ``ell = ell_0J + c ln F`` with ``c = (2/3 - b)/(5/3 - b)``; the map has slope ``c eta`` in
    [0, c), so it contracts when ``b <= 2/3`` (Jarrett: ``b = -0.16``, ``c = 0.4526``), and the
    iterates rise monotonically from ``ell_0J`` (as in Proposition 3.2). Stopping at
    ``|ell_m - ell_(m-1)| <= tau_l (1 - c)/c`` guarantees ``|ell - ell*| <= tau_l``.

    Returns
    -------
    tuple of Tensor
        ``(ell, nu_J, iterations)`` with ``nu_J = ln c_J' + a s + b (ell + k)`` at the root.
    """
    _check_phi_inputs(w, r, torch.zeros_like(w), q, s)
    a_j, b_j = jarrett.a_j, jarrett.b_j
    if not b_j <= 2.0 / 3.0:
        raise ValueError("Picard contraction for rho_J needs b_J <= 2/3 (Prop 7.5 needs < 5/3)")
    c = (2.0 / 3.0 - b_j) / (5.0 / 3.0 - b_j)
    ln_c = math.log(jarrett.c_j_si)
    with torch.no_grad():
        ell0 = (q + ln_c - w - (0.5 - a_j) * s) / (5.0 / 3.0 - b_j)
        if not cfg.kappa:
            raise ValueError("M3 uses full kappa (protocol 4.1)")
        if c == 0.0:
            ell, iters = ell0.clone(), torch.zeros_like(ell0, dtype=torch.int64)
        else:
            ell, iters = _picard(ell0, w, r, c, cfg.tau_l * (1.0 - c) / c, cfg)
        k = log_kappa(ell, w, r, level=cfg.quad_level, trunc=cfg.quad_trunc)
        nu_j = ln_c + a_j * s + b_j * (ell + k)
    return ell, nu_j, iters

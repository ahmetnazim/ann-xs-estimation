"""Power-law section geometry: a, beta, kappa = 1/F, eta, k = ln kappa, k_r and t(r).

Sources: proofs section 2: Proposition 2.1 (``dbar = d_max r/(r+1)``,
``t = ln(d_max/dbar) = ln(1 + 1/r)``), Proposition 2.2 (``beta = 2(r+1)a``, ``kappa = 1/F``),
Proposition 2.7 (``eta = d ln F / d ln beta = beta F_beta / F`` in [0, 1)) and Proposition 2.8
(``k_r = dk/dr|_a = -(2 a F_beta + F_r)/F``); section 3.4, Algorithm Phi (the same formulas);
section 7.1 (shape target ``t``); Proposition 6.1 (``r = 1/(e^t - 1)``, ``dt/dr = -1/(r(r+1))``).

Log coordinates (proofs 1.1.1): ``ell = ln dbar``, ``w = ln W``, so ``a = dbar/W = exp(ell - w)``.
``ln F`` is ``log1p(F - 1)`` with ``F - 1`` integrated directly (section 3.5 numerical note).

Torch-only, float64, CPU, row-wise. Owning phase: 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

from src.physics.quadrature import perimeter_integrals

__all__ = ["SectionTerms", "check_rows", "dt_dr", "log_kappa", "r_of_t", "section_terms", "t_of_r"]


def check_rows(**tensors: Tensor) -> int:
    """Assert 1-D float64 CPU finite tensors of one common length; return that length ``B``."""
    n: int | None = None
    for name, x in tensors.items():
        if x.dtype != torch.float64 or x.device.type != "cpu":
            raise TypeError(f"{name} must be float64 on CPU (protocol 4.2, O-8)")
        if x.dim() != 1:
            raise ValueError(f"{name} must be 1-D (B,), got shape {tuple(x.shape)}")
        if n is None:
            n = int(x.shape[0])
        elif int(x.shape[0]) != n:
            raise ValueError(f"{name} has length {int(x.shape[0])}, expected {n}")
        if not bool(torch.isfinite(x).all()):
            raise ValueError(f"non-finite values in {name}")
    if n is None:
        raise ValueError("no tensors given")
    return n


@dataclass(frozen=True)
class SectionTerms:
    """Row-wise section quantities at one depth ``ell`` (proofs section 2, Algorithm Phi).

    Attributes
    ----------
    a, beta
        ``dbar / W`` and ``2 (r + 1) a`` (Proposition 2.2).
    fm1, log_f
        ``F - 1`` and ``ln F = log1p(F - 1)``; ``k = ln kappa = -ln F``.
    eta
        ``beta F_beta / F`` in [0, 1) (Proposition 2.7).
    k_r
        ``dk/dr`` at fixed ``a``: ``-(2 a F_beta + F_r) / F`` (Proposition 2.8).
    """

    a: Tensor
    beta: Tensor
    fm1: Tensor
    log_f: Tensor
    eta: Tensor
    k_r: Tensor

    @property
    def k(self) -> Tensor:
        """``k = ln kappa = -ln F`` (Proposition 2.2)."""
        return -self.log_f

    @property
    def kappa(self) -> Tensor:
        """``kappa = W / P = 1 / F`` (Proposition 2.2)."""
        return torch.exp(-self.log_f)


def section_terms(ell: Tensor, w: Tensor, r: Tensor, *, level: int, trunc: float) -> SectionTerms:
    """Evaluate :class:`SectionTerms` at ``(ell, w, r)`` (Propositions 2.2, 2.7, 2.8).

    Differentiable by autograd through the quadrature (used only for cross-checks; the physics
    layer uses the analytic partials returned here).
    """
    check_rows(ell=ell, w=w, r=r)
    a = torch.exp(ell - w)
    beta = 2.0 * (r + 1.0) * a
    fm1, f_beta, f_r = perimeter_integrals(beta, r, level=level, trunc=trunc)
    f = 1.0 + fm1
    return SectionTerms(
        a=a,
        beta=beta,
        fm1=fm1,
        log_f=torch.log1p(fm1),
        eta=beta * f_beta / f,
        k_r=-(2.0 * a * f_beta + f_r) / f,
    )


class _LogKappa(torch.autograd.Function):
    """``k(e^(ell - w), r)`` with the analytic partials ``(-eta, +eta, k_r)`` (Props 2.7, 2.8)."""

    @staticmethod
    def forward(
        ctx: FunctionCtx, ell: Tensor, w: Tensor, r: Tensor, level: int, trunc: float
    ) -> Tensor:
        """Evaluate ``k`` without a graph and save ``eta, k_r`` for the backward pass."""
        with torch.no_grad():
            terms = section_terms(ell, w, r, level=level, trunc=trunc)
        ctx.save_for_backward(terms.eta, terms.k_r)
        return terms.k

    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor, Tensor, Tensor, None, None]:
        """Chain rule with ``dk/d ln a = -eta`` and ``dk/dr|_a = k_r``."""
        eta, k_r = ctx.saved_tensors  # type: ignore[attr-defined]
        # dk/d ln a = -eta (proof of Proposition 3.1); a = e^(ell - w); dk/dr|_a = k_r.
        return -grad * eta, grad * eta, grad * k_r, None, None


def log_kappa(ell: Tensor, w: Tensor, r: Tensor, *, level: int, trunc: float) -> Tensor:
    """``k = ln kappa(e^(ell - w), r)``, differentiable with analytic partials (Prop 8.4).

    Used by the explicit Manning residual (M4, proofs 7.7) and by ``rho`` (3.1).
    """
    out: Tensor = _LogKappa.apply(ell, w, r, level, trunc)
    return out


def t_of_r(r: Tensor) -> Tensor:
    """Shape target ``t = ln(d_max / dbar) = ln(1 + 1/r)`` (Proposition 2.1, section 7.1)."""
    return torch.log1p(1.0 / r)


def dt_dr(r: Tensor) -> Tensor:
    """``dt/dr = -1 / (r (r + 1))`` (section 3.4, derived outputs)."""
    return -1.0 / (r * (r + 1.0))


def r_of_t(t: Tensor) -> Tensor:
    """Inverse shape map ``r = 1 / (e^t - 1)`` for ``t > 0`` (Proposition 6.1)."""
    if not bool((t > 0).all()):
        raise ValueError("t must be > 0 (d_max > dbar; Proposition 2.1)")
    return 1.0 / torch.expm1(t)

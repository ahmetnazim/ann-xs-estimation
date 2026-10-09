"""Tanh-sinh quadrature of F - 1, F_beta and F_r for the power-law section (proofs section 2).

Sources: proofs Proposition 2.2 (``F = P / W``), Proposition 2.5 (tau-form for r < 1 and its
parameter derivatives), Proposition 2.8 (``F_beta``, ``F_r``), the section 3.5 numerical note
(integrate ``F - 1`` directly; ``ln F = log1p(F - 1)``) and Proposition 8.4 (``F_beta`` and
``F_r`` come from the same substitution as ``F``); protocol section 4.2 (P3-12: a
double-exponential rule; tolerance ``tau_Q``). Rule resolution: params ``solver.quad_level`` and
``solver.quad_trunc`` (P6-D3).

Forms. Every integrand is bounded on [0, 1] (Propositions 2.5 and 2.8) and free of cancellation:

* **u-form**, used for ``r >= 1``, with ``s = beta^2 u^(2r-2)``::

      F - 1   = int_0^1 s / (sqrt(1 + s) + 1) du
      F_beta  = beta^-1 int_0^1 s / sqrt(1 + s) du
      F_r     = int_0^1 ln(u) s / sqrt(1 + s) du

* **tau-form**, used for ``r < 1``, with ``p = 2/r - 2 > 0``, ``g = tau^(p/2)`` and
  ``h = sqrt(tau^p + beta^2)``::

      F - 1   = r^-1 int_0^1 beta^2 / (h + g) dtau
      F_beta  = r^-1 int_0^1 beta / h dtau
      F_r     = -(F - 1)/r + beta^2 r^-3 int_0^1 g ln(tau) / (h (h + g)) dtau

  The first line uses ``r^-1 int_0^1 g dtau = 1`` (because ``p/2 + 1 = 1/r``). The last line is
  an algebraic rewrite of Proposition 2.5, ``F_r = -F/r - r^-3 int tau^p ln(tau) / h dtau``: with
  ``int_0^1 g ln(tau) dtau = -r^2``, the two O(1) terms of Proposition 2.5 cancel as
  ``beta -> 0``. Subtracting that integral exactly removes the cancellation::

      1 + r^-2 int tau^p ln(tau)/h = r^-2 int g ln(tau) (g/h - 1)
                                   = -r^-2 int g ln(tau) beta^2 / (h (h + g)).

The rule on [0, 1] is ``u_k = 1 / (1 + exp(-pi sinh t_k))`` with ``t_k = k 2^-level``,
``|t_k| <= trunc``, and weights ``2^-level pi cosh(t_k) u_k (1 - u_k)``. ``ln u_k`` is formed
directly from ``t_k``, so powers ``u^c = exp(c ln u)`` stay accurate at the endpoints. The nodes
are fixed, so the result for one row never depends on the other rows (Proposition 3.4).

Torch-only (float64, CPU), differentiable by autograd. The physics layer uses analytic
partials (:mod:`src.physics.shape`); autograd through this module serves as an independent
cross-check (P6-D2 (b)). Owning phase: 6.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Final

import torch
from torch import Tensor

__all__ = ["MAX_TRUNC", "perimeter_fm1", "perimeter_integrals", "tanh_sinh_rule"]

#: Largest truncation for which every node stays strictly inside (0, 1) in float64
#: (``pi sinh(6) ~ 634 < 745``, the underflow limit of ``exp``).
MAX_TRUNC: Final[float] = 6.0
_MAX_LEVEL: Final[int] = 12


@lru_cache(maxsize=8)
def tanh_sinh_rule(level: int, trunc: float) -> tuple[Tensor, Tensor, Tensor]:
    """Nodes ``u``, ``ln u`` and weights of the tanh-sinh rule on [0, 1] (float64, read-only).

    Parameters
    ----------
    level
        Step ``h = 2^-level`` in the ``t`` variable (``1 <= level <= 12``).
    trunc
        Truncation ``|t| <= trunc`` (``0 < trunc <= MAX_TRUNC``).

    Returns
    -------
    tuple of Tensor
        ``(u, ln_u, weight)``, each of shape ``(N,)`` with ``N = 2 floor(trunc / h) + 1``.
        The cached tensors must never be modified in place.
    """
    if not 1 <= level <= _MAX_LEVEL:
        raise ValueError(f"quad level must be in [1, {_MAX_LEVEL}], got {level}")
    if not 0.0 < trunc <= MAX_TRUNC:
        raise ValueError(f"quad trunc must be in (0, {MAX_TRUNC}], got {trunc}")
    h = 2.0 ** (-level)
    k = math.floor(trunc / h)
    t = torch.arange(-k, k + 1, dtype=torch.float64) * h
    x = math.pi * torch.sinh(t.abs())
    e = torch.exp(-x)  # in (0, 1]
    pos = t >= 0
    u = torch.where(pos, 1.0 / (1.0 + e), e / (1.0 + e))
    ln_u = torch.where(pos, -torch.log1p(e), -x - torch.log1p(e))
    weight = h * math.pi * torch.cosh(t) * e / (1.0 + e) ** 2
    if not bool((u > 0).all()):
        raise FloatingPointError("tanh-sinh node underflow; reduce trunc")
    return u, ln_u, weight


def _check_rows(beta: Tensor, r: Tensor) -> None:
    for name, x in (("beta", beta), ("r", r)):
        if x.dtype != torch.float64 or x.device.type != "cpu":
            raise TypeError(f"{name} must be float64 on CPU (protocol 4.2, O-8)")
        if x.dim() != 1:
            raise ValueError(f"{name} must be 1-D (B,), got shape {tuple(x.shape)}")
        if not bool(torch.isfinite(x).all()):
            raise ValueError(f"non-finite {name}")
    if beta.shape != r.shape:
        raise ValueError(f"beta {tuple(beta.shape)} and r {tuple(r.shape)} differ in shape")
    if not bool((beta > 0).all()):
        raise ValueError("beta must be > 0 (a = exp(ell - w) > 0)")
    if not bool((r > 0).all()):
        raise ValueError("r must be > 0 (Proposition 2.2)")


def _u_form(
    beta: Tensor, r: Tensor, rule: tuple[Tensor, Tensor, Tensor], derivs: bool
) -> tuple[Tensor, ...]:
    _, ln_u, wgt = rule
    log_s = 2.0 * torch.log(beta)[:, None] + (2.0 * r - 2.0)[:, None] * ln_u[None, :]
    s = torch.exp(log_s)
    root = torch.sqrt(1.0 + s)
    fm1 = (s / (root + 1.0) * wgt).sum(dim=1)
    if not derivs:
        return (fm1,)
    q = s / root  # s / sqrt(1 + s)
    f_beta = (q * wgt).sum(dim=1) / beta
    f_r = (ln_u[None, :] * q * wgt).sum(dim=1)
    return fm1, f_beta, f_r


def _tau_form(
    beta: Tensor, r: Tensor, rule: tuple[Tensor, Tensor, Tensor], derivs: bool
) -> tuple[Tensor, ...]:
    _, ln_u, wgt = rule
    half_p = (1.0 / r - 1.0)[:, None]  # p / 2 = 1/r - 1 > 0
    g = torch.exp(half_p * ln_u[None, :])
    b = beta[:, None]
    h = torch.sqrt(g * g + b * b)
    fm1 = (b * b / (h + g) * wgt).sum(dim=1) / r
    if not derivs:
        return (fm1,)
    f_beta = (b / h * wgt).sum(dim=1) / r
    corr = (g * ln_u[None, :] / (h * (h + g)) * wgt).sum(dim=1)
    f_r = -fm1 / r + beta * beta * corr / r**3
    return fm1, f_beta, f_r


def _evaluate(beta: Tensor, r: Tensor, level: int, trunc: float, derivs: bool) -> list[Tensor]:
    _check_rows(beta, r)
    rule = tanh_sinh_rule(level, trunc)
    n = beta.shape[0]
    iu = torch.nonzero(r >= 1.0).flatten()
    it = torch.nonzero(r < 1.0).flatten()
    parts_u = _u_form(beta[iu], r[iu], rule, derivs) if iu.numel() else None
    parts_t = _tau_form(beta[it], r[it], rule, derivs) if it.numel() else None
    outs: list[Tensor] = []
    for j in range(3 if derivs else 1):
        out = beta.new_zeros(n)
        if parts_u is not None:
            out = out.index_put((iu,), parts_u[j])
        if parts_t is not None:
            out = out.index_put((it,), parts_t[j])
        if not bool(torch.isfinite(out).all()):
            raise FloatingPointError("non-finite quadrature result (beta outside float64 range)")
        outs.append(out)
    return outs


def perimeter_fm1(beta: Tensor, r: Tensor, *, level: int, trunc: float) -> Tensor:
    """Return ``F - 1`` only (the Picard loop needs nothing else; Algorithm Phi, forward).

    Identical, row by row, to the first output of :func:`perimeter_integrals`.
    """
    return _evaluate(beta, r, level, trunc, derivs=False)[0]


def perimeter_integrals(
    beta: Tensor, r: Tensor, *, level: int, trunc: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(F - 1, F_beta, F_r)`` row-wise (proofs Propositions 2.2, 2.5, 2.8).

    Parameters
    ----------
    beta, r
        ``(B,)`` float64 CPU tensors with ``beta > 0`` and ``r > 0``.
    level, trunc
        Tanh-sinh resolution (params ``solver.quad_level`` / ``solver.quad_trunc``).

    Returns
    -------
    tuple of Tensor
        ``F - 1 >= 0``, ``F_beta = dF/dbeta`` and ``F_r = dF/dr`` at fixed ``beta``,
        each ``(B,)``. The u-form is used where ``r >= 1``, the tau-form where ``r < 1``.

    Raises
    ------
    TypeError, ValueError
        Wrong dtype/device/shape, non-finite input, ``beta <= 0`` or ``r <= 0``.
    FloatingPointError
        A non-finite result (``beta`` beyond the float64 range of ``beta^2``).
    """
    fm1, f_beta, f_r = _evaluate(beta, r, level, trunc, derivs=True)
    return fm1, f_beta, f_r

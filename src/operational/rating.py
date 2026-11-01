"""Calibrated-roughness Dingman-Manning synthetic rating curve (OPS-1, docs/4 section 2).

A predicted anchor section ``(W0, dbar0, dmax0)`` with Dingman exponent
``r = dbar0 / (dmax0 - dbar0)`` (proofs Proposition 2.1) is extended self-similarly to any thalweg
stage ``y``: ``dmax(y) = y``, ``dbar(y) = r/(r+1) y``, ``W(y) = W0 (y / dmax0)^(1/r)``. With
``lam = ln(y / dmax0)`` the aspect ratio is ``ln a(lam) = ln a0 + lam (1 - 1/r)``.

Manning in log form (proofs Proposition 6.2) with roughness calibrated so the curve passes through
the anchor ``(Q_a, dmax0)`` cancels ``nu`` and ``S`` and leaves (OPS-E1)::

    f(lam) = (1/r + 5/3) lam + (2/3) [k(a(lam), r) - k(a0, r)] - ln(Q / Q_a) = 0,

with ``k = ln kappa = -ln F`` (Proposition 2.2). Proposition OPS-1: ``f'(lam) = 1/r + 5/3 -
(2/3) eta (1 - 1/r) > 0`` with ``eta`` in [0, 1) (Proposition 2.7), so the root is unique and lies
in ``[L / D_hi, L / D_lo]`` (ordered by the sign of ``L``), ``D_lo, D_hi`` = min, max of
``1/r + 5/3`` and ``1 + 5/(3r)``. The solver is a safeguarded Newton iteration inside that bracket.

Torch float64 CPU for the quadrature (:func:`src.physics.shape.section_terms`); NumPy interface
for the evaluation code. Owning phase: OPS-1 (post-release extension).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor

from src.physics.shape import check_rows, section_terms

__all__ = [
    "RatingSolution",
    "derivative_bounds",
    "rating_log_q_ratio",
    "solve_lambda",
    "solve_lambda_np",
    "stage_change",
    "width_log_change",
]

FloatArray = NDArray[np.float64]
FIVE_THIRDS = 5.0 / 3.0
TWO_THIRDS = 2.0 / 3.0


@dataclass(frozen=True)
class RatingSolution:
    """Root of OPS-E1 per row and the number of iterations used by the slowest row."""

    lam: Tensor
    iterations: int


def derivative_bounds(r: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(D_lo, D_hi)``, the bounds of ``f'(lam)`` over ``eta`` in [0, 1) (Prop. OPS-1)."""
    c1 = 1.0 / r + FIVE_THIRDS
    c2 = 1.0 + FIVE_THIRDS / r
    return torch.minimum(c1, c2), torch.maximum(c1, c2)


def _k(ln_a: Tensor, r: Tensor, level: int, trunc: float) -> tuple[Tensor, Tensor]:
    """``k = -ln F`` and ``eta`` at aspect ratio ``a = exp(ln_a)`` (Props 2.2, 2.7)."""
    terms = section_terms(ln_a, torch.zeros_like(ln_a), r, level=level, trunc=trunc)
    return -terms.log_f, terms.eta


def rating_log_q_ratio(
    lam: Tensor, r: Tensor, ln_a0: Tensor, *, level: int, trunc: float
) -> Tensor:
    """Forward map ``ln(Q / Q_a) = (1/r + 5/3) lam + (2/3)(k(lam) - k(0))`` (OPS-E1)."""
    check_rows(lam=lam, r=r, ln_a0=ln_a0)
    with torch.no_grad():
        k0, _ = _k(ln_a0, r, level, trunc)
        k1, _ = _k(ln_a0 + lam * (1.0 - 1.0 / r), r, level, trunc)
        return (1.0 / r + FIVE_THIRDS) * lam + TWO_THIRDS * (k1 - k0)


def solve_lambda(
    ln_q_ratio: Tensor,
    r: Tensor,
    ln_a0: Tensor,
    *,
    level: int,
    trunc: float,
    tol: float,
    max_iter: int,
) -> RatingSolution:
    """Solve OPS-E1 for ``lam = ln(y / dmax0)`` row by row (safeguarded Newton).

    Parameters
    ----------
    ln_q_ratio
        ``L = ln(Q / Q_a)`` per row.
    r
        Dingman exponent (``r > 0``).
    ln_a0
        ``ln(dbar0 / W0)`` of the anchor section.
    level, trunc
        Tanh-sinh resolution (``params.yaml`` ``solver.quad_level`` / ``solver.quad_trunc``).
    tol, max_iter
        Stop when every ``|lam_k - lam_(k-1)| <= tol``; raise after ``max_iter`` iterations.

    Raises
    ------
    ValueError
        Non-positive ``r``, non-positive ``tol``, or shape/dtype errors (``check_rows``).
    RuntimeError
        No convergence within ``max_iter``.
    """
    check_rows(ln_q_ratio=ln_q_ratio, r=r, ln_a0=ln_a0)
    if not bool((r > 0).all()):
        raise ValueError("r must be > 0")
    if not tol > 0 or max_iter < 1:
        raise ValueError("tol must be > 0 and max_iter >= 1")
    with torch.no_grad():
        big_l = ln_q_ratio
        d_lo, d_hi = derivative_bounds(r)
        pos = big_l >= 0
        lo = torch.where(pos, big_l / d_hi, big_l / d_lo)
        hi = torch.where(pos, big_l / d_lo, big_l / d_hi)
        c1 = 1.0 / r + FIVE_THIRDS
        slope = 1.0 - 1.0 / r
        k0, _ = _k(ln_a0, r, level, trunc)
        lam = torch.clamp(big_l / c1, min=lo, max=hi)
        for it in range(1, max_iter + 1):
            k1, eta = _k(ln_a0 + lam * slope, r, level, trunc)
            f = c1 * lam + TWO_THIRDS * (k1 - k0) - big_l
            fp = c1 - TWO_THIRDS * eta * slope
            lo = torch.where(f < 0, lam, lo)
            hi = torch.where(f > 0, lam, hi)
            new = lam - f / fp
            outside = (new < lo) | (new > hi)
            new = torch.where(outside, 0.5 * (lo + hi), new)
            step = (new - lam).abs().max().item() if lam.numel() else 0.0
            lam = new
            if step <= tol:
                return RatingSolution(lam=lam, iterations=it)
    raise RuntimeError(f"OPS-E1 did not converge in {max_iter} iterations (last step {step:.3e})")


def solve_lambda_np(
    ln_q_ratio: ArrayLike,
    r: ArrayLike,
    ln_a0: ArrayLike,
    *,
    level: int,
    trunc: float,
    tol: float,
    max_iter: int,
    chunk_rows: int,
) -> tuple[FloatArray, int]:
    """NumPy wrapper of :func:`solve_lambda` in chunks of ``chunk_rows`` rows.

    Returns ``(lam, max_iterations)``; an empty input returns an empty array and ``0``.
    """
    lq = np.asarray(ln_q_ratio, dtype=np.float64)
    rr = np.asarray(r, dtype=np.float64)
    la = np.asarray(ln_a0, dtype=np.float64)
    if not (lq.ndim == rr.ndim == la.ndim == 1 and lq.shape == rr.shape == la.shape):
        raise ValueError("inputs must be 1-D arrays of equal length")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")
    out = np.empty(lq.shape, dtype=np.float64)
    worst = 0
    for start in range(0, lq.size, chunk_rows):
        sl = slice(start, start + chunk_rows)
        sol = solve_lambda(
            torch.from_numpy(lq[sl].copy()),
            torch.from_numpy(rr[sl].copy()),
            torch.from_numpy(la[sl].copy()),
            level=level,
            trunc=trunc,
            tol=tol,
            max_iter=max_iter,
        )
        out[sl] = sol.lam.numpy()
        worst = max(worst, sol.iterations)
    return out, worst


def stage_change(dmax0: ArrayLike, lam: ArrayLike) -> FloatArray:
    """``Delta h = dmax0 (e^lam - 1)`` in metres (thalweg = water-surface stage change)."""
    d = np.asarray(dmax0, dtype=np.float64)
    out: FloatArray = d * np.expm1(np.asarray(lam, dtype=np.float64))
    return out


def width_log_change(lam: ArrayLike, r: ArrayLike) -> FloatArray:
    """``Delta ln W = lam / r`` (self-similar power-law section)."""
    out: FloatArray = np.asarray(lam, dtype=np.float64) / np.asarray(r, dtype=np.float64)
    return out

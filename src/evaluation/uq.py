"""UQ diagnostics: mixture intervals, coverage, mean log-width, PIT and closed-form CRPS.

Sources: proofs section 9.3 (exact mixture quantile ``Q_p``: unique root of
``Psi(x) = (1/M) sum_m Phi_N((x - mu_m)/sigma_m) = p`` in the bracket
``[min_m(mu_m + z_p sigma_m), max_m(mu_m + z_p sigma_m)]``; interval
``[Q_{alpha/2}, Q_{1-alpha/2}]``)
and section 9.4 (coverage ``C_alpha = sum_i w_i 1{L_i <= ln y_i <= U_i}``; mean log-width
``W_alpha = sum_i w_i (U_i - L_i)``; PIT ``u_i = Psi_i(ln y_i)``; Gaussian-mixture CRPS closed form
with the folded-normal mean ``A(mu, s^2) = 2 s phi(mu/s) + mu (2 Phi(mu/s) - 1)``; ``M = 1``
reduces to ``sigma [z (2 Phi(z) - 1) + 2 phi(z) - pi^(-1/2)]``); protocol section 4.6 (alpha 0.10,
acceptance band [0.85, 0.95], a quality target and not a hypothesis).

Arrays: member means and scales are ``(M, n)`` in log space; observations ``(n,)``. This module is
the NumPy/SciPy counterpart of :mod:`src.models.ensemble` so that stored predictions can be scored
without importing torch. Torch-free and xgboost-free. Owning phase: 7 (P7-D1 (a)).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri

__all__ = [
    "central_interval",
    "coverage",
    "coverage_in_band",
    "crps_gaussian",
    "crps_mixture",
    "folded_normal_mean",
    "mean_width",
    "mixture_cdf",
    "mixture_quantile",
    "pit",
]

FloatArray = NDArray[np.float64]

_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _phi(z: FloatArray) -> FloatArray:
    out: FloatArray = _INV_SQRT_2PI * np.exp(-0.5 * z * z)
    return out


def _members(mu: ArrayLike, sigma: ArrayLike) -> tuple[FloatArray, FloatArray]:
    m = np.asarray(mu, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] < 1 or m.shape[1] < 1 or s.shape != m.shape:
        raise ValueError(f"mu and sigma must be equal (M, n) arrays, got {m.shape}, {s.shape}")
    if not (bool(np.isfinite(m).all()) and bool(np.isfinite(s).all())):
        raise ValueError("non-finite member means or scales")
    if not bool((s > 0).all()):
        raise ValueError("member scales must be > 0 (proofs 8.5)")
    return m, s


def _obs(y: ArrayLike, n: int) -> FloatArray:
    a = np.asarray(y, dtype=np.float64)
    if a.shape != (n,) or not bool(np.isfinite(a).all()):
        raise ValueError(f"observations must be a finite ({n},) array, got {a.shape}")
    return a


def _w(weights: ArrayLike | None, n: int) -> FloatArray:
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,) or not bool(np.isfinite(w).all()) or bool((w < 0).any()):
        raise ValueError("weights must be a finite non-negative (n,) array")
    total = float(w.sum())
    if not total > 0.0:
        raise ValueError("weights must have a positive sum")
    return w / total


def mixture_cdf(x: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> FloatArray:
    """``Psi(x) = (1/M) sum_m Phi_N((x - mu_m)/sigma_m)`` per column (proofs 9.3)."""
    m, s = _members(mu, sigma)
    xv = _obs(x, m.shape[1])
    out: FloatArray = ndtr((xv[None, :] - m) / s).mean(axis=0)
    return out


def mixture_quantile(
    p: float, mu: ArrayLike, sigma: ArrayLike, *, tol: float = 1e-12, max_iter: int = 200
) -> FloatArray:
    """Exact mixture quantile ``Q_p`` by bisection on the proofs 9.3 bracket.

    Stops when every bracket is narrower than ``tol (1 + |x|)``; raises if ``max_iter`` is
    reached first.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")
    m, s = _members(mu, sigma)
    ends = m + float(ndtri(p)) * s
    lo, hi = ends.min(axis=0), ends.max(axis=0)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if bool(((hi - lo) <= tol * (1.0 + np.abs(mid))).all()):
            out: FloatArray = mid
            return out
        below = mixture_cdf(mid, m, s) < p
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    raise RuntimeError(f"mixture quantile bisection did not converge in {max_iter} steps")


def central_interval(
    mu: ArrayLike, sigma: ArrayLike, alpha: float
) -> tuple[FloatArray, FloatArray]:
    """``[Q_{alpha/2}, Q_{1-alpha/2}]`` in log space (proofs 9.3; alpha = 0.10, protocol 4.6)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return mixture_quantile(alpha / 2.0, mu, sigma), mixture_quantile(1.0 - alpha / 2.0, mu, sigma)


def coverage(
    lower: ArrayLike, upper: ArrayLike, y: ArrayLike, weights: ArrayLike | None = None
) -> float:
    """``C_alpha = sum_i w_i 1{L_i <= ln y_i <= U_i}`` (proofs 9.4; closed interval)."""
    lo = np.asarray(lower, dtype=np.float64)
    n = lo.size
    hi, yv = _obs(upper, n), _obs(y, n)
    lo = _obs(lo, n)
    if bool((lo > hi).any()):
        raise ValueError("lower bound above upper bound")
    w = _w(weights, n)
    return float(np.sum(w * ((lo <= yv) & (yv <= hi))))


def mean_width(lower: ArrayLike, upper: ArrayLike, weights: ArrayLike | None = None) -> float:
    """``W_alpha = sum_i w_i (U_i - L_i)``; ``exp(W_alpha)`` is the multiplicative factor (9.4)."""
    lo = np.asarray(lower, dtype=np.float64)
    n = lo.size
    lo, hi = _obs(lo, n), _obs(upper, n)
    if bool((lo > hi).any()):
        raise ValueError("lower bound above upper bound")
    return float(np.sum(_w(weights, n) * (hi - lo)))


def coverage_in_band(c: float, band: tuple[float, float]) -> bool:
    """Whether coverage lies in the acceptance band ``[0.85, 0.95]`` (protocol 4.6)."""
    lo, hi = band
    return lo <= c <= hi


def pit(y: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> FloatArray:
    """PIT ``u_i = Psi_i(ln y_i)``; uniform under calibration (proofs 9.4)."""
    return mixture_cdf(y, mu, sigma)


def folded_normal_mean(mu: ArrayLike, s2: ArrayLike) -> FloatArray:
    """``A(mu, s^2) = E|Y| = 2 s phi(mu/s) + mu (2 Phi(mu/s) - 1)``, ``Y ~ N(mu, s^2)`` (9.4)."""
    m = np.asarray(mu, dtype=np.float64)
    v = np.asarray(s2, dtype=np.float64)
    if not bool((v > 0).all()):
        raise ValueError("variance must be > 0")
    s = np.sqrt(v)
    z = m / s
    out: FloatArray = 2.0 * s * _phi(z) + m * (2.0 * ndtr(z) - 1.0)
    return out


def crps_gaussian(y: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> FloatArray:
    """``sigma [z (2 Phi(z) - 1) + 2 phi(z) - pi^(-1/2)]``, ``z = (y - mu)/sigma`` (9.4, M = 1)."""
    m, s = _members(
        np.atleast_1d(np.asarray(mu, dtype=np.float64))[None, :],
        np.atleast_1d(np.asarray(sigma, dtype=np.float64))[None, :],
    )
    yv = _obs(y, m.shape[1])
    z = (yv - m[0]) / s[0]
    out: FloatArray = s[0] * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * _phi(z) - _INV_SQRT_PI)
    return out


def crps_mixture(y: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> FloatArray:
    """Closed-form CRPS of the Gaussian mixture in log space (proofs 9.4).

    ``(1/M) sum_m A(x - mu_m, sigma_m^2) - (1/(2 M^2)) sum_{m,m'} A(mu_m - mu_m', sigma_m^2 +
    sigma_m'^2)``.
    """
    m, s = _members(mu, sigma)
    yv = _obs(y, m.shape[1])
    k = m.shape[0]
    v = s * s
    term1 = folded_normal_mean(yv[None, :] - m, v).mean(axis=0)
    diff = m[:, None, :] - m[None, :, :]
    var = v[:, None, :] + v[None, :, :]
    term2 = folded_normal_mean(diff, var).sum(axis=(0, 1)) / (2.0 * k * k)
    out: FloatArray = term1 - term2
    return out

"""Deep-ensemble mixture: moments, exact mixture quantiles, point prediction and n*.

Sources: proofs section 9.2 (``p* = (1/M) sum_m N(mu_m, sigma_m^2)``; ``mu* = mean_m mu_m``;
``sigma*^2 = sigma_ale^2 + sigma_epi^2`` by the law of total variance), section 9.3 (exact
mixture quantile ``Q_p``: the unique root of ``Psi(x) = (1/M) sum_m Phi_N((x - mu_m)/sigma_m)
= p`` in the bracket ``[min_m(mu_m + z_p sigma_m), max_m(mu_m + z_p sigma_m)]``; point
prediction ``exp(mu*)``), Proposition 9.2 (``n* = Phi^-1(mu*)`` with full kappa, via
Proposition 6.2); protocol section 4.2 (P3-13: ``M = 10``; P3-14: ``exp(mu*)`` everywhere).

Squared-loss rungs (M3, M6-NN, P6-Q3 (a)) have no sigma: their mixture has
``sigma_ale = 0`` and only the epistemic spread, and quantiles are not defined for them here.
M3's point prediction is Phi at the mean member ``w`` (P6-Q4 (a);
:meth:`src.models.m3_physics.M3Physics.physics`). Calibration and coverage diagnostics are
Phase 7 (P6-D1).

Torch-only, float64, CPU. Owning phase: 6.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from src.models.heads import ModelOutput
from src.physics.manning_layer import PhiConfig, phi_inverse_nu
from src.physics.shape import r_of_t

__all__ = [
    "Mixture",
    "ensemble_roughness",
    "mixture_cdf",
    "mixture_moments",
    "mixture_quantile",
    "point_prediction",
    "stack_members",
]


def _check(mu: Tensor, sigma: Tensor | None) -> None:
    if mu.dtype != torch.float64 or mu.dim() != 3 or int(mu.shape[0]) < 1:
        raise ValueError("mu must be float64 (M, B, K) with M >= 1")
    if not bool(torch.isfinite(mu).all()):
        raise ValueError("non-finite member means")
    if sigma is not None:
        if sigma.shape != mu.shape or sigma.dtype != torch.float64:
            raise ValueError("sigma must match mu (float64, (M, B, K))")
        if not bool(torch.isfinite(sigma).all()) or not bool((sigma > 0).all()):
            raise ValueError("member sigmas must be finite and > 0 (proofs 8.5)")


def stack_members(
    outputs: Sequence[ModelOutput], *, native: bool = False
) -> tuple[Tensor, Tensor | None]:
    """Stack member outputs to ``(M, B, K)``; ``native=True`` uses M2's native columns.

    All members must agree on whether they carry sigma (one model, one loss family).
    """
    if not outputs:
        raise ValueError("need at least one member")
    if native:
        mus = [o.extras["mu_native"] for o in outputs]
        sigmas: list[Tensor | None] = [o.extras["sigma_native"] for o in outputs]
    else:
        mus = [o.mu for o in outputs]
        sigmas = [o.sigma for o in outputs]
    has = {s is not None for s in sigmas}
    if len(has) != 1:
        raise ValueError("members disagree on sigma")
    mu = torch.stack([m.detach() for m in mus])
    sigma = None if None in sigmas else torch.stack([s.detach() for s in sigmas if s is not None])
    _check(mu, sigma)
    return mu, sigma


@dataclass(frozen=True)
class Mixture:
    """Mixture moments (proofs 9.2), each ``(B, K)``."""

    mu_star: Tensor
    var_ale: Tensor
    var_epi: Tensor

    @property
    def sigma_star(self) -> Tensor:
        """``sigma* = sqrt(sigma_ale^2 + sigma_epi^2)`` (law of total variance)."""
        return torch.sqrt(self.var_ale + self.var_epi)


def mixture_moments(mu: Tensor, sigma: Tensor | None) -> Mixture:
    """``mu* = mean mu_m``, ``sigma_ale^2 = mean sigma_m^2``, ``sigma_epi^2 = mean (mu_m - mu*)^2``.

    Proofs 9.2 (population form, divisor ``M``). ``sigma=None`` gives ``sigma_ale^2 = 0``.
    """
    _check(mu, sigma)
    mu_star = mu.mean(dim=0)
    var_epi = ((mu - mu_star) ** 2).mean(dim=0)
    var_ale = torch.zeros_like(mu_star) if sigma is None else (sigma**2).mean(dim=0)
    return Mixture(mu_star=mu_star, var_ale=var_ale, var_epi=var_epi)


def mixture_cdf(x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """``Psi(x) = (1/M) sum_m Phi_N((x - mu_m)/sigma_m)``, ``x`` of shape ``(B, K)`` (9.3)."""
    _check(mu, sigma)
    if x.shape != mu.shape[1:]:
        raise ValueError(f"x {tuple(x.shape)} must be (B, K) = {tuple(mu.shape[1:])}")
    out: Tensor = torch.special.ndtr((x[None] - mu) / sigma).mean(dim=0)
    return out


def mixture_quantile(
    p: float, mu: Tensor, sigma: Tensor, *, tol: float = 1e-12, max_iter: int = 200
) -> Tensor:
    """Exact mixture quantile ``Q_p`` by bisection on the bracket of proofs 9.3.

    ``Psi`` is continuous and strictly increasing, so the root is unique and lies in
    ``[min_m(mu_m + z_p sigma_m), max_m(mu_m + z_p sigma_m)]``. Stops when every bracket is
    narrower than ``tol (1 + |x|)``; raises if ``max_iter`` is reached first.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")
    _check(mu, sigma)
    z_p = float(torch.special.ndtri(torch.tensor(p, dtype=torch.float64)))
    ends = mu + z_p * sigma
    lo, hi = ends.min(dim=0).values, ends.max(dim=0).values
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if bool(((hi - lo) <= tol * (1.0 + mid.abs())).all()):
            return mid
        below = mixture_cdf(mid, mu, sigma) < p
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    raise RuntimeError(f"mixture quantile bisection did not converge in {max_iter} steps")


def point_prediction(mu_star: Tensor) -> Tensor:
    """``y_hat = exp(mu*)`` (proofs 9.3; protocol 4.2, P3-14)."""
    if not bool(torch.isfinite(mu_star).all()):
        raise ValueError("non-finite mu*")
    return torch.exp(mu_star)


def ensemble_roughness(mu_star: Tensor, phys: Tensor, cfg: PhiConfig) -> Tensor:
    """``nu* = Phi^-1(mu*)`` (Proposition 9.2) with ``r* = r(mu*_t)`` (Proposition 6.1).

    ``mu_star`` is ``(B, 4)`` canonical ``[w, ell, t, ln A]``; ``phys = [ln Q, ln S]``. Raises if
    ``mu*_t <= 0`` (not possible for M3-M6, whose member ``t`` lie in the r band).
    """
    if mu_star.dim() != 2 or int(mu_star.shape[1]) != 4:
        raise ValueError("mu_star must be (B, 4) canonical columns")
    r_star = r_of_t(mu_star[:, 2])
    return phi_inverse_nu(mu_star[:, 1], mu_star[:, 0], r_star, phys[:, 0], phys[:, 1], cfg)

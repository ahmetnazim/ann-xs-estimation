"""Data, NLL and beta-NLL losses with the sigma floor; prior and residual penalties.

Sources: proofs section 7.0 (site weights ``omega``, log residuals ``e = yhat - y``, every term
dimensionless), section 7.2 (``w_A`` term; ``w_A = 0`` primary, 1 sensitivity), section 7.4
(``L_data = sum_i omega_i sum_k w_k e_ik^2``), section 7.5 (A3-SP prior ``lambda_p sum omega
(nu - nu_S)^2``; A3-SR ridge ``lambda_Delta sum omega Delta^2``), section 7.7 / 7.7.1 (M4
penalty ``lambda_M sum omega R_M^2`` with ``lambda_M(e) = min(lambda0 gamma^e, lambda_max)``,
Proposition 7.9), section 8.5 (NLL, sigma floor, beta-NLL ``sg(sigma^(2 beta_N))``) and
section 9.1 (``L_NLL = sum_i omega_i sum_k w_k [e^2 / (2 sigma^2) + ln sigma]``; derived
``ln A`` with ``sigma_A^2 = sigma_W^2 + sigma_dbar^2`` when there is no correlation head);
protocol section 4.2 (P3-06 weights, P3-13: ``sigma_min = 0.01``, ``beta_N = 0.5``, 20-epoch
MSE warm start); P6-Q3 (a).

Every loss is a weighted **sum** over the rows given, ``sum_i omega_i (...)``; with
``omega = 1/|J_tr|`` (ADL-077) the sum over one epoch's mini-batches is the full objective.
Weight decay is not a loss term here: AdamW applies it (P6-D5 (a)). Inputs are validated and
never clipped; a non-positive ``sigma`` raises.

Torch-only, float64, CPU. Owning phase: 6.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

from src.config import LossWeightParams

__all__ = [
    "LossMode",
    "area_moments",
    "data_term",
    "gaussian_nll",
    "loss_mode",
    "m4_cap_epoch",
    "m4_lambda",
    "residual_penalty",
    "ridge_penalty",
    "squared_loss",
    "strickler_prior_penalty",
    "target_weights",
    "weighted_mean_square",
]

#: ``mse`` = L_data (warm start, squared-loss models); ``nll`` / ``beta_nll`` = proofs 9.1 / 8.5.
LossMode = Literal["mse", "nll", "beta_nll"]


def _check_float(name: str, x: Tensor, dims: int) -> None:
    if x.dtype != torch.float64 or x.device.type != "cpu":
        raise TypeError(f"{name} must be float64 on CPU (protocol 4.2, O-8)")
    if x.dim() != dims:
        raise ValueError(f"{name} must be {dims}-D, got shape {tuple(x.shape)}")
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"non-finite values in {name}")


def _check_columns(mu: Tensor, y: Tensor, omega: Tensor, weights: Tensor) -> None:
    _check_float("mu", mu, 2)
    _check_float("y", y, 2)
    _check_float("omega", omega, 1)
    _check_float("weights", weights, 1)
    b, k = int(mu.shape[0]), int(mu.shape[1])
    if tuple(y.shape) != (b, k) or tuple(omega.shape) != (b,) or tuple(weights.shape) != (k,):
        raise ValueError(
            f"shape mismatch: mu {tuple(mu.shape)}, y {tuple(y.shape)}, "
            f"omega {tuple(omega.shape)}, weights {tuple(weights.shape)}"
        )
    if bool((omega < 0).any()) or bool((weights < 0).any()):
        raise ValueError("omega and weights must be non-negative (proofs 7.0)")


def target_weights(params: LossWeightParams, *, a2: bool = False) -> Tensor:
    """``(w_W, w_dbar, w_t, w_A)`` for the columns ``[ln W, ln dbar, t, ln A]`` (P3-06).

    ``a2=True`` is ablation A2 (``w_t = 0``, proofs 7.10). ``w_A = 1`` is the sensitivity run
    (params ``model.loss_weights``); with ``w_A = 0`` the area column carries no weight.
    """
    w_t = 0.0 if a2 else params.w_t
    return torch.tensor([params.w_w, params.w_dbar, w_t, params.w_a], dtype=torch.float64)


def squared_loss(mu: Tensor, y: Tensor, omega: Tensor, weights: Tensor) -> Tensor:
    """``L_data = sum_i omega_i sum_k w_k (mu_ik - y_ik)^2`` (proofs 7.4; 7.8, 7.9 for M6, M3)."""
    _check_columns(mu, y, omega, weights)
    return (omega[:, None] * weights[None, :] * (mu - y) ** 2).sum()


def gaussian_nll(
    mu: Tensor, sigma: Tensor, y: Tensor, omega: Tensor, weights: Tensor, *, beta_n: float = 0.0
) -> Tensor:
    """Heteroscedastic Gaussian NLL (proofs 9.1); beta-NLL when ``beta_n > 0`` (proofs 8.5).

    Per point ``l = e^2 / (2 sigma^2) + ln sigma``; beta-NLL multiplies it by the stop-gradient
    factor ``sg(sigma^(2 beta_N))`` [Seitzer et al. 2022], so the mean gradient is
    ``-e sigma^(2 beta_N - 2)``. ``beta_n = 0`` is the proper likelihood. The additive constant
    ``ln sqrt(2 pi)`` is omitted, as in proofs 8.5.
    """
    _check_columns(mu, y, omega, weights)
    _check_float("sigma", sigma, 2)
    if tuple(sigma.shape) != tuple(mu.shape):
        raise ValueError(f"sigma {tuple(sigma.shape)} must match mu {tuple(mu.shape)}")
    if not bool((sigma > 0).all()):
        raise ValueError("sigma must be > 0 (use sigma_min + softplus, proofs 8.5)")
    if not 0.0 <= beta_n <= 1.0:
        raise ValueError("beta_n must lie in [0, 1] (proofs 8.5)")
    point = (mu - y) ** 2 / (2.0 * sigma**2) + torch.log(sigma)
    if beta_n > 0.0:
        point = point * sigma.detach() ** (2.0 * beta_n)
    return (omega[:, None] * weights[None, :] * point).sum()


def loss_mode(epoch: int, warm_start_epochs: int, beta_n: float) -> LossMode:
    """MSE for ``epoch < warm_start_epochs`` (P3-13), then beta-NLL (``beta_n > 0``) or NLL."""
    if epoch < 0:
        raise ValueError("epoch must be >= 0")
    if epoch < warm_start_epochs:
        return "mse"
    return "beta_nll" if beta_n > 0.0 else "nll"


def data_term(
    mode: LossMode,
    mu: Tensor,
    y: Tensor,
    omega: Tensor,
    weights: Tensor,
    *,
    sigma: Tensor | None = None,
    beta_n: float = 0.0,
) -> Tensor:
    """Dispatch: ``mse`` -> :func:`squared_loss`; ``nll`` / ``beta_nll`` -> :func:`gaussian_nll`.

    During the MSE warm start the sigma heads receive no gradient (proofs 8.5, mitigation 3).
    """
    if mode == "mse":
        return squared_loss(mu, y, omega, weights)
    if sigma is None:
        raise ValueError(f"mode {mode!r} needs sigma")
    return gaussian_nll(mu, sigma, y, omega, weights, beta_n=beta_n if mode == "beta_nll" else 0.0)


def area_moments(
    mu_w: Tensor, mu_ell: Tensor, sigma_w: Tensor, sigma_dbar: Tensor
) -> tuple[Tensor, Tensor]:
    """Return the derived ``ln A`` moments ``(mu_W + mu_dbar, sqrt(sigma_W^2 + sigma_dbar^2))``.

    Proofs 9.1 (derived targets; no correlation head, ``varrho = 0``, P3-13). Used for the
    ``w_A`` column of the likelihood (Batch B, Q-B1); ``mu_A`` is exact by Theorem 4.1.
    """
    return mu_w + mu_ell, torch.sqrt(sigma_w**2 + sigma_dbar**2)


def weighted_mean_square(r: Tensor, omega: Tensor) -> Tensor:
    """``sum_i omega_i r_i^2`` for ``(B,)`` tensors (proofs 7.7: ``P``; 7.5 penalties)."""
    _check_float("r", r, 1)
    _check_float("omega", omega, 1)
    if r.shape != omega.shape:
        raise ValueError(f"r {tuple(r.shape)} and omega {tuple(omega.shape)} differ")
    if bool((omega < 0).any()):
        raise ValueError("omega must be non-negative")
    return (omega * r**2).sum()


def strickler_prior_penalty(nu: Tensor, nu_s: Tensor, omega: Tensor, lambda_p: float) -> Tensor:
    """A3-SP: ``lambda_p sum_i omega_i (nu_i - nu_S,i)^2`` (proofs 7.5 a; ``lambda_p = 2``)."""
    if not lambda_p >= 0:
        raise ValueError("lambda_p must be >= 0")
    return lambda_p * weighted_mean_square(nu - nu_s, omega)


def ridge_penalty(delta: Tensor, omega: Tensor, lambda_delta: float) -> Tensor:
    """A3-SR: ``lambda_Delta sum_i omega_i Delta_i^2`` (proofs 7.5 b; ``lambda_Delta = 2.865``)."""
    if not lambda_delta >= 0:
        raise ValueError("lambda_delta must be >= 0")
    return lambda_delta * weighted_mean_square(delta, omega)


def residual_penalty(r_m: Tensor, omega: Tensor, lambda_m: float) -> Tensor:
    """M4: ``lambda_M sum_i omega_i R_M,i^2`` (proofs 7.7; ``R_M`` = eq. 3.1 with a free ell)."""
    if not lambda_m >= 0:
        raise ValueError("lambda_m must be >= 0")
    return lambda_m * weighted_mean_square(r_m, omega)


def m4_lambda(epoch: int, lambda0: float, gamma: float, lambda_max: float) -> float:
    """Capped schedule ``lambda_M(e) = min(lambda0 gamma^e, lambda_max)`` (Proposition 7.9)."""
    if epoch < 0 or not (lambda0 > 0 and gamma > 1 and lambda_max >= lambda0):
        raise ValueError("need epoch >= 0, lambda0 > 0, gamma > 1, lambda_max >= lambda0")
    log_lam = math.log(lambda0) + epoch * math.log(gamma)
    return lambda_max if log_lam >= math.log(lambda_max) else math.exp(log_lam)


def m4_cap_epoch(lambda0: float, gamma: float, lambda_max: float) -> int:
    """``e_cap = ceil(log_gamma(lambda_max / lambda0))`` (Proposition 7.9; E12: 73 at 1e3)."""
    if not (lambda0 > 0 and gamma > 1 and lambda_max >= lambda0):
        raise ValueError("need lambda0 > 0, gamma > 1, lambda_max >= lambda0")
    return math.ceil(math.log(lambda_max / lambda0) / math.log(gamma))

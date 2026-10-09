"""Smoke tests for src/models/losses.py (Phase 6, Batch B). Synthetic; no data."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from src.config import load_settings
from src.models import losses as ls
from src.models.heads import sigma_from_raw

pytestmark = pytest.mark.unit

PROTOCOL = load_settings().protocol
UQ = PROTOCOL.uq
WEIGHTS = PROTOCOL.model.loss_weights
SCHED = PROTOCOL.model.m4_schedule
PRIORS = PROTOCOL.priors


def batch(b: int = 12, k: int = 4, seed: int = 0) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(b, k, dtype=torch.float64, generator=g)
    y = torch.randn(b, k, dtype=torch.float64, generator=g)
    sigma = 0.05 + torch.rand(b, k, dtype=torch.float64, generator=g)
    omega = torch.full((b,), 1.0 / 40.0, dtype=torch.float64)
    return mu, y, sigma, omega


def test_target_weights_and_ablations() -> None:
    assert ls.target_weights(WEIGHTS).tolist() == [1.0, 1.0, 1.0, 0.0]
    assert ls.target_weights(WEIGHTS, a2=True).tolist() == [1.0, 1.0, 0.0, 0.0]
    sens = WEIGHTS.model_copy(update={"w_a": 1.0})
    assert ls.target_weights(sens).tolist() == [1.0, 1.0, 1.0, 1.0]


def test_squared_loss_matches_definition() -> None:
    mu, y, _, omega = batch()
    w = torch.tensor([1.0, 2.0, 0.5, 0.0], dtype=torch.float64)
    want = sum(
        float(omega[i]) * sum(float(w[k]) * float(mu[i, k] - y[i, k]) ** 2 for k in range(4))
        for i in range(12)
    )
    assert math.isclose(float(ls.squared_loss(mu, y, omega, w)), want, rel_tol=1e-13)
    omega0 = omega.clone()
    omega0[3] = 0.0
    y_bad = y.clone()
    y_bad[3] += 100.0
    assert torch.equal(ls.squared_loss(mu, y_bad, omega0, w), ls.squared_loss(mu, y, omega0, w))


def test_nll_and_beta_nll_values_and_gradients() -> None:
    mu, y, sigma, omega = batch(seed=1)
    w = torch.ones(4, dtype=torch.float64)
    point = (mu - y) ** 2 / (2 * sigma**2) + torch.log(sigma)
    nll = ls.gaussian_nll(mu, sigma, y, omega, w)
    assert math.isclose(float(nll), float((omega[:, None] * point).sum()), rel_tol=1e-13)
    beta = UQ.beta_nll
    mu_l, s_l = mu.clone().requires_grad_(), sigma.clone().requires_grad_()
    bn = ls.gaussian_nll(mu_l, s_l, y, omega, w, beta_n=beta)
    factor = sigma ** (2 * beta)
    assert math.isclose(
        float(bn.detach()), float((omega[:, None] * factor * point).sum()), rel_tol=1e-13
    )
    g_mu, g_s = torch.autograd.grad(bn, (mu_l, s_l))
    e = mu - y
    assert torch.allclose(g_mu, omega[:, None] * e * sigma ** (2 * beta - 2), rtol=1e-12, atol=0)
    want_s = omega[:, None] * factor * (-(e**2) / sigma**3 + 1 / sigma)
    assert torch.allclose(g_s, want_s, rtol=1e-12, atol=0)


def test_prop_8_5_bounds_with_sigma_floor() -> None:
    g = torch.Generator().manual_seed(2)
    z = (6 * torch.randn(400, 1, dtype=torch.float64, generator=g)).requires_grad_()
    mu = torch.randn(400, 1, dtype=torch.float64, generator=g).requires_grad_()
    y = torch.randn(400, 1, dtype=torch.float64, generator=g)
    one = torch.ones(400, dtype=torch.float64)
    sigma = sigma_from_raw(z, UQ.sigma_min)
    total = ls.gaussian_nll(mu, sigma, y, one, torch.ones(1, dtype=torch.float64))
    g_mu, g_z = torch.autograd.grad(total, (mu, z))
    e = (mu - y).detach().abs()
    s_min = UQ.sigma_min
    assert bool((g_mu.abs() <= e / s_min**2 + 1e-12).all())
    assert bool((g_z.abs() <= 1 / s_min + e**2 / s_min**3 + 1e-12).all())
    point = (mu - y).detach() ** 2 / (2 * sigma.detach() ** 2) + torch.log(sigma.detach())
    assert bool((point >= math.log(s_min)).all())


def test_loss_mode_warm_start() -> None:
    warm = UQ.warm_start_epochs
    assert [ls.loss_mode(e, warm, UQ.beta_nll) for e in (0, warm - 1)] == ["mse", "mse"]
    assert ls.loss_mode(warm, warm, UQ.beta_nll) == "beta_nll"
    assert ls.loss_mode(warm, warm, 0.0) == "nll"
    with pytest.raises(ValueError, match="epoch"):
        ls.loss_mode(-1, warm, UQ.beta_nll)


def test_data_term_dispatch_and_warm_start_isolates_sigma() -> None:
    mu, y, sigma, omega = batch(seed=3)
    w = ls.target_weights(WEIGHTS)
    s_l = sigma.clone().requires_grad_()
    mse = ls.data_term("mse", mu, y, omega, w, sigma=s_l)
    assert torch.equal(mse, ls.squared_loss(mu, y, omega, w))
    assert not mse.requires_grad
    nll = ls.data_term("nll", mu, y, omega, w, sigma=s_l, beta_n=0.5)
    assert torch.equal(nll, ls.gaussian_nll(mu, s_l, y, omega, w))
    bn = ls.data_term("beta_nll", mu, y, omega, w, sigma=s_l, beta_n=0.5)
    assert torch.equal(bn, ls.gaussian_nll(mu, s_l, y, omega, w, beta_n=0.5))
    (g,) = torch.autograd.grad(bn, s_l)
    assert bool((g[:, 3] == 0).all())  # w_A = 0: the area column carries no gradient
    with pytest.raises(ValueError, match="needs sigma"):
        ls.data_term("nll", mu, y, omega, w)


def test_area_moments() -> None:
    mu_w, mu_l, s_w, s_d = (torch.tensor([v], dtype=torch.float64) for v in (1.0, -0.5, 0.3, 0.4))
    mu_a, s_a = ls.area_moments(mu_w, mu_l, s_w, s_d)
    assert float(mu_a[0]) == 0.5
    assert math.isclose(float(s_a[0]), 0.5, rel_tol=1e-15)


def test_losses_are_unit_invariant_prop_8_1() -> None:
    mu, y, sigma, omega = batch(seed=4)
    w = torch.ones(4, dtype=torch.float64)
    shift = torch.tensor([2.0, -1.0, 0.0, 3.0], dtype=torch.float64)  # 0 for the t column
    for f in (
        lambda m, t: ls.squared_loss(m, t, omega, w),
        lambda m, t: ls.gaussian_nll(m, sigma, t, omega, w, beta_n=0.5),
    ):
        assert torch.allclose(f(mu + shift, y + shift), f(mu, y), rtol=1e-14, atol=0)


def test_penalties() -> None:
    g = torch.Generator().manual_seed(5)
    a, b = torch.randn(2, 20, dtype=torch.float64, generator=g)
    omega = torch.full((20,), 0.05, dtype=torch.float64)
    wms = float((omega * a**2).sum())
    assert math.isclose(float(ls.weighted_mean_square(a, omega)), wms, rel_tol=1e-14)
    lam_p = PRIORS.strickler_prior.lambda_p
    sp = ls.strickler_prior_penalty(a, b, omega, lam_p)
    assert math.isclose(float(sp), lam_p * float((omega * (a - b) ** 2).sum()), rel_tol=1e-14)
    lam_d = PRIORS.strickler_residual.lambda_delta
    assert math.isclose(float(ls.ridge_penalty(a, omega, lam_d)), lam_d * wms, rel_tol=1e-14)
    assert math.isclose(float(ls.residual_penalty(a, omega, 100.0)), 100.0 * wms, rel_tol=1e-14)
    with pytest.raises(ValueError, match="lambda_m"):
        ls.residual_penalty(a, omega, -1.0)


def test_m4_capped_schedule_e12() -> None:
    lam0, gamma = SCHED.lambda0, SCHED.gamma
    caps = [ls.m4_cap_epoch(lam0, gamma, lm) for lm in PROTOCOL.model.inner_grid.m4_lambda_max]
    assert caps == [25, 49, 73]  # E12 gives 73 for lambda_max = 1e3
    assert ls.m4_lambda(0, lam0, gamma, 1e3) == 1.0
    assert math.isclose(ls.m4_lambda(10, lam0, gamma, 1e3), 1.1**10, rel_tol=1e-13)
    assert ls.m4_lambda(72, lam0, gamma, 1e3) < 1e3
    assert ls.m4_lambda(73, lam0, gamma, 1e3) == 1e3
    assert ls.m4_lambda(500, lam0, gamma, 1e3) == 1e3
    assert round(gamma**100) == 13781  # E12: uncapped weight at e = 100
    with pytest.raises(ValueError, match="gamma"):
        ls.m4_lambda(1, lam0, 1.0, 1e3)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"mu": torch.zeros(12, 4, dtype=torch.float32)}, TypeError),
        ({"y": torch.zeros(12, 3, dtype=torch.float64)}, ValueError),
        ({"omega": -torch.ones(12, dtype=torch.float64)}, ValueError),
        ({"sigma": torch.zeros(12, 4, dtype=torch.float64)}, ValueError),
        ({"y": torch.full((12, 4), float("nan"), dtype=torch.float64)}, ValueError),
    ],
)
def test_loss_input_validation(change: dict[str, Tensor], error: type) -> None:
    mu, y, sigma, omega = batch()
    args = {"mu": mu, "y": y, "sigma": sigma, "omega": omega} | change
    w = torch.ones(4, dtype=torch.float64)
    with pytest.raises(error):
        ls.gaussian_nll(args["mu"], args["sigma"], args["y"], args["omega"], w)

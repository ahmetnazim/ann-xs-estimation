"""Tests for the torch ladder (Phase 6, Batch C): M2-MLP, M3, M4, M5, M6-NN and the ensemble.

Synthetic batches only (no data). The batch follows the ADL-077 contract; targets are made
Manning-consistent where a test needs exact label recovery.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from src.config import load_settings
from src.data.normstats import NormStats
from src.models import losses as ls
from src.models.ensemble import (
    ensemble_roughness,
    mixture_cdf,
    mixture_moments,
    mixture_quantile,
    point_prediction,
    stack_members,
)
from src.models.heads import BoundedTransforms, LogFeature, ModelOutput, project_nu
from src.models.m2_mlp import M2MLP, M2Config, native_targets
from src.models.m3_physics import M3Config, M3Physics, m3_physics, m3_r_bar
from src.models.m4_soft import M4Config, M4Soft
from src.models.m5_hard import M5Config, M5Hard
from src.models.m6_two_stage import M6NN, M6Config, m6_labels, m6_stage3, project_t
from src.physics.manning_layer import (
    PhiConfig,
    manning_residual,
    phi,
    solve_jarrett,
    wide_channel_depth,
)
from src.physics.roughness_priors import strickler_nu
from src.physics.shape import r_of_t, t_of_r

pytestmark = pytest.mark.unit

PROTOCOL = load_settings().protocol
BACKBONE = PROTOCOL.model.backbone
TR = BoundedTransforms.from_domain(PROTOCOL.domain)
PHI = PhiConfig.from_solver(PROTOCOL.solver)
TAU = PROTOCOL.solver.tau_l
F = 9
D50_COL = 6
BLIND = torch.tensor([0, 1, 2, 5, 6, 7])  # features 3, 4, 8 play Q_E, S, F_S (routed out)
ROUTED_OUT = (3, 4, 8)
STATS = NormStats(
    names=tuple(f"f{j}" if j != D50_COL else "D50_m" for j in range(F)),
    kinds=tuple("log" if j == D50_COL else "identity" for j in range(F)),
    offsets=(0.0,) * F,
    mean=tuple(math.log(0.02) if j == D50_COL else 0.0 for j in range(F)),
    std=tuple(1.5 if j == D50_COL else 1.0 for j in range(F)),
    n_fit=100,
)
LN_D50 = LogFeature.from_stats(STATS, "D50_m")


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def make_batch(b: int = 32, seed: int = 0, steep: float = 0.3) -> dict[str, Tensor]:
    """Manning-consistent synthetic batch: draw (w, r, nu, q, s), solve ell by Phi."""
    g = gen(seed)
    x = torch.randn(b, F, dtype=torch.float64, generator=g)
    q = 2.0 + torch.randn(b, dtype=torch.float64, generator=g)
    n_steep = int(steep * b)
    s_flat = math.log(1e-4) + 2.0 * torch.rand(b - n_steep, dtype=torch.float64, generator=g)
    s_steep = math.log(0.003) + 2.0 * torch.rand(n_steep, dtype=torch.float64, generator=g)
    s = torch.cat([s_flat, s_steep])
    w = 3.0 + 0.5 * torch.randn(b, dtype=torch.float64, generator=g)
    r = 0.6 + 8.0 * torch.rand(b, dtype=torch.float64, generator=g)
    nu = math.log(0.035) + 0.3 * torch.randn(b, dtype=torch.float64, generator=g)
    ell = phi(w, r, nu, q, s, PHI).ell
    y = torch.stack([w, ell, t_of_r(r), w + ell], dim=1)
    return {
        "x": x,
        "x_blind_idx": BLIND,
        "phys": torch.stack([q, s], dim=1),
        "y": y,
        "omega": torch.full((b,), 1.0 / 100, dtype=torch.float64),
        "row": torch.arange(b),
        "nu_true": nu,
        "r_true": r,
    }


def m5(cfg: M5Config | None = None, seed: int = 0, ln_d50: LogFeature | None = None) -> M5Hard:
    torch.manual_seed(seed)
    return M5Hard(F, BLIND, cfg or M5Config.from_protocol(PROTOCOL), BACKBONE, ln_d50=ln_d50)


def m4(lambda_max: float = 100.0, seed: int = 0) -> M4Soft:
    torch.manual_seed(seed)
    return M4Soft(F, BLIND, M4Config.from_protocol(PROTOCOL, lambda_max=lambda_max), BACKBONE)


def grads_wrt_x(net: torch.nn.Module, batch: dict[str, Tensor], key: str) -> Tensor:
    leaf = batch["x"].clone().requires_grad_()
    out = net({**batch, "x": leaf})
    (g,) = torch.autograd.grad(out.extras[key].sum(), leaf)
    return g


def all_grads_finite(net: torch.nn.Module) -> bool:
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    return bool(grads) and all(bool(torch.isfinite(gr).all()) for gr in grads)


# --- M5 ---------------------------------------------------------------------------------------


def test_m5_config_defaults_follow_params() -> None:
    cfg = M5Config.from_protocol(PROTOCOL)
    assert cfg.weights == (1.0, 1.0, 1.0, 0.0)
    assert cfg.phi.kappa
    assert cfg.prior == "none"
    assert cfg.sigma_min == PROTOCOL.uq.sigma_min
    assert cfg.warm_start_epochs == 20
    assert cfg.beta_n == 0.5


def test_m5_forward_is_physics_exact() -> None:
    batch = make_batch()
    net = m5()
    out = net(batch)
    mu, sigma = out.mu, out.sigma
    assert sigma is not None
    assert tuple(mu.shape) == (32, 4)
    assert tuple(sigma.shape) == (32, 4)
    assert torch.equal(mu[:, 3], mu[:, 0] + mu[:, 1])
    assert torch.equal(mu[:, 2], t_of_r(out.extras["r"]))
    q, s = batch["phys"][:, 0], batch["phys"][:, 1]
    rho = manning_residual(mu[:, 1], mu[:, 0], out.extras["r"], out.extras["nu"], q, s, PHI)
    assert float(rho.detach().abs().max()) <= (5.0 / 3.0) * TAU
    assert bool((sigma >= PROTOCOL.uq.sigma_min).all())
    assert torch.allclose(sigma[:, 3], torch.sqrt(sigma[:, 0] ** 2 + sigma[:, 1] ** 2))
    nu = out.extras["nu"]
    assert bool((nu > TR.nu_min).all())
    assert bool((nu < TR.nu_min + TR.delta_n).all())


def test_m5_objective_switches_from_mse_to_beta_nll() -> None:
    batch = make_batch()
    net = m5()
    out = net(batch)
    w = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    warm = net.objective(batch, out, 0).total
    assert torch.allclose(warm, ls.squared_loss(out.mu, batch["y"], batch["omega"], w))
    assert out.sigma is not None
    late = net.objective(batch, out, 20)
    expect = ls.gaussian_nll(out.mu, out.sigma, batch["y"], batch["omega"], w, beta_n=0.5)
    assert torch.allclose(late.total, expect)
    assert set(late.parts) == {"data"}
    late.total.backward()
    assert all_grads_finite(net)


def test_m5_nu_is_blind_to_routed_out_features() -> None:
    g = grads_wrt_x(m5(), make_batch(), "nu")
    for j in ROUTED_OUT:
        assert bool((g[:, j] == 0).all())
    assert bool((g[:, 0] != 0).any())


def test_m5_ablation_a1_uses_the_wide_channel_depth() -> None:
    batch = make_batch()
    net = m5(M5Config.from_protocol(PROTOCOL, kappa=False))
    out = net(batch)
    q, s = batch["phys"][:, 0], batch["phys"][:, 1]
    ell0 = wide_channel_depth(out.mu[:, 0], out.extras["nu"], q, s)
    assert torch.allclose(out.mu[:, 1], ell0, rtol=0, atol=1e-14)
    assert int(out.extras["iterations"].max()) == 0


def test_m5_ablation_a2_and_w_a_weights() -> None:
    assert M5Config.from_protocol(PROTOCOL, a2=True).weights == (1.0, 1.0, 0.0, 0.0)
    assert M5Config.from_protocol(PROTOCOL, w_a=1.0).weights == (1.0, 1.0, 1.0, 1.0)


def test_m5_strickler_prior_sp() -> None:
    cfg = M5Config.from_protocol(PROTOCOL, prior="SP")
    with pytest.raises(ValueError, match="ln D50"):
        m5(cfg)
    batch = make_batch()
    net = m5(cfg, ln_d50=LN_D50)
    out = net(batch)
    nu_s = strickler_nu(LN_D50(batch["x"]), PROTOCOL.priors.strickler)
    assert torch.allclose(out.extras["nu_s"], nu_s)
    obj = net.objective(batch, out, 0)
    expect = 2.0 * (batch["omega"] * (out.extras["nu"] - nu_s) ** 2).sum()
    assert set(obj.parts) == {"data", "prior"}
    assert torch.allclose(obj.parts["prior"], expect)
    assert torch.allclose(obj.total, obj.parts["data"] + obj.parts["prior"])


def test_m5_strickler_residual_sr() -> None:
    cfg = M5Config.from_protocol(PROTOCOL, prior="SR")
    batch = make_batch()
    net = m5(cfg, ln_d50=LN_D50)
    out = net(batch)
    delta = out.extras["delta"]
    pi_s = ((out.extras["nu_s"] - TR.nu_min) / TR.delta_n).clamp(0.01, 0.99)
    assert torch.allclose(out.extras["nu"], TR.nu(torch.logit(pi_s) + delta))
    obj = net.objective(batch, out, 0)
    assert set(obj.parts) == {"data", "ridge"}
    assert torch.allclose(obj.parts["ridge"], 2.865 * (batch["omega"] * delta**2).sum())


def test_m5_rejects_a_bad_batch() -> None:
    batch = make_batch()
    with pytest.raises(KeyError, match="omega"):
        m5()({k: v for k, v in batch.items() if k != "omega"})
    low_s = batch["phys"].clone()
    low_s[0, 1] = math.log(1e-6)
    with pytest.raises(ValueError, match="S_floor"):
        m5()({**batch, "phys": low_s})


# --- M4 ---------------------------------------------------------------------------------------


def test_m4_config_and_grid_guard() -> None:
    cfg = M4Config.from_protocol(PROTOCOL, lambda_max=1000.0)
    assert (cfg.lambda0, cfg.gamma, cfg.lambda_max) == (1.0, 1.1, 1000.0)
    with pytest.raises(ValueError, match="pre-registered grid"):
        M4Config.from_protocol(PROTOCOL, lambda_max=50.0)
    assert cfg.penalty_weight(0) == 1.0
    assert cfg.penalty_weight(10) == pytest.approx(1.1**10)
    assert cfg.penalty_weight(73) == 1000.0
    assert cfg.penalty_weight(0, monitor=True) == 1000.0


def test_m4_forward_and_objective() -> None:
    batch = make_batch()
    net = m4(lambda_max=10.0)
    out = net(batch)
    mu = out.mu
    assert torch.equal(mu[:, 3], mu[:, 0] + mu[:, 1])
    assert torch.equal(mu[:, 2], t_of_r(out.extras["r"]))
    q, s = batch["phys"][:, 0], batch["phys"][:, 1]
    r_m = manning_residual(mu[:, 1], mu[:, 0], out.extras["r"], out.extras["nu"], q, s, PHI)
    assert torch.allclose(out.extras["R_M"], r_m)
    p = (batch["omega"] * r_m**2).sum()
    for epoch, lam in ((0, 1.0), (5, 1.1**5), (100, 10.0)):
        obj = net.objective(batch, out, epoch)
        assert set(obj.parts) == {"data", "penalty"}
        assert torch.allclose(obj.parts["penalty"], lam * p)
        assert torch.allclose(obj.total, obj.parts["data"] + obj.parts["penalty"])
    mon = net.objective(batch, out, 0, monitor=True)
    assert torch.allclose(mon.parts["penalty"], 10.0 * p)


def test_m4_xi_diagnostic_is_computable_from_the_parts() -> None:
    batch = make_batch()
    net = m4()
    obj = net.objective(batch, net(batch), 30)
    params = [p for p in net.parameters() if p.requires_grad]
    g_data = torch.autograd.grad(obj.parts["data"], params, retain_graph=True, allow_unused=True)
    g_pen = torch.autograd.grad(obj.parts["penalty"], params, allow_unused=True)

    def norm(gs: tuple[Tensor | None, ...]) -> float:
        return math.sqrt(sum(float((g**2).sum()) for g in gs if g is not None))

    xi = norm(g_pen) / norm(g_data)
    assert math.isfinite(xi)
    assert xi > 0


def test_m4_nu_is_blind_and_ell_head_is_free() -> None:
    batch = make_batch()
    g = grads_wrt_x(m4(), batch, "nu")
    for j in ROUTED_OUT:
        assert bool((g[:, j] == 0).all())
    leaf = batch["x"].clone().requires_grad_()
    out = m4()({**batch, "x": leaf})
    (g_ell,) = torch.autograd.grad(out.mu[:, 1].sum(), leaf)
    assert bool((g_ell[:, 3] != 0).any())


# --- M6 ---------------------------------------------------------------------------------------

M6CFG = M6Config.from_protocol(PROTOCOL)


def test_m6_config_t_band() -> None:
    assert M6CFG.t_band == pytest.approx((math.log(1.1), math.log(3.0)), abs=1e-15)
    assert M6CFG.weights == (1.0, 1.0, 1.0)


def test_m6_labels_recover_nu_on_manning_consistent_rows() -> None:
    batch = make_batch(b=64, seed=3)
    lab = m6_labels(batch["y"], batch["phys"], M6CFG)
    assert float((lab.nu - batch["nu_true"]).abs().max()) <= (5.0 / 3.0) * TAU * 1.01
    assert torch.equal(lab.w, batch["y"][:, 0])
    assert torch.equal(lab.t, batch["y"][:, 2])
    assert not bool(lab.r_clipped.any())
    assert torch.allclose(lab.r, batch["r_true"], rtol=1e-12)


def test_m6_labels_clip_r_obs_and_flag_outside_labels() -> None:
    batch = make_batch(b=4, seed=4)
    y = batch["y"].clone()
    y[0, 2] = 0.0  # SA3 F_t row: r = inf -> r_max
    y[1, 2] = 1.5  # F_tout above ln 3 -> r_min
    y[2, 1] = y[2, 1] + 6.0  # far too deep for its discharge: nu_obs above the band
    lab = m6_labels(y, batch["phys"], M6CFG)
    assert lab.r[0].item() == pytest.approx(10.0, rel=1e-13)
    assert lab.r[1].item() == pytest.approx(0.5, rel=1e-13)
    assert lab.r_clipped.tolist() == [True, True, False, False]
    assert bool(lab.nu_outside[2])
    assert lab.t[0].item() == 0.0  # the t label itself is taken directly
    y[3, 2] = -0.1
    with pytest.raises(ValueError, match="t_obs"):
        m6_labels(y, batch["phys"], M6CFG)


def test_m6_stage3_projects_and_closes() -> None:
    batch = make_batch(b=6, seed=5)
    lab = m6_labels(batch["y"], batch["phys"], M6CFG)
    g_nu = lab.nu.clone()
    g_t = lab.t.clone()
    g_nu[0] = 2.0  # above nu_max
    g_t[1] = -0.3  # t <= 0: r undefined without the Q-C1 projection
    out = m6_stage3(lab.w, g_t, g_nu, batch["phys"], M6CFG)
    assert out.sigma is None
    assert out.extras["nu_projected"].tolist() == [True] + [False] * 5
    assert out.extras["t_projected"].tolist() == [False, True] + [False] * 4
    assert out.extras["nu"][0].item() == pytest.approx(math.log(0.6), abs=1e-15)
    assert out.mu[1, 2].item() == pytest.approx(math.log(1.1), abs=1e-15)
    assert torch.equal(out.mu[:, 3], out.mu[:, 0] + out.mu[:, 1])
    rest = slice(2, 6)
    assert torch.allclose(out.mu[rest, 1], batch["y"][rest, 1], rtol=0, atol=10 * TAU)


def test_project_helpers() -> None:
    nu = torch.tensor([-9.0, math.log(0.03), 1.0], dtype=torch.float64)
    out = project_nu(nu, TR)
    assert out.tolist() == pytest.approx([math.log(0.005), math.log(0.03), math.log(0.6)])
    t = torch.tensor([0.0, 0.5, 2.0], dtype=torch.float64)
    assert project_t(t, M6CFG.t_band).tolist() == pytest.approx([math.log(1.1), 0.5, math.log(3)])
    with pytest.raises(ValueError, match="non-finite"):
        project_nu(torch.tensor([float("nan")], dtype=torch.float64), TR)


def test_m6nn_objective_and_routing() -> None:
    torch.manual_seed(0)
    net = M6NN(F, BLIND, M6CFG, BACKBONE)
    batch = make_batch()
    out = net(batch)
    assert not out.mu.requires_grad
    lab = m6_labels(batch["y"], batch["phys"], M6CFG)
    pred = torch.stack([out.extras["g_nu"], out.extras["g_w"], out.extras["g_t"]], dim=1)
    expect = (batch["omega"][:, None] * (pred - lab.stacked()) ** 2).sum()
    obj = net.objective(batch, out, 0)
    assert torch.allclose(obj.total, expect)
    assert torch.equal(net.objective(batch, out, 99, monitor=True).total, obj.total)
    obj.total.backward()
    assert all_grads_finite(net)
    g = grads_wrt_x(net, batch, "g_nu")
    for j in ROUTED_OUT:
        assert bool((g[:, j] == 0).all())


# --- M3 ---------------------------------------------------------------------------------------


def test_m3_r_bar() -> None:
    t = torch.tensor([0.2, 0.4, 0.5, 0.9], dtype=torch.float64).numpy()
    assert m3_r_bar(t) == pytest.approx(1.0 / math.expm1(0.45))
    with pytest.raises(ValueError, match="median"):
        m3_r_bar(torch.zeros(3, dtype=torch.float64).numpy())
    with pytest.raises(ValueError, match="finite"):
        m3_r_bar(torch.tensor([0.1, float("nan")], dtype=torch.float64).numpy())


def m3_cfg(r_bar: float = 2.0, **jarrett: float) -> M3Config:
    cfg = M3Config.from_protocol(PROTOCOL, r_bar=r_bar)
    if jarrett:
        cfg = M3Config(
            phi=cfg.phi,
            transforms=cfg.transforms,
            strickler=cfg.strickler,
            jarrett=cfg.jarrett.model_copy(update=jarrett),
            s_floor=cfg.s_floor,
            r_bar=cfg.r_bar,
        )
    return cfg


def test_m3_physics_branches_and_projection() -> None:
    batch = make_batch(b=40, seed=6, steep=0.5)
    w, phys = batch["y"][:, 0], batch["phys"]
    q, s = phys[:, 0], phys[:, 1]
    ln_d50 = torch.full_like(w, math.log(0.01))
    ln_d50[0] = math.log(1e-7)  # Strickler n below n_min -> projected
    cfg = m3_cfg()
    out = m3_physics(w, ln_d50, phys, cfg)
    steep = out.extras["jarrett"]
    assert steep.tolist() == (s >= math.log(0.002)).tolist()
    r = torch.full_like(w, 2.0)
    flat = ~steep
    nu_flat = project_nu(strickler_nu(ln_d50, cfg.strickler), TR)[flat]
    assert torch.allclose(out.extras["nu"][flat], nu_flat)
    assert bool(out.extras["nu_projected"][0])
    ell_flat = phi(w[flat], r[flat], nu_flat, q[flat], s[flat], PHI).ell
    assert torch.allclose(out.mu[flat, 1], ell_flat, rtol=0, atol=1e-14)
    ell_j, nu_j, _ = solve_jarrett(w[steep], r[steep], q[steep], s[steep], cfg.jarrett, PHI)
    assert torch.allclose(out.mu[steep, 1], ell_j, rtol=0, atol=1e-14)
    assert torch.allclose(out.extras["nu"][steep], nu_j)
    assert torch.allclose(out.mu[:, 2], torch.full_like(w, math.log1p(0.5)))
    assert torch.equal(out.mu[:, 3], out.mu[:, 0] + out.mu[:, 1])
    assert out.sigma is None


def test_m3_out_of_band_jarrett_rows_are_resolved_by_phi() -> None:
    batch = make_batch(b=20, seed=7, steep=1.0)
    w, phys = batch["y"][:, 0], batch["phys"]
    cfg = m3_cfg(c_j_si=50.0)  # pushes n_J above n_max
    out = m3_physics(w, torch.full_like(w, math.log(0.01)), phys, cfg)
    assert bool(out.extras["nu_projected"].all())
    nu_max = TR.nu_min + TR.delta_n
    assert torch.allclose(out.extras["nu"], torch.full_like(w, nu_max))
    r = torch.full_like(w, 2.0)
    ell = phi(w, r, out.extras["nu"], phys[:, 0], phys[:, 1], PHI).ell
    assert torch.allclose(out.mu[:, 1], ell, rtol=0, atol=1e-14)
    assert out.extras["jarrett_invalid"].dtype == torch.bool


def test_m3_regressor_trains_w_only() -> None:
    torch.manual_seed(0)
    net = M3Physics(F, m3_cfg(), BACKBONE, LN_D50)
    batch = make_batch(steep=0.5)
    out = net(batch)
    assert out.mu.requires_grad
    obj = net.objective(batch, out, 0)
    expect = (batch["omega"] * (out.mu[:, 0] - batch["y"][:, 0]) ** 2).sum()
    assert torch.allclose(obj.total, expect)
    obj.total.backward()
    assert all_grads_finite(net)
    w_mean = out.mu[:, 0].detach()
    again = net.physics(w_mean, batch)
    assert torch.allclose(again.mu, out.mu.detach())


# --- M2-MLP -----------------------------------------------------------------------------------


def test_m2_native_and_canonical_views() -> None:
    torch.manual_seed(0)
    net = M2MLP(F, M2Config.from_protocol(PROTOCOL), BACKBONE)
    batch = make_batch()
    out = net(batch)
    mu_n, sig_n = out.extras["mu_native"], out.extras["sigma_native"]
    assert out.sigma is not None
    assert torch.allclose(out.mu[:, 2], mu_n[:, 2] - mu_n[:, 1])
    assert torch.allclose(out.ln_dmax, mu_n[:, 2])
    assert torch.allclose(out.sigma[:, 2], torch.sqrt(sig_n[:, 2] ** 2 + sig_n[:, 1] ** 2))
    y_n = native_targets(batch["y"])
    assert torch.allclose(y_n[:, 2], batch["y"][:, 1] + batch["y"][:, 2])
    ones = torch.ones(4, dtype=torch.float64)
    warm = net.objective(batch, out, 0).total
    assert torch.allclose(warm, ls.squared_loss(mu_n, y_n, batch["omega"], ones))
    late = net.objective(batch, out, 25).total
    expect = ls.gaussian_nll(mu_n, sig_n, y_n, batch["omega"], ones, beta_n=0.5)
    assert torch.allclose(late, expect)
    late.backward()
    assert all_grads_finite(net)
    assert net.heads.blind_trunk is None


# --- ensemble ---------------------------------------------------------------------------------


def test_mixture_moments_law_of_total_variance() -> None:
    mu = torch.tensor([[[0.0]], [[2.0]]], dtype=torch.float64)
    sigma = torch.tensor([[[1.0]], [[3.0]]], dtype=torch.float64)
    mix = mixture_moments(mu, sigma)
    assert mix.mu_star.item() == 1.0
    assert mix.var_ale.item() == 5.0
    assert mix.var_epi.item() == 1.0
    assert mix.sigma_star.item() == pytest.approx(math.sqrt(6.0))
    no_sigma = mixture_moments(mu, None)
    assert no_sigma.var_ale.item() == 0.0


def test_mixture_quantile_exact() -> None:
    g = gen(11)
    mu = torch.randn(10, 7, 4, dtype=torch.float64, generator=g)
    sigma = 0.05 + torch.rand(10, 7, 4, dtype=torch.float64, generator=g)
    for p in (0.05, 0.5, 0.95):
        qp = mixture_quantile(p, mu, sigma)
        assert torch.allclose(mixture_cdf(qp, mu, sigma), torch.full_like(qp, p), atol=1e-11)
        z = float(torch.special.ndtri(torch.tensor(p, dtype=torch.float64)))
        ends = mu + z * sigma
        assert bool((qp >= ends.min(0).values - 1e-12).all())
        assert bool((qp <= ends.max(0).values + 1e-12).all())
    one = mixture_quantile(0.95, mu[:1], sigma[:1])
    z95 = float(torch.special.ndtri(torch.tensor(0.95, dtype=torch.float64)))
    assert torch.allclose(one, mu[0] + z95 * sigma[0], rtol=0, atol=1e-12)
    sym = torch.tensor([[[-1.0]], [[1.0]]], dtype=torch.float64)
    ones = torch.ones_like(sym)
    assert mixture_quantile(0.5, sym, ones).item() == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="p must"):
        mixture_quantile(1.0, mu, sigma)
    with pytest.raises(ValueError, match="sigma"):
        mixture_quantile(0.5, mu, -sigma)


def test_point_prediction_and_stack_members() -> None:
    nets = [m5(seed=k) for k in range(3)]
    batch = make_batch()
    outs = [n(batch) for n in nets]
    mu, sigma = stack_members(outs)
    assert tuple(mu.shape) == (3, 32, 4)
    assert sigma is not None
    assert not mu.requires_grad
    mix = mixture_moments(mu, sigma)
    assert torch.allclose(point_prediction(mix.mu_star), torch.exp(mix.mu_star))
    assert torch.allclose(mix.mu_star[:, 3], mix.mu_star[:, 0] + mix.mu_star[:, 1])
    bare = ModelOutput(mu=outs[0].mu, sigma=None, extras={})
    with pytest.raises(ValueError, match="disagree"):
        stack_members([outs[0], bare])
    with pytest.raises(ValueError, match="at least one"):
        stack_members([])


def test_ensemble_roughness_inverts_phi() -> None:
    batch = make_batch()
    out = m5()(batch)
    mu, sigma = stack_members([out])
    nu_star = ensemble_roughness(mixture_moments(mu, sigma).mu_star, batch["phys"], PHI)
    assert float((nu_star - out.extras["nu"].detach()).abs().max()) <= (5.0 / 3.0) * TAU * 1.01
    bad = mu[0].clone()
    bad[0, 2] = 0.0
    with pytest.raises(ValueError, match="t must be > 0"):
        ensemble_roughness(bad, batch["phys"], PHI)
    assert torch.allclose(r_of_t(mu[0, :, 2]), out.extras["r"])

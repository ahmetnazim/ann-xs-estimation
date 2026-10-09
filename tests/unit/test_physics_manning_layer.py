"""Smoke tests for src/physics/{shape,manning_layer}.py (Phase 6, Batch A). Synthetic; no data.

Tier 3 of the E-tier acceptance (P6-Q1 (a)): the hand-computed examples of proofs 10.2
(E1, E2, E11; E3 and E5 live with the quadrature and roughness tests). The examples are
printed to 5-6 significant figures with the last digit rounded, so a printed value is accepted
within **one unit of its last printed digit** (Batch A review: E1's printed V = 0.71101 is
7e-6 from its own Q_E / A; the "Manning back-check" lines are hand-rounding artefacts and are
replaced by the exact residual bound |rho| <= (5/3) tau_l of Theorem 4.1).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import Tensor

from src.config import load_settings
from src.physics import manning_layer as ml
from src.physics.quadrature import perimeter_fm1
from src.physics.shape import (
    dt_dr,
    log_kappa,
    r_of_t,
    section_terms,
    t_of_r,
)

pytestmark = pytest.mark.unit

PROTOCOL = load_settings().protocol
SOLVER = PROTOCOL.solver
JARRETT = PROTOCOL.priors.jarrett
CFG = ml.PhiConfig.from_solver(SOLVER)
#: Tight solver used for derivative checks, so that the O(tau_l) root error is negligible.
TIGHT = ml.PhiConfig(
    tau_l=1e-14, k_max=80, quad_level=SOLVER.quad_level, quad_trunc=SOLVER.quad_trunc
)
A1 = ml.PhiConfig.from_solver(SOLVER, kappa=False)
LAMBDA = 0.3048


def t64(*values: float) -> Tensor:
    return torch.tensor(values, dtype=torch.float64)


def logs(w: float, r: float, n: float, q: float, s: float) -> tuple[Tensor, ...]:
    """Log coordinates (proofs 1.1.1) for one reach: (w, r, nu, q, s)."""
    return t64(math.log(w)), t64(r), t64(math.log(n)), t64(math.log(q)), t64(math.log(s))


def assert_printed(got: float, printed: str) -> None:
    """Tier-3 check: ``got`` within one unit of the last printed digit of ``printed``."""
    decimals = len(printed.split(".")[1])
    assert abs(got - float(printed)) <= 10.0**-decimals, (got, printed)


def picard_iterates(w: Tensor, r: Tensor, nu: Tensor, q: Tensor, s: Tensor, m: int) -> list[float]:
    """First ``m`` Picard iterates ``dbar_m`` of Algorithm Phi (independent re-implementation)."""
    ell0 = ml.wide_channel_depth(w, nu, q, s)
    ell, out = ell0, []
    for _ in range(m):
        beta = 2.0 * (r + 1.0) * torch.exp(ell - w)
        ell = ell0 + 0.4 * torch.log1p(
            perimeter_fm1(beta, r, level=CFG.quad_level, trunc=CFG.quad_trunc)
        )
        out.append(math.exp(float(ell[0])))
    return out


def random_batch(n: int, seed: int) -> tuple[Tensor, ...]:
    """Synthetic D9-domain reaches: W 1-500 m, r 0.5-10, n 0.015-0.25, Q 0.01-1e4, S 1e-5-0.1."""
    g = torch.Generator().manual_seed(seed)

    def u(lo: float, hi: float) -> Tensor:
        return torch.empty(n, dtype=torch.float64).uniform_(lo, hi, generator=g)

    return (
        u(0.0, math.log(500.0)),
        u(0.5, 10.0),
        u(math.log(0.015), math.log(0.25)),
        u(math.log(0.01), math.log(1e4)),
        u(math.log(1e-5), math.log(0.1)),
    )


# --- Tier 3: hand-computed examples (proofs 10.2) -------------------------------------------


def test_e1_wide_parabola() -> None:
    w, r, nu, q, s = logs(20.0, 2.0, 0.035, 10.0, 1e-3)
    assert_printed(math.exp(float(ml.wide_channel_depth(w, nu, q, s)[0])), "0.70117")
    for got, printed in zip(
        picard_iterates(w, r, nu, q, s, 3), ("0.703220", "0.703232", "0.703232"), strict=True
    ):
        assert_printed(got, printed)
    res = ml.phi(w, r, nu, q, s, CFG)
    ell, ln_dmax, ln_a, ln_v = (float(v) for v in res.out[0])
    terms = section_terms(res.ell, w, r, level=CFG.quad_level, trunc=CFG.quad_trunc)
    eta = float(terms.eta[0])
    assert_printed(math.exp(ell), "0.70323")
    assert_printed(1.0 + float(terms.fm1[0]), "1.007369")
    assert_printed(float(terms.kappa[0]), "0.99268")
    assert_printed(math.exp(ln_dmax), "1.05485")
    assert_printed(math.exp(ln_a), "14.0646")
    assert_printed(math.exp(ln_v), "0.71101")
    assert_printed(eta, "0.01454")
    assert_printed(0.4 * eta, "0.00581")
    assert_printed(1.0 / (5.0 / 3.0 - 2.0 * eta / 3.0), "0.60351")


def test_e2_narrow_triangle() -> None:
    w, r, nu, q, s = logs(3.0, 1.0, 0.04, 1.0, 0.01)
    assert_printed(math.exp(float(ml.wide_channel_depth(w, nu, q, s)[0])), "0.298513")
    printed = ("0.307423", "0.307931", "0.307960", "0.307962")
    for got, want in zip(picard_iterates(w, r, nu, q, s, 4), printed, strict=True):
        assert_printed(got, want)
    res = ml.phi(w, r, nu, q, s, CFG)
    dbar = math.exp(float(res.ell[0]))
    assert_printed(dbar, "0.30796")
    assert 0.30742 <= dbar <= 0.31351  # Proposition 3.2 bracket as printed
    terms = section_terms(res.ell, w, r, level=CFG.quad_level, trunc=CFG.quad_trunc)
    kappa, eta = float(terms.kappa[0]), float(terms.eta[0])
    d_max = math.exp(float(res.out[0, 1]))
    assert_printed(kappa, "0.92505")
    assert_printed(3.0 / (3.0 + 2.0 * d_max), "0.70891")
    assert kappa > 3.0 / (3.0 + 2.0 * d_max)
    assert_printed(eta, "0.14428")
    assert_printed(0.4 * eta, "0.0577")
    assert_printed(1.0 / (5.0 / 3.0 - 2.0 * eta / 3.0), "0.63675")
    assert_printed(kappa ** (-2.0 / 3.0), "1.0533")


def test_e11_unit_equivariance() -> None:
    w, r, nu, q, s = logs(20.0, 2.0, 0.035, 10.0, 1e-3)
    chi = -math.log(LAMBDA)  # ln chi_L, chi_L = 1 / 0.3048; chi_T = 1
    w_ft, nu_ft, q_ft = w + chi, nu - chi / 3.0, q + 3.0 * chi
    assert_printed(math.exp(float(w_ft[0])), "65.6168")
    assert_printed(math.exp(float(q_ft[0])), "353.1467")
    assert_printed(math.exp(float(nu_ft[0])), "0.0235544")
    d_ft = math.exp(float(ml.wide_channel_depth(w_ft, nu_ft, q_ft, s)[0]))
    assert_printed(d_ft, "2.30042")
    assert_printed(d_ft * LAMBDA, "0.701169")
    # Proposition 3.5 for the full Phi: outputs shift by (1, 1, 2, 1) ln chi_L.
    si = ml.phi(w, r, nu, q, s, TIGHT).out
    ft = ml.phi(w_ft, r, nu_ft, q_ft, s, TIGHT).out
    assert torch.allclose(ft - si, chi * t64(1.0, 1.0, 2.0, 1.0)[None, :], rtol=0, atol=1e-12)


# --- Theorem 4.1, Propositions 3.2-3.4 ------------------------------------------------------


def test_theorem_4_1_identities_and_residual() -> None:
    w, r, nu, q, s = random_batch(300, seed=11)
    res = ml.phi(w, r, nu, q, s, CFG)
    ell, ln_dmax, ln_a, ln_v = res.out.unbind(dim=1)
    assert torch.equal(ln_a, w + ell)
    assert torch.equal(ln_v, q - w - ell)
    assert torch.allclose(torch.exp(ln_dmax - ell), 1.0 + 1.0 / r, rtol=1e-14, atol=0)
    rho = ml.manning_residual(ell, w, r, nu, q, s, CFG)
    assert float(rho.abs().max()) <= (5.0 / 3.0) * CFG.tau_l
    assert int(res.iterations.min()) >= 1
    assert int(res.iterations.max()) <= CFG.k_max


def test_prop_3_2_bracket_and_monotone_start() -> None:
    w, r, nu, q, s = random_batch(300, seed=12)
    ell = ml.phi(w, r, nu, q, s, CFG).ell
    ell0 = ml.wide_channel_depth(w, nu, q, s)
    k0 = section_terms(ell0, w, r, level=CFG.quad_level, trunc=CFG.quad_trunc).k
    assert bool((k0 <= 0).all())
    assert bool((ell >= ell0 + 0.4 * k0.abs() - CFG.tau_l).all())
    assert bool((ell <= ell0 + (2.0 / 3.0) * k0.abs() + CFG.tau_l).all())


def ift_grads(cfg: ml.PhiConfig, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    leaves = tuple(x.clone().requires_grad_() for x in inputs)
    ell = ml.phi(*leaves, cfg).ell
    return torch.autograd.grad(ell.sum(), leaves)


def test_prop_3_3_ranges_and_3_6_bounds() -> None:
    inputs = random_batch(300, seed=13)
    gw, gr, gnu, gq, gs = ift_grads(CFG, inputs)
    r = inputs[1]
    assert torch.equal(gq, gnu)
    assert bool((gq >= 0.6).all())
    assert bool((gq < 1.0).all())
    assert bool((gw > -5.0 / 3.0).all())
    assert bool((gw <= -0.6).all())
    assert bool((gs.abs() >= 0.3).all())
    assert bool((gs.abs() < 0.5).all())
    assert torch.allclose(gs, -0.5 * gq, rtol=0, atol=0)
    assert bool((gr.abs() <= 2.0 * (2.0 * r + 1.0) / (3.0 * r * (r + 1.0))).all())


def test_ift_gradient_matches_central_differences() -> None:
    inputs = random_batch(12, seed=14)
    grads = ift_grads(TIGHT, inputs)
    h = 1e-5
    for j in range(5):
        up = list(inputs)
        dn = list(inputs)
        up[j] = inputs[j] + h
        dn[j] = inputs[j] - h
        fd = (ml.phi(*up, TIGHT).ell - ml.phi(*dn, TIGHT).ell) / (2.0 * h)
        assert torch.allclose(grads[j], fd, rtol=0, atol=1e-8), j


def test_gradcheck_full_output() -> None:
    leaves = tuple(x.clone().requires_grad_() for x in random_batch(4, seed=15))
    assert torch.autograd.gradcheck(lambda *z: ml.phi(*z, TIGHT).out, leaves, eps=1e-6, atol=1e-7)


def test_one_step_surrogate_equals_ift() -> None:
    inputs = random_batch(40, seed=16)
    leaves = tuple(x.clone().requires_grad_() for x in inputs)
    surrogate = ml.phi_one_step(*leaves, TIGHT)
    g_sur = torch.autograd.grad(surrogate.sum(), leaves)
    ell = ml.phi(*inputs, TIGHT).ell
    assert torch.allclose(surrogate.detach(), ell, rtol=0, atol=(5.0 / 3.0) * TIGHT.tau_l + 1e-15)
    for a, b in zip(g_sur, ift_grads(TIGHT, inputs), strict=True):
        assert torch.allclose(a, b, rtol=1e-12, atol=1e-14)


def test_prop_3_4_permutation_and_subset_invariance() -> None:
    inputs = random_batch(64, seed=17)
    full = ml.phi(*inputs, CFG).out
    g_full = ift_grads(CFG, inputs)
    perm = torch.randperm(64, generator=torch.Generator().manual_seed(3))
    permuted = tuple(x[perm] for x in inputs)
    assert torch.equal(ml.phi(*permuted, CFG).out, full[perm])
    for a, b in zip(ift_grads(CFG, permuted), g_full, strict=True):
        assert torch.equal(a, b[perm])
    sub = torch.tensor([5, 0, 63, 17])
    assert torch.equal(ml.phi(*(x[sub] for x in inputs), CFG).out, full[sub])


def test_block_diagonal_jacobian() -> None:
    leaves = tuple(x.clone().requires_grad_() for x in random_batch(6, seed=18))
    ell = ml.phi(*leaves, CFG).ell
    grads = torch.autograd.grad(ell[2], leaves)
    for g in grads:
        assert bool((g[torch.arange(6) != 2] == 0).all())


def test_non_convergence_raises() -> None:
    w, r, nu, q, s = logs(3.0, 1.0, 0.04, 1.0, 0.01)
    short = ml.PhiConfig(tau_l=1e-10, k_max=2, quad_level=CFG.quad_level, quad_trunc=CFG.quad_trunc)
    with pytest.raises(ml.PhiConvergenceError):
        ml.phi(w, r, nu, q, s, short)


def test_ablation_a1_closed_form_and_gradients() -> None:
    inputs = random_batch(20, seed=19)
    w, r, nu, q, s = inputs
    res = ml.phi(w, r, nu, q, s, A1)
    assert torch.equal(res.ell, ml.wide_channel_depth(w, nu, q, s))
    assert int(res.iterations.max()) == 0
    gw, gr, gnu, gq, gs = ift_grads(A1, inputs)
    for g, want in ((gw, -0.6), (gr, 0.0), (gnu, 0.6), (gq, 0.6), (gs, -0.3)):
        assert torch.allclose(g, torch.full_like(g, want), rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    ("bad", "error"),
    [
        ({"r": t64(0.0)}, ValueError),
        ({"q": t64(float("inf"))}, ValueError),
        ({"w": torch.tensor([1.0], dtype=torch.float32)}, TypeError),
        ({"nu": t64(1.0, 2.0)}, ValueError),
    ],
)
def test_phi_rejects_invalid_input(bad: dict[str, Tensor], error: type) -> None:
    args = dict(zip(("w", "r", "nu", "q", "s"), logs(20.0, 2.0, 0.035, 10.0, 1e-3), strict=True))
    args.update(bad)
    with pytest.raises(error):
        ml.phi(args["w"], args["r"], args["nu"], args["q"], args["s"], CFG)


def test_check_phys_floor_is_exact() -> None:
    floor = PROTOCOL.domain.s_floor
    s_at = t64(float(np.log(floor)))
    ml.check_phys(t64(1.0), s_at, s_floor=floor)
    below = t64(float(np.nextafter(np.log(floor), -np.inf)))
    with pytest.raises(ValueError, match="S_floor"):
        ml.check_phys(t64(1.0), below, s_floor=floor)
    with pytest.raises(ValueError, match="non-finite"):
        ml.check_phys(t64(float("nan")), s_at, s_floor=floor)


# --- Proposition 6.2, Proposition 7.5, shape helpers ----------------------------------------


def test_inverse_nu_round_trip() -> None:
    w, r, nu, q, s = random_batch(200, seed=20)
    ell = ml.phi(w, r, nu, q, s, CFG).ell
    nu_back = ml.phi_inverse_nu(ell, w, r, q, s, CFG)
    assert float((nu_back - nu).abs().max()) <= (5.0 / 3.0) * CFG.tau_l


def test_jarrett_root_and_consistency() -> None:
    w, r, _, q, s = random_batch(200, seed=21)
    ell_j, nu_j, iters = ml.solve_jarrett(w, r, q, s, JARRETT, TIGHT)
    assert int(iters.max()) <= TIGHT.k_max
    # rho_J(ell_J) = rho(ell_J; nu_J): the explicit inverse returns nu_J up to the root error.
    rho_j = ml.phi_inverse_nu(ell_j, w, r, q, s, TIGHT) - nu_j
    assert float(rho_j.abs().max()) <= (5.0 / 3.0 - JARRETT.b_j) * TIGHT.tau_l + 1e-13
    # Phi with the Jarrett roughness reproduces the Jarrett depth.
    assert torch.allclose(ml.phi(w, r, nu_j, q, s, TIGHT).ell, ell_j, rtol=0, atol=1e-12)
    # nu_J = ln c_J' + a s + b ln R with R = kappa dbar (protocol 4.1 SI form).
    terms = section_terms(ell_j, w, r, level=TIGHT.quad_level, trunc=TIGHT.quad_trunc)
    ln_r_hyd = ell_j + terms.k
    want = math.log(JARRETT.c_j_si) + JARRETT.a_j * s + JARRETT.b_j * ln_r_hyd
    assert torch.allclose(nu_j, want, rtol=0, atol=1e-14)


def test_jarrett_rejects_non_contracting_exponent() -> None:
    w, r, _, q, s = random_batch(3, seed=22)
    steep = JARRETT.model_copy(update={"b_j": 1.0})
    with pytest.raises(ValueError, match="b_J"):
        ml.solve_jarrett(w, r, q, s, steep, CFG)
    with pytest.raises(ValueError, match="full kappa"):
        ml.solve_jarrett(w, r, q, s, JARRETT, A1)


def test_log_kappa_analytic_partials_match_autograd() -> None:
    w, r, nu, q, s = random_batch(30, seed=23)
    ell = ml.phi(w, r, nu, q, s, CFG).ell
    a_leaves = tuple(x.clone().requires_grad_() for x in (ell, w, r))
    b_leaves = tuple(x.clone().requires_grad_() for x in (ell, w, r))
    k_analytic = log_kappa(*a_leaves, level=CFG.quad_level, trunc=CFG.quad_trunc)
    k_autograd = section_terms(*b_leaves, level=CFG.quad_level, trunc=CFG.quad_trunc).k
    assert torch.equal(k_analytic.detach(), k_autograd.detach())
    ga = torch.autograd.grad(k_analytic.sum(), a_leaves)
    gb = torch.autograd.grad(k_autograd.sum(), b_leaves)
    for x, y in zip(ga, gb, strict=True):
        assert torch.allclose(x, y, rtol=1e-10, atol=1e-16)


def test_shape_maps() -> None:
    r = t64(0.5, 1.0, 2.0, 10.0)
    t = t_of_r(r)
    assert torch.allclose(r_of_t(t), r, rtol=1e-14, atol=0)
    assert torch.allclose(t, torch.log(1.0 + 1.0 / r), rtol=1e-15, atol=0)
    leaf = r.clone().requires_grad_()
    (g,) = torch.autograd.grad(t_of_r(leaf).sum(), leaf)
    assert torch.allclose(g, dt_dr(r), rtol=1e-14, atol=0)
    assert_printed(float(t_of_r(t64(10.0))[0]), "0.09531")  # E10 lower end, t(r = 10)
    assert_printed(float(t_of_r(t64(0.5))[0]), "1.09861")  # E10 upper end, t(r = 1/2)
    with pytest.raises(ValueError, match="t must be > 0"):
        r_of_t(t64(0.0))

"""Smoke tests for src/physics/quadrature.py (Phase 6, Batch A). Synthetic; no data.

E-tier acceptance (P6-Q1 (a)): tier 1 = mpmath reference (dps 40, both substitutions);
tier 2 = closed forms of Proposition 2.4 and the r = 1 seam; both at tau_Q from params.
The full (r, beta) level study is reproduced by scripts/verify_models.py (Batch E).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import mpmath as mp
import pytest
import torch

from src.config import load_settings
from src.physics import quadrature as qd

pytestmark = pytest.mark.unit

SOLVER = load_settings().protocol.solver
LEVEL, TRUNC, TAU_Q = SOLVER.quad_level, SOLVER.quad_trunc, SOLVER.tau_q


def t64(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def integrals(beta: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return qd.perimeter_integrals(beta, r, level=LEVEL, trunc=TRUNC)


def mp_reference(beta: float, r: float) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    """(F - 1, F_beta, F_r) by the tau-form at 40 digits, split at the s = 1 transition."""
    with mp.workdps(40):
        b, rr = mp.mpf(beta), mp.mpf(r)
        hp = 1 / rr - 1
        pts = [0, 1]
        if rr != 1:
            ts = b ** (1 / hp)
            if 0 < ts < 1:
                pts = [0, ts, 1]

        def h(t: mp.mpf) -> mp.mpf:
            return mp.sqrt(t ** (2 * hp) + b**2)

        fm1 = mp.quad(lambda t: b**2 / (h(t) + t**hp), pts, maxdegree=10) / rr
        fb = mp.quad(lambda t: b / h(t), pts, maxdegree=10) / rr
        corr = mp.quad(lambda t: t**hp * mp.log(t) / (h(t) * (h(t) + t**hp)), pts, maxdegree=10)
        return fm1, fb, -fm1 / rr + b**2 * corr / rr**3


def rel(got: float, ref: mp.mpf) -> float:
    return float(abs(mp.mpf(got) - ref) / abs(ref))


def test_rule_inside_unit_interval_and_weights_sum_to_one() -> None:
    u, ln_u, wgt = qd.tanh_sinh_rule(LEVEL, TRUNC)
    assert u.dtype == torch.float64
    assert u.shape == wgt.shape == ln_u.shape
    assert bool((u > 0).all())
    assert bool((u <= 1).all())
    assert math.isclose(float(wgt.sum()), 1.0, abs_tol=1e-14)
    assert torch.allclose(torch.exp(ln_u), u, rtol=1e-13, atol=0)  # exp error ~ |ln u| eps
    assert u.shape[0] == 2 * math.floor(TRUNC * 2**LEVEL) + 1


@pytest.mark.parametrize(
    ("r", "closed_form"),
    [
        (1.0, lambda b: mp.sqrt(1 + b**2)),  # Prop 2.4 (a)
        (2.0, lambda b: (mp.sqrt(1 + b**2) + mp.asinh(b) / b) / 2),  # Prop 2.4 (b)
        (0.5, lambda b: mp.sqrt(1 + b**2) + b**2 * mp.asinh(1 / b)),  # Prop 2.4 (d)
    ],
)
def test_closed_forms_prop_2_4(r: float, closed_form: Callable[[mp.mpf], mp.mpf]) -> None:
    betas = [10.0**e for e in (-6, -4, -2, -1, 0, 1, 2)]
    r_col = torch.full((7,), r, dtype=torch.float64)
    fm1, f_beta, _ = integrals(torch.tensor(betas, dtype=torch.float64), r_col)
    with mp.workdps(40):
        for i, b in enumerate(betas):
            ref = closed_form(mp.mpf(b)) - 1
            dref = mp.diff(closed_form, mp.mpf(b))
            assert rel(float(fm1[i]), ref) <= TAU_Q / 10
            assert rel(float(f_beta[i]), dref) <= TAU_Q / 10


def test_seam_at_r_equal_one() -> None:
    beta = t64(1e-6, 1e-3, 0.3, 3.0, 300.0)
    rule = qd.tanh_sinh_rule(LEVEL, TRUNC)
    one = torch.ones(5, dtype=torch.float64)
    u_form = qd._u_form(beta, one, rule, True)
    tau_form = qd._tau_form(beta, one, rule, True)
    for a, b in zip(u_form, tau_form, strict=True):
        assert bool(((a - b).abs() <= TAU_Q * b.abs()).all())
    below = integrals(beta, torch.full((5,), 1.0 - 1e-12, dtype=torch.float64))
    at = integrals(beta, one)
    for a, b in zip(below, at, strict=True):
        assert bool(((a - b).abs() <= TAU_Q * b.abs()).all())


@pytest.mark.parametrize(("r", "printed"), [(0.5, "1.03497"), (1.0, "1.01980"), (2.0, "1.02606")])
def test_e3_hand_values(r: float, printed: str) -> None:
    """Tier 3 (proofs 10.2 E3): delta = d_max / W = 0.1, so beta = 2 r delta; one-unit tolerance."""
    fm1, _, _ = integrals(t64(2.0 * r * 0.1), t64(r))
    assert abs(1.0 + float(fm1[0]) - float(printed)) <= 1e-5
    assert 1.0 <= 1.0 + float(fm1[0]) <= 1.2


@pytest.mark.parametrize(
    ("beta", "r"),
    [
        (1e-8, 0.7),
        (1e-3, 0.5),
        (0.2, 0.75),
        (2.0, 0.99),
        (1e-5, 1.01),
        (0.05, 1.25),
        (5.0, 1.49),
        (30.0, 3.0),
        (1e-4, 10.0),
        (1e3, 10.0),
    ],
)
def test_against_mpmath_reference(beta: float, r: float) -> None:
    fm1, f_beta, f_r = integrals(t64(beta), t64(r))
    ref = mp_reference(beta, r)
    for got, want in zip((fm1, f_beta, f_r), ref, strict=True):
        assert rel(float(got[0]), want) <= TAU_Q / 10


def test_bounds_props_2_3_2_7_2_8() -> None:
    g = torch.Generator().manual_seed(1)
    beta = torch.exp(torch.empty(400, dtype=torch.float64).uniform_(-15, 6, generator=g))
    r = torch.empty(400, dtype=torch.float64).uniform_(0.5, 10, generator=g)
    fm1, f_beta, f_r = integrals(beta, r)
    f = 1 + fm1
    assert bool((fm1 > 0).all())
    assert bool((fm1 < beta / r).all())  # 1 < F < 1 + 2 delta
    assert bool((f_beta >= 0).all())
    assert bool((f_beta <= 1 / r).all())
    assert bool((f_r <= 0).all())
    assert bool((f_r.abs() <= beta / r**2).all())
    eta = beta * f_beta / f
    assert bool((eta >= 0).all())
    assert bool((eta < 1).all())


def test_row_independence_and_fm1_path() -> None:
    beta = t64(1e-7, 0.01, 0.5, 4.0, 70.0, 1e-2)
    r = t64(0.5, 0.8, 1.0, 2.0, 9.0, 0.999)
    fm1, _, f_r = integrals(beta, r)
    for i in range(6):
        one = integrals(beta[i : i + 1], r[i : i + 1])
        assert torch.equal(one[0], fm1[i : i + 1])
        assert torch.equal(one[2], f_r[i : i + 1])
    assert torch.equal(qd.perimeter_fm1(beta, r, level=LEVEL, trunc=TRUNC), fm1)


def test_autograd_through_rule_matches_analytic_partials() -> None:
    beta = t64(1e-4, 0.03, 0.7, 6.0, 0.4).requires_grad_()
    r = t64(0.55, 0.9, 1.0, 2.5, 7.0).requires_grad_()
    fm1, f_beta, f_r = integrals(beta, r)
    gb, gr = torch.autograd.grad(fm1.sum(), (beta, r))
    assert torch.allclose(gb, f_beta.detach(), rtol=1e-12, atol=0)
    assert torch.allclose(gr, f_r.detach(), rtol=1e-11, atol=1e-18)


@pytest.mark.parametrize(
    ("beta", "r", "error"),
    [
        (t64(0.0), t64(1.0), ValueError),
        (t64(1.0), t64(0.0), ValueError),
        (t64(float("nan")), t64(1.0), ValueError),
        (
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
            TypeError,
        ),
        (t64(1.0, 2.0).reshape(1, 2), t64(1.0, 2.0).reshape(1, 2), ValueError),
        (t64(1.0, 2.0), t64(1.0), ValueError),
    ],
)
def test_rejects_invalid_input(beta: torch.Tensor, r: torch.Tensor, error: type) -> None:
    with pytest.raises(error):
        integrals(beta, r)


def test_rule_parameter_validation() -> None:
    with pytest.raises(ValueError, match="quad level"):
        qd.tanh_sinh_rule(0, 4.0)
    with pytest.raises(ValueError, match="quad trunc"):
        qd.tanh_sinh_rule(6, 7.0)

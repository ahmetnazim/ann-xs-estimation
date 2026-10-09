"""Theorem 8.2 and Corollary 8.3: uniform Jacobian bounds of the physics layer (Phase 7, Step 4).

Oracle: the docs/2 section 8 bounds, evaluated in float64 from the params bands ``Delta_n``,
``Delta_r`` and ``r_min`` (never from the code under test). Jacobians are autograd Jacobians of
``(w, ell, t)`` and ``ln A`` with respect to the network coordinates ``(z_W, z_n, z_r)`` through
the logistic maps of docs/2 7.6 and Phi (rows are independent, Prop 3.4, so the gradient of a
batch sum is the per-row partial).

Floating-point operationalizations (docs/3 section 12): O5 rows (float ``eta >= 1 - 4 eps``,
checked against mpmath) may reach the eta -> 1 end of each bound within the O8 interval bounds
(P7-B1 (a)); where float ``sigma(z)`` rounds to 0 or 1 the transform derivative is exactly 0
(O9, P7-B2 (a); docs/2 Prop 3.6 table), elsewhere the strict bounds hold.

The E8 number ``||J||_F < 2.349`` uses the provisional bands of docs/2 1.3; the Corollary 8.3
bound is evaluated here at the params bands instead.
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from torch import Tensor

from domain_strategies import (
    BATCH_SIZES,
    O8_DW_MAX,
    O8_DZN_MAX,
    O8_LNA_MAX,
    QUAD_FORMS,
    R_MIN,
    SOLVER,
    TR,
    LogitRows,
    closed_end_rows,
    fixed_logit_rows,
    in_form,
    logit_rows,
    require,
    saturated,
)
from src.physics.manning_layer import PhiConfig, phi
from src.physics.shape import t_of_r

pytestmark = pytest.mark.physics

PHI = PhiConfig.from_solver(SOLVER)
DN, DR = TR.delta_n, TR.delta_r
T8 = "docs/2 Theorem 8.2"
#: Corollary 8.3 with Omega = I: ||J||_F^2 bound (docs/2 section 8), at the params bands.
JF2_BOUND = 1.0 + 25.0 / 9.0 + DN**2 / 16.0 + DR**2 / 9.0 + DR**2 / (16.0 * (1.0 + R_MIN) ** 2)
JF2_BOUND_O8 = (
    1.0 + O8_DW_MAX**2 + O8_DZN_MAX**2 + DR**2 / 9.0 + DR**2 / (16.0 * (1.0 + R_MIN) ** 2)
)
#: Fast-suite budget of the trimmed Corollary 8.3 test; the full budget runs under ``slow``.
FAST_EXAMPLES = max(10, settings.default.max_examples // 4)


def jacobian(lr: LogitRows) -> dict[str, Tensor]:
    """Autograd partials of ``ell``, ``t`` and ``ln A`` w.r.t. ``(z_W, z_n, z_r)``."""
    z_w, z_n, z_r = (x.clone().requires_grad_() for x in (lr.z_w, lr.z_n, lr.z_r))
    r = TR.r(z_r)
    out = phi(z_w, r, TR.nu(z_n), lr.q, lr.s, PHI).out
    ell, ln_a = out[:, 0], out[:, 2]
    t = t_of_r(r)  # M5's shape output mu[:, 2] (docs/2 7.1)
    g_ell = torch.autograd.grad(ell.sum(), (z_w, z_n, z_r), retain_graph=True)
    g_t = torch.autograd.grad(t.sum(), (z_w, z_n, z_r), retain_graph=True, allow_unused=True)
    (g_a,) = torch.autograd.grad(ln_a.sum(), z_w)
    structural = "t = t(r(z_r)) depends on z_r only (docs/2 Theorem 8.2, J of (w, ell, t))"
    assert g_t[0] is None, f"J[2, 0] = d t/d z_W is not structurally 0: {structural}"
    assert g_t[1] is None, f"J[2, 1] = d t/d z_n is not structurally 0: {structural}"
    assert g_t[2] is not None, f"J[2, 2] = d t/d z_r is missing from the graph: {structural}"
    return {
        "ell": ell.detach(),
        "dw": g_ell[0],
        "dn": g_ell[1],
        "dr": g_ell[2],
        "dt": g_t[2],
        "dlna": g_a,
        "r": r.detach(),
    }


def check_theorem_8_2(lr: LogitRows) -> None:
    """All Theorem 8.2 bounds on one batch (strict, with the O8 and O9 rules)."""
    rows = lr.rows()
    j = jacobian(lr)
    dw, dn, dr, dt, dlna, r = j["dw"], j["dn"], j["dr"], j["dt"], j["dlna"], j["r"]
    at_end = closed_end_rows(j["ell"], lr.z_w, r, rows)  # O5 rows (mpmath-checked)
    sat_n, sat_r = saturated(lr.z_n), saturated(lr.z_r)
    # d ell / d z_W in (-5/3, -3/5]; O5 rows down to -O8_DW_MAX.
    lo_ok = (dw > -5.0 / 3.0) | (at_end & (dw >= -O8_DW_MAX))
    require(
        lo_ok & (dw <= -0.6),
        torch.maximum(-5.0 / 3.0 - dw, dw + 0.6),
        "in (-5/3, -3/5] (O8 on O5 rows)",
        "d ell/d z_W",
        T8,
        rows,
    )
    # 0 < d ell / d z_n <= Delta_n / 4; exactly 0 where sigma(z_n) rounds (O9).
    pos_ok = torch.where(sat_n, dn == 0.0, dn > 0.0)
    require(pos_ok, dn.abs(), "> 0 (= 0 where sigma rounds, O9)", "d ell/d z_n sign", T8, rows)
    hi_ok = (dn <= DN / 4.0) | (at_end & (dn <= O8_DZN_MAX))
    require(hi_ok, dn - DN / 4.0, f"<= Delta_n/4 = {DN / 4.0!r} (O8)", "d ell/d z_n", T8, rows)
    # |d ell / d z_r| <= (2r + 1) Delta_r / (6 (r + 1)) < Delta_r / 3.
    b_r = (2.0 * r + 1.0) * DR / (6.0 * (r + 1.0))
    require(dr.abs() <= b_r, dr.abs() - b_r, "<= (2r+1) Delta_r/(6(r+1))", "d ell/d z_r", T8, rows)
    require(b_r < DR / 3.0, b_r - DR / 3.0, "< Delta_r/3", "z_r bound", T8, rows)
    # Wide-channel form: |d ell / d z_r| <= a (2r + 1) Delta_r / (3r) (first bound of Prop 2.8).
    a = torch.exp(j["ell"] - lr.z_w)
    b_wide = a * (2.0 * r + 1.0) * DR / (3.0 * r)
    require(
        dr.abs() <= b_wide, dr.abs() - b_wide, "<= a(2r+1) Delta_r/(3r)", "d ell/d z_r", T8, rows
    )
    # |d t / d z_r| <= Delta_r / (4 (1 + r_min)); exactly 0 where sigma(z_r) rounds (O9).
    b_t = DR / (4.0 * (1.0 + R_MIN))
    require(dt.abs() <= b_t, dt.abs() - b_t, f"<= {b_t!r}", "d t/d z_r", T8, rows)
    # d t / d z_r = -Delta_r sigma (1 - sigma) / (r + 1) (proof of Theorem 8.2): exactly 0 where
    # sigma(z_r) rounds to 0 or 1 (O9) and non-zero elsewhere. d ell / d z_r = 0 too there (O9).
    ok = torch.where(sat_r, dt == 0.0, dt != 0.0)
    require(ok, dt.abs(), "!= 0 (= 0 where sigma rounds, O9)", "d t/d z_r support", T8, rows)
    require(~sat_r | (dr == 0.0), dr.abs(), "= 0 where sigma rounds (O9)", "d ell/d z_r", T8, rows)
    # |d ln A / d z_W| <= 2/3, and d ln A / d w <= 2/5 (Prop 3.3 derived outputs).
    a_ok = (dlna.abs() <= 2.0 / 3.0) | (at_end & (dlna.abs() <= O8_LNA_MAX))
    require(a_ok, dlna.abs() - 2.0 / 3.0, "<= 2/3 (O8 on O5 rows)", "|d ln A/d z_W|", T8, rows)
    require(dlna <= 0.4, dlna - 0.4, "<= 2/5", "d ln A/d w", "docs/2 Prop 3.3", rows)
    # Corollary 8.3: ||J||_F^2 = 1 + dw^2 + dn^2 + dr^2 + dt^2 (J of (w, ell, t)).
    jf2 = 1.0 + dw**2 + dn**2 + dr**2 + dt**2
    jf_ok = (jf2 <= JF2_BOUND) | (at_end & (jf2 <= JF2_BOUND_O8))
    require(jf_ok, jf2 - JF2_BOUND, f"<= {JF2_BOUND!r}", "||J||_F^2", "docs/2 Cor 8.3", rows)


@given(lr=logit_rows())
def test_theorem_8_2_bounds(lr: LogitRows) -> None:
    """docs/2 Theorem 8.2 on hypothesis rows of D in logit coordinates (atoms z = +/-40)."""
    check_theorem_8_2(lr)


def test_theorem_8_2_uniform_over_aspect_ratio() -> None:
    """docs/2 Theorem 8.2 is uniform in W, Q_E, S and the aspect ratio: deterministic sweep.

    q from ln 1e-4 to 11.5 (13 points) at W in {0.1, 1, 5000} m, S in {S_floor, 0.1} and the
    logit atoms z in {-40, 0, 40}; beta spans the wide-channel limit to the O5 closed end.
    """
    q = torch.linspace(math.log(1e-4), 11.5, 13, dtype=torch.float64)
    cols = torch.cartesian_prod(
        torch.log(torch.tensor([0.1, 1.0, 5000.0], dtype=torch.float64)),
        torch.tensor([-40.0, 0.0, 40.0], dtype=torch.float64),
        torch.tensor([-40.0, 0.0, 40.0], dtype=torch.float64),
        q,
        torch.log(torch.tensor([1e-5, 0.1], dtype=torch.float64)),
    )
    for chunk in cols.split(256):
        check_theorem_8_2(LogitRows(*chunk.unbind(1)))


@pytest.mark.parametrize("form", sorted(QUAD_FORMS))
@pytest.mark.parametrize("b", BATCH_SIZES)
def test_theorem_8_2_batch_sizes_and_quadrature_forms(b: int, form: str) -> None:
    """Theorem 8.2 for B in {1, 16, 64, 256} on each quadrature form (docs/2 3.5: u-form
    r >= 1, tau-form r < 1), logit atoms in the first rows (PHASE_07 refinement 1; P7-D6).
    """
    lr = fixed_logit_rows(b, form, seed=3000 + b)
    rows = lr.rows()
    side = in_form(rows.r, form)
    require(side, (rows.r - 1.0).abs(), f"{form}-form side", "fixture r", "docs/2 3.5", rows)
    check_theorem_8_2(lr)


def check_corollary_8_3(lr: LogitRows) -> None:
    """Corollary 8.3 (Omega = I): ||grad_z L_i|| <= C_Phi ||e_i|| for L_i = |e_i|^2.

    ``e_i = (w - y_w, ell - y_ell, t - y_t)`` with seeded targets; ``C_Phi = 2 ||J||_F bound``.
    """
    rows = lr.rows()
    g = torch.Generator().manual_seed(lr.z_w.numel())
    y = 3.0 * torch.randn(lr.z_w.numel(), 3, dtype=torch.float64, generator=g)
    z_w, z_n, z_r = (x.clone().requires_grad_() for x in (lr.z_w, lr.z_n, lr.z_r))
    r = TR.r(z_r)
    ell = phi(z_w, r, TR.nu(z_n), lr.q, lr.s, PHI).ell
    e = torch.stack([z_w, ell, t_of_r(r)], dim=1) - y
    loss = (e**2).sum()
    grads = torch.autograd.grad(loss, (z_w, z_n, z_r))
    norm = torch.stack(grads, dim=1).norm(dim=1)
    at_end = closed_end_rows(ell.detach(), lr.z_w, r.detach(), rows)
    c_phi = 2.0 * torch.where(at_end, math.sqrt(JF2_BOUND_O8), math.sqrt(JF2_BOUND))
    bound = c_phi * e.detach().norm(dim=1)
    require(norm <= bound, norm - bound, "<= C_Phi ||e||", "||grad_z L||", "docs/2 Cor 8.3", rows)


@given(lr=logit_rows(max_size=16))
@settings(max_examples=FAST_EXAMPLES)
def test_corollary_8_3_bounded_gradient(lr: LogitRows) -> None:
    """Corollary 8.3 bounded gradient (docs/2 section 8); fast-suite budget."""
    check_corollary_8_3(lr)


@pytest.mark.slow
@given(lr=logit_rows(max_size=16))
def test_corollary_8_3_bounded_gradient_full_budget(lr: LogitRows) -> None:
    """docs/2 Corollary 8.3 at the full profile budget (ci 200, thorough 5000; P7-D6)."""
    check_corollary_8_3(lr)

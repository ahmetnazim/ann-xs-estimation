"""Theorem 4.1 invariants 1-7 and the conservation identities of docs/2 (Phase 7, Step 2).

Oracle: docs/2 statements only; expected values are closed forms, bounds or mpmath
(``reference_oracles``), never the code under test. Tolerances: P7-D5 (c) (ADL-101) and the
docs/3 section 12 amendments (invariant 7 by P7-Q3 (a); operationalizations O1-O4 there).

Covered: Phi directly; M5 and its ablations (A1, A2, A3-SP, A3-SR, w_A = 1); M3 and M6 stage 3
through Phi; the ensemble point prediction ``exp(mu*)`` (Proposition 9.2); Propositions 2.3, 2.7,
3.2 and 6.2; the section 9 mixture (moments, exact quantiles). M4 and M2-MLP are not claimed to
satisfy Theorem 4.1 (docs/2 section 7.7 and 9.1); their documented status is tested instead.

PHASE_07 refinement: explicit parametrization over B in {1, 16, 64, 256}, every M5 ablation
switch and both quadrature forms (docs/2 3.5: u-form r >= 1, tau-form r < 1); the Phi property
test runs at a quarter budget in the fast suite and at the full profile budget under ``slow``
(P7-D6; ``thorough`` = 5000 examples at the gate).
"""

from __future__ import annotations

import math

import mpmath as mp
import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import norm
from torch import Tensor

import reference_oracles as ro
from domain_strategies import (
    BATCH_SIZES,
    BLIND_IDX,
    D_MIN,
    EPS,
    ETA_MAX,
    LN_D50,
    LN_S_FLOOR,
    N_FEATURES,
    N_MAX,
    N_MIN,
    PROTOCOL,
    QUAD_FORMS,
    R_MAX,
    R_MIN,
    SLOPE_MAX,
    SOLVER,
    SPOT_ROWS,
    TAU_L,
    TR,
    Rows,
    closed_end_rows,
    fixed_rows,
    in_form,
    model_batch,
    phi_rows,
    require,
    require_equal,
    require_in,
    require_le,
)
from src.models.ensemble import (
    ensemble_roughness,
    mixture_moments,
    mixture_quantile,
    stack_members,
)
from src.models.heads import ModelOutput
from src.models.m2_mlp import M2MLP, M2Config
from src.models.m3_physics import M3Config, m3_physics
from src.models.m4_soft import M4Config, M4Soft
from src.models.m5_hard import M5Config, M5Hard
from src.models.m6_two_stage import M6Config, m6_stage3
from src.physics.manning_layer import PhiConfig, manning_residual, phi, phi_inverse_nu
from src.physics.quadrature import perimeter_fm1
from src.physics.shape import section_terms

pytestmark = pytest.mark.physics

BACKBONE = PROTOCOL.model.backbone
PHI = PhiConfig.from_solver(SOLVER)
PHI_A1 = PhiConfig.from_solver(SOLVER, kappa=False)
RHO_BOUND = 5.0 / 3.0 * TAU_L  # Theorem 4.1 invariant 3
ABS_LOG = 1e-14  # P7-D5: invariant 2 and exp(mu*) in log space (absolute)
REL_LIN = 1e-13  # P7-D5: invariants 1-2 in linear space (relative)
REL_INV4 = 1e-14  # P7-D5: invariant 4
MODEL_EXAMPLES = max(10, settings.default.max_examples // 8)
SLOW_EXAMPLES = max(10, settings.default.max_examples // 4)
#: Fast-suite budget of the trimmed Phi property test; the full budget runs under ``slow``.
FAST_EXAMPLES = max(10, settings.default.max_examples // 4)
NU_HI = TR.nu_min + TR.delta_n
LN_R_HI = TR.ln_r_min + TR.delta_r


# ----------------------------------------------------------------------------- shared checks
def check_theorem_4_1(
    label: str,
    rows: Rows,
    *,
    w: Tensor,
    ell: Tensor,
    t: Tensor,
    ln_a: Tensor,
    nu: Tensor,
    r: Tensor,
    cfg: PhiConfig,
    rho_bound: float | None = RHO_BOUND,
    nu_band: bool = True,
    dldq: Tensor | None = None,
) -> None:
    """Theorem 4.1 on canonical outputs ``[w, ell, t, ln A]`` with ``ln V = q - ln A``.

    ``rho_bound=None`` checks invariant 3 as the A1 closed form instead (bit-equal).
    """
    q, s = rows.q, rows.s
    ref = f"docs/2 Theorem 4.1 ({label})"
    # 1. A = W dbar: ln A bit-equal to w + ell; relative <= 1e-13 in linear space.
    require_equal(ln_a, w + ell, "inv. 1 ln A = w + ell", ref, rows)
    a_lin = torch.exp(ln_a)
    require_le(
        (a_lin - torch.exp(w) * torch.exp(ell)).abs() / a_lin, REL_LIN, "inv. 1 linear", ref, rows
    )
    # 2. Q_E = A V with ln V = q - ln A; relative <= 1e-13 in linear space.
    ln_v = q - ln_a
    q_lin = torch.exp(q)
    require_le((a_lin * torch.exp(ln_v) - q_lin).abs() / q_lin, REL_LIN, "inv. 2 linear", ref, rows)
    # 3. |rho| <= (5/3) tau_l at the returned ell; A1: ell is the closed form.
    if rho_bound is None:
        require_equal(ell, 0.6 * (q + nu - w - 0.5 * s), "inv. 3 (A1 closed form)", ref, rows)
    else:
        res = manning_residual(ell.detach(), w.detach(), r.detach(), nu.detach(), q, s, cfg)
        require_le(res.abs(), rho_bound, "inv. 3 |rho|", ref, rows)
    # 4. d_max / dbar = exp(t) = 1 + 1/r, strictly > 1.
    ratio = torch.exp(t)
    one_r = 1.0 + 1.0 / r
    require_le((ratio - one_r).abs() / one_r, REL_INV4, "inv. 4 d_max/dbar = 1 + 1/r", ref, rows)
    require(ratio > 1.0, 1.0 - ratio, "> 1", "inv. 4 strict", ref, rows)
    # 5. positivity: W, dbar, A, n, r, V, d_max finite and > 0.
    for name, x in (
        ("W", torch.exp(w)),
        ("dbar", torch.exp(ell)),
        ("A", a_lin),
        ("n", torch.exp(nu)),
        ("r", r),
        ("V", torch.exp(ln_v)),
        ("d_max", torch.exp(ell + t)),
    ):
        xd = x.detach()
        require(torch.isfinite(xd) & (xd > 0), -xd, "finite and > 0", f"inv. 5 {name}", ref, rows)
    # 6. d ln dbar / d ln Q in [3/5, 1); closed end [1, 1/(1 - 3 eps)] only under O5.
    if dldq is not None:
        at_end = torch.zeros_like(dldq, dtype=torch.bool)
        if bool((dldq >= 1.0).any()):
            at_end = closed_end_rows(ell, w, r, rows) & (dldq <= SLOPE_MAX)
        require(
            (dldq >= 0.6) & ((dldq < 1.0) | at_end),
            torch.maximum(0.6 - dldq, dldq - 1.0),
            "in [3/5, 1) (closed end only under O5)",
            "inv. 6",
            ref,
            rows,
        )
    # 7. own band exactly (logistic or Pi); params bounds within 2 ulp (docs/3 section 12).
    if nu_band:
        require_in(nu, TR.nu_min, NU_HI, "inv. 7 nu in own band", ref, rows)
    require_in(torch.exp(nu), N_MIN, N_MAX, "inv. 7 n vs params", ref, rows, ulps=2)
    require_in(r, R_MIN, R_MAX, "inv. 7 r vs params", ref, rows, ulps=2)


def dl_dq(rows: Rows, cfg: PhiConfig) -> tuple[Tensor, Tensor]:
    """Phi outputs and per-row ``d ell / d q`` by autograd (rows are independent, Prop 3.4)."""
    q = rows.q.clone().requires_grad_()
    out = phi(rows.w, rows.r, rows.nu, q, rows.s, cfg).out
    (g,) = torch.autograd.grad(out[:, 0].sum(), q)
    return out.detach(), g


# ----------------------------------------------------------------------------- Phi
def check_phi_theorem_4_1(rows: Rows) -> None:
    """Theorem 4.1 invariants 1-7 on Phi itself, plus its own (B, 4) columns (docs/2 3.1)."""
    out, g = dl_dq(rows, PHI)
    shape_ok = tuple(out.shape) == (rows.n, 4) and out.dtype == torch.float64
    assert shape_ok, (
        f"Phi out {tuple(out.shape)} {out.dtype} != ({rows.n}, 4) float64 "
        "[ell, ln d_max, ln A, ln V] (docs/2 3.1; PHASE_07 requirement 3)"
    )
    ell, ln_dmax, ln_a, ln_v = out.unbind(1)
    w, q = rows.w, rows.q
    ref = "docs/2 section 3.1, Theorem 4.1"
    require_equal(ln_a, w + ell, "inv. 1 ln A = w + ell (bit)", ref, rows)
    require_equal(ln_v, (q - w) - ell, "inv. 2 ln V = (q - w) - ell (bit)", ref, rows)
    require_le((ln_v - (q - ln_a)).abs(), ABS_LOG, "inv. 2 ln V vs q - ln A", ref, rows)
    check_theorem_4_1(
        "Phi", rows, w=w, ell=ell, t=ln_dmax - ell, ln_a=ln_a, nu=rows.nu, r=rows.r, cfg=PHI,
        dldq=g,
    )  # fmt: skip
    # Theorem 4.1 note: d ln V / d q = 1 - d ell / d q in (0, 2/5]; closed end only under O5.
    dv = 1.0 - g
    at_end = torch.zeros_like(dv, dtype=torch.bool)
    if bool((dv <= 0).any()):
        at_end = closed_end_rows(ell, w, rows.r, rows) & (dv >= 1.0 - SLOPE_MAX)
    require(
        ((dv > 0) | at_end) & (dv <= 0.4),
        torch.maximum(-dv, dv - 0.4),
        "in (0, 2/5] (closed end only under O5)",
        "dlnV/dq",
        ref,
        rows,
    )


@given(rows=phi_rows())
@settings(max_examples=FAST_EXAMPLES)
def test_phi_theorem_4_1(rows: Rows) -> None:
    """Theorem 4.1 invariants 1-7 on Phi (docs/2 3.1, Theorem 4.1); fast-suite budget."""
    check_phi_theorem_4_1(rows)


@pytest.mark.slow
@given(rows=phi_rows())
def test_phi_theorem_4_1_full_budget(rows: Rows) -> None:
    """Theorem 4.1 on Phi at the full profile budget (ci 200, thorough 5000; P7-D6)."""
    check_phi_theorem_4_1(rows)


@pytest.mark.parametrize("form", sorted(QUAD_FORMS))
@pytest.mark.parametrize("b", BATCH_SIZES)
def test_phi_theorem_4_1_batch_sizes_and_quadrature_forms(b: int, form: str) -> None:
    """Theorem 4.1 on Phi for B in {1, 16, 64, 256} on each quadrature form (docs/2 3.5:
    u-form r >= 1, tau-form r < 1), P7-D6 atoms in the first rows (PHASE_07 refinement 1).
    """
    rows = fixed_rows(b, form, seed=b)
    side = in_form(rows.r, form)
    require(side, (rows.r - 1.0).abs(), f"{form}-form side", "fixture r", "docs/2 3.5", rows)
    check_phi_theorem_4_1(rows)


@given(rows=phi_rows())
def test_phi_a1_closed_form(rows: Rows) -> None:
    """Ablation A1 (docs/2 3.2): ell bit-equal to (3/5)(q + nu - w - s/2); slope exactly 3/5."""
    out, g = dl_dq(rows, PHI_A1)
    ell = out[:, 0]
    check_theorem_4_1(
        "Phi A1", rows, w=rows.w, ell=ell, t=out[:, 1] - ell, ln_a=out[:, 2], nu=rows.nu,
        r=rows.r, cfg=PHI_A1, rho_bound=None, dldq=g,
    )  # fmt: skip
    require_equal(g, torch.full_like(g, 0.6), "A1 d ell/dq = 3/5", "docs/2 3.2, inv. 6", rows)


@given(rows=phi_rows())
def test_props_2_3_and_2_7_at_the_root(rows: Rows) -> None:
    """Prop 2.3 (W/(W + 2 d_max) < kappa < 1, i.e. 0 < F - 1 < 2 delta) and Prop 2.7/3.1.

    ``eta in [0, 1)`` and ``D = 5/3 - (2/3) eta in (1, 5/3]`` at the converged depth; the
    closed ends ``eta in [1, 1 + 4 eps]``, ``D in [1 - (8/3) eps, 1]`` only under O5
    (docs/3 section 12, P7-A3 (a)).
    """
    ell = phi(*rows.args(), PHI).ell
    terms = section_terms(ell, rows.w, rows.r, level=SOLVER.quad_level, trunc=SOLVER.quad_trunc)
    delta = terms.a * (rows.r + 1.0) / rows.r  # d_max / W (Prop 2.1)
    fm1, eta = terms.fm1, terms.eta
    p23, p27, p31 = "docs/2 Prop 2.3", "docs/2 Prop 2.7", "docs/2 Prop 3.1"
    require(fm1 > 0, -fm1, "> 0", "Prop 2.3 kappa < 1", p23, rows)
    require(fm1 < 2 * delta, fm1 - 2 * delta, "< 2 delta", "Prop 2.3 lower bound", p23, rows)
    d = 5.0 / 3.0 - (2.0 / 3.0) * eta
    at_end = torch.zeros_like(eta, dtype=torch.bool)
    if bool(((eta >= 1.0) | (d <= 1.0)).any()):
        at_end = closed_end_rows(ell, rows.w, rows.r, rows)
    eta_ok = (eta >= 0) & ((eta < 1) | (at_end & (eta <= ETA_MAX)))
    require(eta_ok, torch.maximum(-eta, eta - 1), "in [0, 1) (O5 closed end)", "eta", p27, rows)
    d_ok = ((d > 1) | (at_end & (d >= D_MIN))) & (d <= 5.0 / 3.0)
    require(d_ok, torch.maximum(1 - d, d - 5 / 3), "in (1, 5/3] (O5 closed end)", "D", p31, rows)


def test_o5_reproducer_rows() -> None:
    """P7-A1/A2/A3 reproducers (docs/3 section 12, O5): r = 1/2, n = 0.6, S = S_floor.

    W = 1 m, q = 11 (beta ~ 1.1e8): exact 1 - eta = 2.6e-17, float eta = 1, slope = 1.
    W = e^-2.28125 m, q = 4: exact 1 - eta = 1.65e-16, computed eta = 1.0 (0.74 eps off).
    W = e^-2.25 m, q = 6: exact 1 - eta = 3.6e-18, computed eta = 1 + eps (P7-A3).
    All three are accurate to 4 eps and accepted. W = 1 m, q = 8 (beta ~ 5.6e6,
    1 - eta = 1.06e-14) is not near the closed end and must stay strictly inside.
    """
    nu_hi = TR.nu_min + TR.delta_n
    rows = Rows(
        w=torch.tensor([0.0, -2.28125, -2.25, 0.0], dtype=torch.float64),
        r=torch.full((4,), 0.5, dtype=torch.float64),
        nu=torch.full((4,), nu_hi, dtype=torch.float64),
        q=torch.tensor([11.0, 4.0, 6.0, 8.0], dtype=torch.float64),
        s=torch.full((4,), LN_S_FLOOR, dtype=torch.float64),
    )
    out, g = dl_dq(rows, PHI)
    allowed = closed_end_rows(out[:, 0], rows.w, rows.r, rows)
    want = [True, True, True, False]
    wrong = [i for i, (a, b) in enumerate(zip(allowed.tolist(), want, strict=True)) if a != b]
    assert not wrong, (
        f"O5 classification {allowed.tolist()} != {want} (docs/3 section 12, O5); "
        + "; ".join(rows.describe(i) for i in wrong)
    )
    p_inv6 = "docs/2 Theorem 4.1 inv. 6; docs/3 section 12 O5"
    ok_end = (g >= 1.0) & (g <= SLOPE_MAX)
    mag_end = torch.maximum(1.0 - g, g - SLOPE_MAX)
    ok = torch.where(torch.arange(4) < 3, ok_end, g < 1.0)
    mag = torch.where(torch.arange(4) < 3, mag_end, g - 1.0)
    require(ok, mag, f"rows 0-2 in [1, {SLOPE_MAX!r}], row 3 < 1", "d ell/dq", p_inv6, rows)


def _picard_history(rows: Rows, cfg: PhiConfig) -> list[Tensor]:
    """Independent Picard loop (docs/2 Prop 3.2 (2)) on all rows, without freezing."""
    w, r, nu, q, s = rows.args()
    l0 = 0.6 * (q + nu - w - 0.5 * s)
    hist = [l0]
    done = torch.zeros_like(l0, dtype=torch.bool)
    for _ in range(cfg.k_max):
        beta = 2.0 * (r + 1.0) * torch.exp(hist[-1] - w)
        fm1 = perimeter_fm1(beta, r, level=cfg.quad_level, trunc=cfg.quad_trunc)
        hist.append(l0 + 0.4 * torch.log1p(fm1))
        done |= (hist[-1] - hist[-2]).abs() <= 1.5 * cfg.tau_l
        if bool(done.all()):
            break
    return hist


@given(rows=phi_rows())
@settings(max_examples=SLOW_EXAMPLES)
def test_prop_3_2_bracket_contraction_monotone(rows: Rows) -> None:
    """Prop 3.2 (1) bracket, (2) contraction, (3) a-posteriori bound, (4) monotone iterates.

    Also Prop 3.4 (c): Phi's per-row stopping index and value equal those of an independent
    loop (bit-equal). Contraction uses the rounding allowance O2 of docs/3 section 12.
    """
    res = phi(*rows.args(), PHI)
    hist = torch.stack(_picard_history(rows, PHI))  # (m+1, B)
    inc = hist[1:] - hist[:-1]
    stop = (inc.abs() <= 1.5 * TAU_L).to(torch.int64).argmax(dim=0) + 1  # first m, per row
    idx = torch.arange(rows.n)
    ref = "docs/2 Prop 3.2"
    require_equal(res.iterations, stop, "stopping index (Prop 3.4 c)", "docs/2 Prop 3.4", rows)
    require_equal(res.ell, hist[stop, idx], "per-row value (Prop 3.4 c)", "docs/2 Prop 3.4", rows)
    l0, ell = hist[0], res.ell
    k0_abs = (hist[1] - l0) / 0.4  # |k_0| = ln F(beta_0)
    lo, hi = l0 + 0.4 * k0_abs, l0 + (2.0 / 3.0) * k0_abs
    require(
        (ell >= lo) & (ell <= hi),
        torch.maximum(lo - ell, ell - hi),
        "in bracket",
        "(1) bracket",
        ref,
        rows,
    )
    for m in range(1, int(stop.max()) + 1):
        live = stop >= m
        d = inc[m - 1]
        require(~live | (d >= 0), -d, ">= 0", f"(4) monotone, step {m}", ref, rows)
        post = (hist[m] - ell).abs() - ((2.0 / 3.0) * d.abs() + TAU_L)
        require(~live | (post <= 0), post, "<= 0", f"(3) a posteriori, step {m}", ref, rows)
        if m >= 2:
            prev = inc[m - 2]
            scale = torch.maximum(hist[m].abs(), hist[m - 1].abs()).maximum(hist[m - 2].abs())
            allow = 4.0 * EPS * torch.maximum(scale, torch.ones_like(scale))
            excess = d.abs() - (0.4 * prev.abs() + allow)
            require(~live | (excess <= 0), excess, "<= 0", f"(2) contraction, step {m}", ref, rows)


@given(rows=phi_rows())
def test_prop_6_2_round_trip(rows: Rows) -> None:
    """docs/2 Prop 6.2: Phi^-1(Phi(nu)) = nu to (5/3) tau_l (identity nu - Phi^-1 = -rho)."""
    ell = phi(*rows.args(), PHI).ell
    nu_back = phi_inverse_nu(ell, rows.w, rows.r, rows.q, rows.s, PHI)
    require_le((nu_back - rows.nu).abs(), RHO_BOUND, "round trip", "docs/2 Prop 6.2", rows)
    ell_a1 = phi(*rows.args(), PHI_A1).ell
    back_a1 = phi_inverse_nu(ell_a1, rows.w, rows.r, rows.q, rows.s, PHI_A1)
    scale = rows.q.abs() + rows.nu.abs() + rows.w.abs() + rows.s.abs()
    require_le(
        (back_a1 - rows.nu).abs() - 8 * EPS * scale, 0.0, "A1 round trip", "docs/2 Prop 6.2", rows
    )


@given(z=st.lists(st.floats(-40.0, 40.0) | st.sampled_from([-40.0, 40.0]), min_size=1, max_size=32))
def test_inv_7_transforms_own_band(z: list[float]) -> None:
    """Invariant 7 / docs/2 7.6: nu and ln r lie in the transform's own closed band exactly."""
    zt = torch.tensor(z, dtype=torch.float64)
    rows = Rows(w=zt, r=torch.ones_like(zt), nu=zt, q=zt, s=zt)  # for the row report only
    require_in(TR.nu(zt), TR.nu_min, NU_HI, "nu band", "docs/2 7.6, inv. 7", rows)
    require_in(TR.log_r(zt), TR.ln_r_min, LN_R_HI, "ln r band", "docs/2 7.6, inv. 7", rows)
    require_in(TR.r(zt), R_MIN, R_MAX, "r vs params", "docs/3 section 12 (P7-Q3)", rows, ulps=2)


# ----------------------------------------------------------------------------- inv. 3 vs mpmath


@pytest.mark.parametrize(("tau", "label"), [(TAU_L, "params tau_l"), (1e-14, "tight")])
def test_inv_3_against_mpmath_residual(tau: float, label: str) -> None:
    """Invariant 3 with an exact (mpmath, 40 digits) rho at Phi's float64 ell (docs/2 eq. 3.1)."""
    rows = Rows.from_si(SPOT_ROWS)
    cfg = PhiConfig(tau_l=tau, k_max=80, quad_level=SOLVER.quad_level, quad_trunc=SOLVER.quad_trunc)
    ell = phi(*rows.args(), cfg).ell
    rho_mp = torch.tensor(
        [
            abs(
                float(
                    ro.rho(*(float(x[i]) for x in (ell, rows.w, rows.r, rows.nu, rows.q, rows.s)))
                )
            )
            for i in range(rows.n)
        ],
        dtype=torch.float64,
    )
    require_le(rho_mp, 5.0 / 3.0 * tau, f"inv. 3 |rho_mp| ({label})", "docs/2 Theorem 4.1", rows)


# ----------------------------------------------------------------------------- M5 and ablations
M5_VARIANTS: dict[str, M5Config] = {
    "primary": M5Config.from_protocol(PROTOCOL),
    "A1": M5Config.from_protocol(PROTOCOL, kappa=False),
    "A2": M5Config.from_protocol(PROTOCOL, a2=True),
    "A3-SP": M5Config.from_protocol(PROTOCOL, prior="SP"),
    "A3-SR": M5Config.from_protocol(PROTOCOL, prior="SR"),
    "w_A=1": M5Config.from_protocol(PROTOCOL, w_a=1.0),
}


def build_m5(variant: str, seed: int) -> M5Hard:
    torch.manual_seed(seed)
    cfg = M5_VARIANTS[variant]
    return M5Hard(N_FEATURES, BLIND_IDX, cfg, BACKBONE, ln_d50=LN_D50)


def m5_forward(net: M5Hard, rows: Rows, seed: int) -> tuple[ModelOutput, Tensor]:
    batch = model_batch(rows, seed)
    q = rows.q.clone().requires_grad_()
    batch["phys"] = torch.stack([q, rows.s], dim=1)
    out = net(batch)
    (g,) = torch.autograd.grad(out.mu[:, 1].sum(), q)
    return out, g


def check_m5(variant: str, net: M5Hard, rows: Rows, seed: int) -> None:
    """Run M5 on ``rows`` (``q, s``) and check Theorem 4.1 on its own heads' ``(w, r, nu)``."""
    out, g = m5_forward(net, rows, seed)
    mu = out.mu.detach()
    nu, r = out.extras["nu"].detach(), out.extras["r"].detach()
    rows_m = Rows(w=mu[:, 0], r=r, nu=nu, q=rows.q, s=rows.s)
    cfg = net.cfg.phi
    check_theorem_4_1(
        f"M5 {variant}", rows_m, w=mu[:, 0], ell=mu[:, 1], t=mu[:, 2], ln_a=mu[:, 3],
        nu=nu, r=r, cfg=cfg, rho_bound=RHO_BOUND if cfg.kappa else None, dldq=g,
    )  # fmt: skip


@pytest.mark.parametrize("variant", sorted(M5_VARIANTS))
@given(rows=phi_rows(max_size=16), seed=st.integers(0, 2**16))
@settings(max_examples=MODEL_EXAMPLES)
def test_m5_theorem_4_1(variant: str, rows: Rows, seed: int) -> None:
    """Theorem 4.1 for M5 and the docs/2 7.10 ablations (A1, A2, A3-SP, A3-SR) and w_A = 1."""
    check_m5(variant, build_m5(variant, seed % 7), rows, seed)


@pytest.mark.parametrize("b", BATCH_SIZES)
@pytest.mark.parametrize("variant", sorted(M5_VARIANTS))
def test_m5_theorem_4_1_batch_sizes(variant: str, b: int) -> None:
    """Theorem 4.1 for M5 and every ablation switch (docs/2 7.10) at B in {1, 16, 64, 256}
    (PHASE_07 refinement 1). ``q, s`` come from :func:`fixed_rows` (S_floor atom included);
    ``(w, r, nu)`` come from M5's own heads, so the quadrature form is the network's choice.
    """
    check_m5(variant, build_m5(variant, b % 7), fixed_rows(b, "u", seed=1000 + b), b)


@pytest.mark.parametrize("variant", ["primary", "A1"])
@pytest.mark.parametrize(
    ("z_r", "z_n"), [(-40.0, -40.0), (-40.0, 40.0), (40.0, -40.0), (40.0, 40.0)]
)
def test_m5_logit_atoms(variant: str, z_r: float, z_n: float) -> None:
    """P7-D6 atoms through M5: heads forced to z = +/-40 (sigma rounds to 0 or 1); Theorem 4.1."""
    net = build_m5(variant, 0)
    with torch.no_grad():
        net.heads.full_head.weight[1].zero_()
        net.heads.full_head.bias[1] = z_r  # "z_r" is full output 1
        assert net.heads.blind_head is not None
        net.heads.blind_head.weight.zero_()
        net.heads.blind_head.bias.fill_(z_n)
    base = Rows.from_si([(20.0, 2.0, 0.035, 10.0, 1e-3)] * 4)
    s = torch.tensor([LN_S_FLOOR, -7.0, -3.0, math.log(0.1)], dtype=torch.float64)
    q = base.q + torch.tensor([-6.0, 0.0, 6.0, 11.0], dtype=torch.float64)
    check_m5(variant, net, Rows(w=base.w, r=base.r, nu=base.nu, q=q, s=s), 0)


# ----------------------------------------------------------------------------- M4, M2: status
def test_m4_documented_status() -> None:
    """M4 (docs/2 7.7): invariants 1, 4, 7 by construction; invariant 3 is not claimed.

    Manning is a training penalty on R_M; at a test reach it is controlled only by
    generalization (docs/2 7.7, "Satisfaction is on the training sample only").
    """
    torch.manual_seed(0)
    net = M4Soft(
        N_FEATURES, BLIND_IDX, M4Config.from_protocol(PROTOCOL, lambda_max=100.0), BACKBONE
    )
    rows = Rows.from_si([(20.0, 2.0, 0.035, 10.0, 1e-3), (3.0, 1.0, 0.04, 1.0, 0.01)] * 4)
    out = net(model_batch(rows, 1))
    mu, nu, r = out.mu.detach(), out.extras["nu"].detach(), out.extras["r"].detach()
    rows_m = Rows(w=mu[:, 0], r=r, nu=nu, q=rows.q, s=rows.s)
    ref = "docs/2 7.7 (M4)"
    require_equal(mu[:, 3], mu[:, 0] + mu[:, 1], "inv. 1 by construction", ref, rows_m)
    require_le(
        (torch.exp(mu[:, 2]) - (1 + 1 / r)).abs() / (1 + 1 / r), REL_INV4, "inv. 4", ref, rows_m
    )
    require_in(nu, TR.nu_min, NU_HI, "inv. 7 nu band", ref, rows_m)
    require_in(r, R_MIN, R_MAX, "inv. 7 r", ref, rows_m, ulps=2)
    r_m = out.extras["R_M"].detach()
    require(torch.isfinite(r_m), r_m.abs(), "finite", "M4 R_M", ref, rows_m)
    worst = float(r_m.abs().max())
    assert worst > RHO_BOUND, (
        f"untrained M4: max |R_M| = {worst:.3e} <= (5/3) tau_l = {RHO_BOUND:.3e}; invariant 3 is "
        "not claimed for M4 (docs/2 7.7), so a Manning-exact untrained M4 is unexpected"
    )


def test_m2_documented_status() -> None:
    """M2-MLP (docs/2 9.1, P6): canonical t = m_dmax - m_dbar exactly; A closure not enforced."""
    torch.manual_seed(0)
    net = M2MLP(N_FEATURES, M2Config.from_protocol(PROTOCOL), BACKBONE)
    rows = Rows.from_si([(20.0, 2.0, 0.035, 10.0, 1e-3)] * 8)
    out = net(model_batch(rows, 2))
    mu, nat = out.mu.detach(), out.extras["mu_native"].detach()
    require_equal(mu[:, 2], nat[:, 2] - nat[:, 1], "t = m_dmax - m_dbar", "M2 canonical", rows)
    gap = (mu[:, 3] - (mu[:, 0] + mu[:, 1])).abs()
    assert float(gap.max()) > 0.0, (
        f"M2: max |mu_A - (mu_W + mu_dbar)| = {float(gap.max()):.3e}, expected > 0 (independent "
        f"A head, closure not claimed; docs/2 9.1); {rows.describe(0)}"
    )


# ----------------------------------------------------------------------------- M3, M6 via Phi
def check_m3(rows: Rows, d50: Tensor, r_bar: float) -> None:
    """M3 (docs/2 7.9) on ``rows`` (``w, q, s``) with ``ln D50 = d50`` and ``r = r_bar``."""
    cfg = M3Config.from_protocol(PROTOCOL, r_bar=r_bar)
    out = m3_physics(rows.w, d50, rows.phys(), cfg)
    mu, nu = out.mu, out.extras["nu"]
    r = torch.full_like(nu, r_bar)
    rows_m = Rows(w=rows.w, r=r, nu=nu, q=rows.q, s=rows.s)
    check_theorem_4_1(
        "M3", rows_m, w=mu[:, 0], ell=mu[:, 1], t=mu[:, 2], ln_a=mu[:, 3], nu=nu, r=r,
        cfg=cfg.phi, rho_bound=(5.0 / 3.0 - PROTOCOL.priors.jarrett.b_j) * TAU_L,
    )  # fmt: skip
    via_phi = ~out.extras["jarrett"] | out.extras["nu_projected"]
    res = manning_residual(mu[:, 1], rows.w, r, nu, rows.q, rows.s, cfg.phi).abs()
    require_le(
        torch.where(via_phi, res, 0.0), RHO_BOUND, "inv. 3 via Phi rows", "docs/2 7.9", rows_m
    )


def check_m6(rows: Rows, gt: Tensor, gn: Tensor) -> Rows:
    """M6 stage 3 (docs/2 7.8) on ``rows`` with ``g_t = gt``, ``g_nu = gn``; returns its rows."""
    cfg = M6Config.from_protocol(PROTOCOL)
    out = m6_stage3(rows.w, gt, gn, rows.phys(), cfg)
    mu, nu, r = out.mu, out.extras["nu"], out.extras["r"]
    rows_m = Rows(w=rows.w, r=r, nu=nu, q=rows.q, s=rows.s)
    check_theorem_4_1(
        "M6 stage 3", rows_m, w=mu[:, 0], ell=mu[:, 1], t=mu[:, 2], ln_a=mu[:, 3], nu=nu, r=r,
        cfg=cfg.phi,
    )  # fmt: skip
    return rows_m


@given(
    rows=phi_rows(max_size=16),
    ln_d50=st.lists(st.floats(math.log(1e-7), math.log(4.0)), min_size=16, max_size=16),
    r_bar=st.sampled_from([R_MIN, R_MAX]) | st.floats(R_MIN, R_MAX),
)
@settings(max_examples=MODEL_EXAMPLES)
def test_m3_through_phi(rows: Rows, ln_d50: list[float], r_bar: float) -> None:
    """M3 (docs/2 7.9): Theorem 4.1 on Strickler / projected rows (through Phi).

    Jarrett in-band rows solve rho_J = 0 (Prop 7.5); since rho(ell; nu_J) = rho_J(ell), their
    bound is (5/3 - b_J) tau_l (docs/3 section 12, O3).
    """
    check_m3(rows, torch.tensor(ln_d50[: rows.n], dtype=torch.float64), r_bar)


@given(
    rows=phi_rows(max_size=16),
    g_t=st.lists(st.floats(0.01, 2.0), min_size=16, max_size=16),
    g_nu=st.lists(st.floats(math.log(1e-3), math.log(2.0)), min_size=16, max_size=16),
)
@settings(max_examples=MODEL_EXAMPLES)
def test_m6_stage3_through_phi(rows: Rows, g_t: list[float], g_nu: list[float]) -> None:
    """M6 stage 3 (docs/2 7.8): Pi(g_nu), Pi_t(g_t), ell = Phi; Theorem 4.1 at full tolerance."""
    gt = torch.tensor(g_t[: rows.n], dtype=torch.float64)
    gn = torch.tensor(g_nu[: rows.n], dtype=torch.float64)
    check_m6(rows, gt, gn)


#: Per quadrature form: M3's r_bar (the form's lower atom) and M6's g_t range inside the t band
#: [ln 1.1, ln 3]. t = ln(1 + 1/r) (docs/2 2.1), so r >= 1 iff t <= ln 2; the 1e-9 margin keeps
#: rounding on the form's side.
M3_R_BAR = {"u": 1.0, "tau": R_MIN}
M6_T_RANGE = {
    "u": (math.log(1.1), math.log(2.0) - 1e-9),
    "tau": (math.log(2.0) + 1e-9, math.log(3.0)),
}


@pytest.mark.parametrize("form", sorted(QUAD_FORMS))
@pytest.mark.parametrize("b", BATCH_SIZES)
def test_m3_m6_through_phi_batch_sizes_and_quadrature_forms(b: int, form: str) -> None:
    """M3 (docs/2 7.9) and M6 stage 3 (docs/2 7.8) through Phi for B in {1, 16, 64, 256} on
    each quadrature form (docs/2 3.5); Theorem 4.1 as in the hypothesis tests above.
    """
    rows = fixed_rows(b, form, seed=2000 + b)
    g = torch.Generator().manual_seed(b)
    d50 = torch.empty(b, dtype=torch.float64).uniform_(math.log(1e-7), math.log(4.0), generator=g)
    check_m3(rows, d50, M3_R_BAR[form])
    gt = torch.empty(b, dtype=torch.float64).uniform_(*M6_T_RANGE[form], generator=g)
    gn = torch.empty(b, dtype=torch.float64).uniform_(math.log(1e-3), math.log(2.0), generator=g)
    rows_m6 = check_m6(rows, gt, gn)
    side = in_form(rows_m6.r, form)
    require(side, (rows_m6.r - 1.0).abs(), f"{form}-form side", "M6 r", "docs/2 3.5", rows_m6)


# ----------------------------------------------------------------------------- ensemble (9.2)
@given(rows=phi_rows(max_size=8), seed=st.integers(0, 2**16))
@settings(max_examples=MODEL_EXAMPLES)
def test_ensemble_point_prediction_prop_9_2(rows: Rows, seed: int) -> None:
    """docs/2 Prop 9.2: mu*_A = mu*_W + mu*_dbar (<= 1e-14), mu*_V = q - mu*_A; n* bounds.

    Full kappa: |nu* - nu_bar| <= (2/3) ln(1 + 2 delta_max). A1: nu* = nu_bar to rounding (O4).
    """
    for variant in ("primary", "A1"):
        outs = [m5_forward(build_m5(variant, m), rows, seed)[0] for m in range(3)]
        mu, sigma = stack_members(outs)
        mix = mixture_moments(mu, sigma)
        ms = mix.mu_star
        ref = f"docs/2 Prop 9.2 ({variant})"
        require_le((ms[:, 3] - (ms[:, 0] + ms[:, 1])).abs(), ABS_LOG, "mu*_A closure", ref, rows)
        ln_v = rows.q - ms[:, 3]
        q_lin = torch.exp(rows.q)
        err = (torch.exp(ms[:, 3]) * torch.exp(ln_v) - q_lin).abs() / q_lin
        require_le(err, REL_LIN, "exp(mu*) Q = A V", ref, rows)
        cfg = PHI if variant == "primary" else PHI_A1
        nu_star = ensemble_roughness(ms, rows.phys(), cfg)
        nu_bar = torch.stack([o.extras["nu"].detach() for o in outs]).mean(dim=0)
        if variant == "primary":
            deltas = [torch.exp(o.mu[:, 1] + o.mu[:, 2] - o.mu[:, 0]).detach() for o in outs]
            deltas.append(torch.exp(ms[:, 1] + ms[:, 2] - ms[:, 0]))
            d_max = torch.stack(deltas).max(dim=0).values
            # nu* - nu_bar = (2/3)(k* - k_bar) + mean_m rho_m with |rho_m| <= (5/3) tau_l (O6).
            scale = (5 / 3) * ms[:, 1].abs() + ms[:, 0].abs() + 0.5 * rows.s.abs() + rows.q.abs()
            bound = (2.0 / 3.0) * torch.log1p(2.0 * d_max) + RHO_BOUND + 16 * EPS * scale
            excess = (nu_star - nu_bar).abs() - bound
            require(excess <= 0, excess, "<= (2/3) ln(1 + 2 delta_max)", "n* bound", ref, rows)
        else:
            scale = (5 / 3) * ms[:, 1].abs() + ms[:, 0].abs() + 0.5 * rows.s.abs() + rows.q.abs()
            excess = (nu_star - nu_bar).abs() - 16 * EPS * (scale + nu_bar.abs())
            require(excess <= 0, excess, "rounding bound (O4)", "A1 nu* = nu_bar", ref, rows)


# ----------------------------------------------------------------------------- mixture (9.3)
@given(
    m=st.integers(1, 6),
    b=st.integers(1, 4),
    p=st.floats(1e-3, 1 - 1e-3),
    seed=st.integers(0, 2**16),
)
def test_mixture_quantile_solves_psi(m: int, b: int, p: float, seed: int) -> None:
    """docs/2 9.3: Psi(Q_p) = p (scipy oracle), Q_p in the bracket, Q_p increasing in p."""
    g = torch.Generator().manual_seed(seed)
    mu = 3.0 * torch.randn(m, b, 4, dtype=torch.float64, generator=g)
    sigma = 0.01 + 2.0 * torch.rand(m, b, 4, dtype=torch.float64, generator=g)
    qp = mixture_quantile(p, mu, sigma)
    psi = norm.cdf((qp.numpy()[None] - mu.numpy()) / sigma.numpy()).mean(axis=0)
    slope = 1.0 / (math.sqrt(2 * math.pi) * float(sigma.min()))
    bound = slope * 1e-12 * (1.0 + np.abs(qp.numpy())) + 4 * EPS
    gap = np.abs(psi - p)
    where = f"(m, B, p, seed) = ({m}, {b}, {p!r}, {seed})"
    k = np.unravel_index(gap.argmax(), gap.shape)
    assert bool((gap <= bound).all()), (
        f"|Psi(Q_p) - p| = {gap.max():.3e} > {bound[k]:.3e} at (row, target) {tuple(map(int, k))},"
        f" {where} (docs/2 9.3; scipy.stats.norm oracle)"
    )
    z = float(norm.ppf(p))
    ends = mu + z * sigma
    tol = 1e-12 * (1 + qp.abs())
    below = (ends.min(0).values - tol - qp).max()
    above = (qp - ends.max(0).values - tol).max()
    assert float(below) <= 0, f"Q_p below the bracket by {float(below):.3e}, {where} (docs/2 9.3)"
    assert float(above) <= 0, f"Q_p above the bracket by {float(above):.3e}, {where} (docs/2 9.3)"
    if p < 0.99:
        step = (mixture_quantile(p + 0.005, mu, sigma) - qp).min()
        assert float(step) > 0, (
            f"min Q_(p+0.005) - Q_p = {float(step):.3e}, required > 0 (increasing), {where} "
            "(docs/2 9.3)"
        )


def test_mixture_variance_against_mpmath() -> None:
    """docs/2 9.2: mu* and sigma*^2 = sigma_ale^2 + sigma_epi^2 equal the mixture's own moments."""
    mu = torch.tensor([[[0.3]], [[-1.1]], [[2.4]]], dtype=torch.float64)
    sigma = torch.tensor([[[0.5]], [[1.7]], [[0.05]]], dtype=torch.float64)
    mix = mixture_moments(mu, sigma)
    comps = [
        (mp.mpf(float(a)), mp.mpf(float(s)))
        for a, s in zip(mu.flatten(), sigma.flatten(), strict=True)
    ]
    with mp.workdps(30):

        def pdf(x: mp.mpf) -> mp.mpf:
            return sum(mp.npdf(x, a, s) for a, s in comps) / len(comps)

        pts = sorted({float(a) for a, _ in comps} | {-mp.inf, mp.inf}, key=float)
        mean = mp.quad(lambda x: x * pdf(x), pts)
        var = mp.quad(lambda x: (x - mean) ** 2 * pdf(x), pts)
    got_mean = float(mix.mu_star[0, 0])
    got_var = float(mix.sigma_star[0, 0]) ** 2
    where = "components (mu, sigma) = (0.3, 0.5), (-1.1, 1.7), (2.4, 0.05)"
    e_mean, e_var = ro.rel(got_mean, mean), ro.rel(got_var, var)
    assert e_mean <= 1e-12, (
        f"mu* = {got_mean!r} vs mpmath {mp.nstr(mean, 17)}: rel {e_mean:.3e} > 1e-12 "
        f"(docs/2 9.2); {where}"
    )
    assert e_var <= 1e-12, (
        f"sigma*^2 = {got_var!r} vs mpmath {mp.nstr(var, 17)}: rel {e_var:.3e} > 1e-12 "
        f"(docs/2 9.2); {where}"
    )

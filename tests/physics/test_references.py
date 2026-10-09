"""Reference values: docs/2 section 10.2 (E1, E2, E3, E5, E11; there is no E4) and quadrature.

Acceptance tiers (P6-Q1 (a), ADL-080; P7-D5 (c)):
- tier 1: an independent mpmath computation (``reference_oracles``, 40 digits), relative <= tau_Q;
- tier 2: docs/2 closed forms (Proposition 2.4) and the ``r = 1`` seam, relative <= tau_Q;
- tier 3: printed hand values, within one unit of the last printed digit.
The "Manning back-check" lines of E1/E2 are replaced by the invariant 3 bound (ADL-080;
``test_conservation.py::test_inv_3_against_mpmath_residual``).

Quadrature: ``F - 1``, ``F_beta``, ``F_r`` against the Proposition 2.2 u-integral over a fast
6 x 6 sub-grid (with the beta atoms 1e-8 and 1e3) and, under ``slow``, the 25 r x 23 beta grid
of the P6-D3 level study.
"""

from __future__ import annotations

import math

import mpmath as mp
import pytest
import torch

import reference_oracles as ro
from domain_strategies import SOLVER, TAU_L, TAU_Q, Rows, t64
from src.config import FT_TO_M
from src.data.unit_map import CFS_TO_CMS, FT2_TO_M2
from src.physics import quadrature as qd
from src.physics.manning_layer import PhiConfig, phi, wide_channel_depth
from src.physics.quadrature import perimeter_fm1
from src.physics.roughness_priors import LAMBDA_L
from src.physics.shape import section_terms

pytestmark = pytest.mark.physics

LEVEL, TRUNC = SOLVER.quad_level, SOLVER.quad_trunc
PHI = PhiConfig.from_solver(SOLVER)
PHI_A1 = PhiConfig.from_solver(SOLVER, kappa=False)

E1 = (20.0, 2.0, 0.035, 10.0, 1e-3)  # (W m, r, n, Q_E m3/s, S)
E2 = (3.0, 1.0, 0.04, 1.0, 0.01)

R_GRID = (
    0.5, 0.501, 0.55, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 1.0, 1.001, 1.01, 1.1,
    1.25, 1.4, 1.49, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0,
)  # fmt: skip
B_GRID = tuple(10.0 ** (e / 2) for e in range(-16, 7))  # 1e-8 .. 1e3, half decades
R_FAST = (0.5, 0.7, 1.0, 1.01, 2.0, 10.0)
B_FAST = (1e-8, 1e-4, 1e-2, 1.0, 1e2, 1e3)


def check_tier3(name: str, got: float, printed: str, ref: str) -> None:
    ok, err, unit = ro.tier3(got, printed)
    assert ok, f"{name}: |{got!r} - {printed}| = {err:.3e} > one unit {unit:g} ({ref}, tier 3)"


def check_rel(name: str, got: float, want: object, tol: float, ref: str) -> None:
    err = ro.rel(got, want)
    assert err <= tol, f"{name}: relative error {err:.3e} > {tol:.1e} vs mpmath ({ref})"


def code_iterates(rows: Rows, m: int) -> list[float]:
    """``dbar`` after Picard steps 0..m with the code's quadrature (docs/2 Prop 3.2 (2))."""
    w, r, nu, q, s = rows.args()
    l0 = 0.6 * (q + nu - w - 0.5 * s)
    ells = [l0]
    for _ in range(m):
        beta = 2.0 * (r + 1.0) * torch.exp(ells[-1] - w)
        ells.append(l0 + 0.4 * torch.log1p(perimeter_fm1(beta, r, level=LEVEL, trunc=TRUNC)))
    return [math.exp(float(e[0])) for e in ells]


def solved(example: tuple[float, ...]) -> dict[str, float]:
    """Phi at an E-example: dbar, d_max, A, V, F, kappa, eta and d ln dbar / d ln Q."""
    rows = Rows.from_si([example])
    q = rows.q.clone().requires_grad_()
    out = phi(rows.w, rows.r, rows.nu, q, rows.s, PHI).out
    (g,) = torch.autograd.grad(out[:, 0].sum(), q)
    terms = section_terms(out[:, 0].detach(), rows.w, rows.r, level=LEVEL, trunc=TRUNC)
    o = out.detach()[0]
    return {
        "ell": float(o[0]),
        "dbar": math.exp(float(o[0])),
        "dmax": math.exp(float(o[1])),
        "A": math.exp(float(o[2])),
        "V": math.exp(float(o[3])),
        "F": 1.0 + float(terms.fm1[0]),
        "fm1": float(terms.fm1[0]),
        "kappa": float(terms.kappa[0]),
        "eta": float(terms.eta[0]),
        "slope": float(g[0]),
    }


def mp_example(example: tuple[float, ...]) -> dict[str, object]:
    """Tier-1 oracle: the mpmath root of eq. 3.1 with the Prop 2.4 closed-form F."""
    big_w, r, n, big_q, big_s = example
    with mp.workdps(ro.MP_DPS):
        w, nu, q, s = (mp.log(mp.mpf(v)) for v in (big_w, n, big_q, big_s))
        ell = ro.root(w, r, nu, q, s)
        beta = 2 * (mp.mpf(r) + 1) * mp.exp(ell - w)
        f = ro.f_closed(beta, r)
        eta = ro.eta(beta, r)
        dbar = mp.exp(ell)
        return {
            "ell": ell,
            "dbar": dbar,
            "dmax": dbar * (1 + 1 / mp.mpf(r)),
            "A": mp.mpf(big_w) * dbar,
            "V": mp.mpf(big_q) / (mp.mpf(big_w) * dbar),
            "F": f,
            "fm1": f - 1,
            "kappa": 1 / f,
            "eta": eta,
            "slope": 1 / (mp.mpf(5) / 3 - mp.mpf(2) / 3 * eta),
            "iterates": [mp.exp(x) for x in ro.picard(w, r, nu, q, s, 4)],
        }


# ----------------------------------------------------------------------------- E1, E2
@pytest.mark.parametrize(("name", "example"), [("E1", E1), ("E2", E2)])
def test_e1_e2_tier1_against_mpmath(name: str, example: tuple[float, ...]) -> None:
    """Tier 1 (docs/2 10.2 E1, E2): Phi outputs and section terms vs an mpmath Algorithm Phi."""
    got, want = solved(example), mp_example(example)
    ref = f"docs/2 10.2 {name}"
    for key in ("dbar", "dmax", "A", "V", "fm1", "kappa", "eta", "slope"):
        check_rel(f"{name} {key}", got[key], want[key], TAU_Q, ref)
    assert abs(got["ell"] - float(want["ell"])) <= TAU_L, (
        f"{name}: |ell - ell*_mp| = {abs(got['ell'] - float(want['ell'])):.3e} > tau_l "
        f"(docs/2 Prop 3.2 stopping rule)"
    )
    iters = code_iterates(Rows.from_si([example]), 4)
    for m, (a, b) in enumerate(zip(iters, want["iterates"], strict=True)):  # type: ignore[call-overload]
        check_rel(f"{name} iterate {m}", a, b, TAU_Q, f"{ref}, docs/2 Prop 3.2 (2)")


def test_e1_tier3_printed() -> None:
    """Tier 3 (docs/2 10.2 E1, wide parabola): every printed value within one unit."""
    got = solved(E1)
    it = code_iterates(Rows.from_si([E1]), 3)
    ref = "docs/2 10.2 E1"
    for name, val, printed in [
        ("dbar_kappa=1", it[0], "0.70117"),
        ("iterate 1", it[1], "0.703220"),
        ("iterate 2", it[2], "0.703232"),
        ("iterate 3", it[3], "0.703232"),
        ("dbar*", got["dbar"], "0.70323"),
        ("F", got["F"], "1.007369"),
        ("kappa*", got["kappa"], "0.99268"),
        ("d_max", got["dmax"], "1.05485"),
        ("A", got["A"], "14.0646"),
        ("V", got["V"], "0.71101"),
        ("eta", got["eta"], "0.01454"),
        ("L = (2/5) eta", 0.4 * got["eta"], "0.00581"),
        ("d ln dbar / d ln Q", got["slope"], "0.60351"),
    ]:
        check_tier3(f"E1 {name}", val, printed, ref)


def test_e2_tier3_printed() -> None:
    """Tier 3 (docs/2 10.2 E2, narrow triangle), incl. the Prop 3.2 bracket and Prop 2.3 bound."""
    got = solved(E2)
    it = code_iterates(Rows.from_si([E2]), 4)
    ref = "docs/2 10.2 E2"
    rows = Rows.from_si([E2])
    l0 = 0.6 * (rows.q + rows.nu - rows.w - 0.5 * rows.s)
    beta0 = 2.0 * (rows.r + 1.0) * torch.exp(l0 - rows.w)
    k0 = float(torch.log1p(perimeter_fm1(beta0, rows.r, level=LEVEL, trunc=TRUNC))[0])
    lo, hi = math.exp(float(l0[0]) + 0.4 * k0), math.exp(float(l0[0]) + (2.0 / 3.0) * k0)
    assert lo <= got["dbar"] <= hi, f"E2 dbar* {got['dbar']!r} outside [{lo!r}, {hi!r}] (Prop 3.2)"
    for name, val, printed in [
        ("dbar_kappa=1", it[0], "0.298513"),
        ("iterate 1", it[1], "0.307423"),
        ("iterate 2", it[2], "0.307931"),
        ("iterate 3", it[3], "0.307960"),
        ("iterate 4", it[4], "0.307962"),
        ("dbar*", got["dbar"], "0.30796"),
        ("bracket low", lo, "0.30742"),
        ("bracket high", hi, "0.31351"),
        ("kappa*", got["kappa"], "0.92505"),
        ("W/(W + 2 d_max)", E2[0] / (E2[0] + 2.0 * got["dmax"]), "0.70891"),
        ("eta", got["eta"], "0.14428"),
        ("L = (2/5) eta", 0.4 * got["eta"], "0.0577"),
        ("d ln dbar / d ln Q", got["slope"], "0.63675"),
        ("A1 factor kappa^(-2/3)", got["kappa"] ** (-2.0 / 3.0), "1.0533"),
    ]:
        check_tier3(f"E2 {name}", val, printed, ref)
    assert got["kappa"] > E2[0] / (E2[0] + 2.0 * got["dmax"]), "E2 kappa above bound (Prop 2.3)"


# ----------------------------------------------------------------------------- E3, E5, E11
@pytest.mark.parametrize(("r", "printed"), [(0.5, "1.03497"), (1.0, "1.01980"), (2.0, "1.02606")])
def test_e3_closed_forms_at_delta_0_1(r: float, printed: str) -> None:
    """docs/2 10.2 E3: F at delta = 0.1 (beta = 2 r delta); tier 2 vs Prop 2.4 and tier 3."""
    beta = 2.0 * r * 0.1
    fm1 = float(perimeter_fm1(t64(beta), t64(r), level=LEVEL, trunc=TRUNC)[0])
    with mp.workdps(ro.MP_DPS):
        check_rel(f"E3 F-1 r={r}", fm1, ro.f_closed(mp.mpf(beta), r) - 1, TAU_Q, "docs/2 Prop 2.4")
    check_tier3(f"E3 F r={r}", 1.0 + fm1, printed, "docs/2 10.2 E3")
    assert 1.0 <= 1.0 + fm1 <= 1.2, "E3: F outside [1, 1.2] (docs/2 10.2)"


def test_e5_unit_constants() -> None:
    """docs/2 10.2 E5 and 1.2: lambda_L^3 exact; k_US = 0.3048^(-1/3); k_US^(3/5) (tier 3)."""
    with mp.workdps(ro.MP_DPS):
        lam = mp.mpf("0.3048")
        assert lam**3 == mp.mpf("0.028316846592"), "lambda_L^3 is exact (docs/2 10.2 E5)"
        k_us = lam ** (-mp.mpf(1) / 3)
        check_tier3("E5 k_US", float(k_us), "1.48592", "docs/2 10.2 E5")
        check_tier3("E5 k_US^(3/5)", float(k_us ** (mp.mpf(3) / 5)), "1.26822", "docs/2 10.2 E5")
    assert FT_TO_M == LAMBDA_L == 0.3048, "lambda_L = 0.3048 m/ft exactly (docs/2 1.2)"
    assert float(mp.mpf("0.028316846592")) == CFS_TO_CMS, "CFS_TO_CMS = lambda_L^3 (docs/2 1.2)"
    assert float(mp.mpf("0.09290304")) == FT2_TO_M2, "FT2_TO_M2 = lambda_L^2 (docs/2 1.2)"


def test_e5_mishandled_k_us_depth_factor() -> None:
    """docs/2 D8 / E5: a mishandled k_US scales the A1 depth by k_US^(3/5) = 1.26822."""
    rows = Rows.from_si([E1])
    with mp.workdps(ro.MP_DPS):
        ln_k = mp.log(mp.mpf("0.3048") ** (-mp.mpf(1) / 3))
        want = mp.exp(mp.mpf(3) / 5 * ln_k)
    base = phi(*rows.args(), PHI_A1).ell
    shifted = phi(rows.w, rows.r, rows.nu + float(ln_k), rows.q, rows.s, PHI_A1).ell
    factor = math.exp(float(shifted[0] - base[0]))
    check_rel("E5 depth factor", factor, want, 1e-12, "docs/2 D8, E5")
    check_tier3("E5 depth factor", factor, "1.26822", "docs/2 10.2 E5")


def test_e11_unit_equivariance_hand_values() -> None:
    """docs/2 10.2 E11 (Prop 3.5): the E1 wide-channel depth in ft units; tier 3 and tier 1."""
    big_w, _, n, big_q, big_s = E1
    w_ft, q_ft = big_w / FT_TO_M, big_q / CFS_TO_CMS
    n_ft = n * FT_TO_M ** (1.0 / 3.0)
    ref = "docs/2 10.2 E11"
    check_tier3("E11 W_ft", w_ft, "65.6168", ref)
    check_tier3("E11 Q_ft", q_ft, "353.1467", ref)
    check_tier3("E11 n_ft", n_ft, "0.0235544", ref)
    check_tier3("E11 Q n / (W S^1/2)", q_ft * n_ft / (w_ft * math.sqrt(big_s)), "4.00878", ref)
    ell_ft = wide_channel_depth(t64(math.log(w_ft)), t64(math.log(n_ft)), t64(math.log(q_ft)),
                                t64(math.log(big_s)))  # fmt: skip
    d_ft = math.exp(float(ell_ft[0]))
    check_tier3("E11 dbar (ft)", d_ft, "2.30042", ref)
    check_tier3("E11 dbar (ft -> m)", d_ft * FT_TO_M, "0.701169", ref)
    with mp.workdps(ro.MP_DPS):
        want_m = mp.exp(ro.ell0(*(mp.log(mp.mpf(v)) for v in (big_w, n, big_q, big_s))))
    check_rel("E11 ft -> m vs SI (Prop 3.5)", d_ft * FT_TO_M, want_m, 1e-12, ref)


# ----------------------------------------------------------------------------- quadrature
def check_quadrature(rs: tuple[float, ...], bs: tuple[float, ...]) -> None:
    grid = [(r, b) for r in rs for b in bs]
    beta = torch.tensor([b for _, b in grid], dtype=torch.float64)
    r = torch.tensor([rr for rr, _ in grid], dtype=torch.float64)
    got = qd.perimeter_integrals(beta, r, level=LEVEL, trunc=TRUNC)
    worst = (0.0, "")
    for i, (rr, b) in enumerate(grid):
        want = ro.f_integrals(b, rr)
        for name, g, w in zip(("F-1", "F_beta", "F_r"), got, want, strict=True):
            err = ro.rel(float(g[i]), w)
            if err > worst[0]:
                worst = (err, f"{name} at (r, beta) = ({rr!r}, {b!r})")
    assert worst[0] <= TAU_Q, (
        f"quadrature: max relative error {worst[0]:.3e} > tau_Q = {TAU_Q:.1e} at {worst[1]} "
        "(docs/2 Prop 2.2, 3.5 numerical note)"
    )


def test_quadrature_fast_subgrid() -> None:
    """docs/2 Props 2.2, 2.5, 2.8: F - 1, F_beta, F_r <= tau_Q on a 6 x 6 (r, beta) sub-grid."""
    check_quadrature(R_FAST, B_FAST)


@pytest.mark.slow
def test_quadrature_full_grid() -> None:
    """The 25 r x 23 beta grid of the P6-D3 level study (docs/2 3.5; protocol 4.2)."""
    check_quadrature(R_GRID, B_GRID)


@pytest.mark.parametrize("r", ro.CLOSED_FORM_R)
def test_prop_2_4_closed_forms(r: float) -> None:
    """docs/2 Prop 2.4 (a), (b), (d): F - 1 and F_beta vs the closed forms, beta atoms included."""
    betas = (1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3)
    fm1, f_beta, _ = qd.perimeter_integrals(
        torch.tensor(betas, dtype=torch.float64), torch.full((len(betas),), r, dtype=torch.float64),
        level=LEVEL, trunc=TRUNC,
    )  # fmt: skip
    with mp.workdps(ro.MP_DPS):
        for i, b in enumerate(betas):
            bm = mp.mpf(b)
            check_rel(
                f"F-1 r={r} beta={b}", float(fm1[i]), ro.f_closed(bm, r) - 1, TAU_Q, "Prop 2.4"
            )
            dref = mp.diff(lambda x: ro.f_closed(x, r), bm)
            check_rel(f"F_beta r={r} beta={b}", float(f_beta[i]), dref, TAU_Q, "Prop 2.4")


def test_oracle_self_consistency() -> None:
    """The u-integral oracle reproduces the Prop 2.4 closed forms far below tau_Q.

    Includes ``1 - eta`` up to beta = 1e8 (the P7-A1 regime; docs/3 section 12, O5).
    """
    for r in ro.CLOSED_FORM_R:
        for b in (1e-6, 0.3, 50.0):
            fm1 = ro.f_integrals(b, r)[0]
            with mp.workdps(ro.MP_DPS):
                assert ro.rel(float(fm1), ro.f_closed(mp.mpf(b), r) - 1) <= 1e-15
        for b in (1e4, 1.1253e8):
            gap = ro.one_minus_eta(b, r)
            with mp.workdps(ro.MP_DPS):
                want = 1 - ro.eta(mp.mpf(b), r)  # closed form, mp.diff
                err = abs(gap - want) / abs(want)
            assert err <= 1e-10, f"1 - eta oracle at (r, beta) = ({r}, {b}): rel {float(err):.2e}"


def test_seam_at_r_equal_one() -> None:
    """docs/2 3.5: the u-form (r >= 1) and tau-form (r < 1) agree across r = 1 within tau_Q."""
    beta = t64(1e-8, 1e-3, 0.3, 3.0, 300.0, 1e3)
    at = qd.perimeter_integrals(beta, torch.ones(6, dtype=torch.float64), level=LEVEL, trunc=TRUNC)
    for side in (1.0 - 1e-12, 1.0 + 1e-12):
        near = qd.perimeter_integrals(beta, torch.full((6,), side, dtype=torch.float64),
                                      level=LEVEL, trunc=TRUNC)  # fmt: skip
        for name, a, b in zip(("F-1", "F_beta", "F_r"), near, at, strict=True):
            err = float(((a - b).abs() / b.abs()).max())
            assert err <= TAU_Q, f"seam {name} at r = {side!r}: {err:.3e} > tau_Q (docs/2 3.5)"
    with mp.workdps(ro.MP_DPS):
        for i, b in enumerate(beta.tolist()):
            check_rel(f"seam F-1 beta={b}", float(at[0][i]), ro.f_closed(mp.mpf(b), 1.0) - 1, TAU_Q,
                      "Prop 2.4 (a)")  # fmt: skip

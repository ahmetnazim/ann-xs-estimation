"""Prop 3.6 boundary behaviour, fuzzing inside D and the out-of-D error contract (Phase 7, Step 3).

Oracle: docs/2 statements only (Props 2.4, 2.8, 3.3, 3.6, section 1.3 and D9); expected values
are closed forms, bounds or mpmath (``reference_oracles``). Floating-point rules: docs/3
section 12 (O5 closed end with the O8 bounds; O9 saturated logistic).

Error contract (docs/2 1.3, D9; ``src.physics.shape.check_rows``, ``manning_layer.check_phys``,
``models.heads.check_batch``): non-finite inputs (``Q_E <= 0`` gives ``q = -inf`` or NaN) raise
``ValueError``; a dtype other than float64 raises ``TypeError``; a wrong shape or length raises
``ValueError``; ``r <= 0`` raises ``ValueError``; a missing batch key raises ``KeyError``;
``S < S_floor`` raises ``ValueError`` in every physics model, while Phi itself is defined on
``D_nat`` (``S > 0``, Prop 3.6). Nothing is clipped or masked.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import Tensor

import reference_oracles as ro
from domain_strategies import (
    BLIND_IDX,
    LN_S_FLOOR,
    N_FEATURES,
    O8_DW_MAX,
    PROTOCOL,
    S_MAX,
    SLOPE_MAX,
    SOLVER,
    STATS,
    TAU_Q,
    TR,
    Rows,
    closed_end_rows,
    model_batch,
    phi_rows,
    q_values,
    r_values,
    require,
    require_equal,
    require_le,
    saturated,
    w_values,
)
from src.models.checkpoint import build_model
from src.models.heads import LadderNet
from src.models.m5_hard import M5Config, M5Hard
from src.models.registry import ModelSpec, nn_grid
from src.physics.manning_layer import PhiConfig, PhiConvergenceError, phi
from src.physics.shape import section_terms

pytestmark = pytest.mark.physics

PHI = PhiConfig.from_solver(SOLVER)
NU_HI = TR.nu_min + TR.delta_n
MP_EXAMPLES = max(10, settings.default.max_examples // 8)
#: Fast-suite budget of the trimmed Prop 3.6 D_nat test; the full budget runs under ``slow``.
FAST_EXAMPLES = max(10, settings.default.max_examples // 4)
P36 = "docs/2 Prop 3.6"


def ell_partials(rows: Rows, cfg: PhiConfig = PHI) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Phi outputs, iterations and the per-row ``d ell / d(w, r, nu, q, s)`` (Prop 3.4)."""
    leaves = [x.clone().requires_grad_() for x in rows.args()]
    res = phi(*leaves, cfg)
    grads = torch.autograd.grad(res.out[:, 0].sum(), leaves)
    names = ("w", "r", "nu", "q", "s")
    return res.out.detach(), res.iterations, dict(zip(names, grads, strict=True))


# ----------------------------------------------------------------------------- inside D
@given(rows=phi_rows())
def test_inside_domain_finite_converged_and_differentiable(rows: Rows) -> None:
    """Inside D (P7-D6 atoms): finite outputs, Picard stops within K_max, finite gradients.

    docs/2 Prop 3.1 (unique root), Prop 3.2 (contraction, stopping rule), Prop 3.6 (C^1).
    """
    leaves = [x.clone().requires_grad_() for x in rows.args()]
    res = phi(*leaves, PHI)
    out, it = res.out, res.iterations
    require(torch.isfinite(out).all(dim=1), out.abs().amax(dim=1), "finite", "Phi out", P36, rows)
    ok_it = (it >= 1) & (it <= SOLVER.k_max)
    require(ok_it, it.to(torch.float64), f"in [1, {SOLVER.k_max}]", "iterations", P36, rows)
    names = ["w", "r", "nu", "q", "s"]
    for name, g in zip(names, torch.autograd.grad(out.sum(), leaves), strict=True):
        require(torch.isfinite(g), g.abs(), "finite", f"d(sum out)/d{name}", P36, rows)


nu_nat = st.one_of(st.sampled_from([TR.nu_min, NU_HI]), st.floats(TR.nu_min - 3.0, NU_HI + 3.0))
s_nat = st.one_of(
    st.sampled_from([LN_S_FLOOR, float(np.nextafter(LN_S_FLOOR, -math.inf))]),
    st.floats(math.log(1e-9), math.log(S_MAX)),
)


@st.composite
def nat_rows(draw: st.DrawFn) -> Rows:
    """Rows of D_nat beyond D in (nu, S): nu outside the band and S below S_floor (Prop 3.6)."""
    table = draw(
        st.lists(st.tuples(w_values, r_values, nu_nat, q_values, s_nat), min_size=1, max_size=32)
    )
    cols = [torch.tensor(c, dtype=torch.float64) for c in zip(*table, strict=True)]
    return Rows(w=cols[0], r=cols[1], nu=cols[2], q=cols[3], s=cols[4])


def check_prop_3_6_d_nat(rows: Rows) -> None:
    """Prop 3.6 on D_nat and the Prop 3.3 ranges: |d ell/dq| = |d ell/dnu| < 1, |d ell/ds| < 1/2,
    |d ell/dw| < 5/3, |d ell/dr| <= 2(2r + 1)/(3r(r + 1)); 1/D in [3/5, 1), d ell/dw in
    (-5/3, -3/5]. O5 rows use the O8 closed-end bounds (docs/3 section 12).
    """
    out, _, g = ell_partials(rows)
    p33 = "docs/2 Prop 3.3"
    require_equal(g["nu"], g["q"], "d ell/dnu = d ell/dq", p33, rows)
    require_equal(g["s"], -0.5 * g["q"], "d ell/ds = -1/(2D)", p33, rows)
    at_end = closed_end_rows(out[:, 0], rows.w, rows.r, rows)
    gq, gw, gr, r = g["q"], g["w"], g["r"], rows.r
    q_ok = (gq >= 0.6) & ((gq < 1.0) | (at_end & (gq <= SLOPE_MAX)))
    require(q_ok, torch.maximum(0.6 - gq, gq - 1.0), "in [3/5, 1) (O8)", "1/D", P36, rows)
    w_ok = (gw <= -0.6) & ((gw > -5.0 / 3.0) | (at_end & (gw >= -O8_DW_MAX)))
    require(w_ok, torch.maximum(gw + 0.6, -5 / 3 - gw), "(-5/3, -3/5] (O8)", "d ell/dw", P36, rows)
    b_r = 2.0 * (2.0 * r + 1.0) / (3.0 * r * (r + 1.0))
    require(gr.abs() <= b_r, gr.abs() - b_r, "<= 2(2r+1)/(3r(r+1))", "|d ell/dr|", P36, rows)


@given(rows=nat_rows())
@settings(max_examples=FAST_EXAMPLES)
def test_prop_3_6_uniform_bounds_on_d_nat(rows: Rows) -> None:
    """Prop 3.6 / Prop 3.3 bounds on D_nat (docs/2; O5/O8 of docs/3 section 12); fast budget."""
    check_prop_3_6_d_nat(rows)


@pytest.mark.slow
@given(rows=nat_rows())
def test_prop_3_6_uniform_bounds_on_d_nat_full_budget(rows: Rows) -> None:
    """Prop 3.6 / Prop 3.3 bounds on D_nat at the full profile budget (ci 200, thorough 5000;
    P7-D6).
    """
    check_prop_3_6_d_nat(rows)


@given(rows=phi_rows(max_size=4))
@settings(max_examples=MP_EXAMPLES)
def test_prop_3_6_r_half_matches_prop_2_4_closed_form(rows: Rows) -> None:
    """Prop 3.6 boundary r = r_min = 1/2: the depth equals the root of eq. 3.1 with the
    Prop 2.4 (d) closed form F = sqrt(1 + b^2) + b^2 asinh(1/b) (mpmath, 40 digits) to tau_Q
    (tiers 1-2 rule, docs/3 section 12.1).
    """
    rows = Rows(w=rows.w, r=torch.full_like(rows.w, 0.5), nu=rows.nu, q=rows.q, s=rows.s)
    ell = phi(*rows.args(), PHI).ell
    ref = [
        float(ro.root(float(rows.w[i]), 0.5, float(rows.nu[i]), float(rows.q[i]), float(rows.s[i])))
        for i in range(rows.n)
    ]
    err = (ell - torch.tensor(ref, dtype=torch.float64)).abs()
    require_le(err, TAU_Q, "|ell - ell*_mp| at r = 1/2", "docs/2 Props 2.4 (d), 3.6", rows)


def test_prop_3_6_eta_to_one_gives_d_to_one() -> None:
    """Prop 3.6 table, kappa -> 0 (deep, narrow): eta -> 1 and D -> 1, D > 1 (or O5 closed end).

    W = 1 m, n = 0.6, S = S_floor, q in {2, 6, 10, 14} at r in {1/2, 1, 2, 10}; float D against
    D_mp = 1 + (2/3)(1 - eta)_mp to (2/3) tau_Q. The limit itself is witnessed on the mpmath
    gap (1 - eta)_mp: positive on every row (Prop 2.7) and contracted by at least 1e3 from
    q = 2 to q = 14. No absolute end threshold is imposed: docs/2 states the limit, not a rate
    at finite q (at W = 1 m, r = 10 the gap at q = 14 is ~1e-10).
    """
    q = torch.tensor([2.0, 6.0, 10.0, 14.0], dtype=torch.float64)
    for r_val in (0.5, 1.0, 2.0, 10.0):
        n = q.numel()
        rows = Rows(
            w=torch.zeros(n, dtype=torch.float64),
            r=torch.full((n,), r_val, dtype=torch.float64),
            nu=torch.full((n,), NU_HI, dtype=torch.float64),
            q=q,
            s=torch.full((n,), LN_S_FLOOR, dtype=torch.float64),
        )
        ell = phi(*rows.args(), PHI).ell
        terms = section_terms(ell, rows.w, rows.r, level=SOLVER.quad_level, trunc=SOLVER.quad_trunc)
        d = 5.0 / 3.0 - (2.0 / 3.0) * terms.eta
        gap = [ro.one_minus_eta(float(b), r_val) for b in terms.beta]
        d_mp = torch.tensor([1.0 + (2.0 / 3.0) * float(g) for g in gap], dtype=torch.float64)
        require_le((d - d_mp).abs(), (2.0 / 3.0) * TAU_Q, f"D vs mpmath (r = {r_val})", P36, rows)
        at_end = closed_end_rows(ell, rows.w, rows.r, rows)
        require((d > 1.0) | at_end, 1.0 - d, "> 1 (O5 closed end)", "D", "docs/2 Prop 3.1", rows)
        assert all(g > 0 for g in gap), f"(1 - eta)_mp <= 0 at r = {r_val} (docs/2 Prop 2.7)"
        ratio = float(gap[-1] / gap[0])
        assert ratio <= 1e-3, (
            f"(1 - eta)_mp(q = 14) / (1 - eta)_mp(q = 2) = {ratio:.3e} > 1e-3 at r = {r_val}: "
            f"D not -> 1 ({P36}); gap(q = 14) = {float(gap[-1]):.3e}"
        )


def test_prop_3_6_kappa_to_one_limits() -> None:
    """Prop 3.6 table, kappa -> 1 (a -> 0): eta <= beta/r, |k_r| <= 2a(2r+1)/r^2 (Prop 2.8), and
    the partials approach (3/5, -3/10, -3/5, 0) at the docs/2 rates:
    |1/D - 3/5| <= (2/5) eta, |d ell/dw + 3/5| <= (16/15) eta, |d ell/ds + 3/10| <= eta / 5.
    """
    table = [
        (5000.0, r, PROTOCOL.domain.n_bounds[0], qe, 0.1)
        for r in (0.5, 1.0, 2.0, 10.0)
        for qe in (1e-4, 1e-3, 1e-2)
    ]
    rows = Rows.from_si(table)
    out, _, g = ell_partials(rows)
    terms = section_terms(
        out[:, 0], rows.w, rows.r, level=SOLVER.quad_level, trunc=SOLVER.quad_trunc
    )
    r, a, beta, eta = rows.r, terms.a, terms.beta, terms.eta
    require(eta <= beta / r, eta - beta / r, "<= beta/r", "eta", P36, rows)
    b_kr = 2.0 * a * (2.0 * r + 1.0) / r**2
    require(terms.k_r.abs() <= b_kr, terms.k_r.abs() - b_kr, "<= 2a(2r+1)/r^2", "k_r", P36, rows)
    for name, got, want, rate in (
        ("d ell/dq", g["q"], 0.6, 0.4),
        ("d ell/dw", g["w"], -0.6, 16.0 / 15.0),
        ("d ell/ds", g["s"], -0.3, 0.2),
    ):
        dev = (got - want).abs()
        require(dev <= rate * eta, dev - rate * eta, f"<= {rate:.4g} eta", name, P36, rows)
    dr_bound = (2.0 / 3.0) * b_kr
    require(
        g["r"].abs() <= dr_bound, g["r"].abs() - dr_bound, "<= (2/3) b_kr", "d ell/dr", P36, rows
    )


@pytest.mark.parametrize("z_r", [-1000.0, -40.0, 40.0, 1000.0])
@pytest.mark.parametrize("z_n", [-1000.0, -40.0, 40.0, 1000.0])
def test_sigma_rounding_still_defines_phi(z_n: float, z_r: float) -> None:
    """Prop 3.6 table (nu and r bounds): if sigma(z) rounds to 0 or 1, Phi is still defined and
    the chain-rule gradient vanishes exactly; it is never undefined (O9, docs/3 section 12).
    """
    base = Rows.from_si(
        [
            (20.0, 2.0, 0.035, 10.0, 1e-3),
            (3.0, 1.0, 0.04, 1.0, 0.01),
            (5000.0, 1.0, 0.04, 1e-4, 0.1),
            (0.1, 1.0, 0.04, 1e5, 1e-5),
        ]
    )
    n = base.n
    zn = torch.full((n,), z_n, dtype=torch.float64, requires_grad=True)
    zr = torch.full((n,), z_r, dtype=torch.float64, requires_grad=True)
    leaves = [x.clone().requires_grad_() for x in (base.w, base.q, base.s)]
    r, nu = TR.r(zr), TR.nu(zn)
    rows = Rows(w=base.w, r=r.detach(), nu=nu.detach(), q=base.q, s=base.s)
    out = phi(leaves[0], r, nu, leaves[1], leaves[2], PHI).out
    require(torch.isfinite(out).all(dim=1), out.abs().amax(dim=1), "finite", "Phi out", P36, rows)
    g_zn, g_zr, *g_data = torch.autograd.grad(out[:, 0].sum(), [zn, zr, *leaves])
    for name, gz in (("d ell/d z_n", g_zn), ("d ell/d z_r", g_zr)):
        require(torch.isfinite(gz), gz.abs(), "finite (never NaN)", name, P36, rows)
    # O9: d ell/d z_n = (1/D) Delta_n sigma (1 - sigma) is 0 where sigma rounds, > 0 elsewhere.
    sign_ok = torch.where(saturated(zn), g_zn == 0.0, g_zn > 0.0)
    require(sign_ok, g_zn.abs(), "> 0 (= 0 where sigma rounds, O9)", "d ell/d z_n", P36, rows)
    zero_r = ~saturated(zr) | (g_zr == 0.0)
    require(zero_r, g_zr.abs(), "= 0 where sigma rounds (O9)", "d ell/d z_r", P36, rows)
    for name, g in zip(("w", "q", "s"), g_data, strict=True):
        require(torch.isfinite(g), g.abs(), "finite", f"d ell/d{name}", P36, rows)


# ----------------------------------------------------------------------------- outside D
def _set(row: int, value: float) -> Callable[[Tensor], Tensor]:
    def f(t: Tensor) -> Tensor:
        out = t.clone()
        out[row] = value
        return out

    return f


PHI_CORRUPTIONS: dict[str, tuple[str, str, float | None, type[Exception], str]] = {
    # name: (argument, kind, value, error, message fragment)
    "Q_E = 0 (q = -inf)": ("q", "set", -math.inf, ValueError, "non-finite"),
    "Q_E < 0 (q = NaN)": ("q", "log_neg", None, ValueError, "non-finite"),
    "W NaN": ("w", "set", math.nan, ValueError, "non-finite"),
    "nu +inf": ("nu", "set", math.inf, ValueError, "non-finite"),
    "S = 0 (s = -inf)": ("s", "set", -math.inf, ValueError, "non-finite"),
    "r = 0": ("r", "set", 0.0, ValueError, "r must be > 0"),
    "r < 0": ("r", "set", -0.5, ValueError, "r must be > 0"),
    "float32": ("w", "float32", None, TypeError, "float64"),
    "2-D shape": ("q", "2d", None, ValueError, "1-D"),
    "length mismatch": ("nu", "short", None, ValueError, "length"),
}


@given(rows=phi_rows(min_size=2), kind=st.sampled_from(sorted(PHI_CORRUPTIONS)), pick=st.integers())
def test_phi_rejects_out_of_domain_input(rows: Rows, kind: str, pick: int) -> None:
    """docs/2 1.3 and D9: out-of-D input raises the documented error before computing."""
    arg, how, value, error, fragment = PHI_CORRUPTIONS[kind]
    args = dict(zip(("w", "r", "nu", "q", "s"), rows.args(), strict=True))
    row = pick % rows.n
    x = args[arg]
    if how == "set":
        assert value is not None
        args[arg] = _set(row, value)(x)
    elif how == "log_neg":
        args[arg] = _set(row, float(torch.log(torch.tensor(-1.0, dtype=torch.float64))))(x)
    elif how == "float32":
        args[arg] = x.to(torch.float32)
    elif how == "2d":
        args[arg] = x[:, None]
    else:
        args[arg] = x[:-1]
    with pytest.raises(error, match=fragment):
        phi(args["w"], args["r"], args["nu"], args["q"], args["s"], PHI)


def _ladder() -> dict[str, LadderNet]:
    torch.manual_seed(0)
    nets = {}
    for model in ("M2-MLP", "M3", "M4", "M5", "M6-NN"):
        spec = ModelSpec(model)
        nets[model] = build_model(
            spec, PROTOCOL, stats=STATS, blind_idx=BLIND_IDX, hyper=nn_grid(PROTOCOL, model)[0],
            r_bar=2.0 if model == "M3" else None,
        )  # fmt: skip
    return nets


LADDER = _ladder()
PHYSICS_MODELS = ("M3", "M4", "M5", "M6-NN")
BELOW_FLOOR = float(np.nextafter(LN_S_FLOOR, -math.inf))

MODEL_CORRUPTIONS: dict[str, tuple[type[Exception], str, bool]] = {
    # name: (error, message fragment, physics models only)
    "S just below S_floor": (ValueError, "S_floor", True),
    "S = 1e-6": (ValueError, "S_floor", True),
    "Q_E = 0 (q = -inf)": (ValueError, "non-finite values in phys", False),
    "Q_E < 0 (q = NaN)": (ValueError, "non-finite values in phys", False),
    "x NaN": (ValueError, "non-finite values in x", False),
    "x +inf": (ValueError, "non-finite values in x", False),
    "x float32": (TypeError, "float64", False),
    "x width F - 1": (ValueError, "shape", False),
    "phys (B, 3)": (ValueError, "shape", False),
    "missing omega": (KeyError, "omega", False),
}


def _corrupt(batch: dict[str, Tensor], kind: str, row: int) -> dict[str, Tensor]:
    b = dict(batch)
    phys, x = b["phys"].clone(), b["x"].clone()
    if kind == "S just below S_floor":
        phys[row, 1] = BELOW_FLOOR
    elif kind == "S = 1e-6":
        phys[row, 1] = math.log(1e-6)
    elif kind == "Q_E = 0 (q = -inf)":
        phys[row, 0] = -math.inf
    elif kind == "Q_E < 0 (q = NaN)":
        phys[row, 0] = math.nan
    elif kind == "x NaN":
        x[row, 0] = math.nan
    elif kind == "x +inf":
        x[row, 1] = math.inf
    elif kind == "x float32":
        x = x.to(torch.float32)
    elif kind == "x width F - 1":
        x = x[:, : N_FEATURES - 1]
    elif kind == "phys (B, 3)":
        phys = torch.cat([phys, phys[:, :1]], dim=1)
    else:
        del b["omega"]
    b["phys"], b["x"] = phys, x
    return b


@given(
    rows=phi_rows(max_size=8),
    kind=st.sampled_from(sorted(MODEL_CORRUPTIONS)),
    model=st.sampled_from(sorted(LADDER)),
    pick=st.integers(),
)
@settings(max_examples=MP_EXAMPLES)
def test_models_reject_out_of_domain_batches(rows: Rows, kind: str, model: str, pick: int) -> None:
    """docs/2 1.3, D9 and the ADL-077 batch contract at the model level.

    ``S < S_floor`` is rejected by every physics model (``check_phys``); Phi itself stays defined
    there (D_nat, Prop 3.6), which is asserted on the same rows.
    """
    error, fragment, physics_only = MODEL_CORRUPTIONS[kind]
    if physics_only and model not in PHYSICS_MODELS:
        return
    batch = _corrupt(model_batch(rows, pick % 1000), kind, pick % rows.n)
    with pytest.raises(error, match=fragment):
        LADDER[model](batch)
    if physics_only:
        s = batch["phys"][:, 1]
        out = phi(rows.w, rows.r, rows.nu, rows.q, s, PHI).out
        assert bool(torch.isfinite(out).all()), f"Phi must be defined on D_nat ({P36})"


# ----------------------------------------------------------------------------- non-convergence
@given(rows=phi_rows())
def test_phi_convergence_error_when_k_max_is_forced_small(rows: Rows) -> None:
    """Prop 3.2 stopping rule and Prop 3.4 (c): with K_max = max iterations - 1 the solver raises
    PhiConvergenceError (never a silent result); with K_max = max iterations the output is
    bit-identical to the params run (per-row freezing does not depend on K_max). Source:
    docs/2 Props 3.2 and 3.4.
    """
    full = phi(*rows.args(), PHI)
    m = int(full.iterations.max())
    if m >= 2:
        with pytest.raises(PhiConvergenceError, match="did not meet the stopping rule"):
            phi(*rows.args(), replace(PHI, k_max=m - 1))
    again = phi(*rows.args(), replace(PHI, k_max=m))
    require_equal(again.out[:, 0], full.out[:, 0], "ell at K_max = m", "docs/2 Prop 3.4", rows)
    require_equal(
        again.iterations.to(torch.float64),
        full.iterations.to(torch.float64),
        "iterations at K_max = m",
        "docs/2 Prop 3.4",
        rows,
    )


def test_m5_propagates_phi_convergence_error() -> None:
    """The M5 forward never hides a non-converged Phi (docs/2 3.3; protocol 4.2)."""
    torch.manual_seed(0)
    cfg = M5Config.from_protocol(PROTOCOL)
    short = cfg.with_flags(phi=replace(cfg.phi, k_max=1))
    net = M5Hard(N_FEATURES, BLIND_IDX, short, PROTOCOL.model.backbone)
    rows = Rows.from_si([(0.5, 1.0, 0.04, 1e4, 1e-5)] * 3)
    with pytest.raises(PhiConvergenceError):
        net(model_batch(rows, 0))

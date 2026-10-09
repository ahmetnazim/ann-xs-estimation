"""Props 3.4, 3.5 and 1.3: row-wise action, unit equivariance, US/SI commutation (Step 3).

Oracle: the docs/2 statements themselves (permutation and subset maps; the exact output shifts
of Prop 3.5; the unit map U of docs/2 1.2 with the exact international foot). Tolerances
(P7-D5 (c), docs/3 section 12.1): Phi bit-identical under Prop 3.4; whole models within 1e-12
(BLAS kernel choice, ADL-089); Props 3.5 and 1.3 within 1e-12 relative in linear space, i.e.
absolute in log space.
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import Tensor

from domain_strategies import (
    BLIND_IDX,
    PROTOCOL,
    SOLVER,
    STATS,
    Rows,
    model_batch,
    phi_rows,
    require_equal,
    require_le,
)
from src.models.checkpoint import build_model
from src.models.heads import LadderNet
from src.models.registry import ModelSpec, nn_grid
from src.physics.manning_layer import PhiConfig, phi
from src.physics.shape import section_terms

pytestmark = pytest.mark.physics

PHI = PhiConfig.from_solver(SOLVER)
UNITS_TOL = 1e-12  # P7-D5: Props 3.5 and 1.3
MODEL_TOL = 1e-12  # P7-D5: Prop 3.4, whole model
LAMBDA_L = 0.3048  # docs/2 1.2: international foot, exact
LN_K_US = -math.log(LAMBDA_L) / 3.0  # docs/2 Prop 1.2: k_US = lambda_L^(-1/3)
MODEL_EXAMPLES = max(10, settings.default.max_examples // 8)
P34 = "docs/2 Prop 3.4"


def _take(rows: Rows, idx: list[int]) -> Rows:
    i = torch.tensor(idx, dtype=torch.int64)
    return Rows(w=rows.w[i], r=rows.r[i], nu=rows.nu[i], q=rows.q[i], s=rows.s[i])


def _cat(a: Rows, b: Rows) -> Rows:
    return Rows(*(torch.cat([x, y]) for x, y in zip(a.args(), b.args(), strict=True)))


def _grads(rows: Rows) -> Tensor:
    leaves = [x.clone().requires_grad_() for x in rows.args()]
    ell = phi(*leaves, PHI).ell
    return torch.stack(torch.autograd.grad(ell.sum(), leaves), dim=1)


def _same(got: Tensor, want: Tensor, what: str, rows: Rows) -> None:
    for k in range(got.shape[1]):
        require_equal(got[:, k], want[:, k], f"{what} col {k}", P34, rows)


# ----------------------------------------------------------------------------- Prop 3.4
@given(rows=phi_rows(min_size=2), other=phi_rows(max_size=8), data=st.data())
def test_prop_3_4_permutation_subset_and_composition(
    rows: Rows, other: Rows, data: st.DataObject
) -> None:
    """Prop 3.4 (a), (c): Phi_N(Pi Z) = Pi Phi_N(Z), Phi on a subset = subset of Phi, and the
    outputs do not depend on the other rows of the batch (bit-identical, per-row freezing).
    Prop 3.3 partials follow the rows in the same way.
    """
    perm = data.draw(st.permutations(range(rows.n)), label="perm")
    sub = data.draw(st.lists(st.integers(0, rows.n - 1), min_size=1, unique=True), label="subset")
    full = phi(*rows.args(), PHI)
    g_full = _grads(rows)
    p_rows = _take(rows, perm)
    p = phi(*p_rows.args(), PHI)
    _same(p.out, full.out[perm], "permuted out", p_rows)
    _same(p.iterations[:, None].double(), full.iterations[perm][:, None].double(), "iters", p_rows)
    _same(_grads(p_rows), g_full[perm], "permuted grads", p_rows)
    s_rows = _take(rows, sub)
    _same(phi(*s_rows.args(), PHI).out, full.out[sub], "subset out", s_rows)
    mixed = _cat(other, rows)
    _same(phi(*mixed.args(), PHI).out[other.n :], full.out, "composed batch out", rows)


@given(rows=phi_rows(min_size=2, max_size=6))
def test_prop_3_4_block_diagonal_jacobian(rows: Rows) -> None:
    """Prop 3.4 (b): d Phi_N(Z)_i / d Z_j = 0 exactly for i != j (all four outputs, five inputs)."""

    def f(*args: Tensor) -> Tensor:
        return phi(*args, PHI).out

    jac = torch.autograd.functional.jacobian(f, rows.args())  # 5 x (B, 4, B)
    off = ~torch.eye(rows.n, dtype=torch.bool)
    for name, j in zip(["w", "r", "nu", "q", "s"], jac, strict=True):
        leak = j.abs().amax(dim=1)[off]  # (B*(B-1),) over output columns
        assert bool((leak == 0.0).all()), (
            f"Prop 3.4 (b) violated: |d out_i/d {name}_j| = {float(leak.max()):.3e} for i != j; "
            f"required exactly 0; {rows.describe(int(off.nonzero()[int(leak.argmax())][0]))}"
        )


def _ladder() -> dict[str, LadderNet]:
    torch.manual_seed(0)
    return {
        m: build_model(
            ModelSpec(m),
            PROTOCOL,
            stats=STATS,
            blind_idx=BLIND_IDX,
            hyper=nn_grid(PROTOCOL, m)[0],
            r_bar=2.0 if m == "M3" else None,
        )
        for m in ("M2-MLP", "M3", "M4", "M5", "M6-NN")
    }


LADDER = _ladder()


@pytest.mark.parametrize("model", sorted(LADDER))
@given(rows=phi_rows(min_size=2, max_size=16), seed=st.integers(0, 2**16), data=st.data())
@settings(max_examples=MODEL_EXAMPLES)
def test_prop_3_4_whole_model(model: str, rows: Rows, seed: int, data: st.DataObject) -> None:
    """Prop 3.4 scope for the whole torch rung: permuting or subsetting the batch permutes or
    subsets mu and sigma to 1e-12 (BLAS kernels may differ with batch size, ADL-089).
    """
    net = LADDER[model]
    batch = model_batch(rows, seed)
    perm = data.draw(st.permutations(range(rows.n)), label="perm")
    sub = data.draw(st.lists(st.integers(0, rows.n - 1), min_size=1, unique=True), label="subset")
    with torch.no_grad():
        full = net(batch)
        for idx, label in ((perm, "permuted"), (sub, "subset")):
            i = torch.tensor(idx, dtype=torch.int64)
            part = {k: (v[i] if k != "x_blind_idx" else v) for k, v in batch.items()}
            part["omega"] = torch.full((len(idx),), 1.0 / len(idx), dtype=torch.float64)
            got = net(part)
            sel = _take(rows, idx)
            for name, a, b in (("mu", got.mu, full.mu[i]), ("sigma", got.sigma, full.sigma)):
                if a is None or b is None:
                    continue
                b = b if name == "mu" else b[i]
                err = (a - b).abs().amax(dim=1)
                require_le(err, MODEL_TOL, f"{model} {label} {name}", P34, sel)


# ----------------------------------------------------------------------------- Prop 3.5
shift_values = st.one_of(
    st.sampled_from([0.0, math.log(LAMBDA_L), -math.log(LAMBDA_L), math.log(3600.0)]),
    st.floats(-5.0, 5.0),
)


@given(rows=phi_rows(), ln_chi_l=shift_values, ln_chi_t=shift_values)
def test_prop_3_5_unit_equivariance(rows: Rows, ln_chi_l: float, ln_chi_t: float) -> None:
    """Prop 3.5: w + ln chi_L, nu - ln chi_L/3 + ln chi_T, q + 3 ln chi_L - ln chi_T, s, r give
    ell, ln d_max + ln chi_L, ln A + 2 ln chi_L, ln V + ln chi_L - ln chi_T; a, eta, k, D and the
    Prop 3.3 partials are invariant. Tolerance 1e-12 (absolute in log space).
    """
    lx, lt = ln_chi_l, ln_chi_t
    moved = Rows(
        w=rows.w + lx, r=rows.r, nu=rows.nu - lx / 3.0 + lt, q=rows.q + 3.0 * lx - lt, s=rows.s
    )
    a, b = phi(*rows.args(), PHI).out, phi(*moved.args(), PHI).out
    shift = torch.tensor([lx, lx, 2.0 * lx, lx - lt], dtype=torch.float64)
    p35 = "docs/2 Prop 3.5"
    err = (b - a - shift).abs()
    for k, name in enumerate(("ell", "ln d_max", "ln A", "ln V")):
        require_le(err[:, k], UNITS_TOL, f"{name} shift", p35, rows)
    lvl, tr = SOLVER.quad_level, SOLVER.quad_trunc
    ta = section_terms(a[:, 0], rows.w, rows.r, level=lvl, trunc=tr)
    tb = section_terms(b[:, 0], moved.w, moved.r, level=lvl, trunc=tr)
    require_le((tb.a / ta.a - 1.0).abs(), UNITS_TOL, "a = dbar/W invariant", p35, rows)
    require_le((tb.eta - ta.eta).abs(), UNITS_TOL, "eta invariant", p35, rows)
    require_le((tb.log_f - ta.log_f).abs(), UNITS_TOL, "k invariant", p35, rows)
    g_err = (_grads(moved) - _grads(rows)).abs().amax(dim=1)
    require_le(g_err, UNITS_TOL, "Prop 3.3 partials invariant", p35, rows)


# ----------------------------------------------------------------------------- Prop 1.3
@given(rows=phi_rows())
def test_prop_1_3_si_us_commutation(rows: Rows) -> None:
    """Prop 1.3: Phi_SI o U = U o Phi_US with k_US = lambda_L^(-1/3) (Prop 1.2).

    ``rows`` are read in ft-based units (W in ft, Q_E in cfs; n keeps its SI value). Phi_US is
    the US Manning form Q = k_US n^-1 A R^(2/3) S^(1/2), i.e. Phi with nu - ln k_US. U maps
    w + ln lambda_L, q + 3 ln lambda_L and the outputs by ln lambda_L (L), 2 ln lambda_L (A).
    """
    ln_l = math.log(LAMBDA_L)
    us = phi(rows.w, rows.r, rows.nu - LN_K_US, rows.q, rows.s, PHI).out
    si = phi(rows.w + ln_l, rows.r, rows.nu, rows.q + 3.0 * ln_l, rows.s, PHI).out
    u_us = us + torch.tensor([ln_l, ln_l, 2.0 * ln_l, ln_l], dtype=torch.float64)
    err = (si - u_us).abs()
    for k, name in enumerate(("ell", "ln d_max", "ln A", "ln V")):
        require_le(
            err[:, k], UNITS_TOL, f"{name}: Phi_SI(U x) vs U(Phi_US x)", "docs/2 Prop 1.3", rows
        )

"""Gradient verification of Phi and the torch ladder (Phase 7, Step 4; P7-D7 (a), P7-D8 (a)).

Oracle: docs/2 Prop 3.3 (IFT partials), Prop 8.4 (quadrature gradients to tau_Q), section 3.4
(no unrolling; first order only), Theorem 8.2 / Corollary 8.3 (bounded gradients) and
assumption A3.1 (section 3.3). Expected values come from central finite differences of the
forward map, from the mpmath oracle (``reference_oracles.ift_partials``) or from the
independent one-step surrogate (``phi_one_step``, P6-D2 (b)), never from the backward under
test.

- ``gradcheck`` (float64, tight solver ``tau_l = 1e-14``) on Phi's full ``(B, 4)`` output wrt
  ``(w, nu, ln r, q, s)``, on the A1 closed form and on ``log_kappa`` (P7-D7 (a)).
- IFT vs surrogate and IFT vs mpmath to ``tau_Q`` (docs/3 section 12.1 tiers 1-2 rule).
- Higher-order derivatives raise (``create_graph=True`` refused on every route; P7-Q1, P7-B4).
- 100 random forward/backward passes per torch ladder member: finite objective, finite
  gradients, non-zero global gradient norm (Phase 7 gate item 2).
- The A3.1 study (P7-B3 (a), ``@slow``) writes ``reports/a31_fd_study.json``.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from torch import Tensor

import reference_oracles as ro
from domain_strategies import (
    BLIND_IDX,
    LN_S_FLOOR,
    PROTOCOL,
    SOLVER,
    SPOT_ROWS,
    STATS,
    TAU_Q,
    Rows,
    phi_rows,
    require,
    require_le,
)
from src.config import load_settings
from src.data.normstats import NormStats
from src.models.checkpoint import build_model
from src.models.heads import LadderNet
from src.models.registry import ModelSpec, nn_grid, write_json
from src.physics.manning_layer import (
    ManningPhi,
    PhiConfig,
    manning_residual,
    phi,
    phi_one_step,
)
from src.physics.shape import log_kappa, section_terms, t_of_r

pytestmark = pytest.mark.unit

PHI = PhiConfig.from_solver(SOLVER)
TIGHT = PhiConfig(
    tau_l=1e-14, k_max=200, quad_level=SOLVER.quad_level, quad_trunc=SOLVER.quad_trunc
)
TIGHT_A1 = PhiConfig(
    tau_l=1e-14, k_max=200, quad_level=SOLVER.quad_level, quad_trunc=SOLVER.quad_trunc, kappa=False
)
GC_EXAMPLES = max(5, settings.default.max_examples // 20)
GC = {"eps": 1e-6, "atol": 1e-6, "rtol": 1e-4}  # P7-D7 (a)
NAMES = ("w", "r", "nu", "q", "s")
REPO = Path(__file__).resolve().parents[2]


def ift_grads(rows: Rows, cfg: PhiConfig) -> tuple[Tensor, Tensor]:
    """``ell*`` and the per-row IFT gradient ``d ell/d(w, r, nu, q, s)`` (B, 5)."""
    leaves = [x.clone().requires_grad_() for x in rows.args()]
    ell, _ = ManningPhi.apply(*leaves, cfg)
    grads = torch.autograd.grad(ell.sum(), leaves)
    return ell.detach(), torch.stack(grads, dim=1)


def _log_r_rows(rows: Rows) -> tuple[Tensor, ...]:
    return tuple(
        x.clone().requires_grad_() for x in (rows.w, rows.nu, torch.log(rows.r), rows.q, rows.s)
    )


# ----------------------------------------------------------------------------- gradcheck
@given(rows=phi_rows(max_size=4))
@settings(max_examples=GC_EXAMPLES)
def test_gradcheck_phi_full_output(rows: Rows) -> None:
    """P7-D7 (a): gradcheck of Phi's (B, 4) output wrt (w, nu, ln r, q, s) at tau_l = 1e-14.

    docs/2 Prop 3.3 (IFT partials), Theorem 4.1 (affine outputs), section 3.4 (no unrolling).
    """

    def fn(w: Tensor, nu: Tensor, ln_r: Tensor, q: Tensor, s: Tensor) -> Tensor:
        return phi(w, torch.exp(ln_r), nu, q, s, TIGHT).out

    assert torch.autograd.gradcheck(fn, _log_r_rows(rows), **GC), "gradcheck (docs/2 Prop 3.3)"


@given(rows=phi_rows(max_size=4))
@settings(max_examples=GC_EXAMPLES)
def test_gradcheck_a1_and_log_kappa(rows: Rows) -> None:
    """gradcheck of the A1 closed form (docs/2 3.2, kappa = 1) and of ``log_kappa`` wrt
    (ell, w, ln r) with the analytic partials (-eta, +eta, k_r) (docs/2 Props 2.7, 2.8, 8.4).
    """

    def fn_a1(w: Tensor, nu: Tensor, ln_r: Tensor, q: Tensor, s: Tensor) -> Tensor:
        return phi(w, torch.exp(ln_r), nu, q, s, TIGHT_A1).out

    assert torch.autograd.gradcheck(fn_a1, _log_r_rows(rows), **GC), "A1 gradcheck (docs/2 3.2)"
    ell = phi(*rows.args(), TIGHT).ell
    leaves = tuple(x.clone().requires_grad_() for x in (ell, rows.w, torch.log(rows.r)))

    def fn_k(e: Tensor, w: Tensor, ln_r: Tensor) -> Tensor:
        return log_kappa(e, w, torch.exp(ln_r), level=SOLVER.quad_level, trunc=SOLVER.quad_trunc)

    assert torch.autograd.gradcheck(fn_k, leaves, **GC), "log_kappa gradcheck (docs/2 Prop 8.4)"


# ----------------------------------------------------------------------------- IFT vs oracles
@given(rows=phi_rows())
def test_ift_matches_one_step_surrogate(rows: Rows) -> None:
    """docs/2 Prop 8.4: the IFT gradient (analytic F_beta, F_r) equals the surrogate's gradient
    (autograd through the quadrature integrand, P6-D2 (b)) to tau_Q, per row and input.
    """
    _, g_ift = ift_grads(rows, PHI)
    leaves = [x.clone().requires_grad_() for x in rows.args()]
    g_sur = torch.stack(torch.autograd.grad(phi_one_step(*leaves, PHI).sum(), leaves), dim=1)
    err = (g_ift - g_sur).abs().amax(dim=1)
    require_le(err, TAU_Q, "|grad_IFT - grad_surrogate| (max over inputs)", "docs/2 Prop 8.4", rows)


def test_ift_matches_mpmath_partials() -> None:
    """docs/2 Prop 3.3 at Phi's float64 root, against the mpmath u-integrals (40 digits), to
    tau_Q (tiers 1-2 rule, docs/3 section 12.1); tight solver so the root is converged.
    """
    rows = Rows.from_si(SPOT_ROWS)
    ell, g = ift_grads(rows, TIGHT)
    parts = [
        ro.ift_partials(float(ell[i]), float(rows.w[i]), float(rows.r[i])) for i in range(rows.n)
    ]
    want = torch.tensor([[float(p[k]) for k in NAMES] for p in parts], dtype=torch.float64)
    for j, name in enumerate(NAMES):
        require_le(
            (g[:, j] - want[:, j]).abs(), TAU_Q, f"d ell/d{name} vs mpmath", "Prop 3.3", rows
        )


# ----------------------------------------------------------------------------- first order only
FIRST_ORDER = "first order only"


def test_higher_order_derivatives_raise() -> None:
    """docs/2 section 3.4, P7-Q1 and P7-B4 (a): ManningPhi and _LogKappa are first order only;
    any backward with ``create_graph=True`` raises instead of building a wrong second derivative.

    Routes (the F1 / P7-B4 reproducers): ``grad(inputs=q)`` of ``ell^2`` via ``phi()`` (silently
    0 under ``once_differentiable``; FD 0.7108), the linear loss, ``.backward()``, and M4's
    ``R_M`` via ``log_kappa`` (silently 5.235 vs about 2 D^2 = 4.93). First-order use is intact.
    """
    rows = Rows.from_si(SPOT_ROWS[:3])
    leaves = [x.clone().requires_grad_() for x in rows.args()]
    for loss in (lambda e: (e**2).sum(), lambda e: e.sum()):
        with pytest.raises(RuntimeError, match=FIRST_ORDER):
            torch.autograd.grad(loss(phi(*leaves, PHI).ell), [leaves[3]], create_graph=True)
    with pytest.raises(RuntimeError, match=FIRST_ORDER):
        (phi(*leaves, PHI).ell ** 2).sum().backward(create_graph=True)
    ell_d = phi(*rows.args(), PHI).ell.clone().requires_grad_()
    rho = manning_residual(ell_d, *rows.args(), PHI)
    with pytest.raises(RuntimeError, match=FIRST_ORDER):
        torch.autograd.grad((rho**2).sum(), [ell_d], create_graph=True)
    (g_q,) = torch.autograd.grad((phi(*leaves, PHI).ell ** 2).sum(), [leaves[3]])
    assert bool(torch.isfinite(g_q).all()), "first-order gradient must still work (Prop 3.3)"


# ----------------------------------------------------------------------------- 100-pass sweep
A4_DROP = STATS.names.index("D50_m")
A4_STATS = NormStats(
    names=tuple(n for j, n in enumerate(STATS.names) if j != A4_DROP),
    kinds=tuple(k for j, k in enumerate(STATS.kinds) if j != A4_DROP),
    offsets=tuple(o for j, o in enumerate(STATS.offsets) if j != A4_DROP),
    mean=tuple(m for j, m in enumerate(STATS.mean) if j != A4_DROP),
    std=tuple(s for j, s in enumerate(STATS.std) if j != A4_DROP),
    n_fit=STATS.n_fit,
)
A4_BLIND = torch.tensor([j - (j > A4_DROP) for j in BLIND_IDX.tolist() if j != A4_DROP])
MEMBERS: tuple[tuple[str, str | None], ...] = (
    ("M2-MLP", None),
    ("M3", None),
    ("M4", None),
    ("M5", None),
    ("M6-NN", None),
    ("M5", "A1"),
    ("M5", "A2"),
    ("M5", "A3-SP"),
    ("M5", "A3-SR"),
    ("M5", "wA1"),
    ("M5", "A4"),
    ("M2-MLP", "A4"),
    ("M4", "A4"),
    ("M6-NN", "A4"),
)
PASSES = 100
SWEEP_B = 16
WARM = PROTOCOL.uq.warm_start_epochs


def _member(model: str, ablation: str | None) -> tuple[LadderNet, int, Tensor]:
    torch.manual_seed(0)
    a4 = ablation == "A4"
    stats, blind = (A4_STATS, A4_BLIND) if a4 else (STATS, BLIND_IDX)
    net = build_model(
        ModelSpec(model, ablation), PROTOCOL, stats=stats, blind_idx=blind,
        hyper=nn_grid(PROTOCOL, model)[0], r_bar=2.0 if model == "M3" else None,
    )  # fmt: skip
    return net, len(stats.names), blind


def _sweep_batch(seed: int, n_features: int, blind: Tensor) -> dict[str, Tensor]:
    """Seeded rows of D with Phi-truth targets plus noise (``t > 0`` for the M6 labels)."""
    g = torch.Generator().manual_seed(seed)

    def u(lo: float, hi: float) -> Tensor:
        return torch.empty(SWEEP_B, dtype=torch.float64).uniform_(lo, hi, generator=g)

    w, r = u(0.0, math.log(500.0)), u(0.5, 10.0)
    nu, q = u(math.log(0.015), math.log(0.25)), u(math.log(0.01), math.log(1e4))
    s = u(LN_S_FLOOR, math.log(0.1))
    ell = phi(w, r, nu, q, s, PHI).ell
    y = torch.stack([w, ell, t_of_r(r), w + ell], dim=1)
    y = y + 0.05 * torch.randn(SWEEP_B, 4, dtype=torch.float64, generator=g)
    y[:, 2] = y[:, 2].abs()
    return {
        "x": torch.randn(SWEEP_B, n_features, dtype=torch.float64, generator=g),
        "x_blind_idx": blind,
        "phys": torch.stack([q, s], dim=1),
        "y": y,
        "omega": torch.full((SWEEP_B,), 1.0 / SWEEP_B, dtype=torch.float64),
        "row": torch.arange(SWEEP_B),
    }


@pytest.mark.parametrize(
    ("model", "ablation"), MEMBERS, ids=[m if a is None else f"{m}_{a}" for m, a in MEMBERS]
)
def test_100_pass_finite_nonzero_gradients(model: str, ablation: str | None) -> None:
    """Phase 7 gate item 2: 100 random forward/backward passes, finite objective, finite
    gradients on every trainable parameter that receives one, non-zero global gradient norm.

    docs/2 Theorem 8.2 and Corollary 8.3 (Phi cannot cause explosion), section 8.5 (sigma floor);
    epochs alternate between the MSE warm start (0) and the (beta-)NLL phase (docs/3 4.2).
    A small Adam step after each pass moves the parameters off the initialization.
    """
    net, n_features, blind = _member(model, ablation)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3)
    for k in range(PASSES):
        batch = _sweep_batch(1000 * k + 7, n_features, blind)
        epoch = 0 if k % 2 == 0 else WARM
        total = net.objective(batch, net(batch), epoch).total
        where = f"{model} {ablation or ''} pass {k} epoch {epoch}"
        assert bool(torch.isfinite(total)), f"non-finite objective {float(total)!r} at {where}"
        grads = torch.autograd.grad(total, params, allow_unused=True)
        live = [gr for gr in grads if gr is not None]
        bad = [i for i, gr in enumerate(grads) if gr is not None and not bool(gr.isfinite().all())]
        assert not bad, f"non-finite gradient in parameters {bad} at {where} (docs/2 Thm 8.2)"
        norm = float(torch.sqrt(sum((gr**2).sum() for gr in live)))
        assert norm > 0.0, f"zero global gradient norm at {where} (docs/2 Cor 8.3)"
        opt.zero_grad(set_to_none=True)
        for p, gr in zip(params, grads, strict=True):
            p.grad = gr
        opt.step()


# ----------------------------------------------------------------------------- A3.1 study
A31_TAUS = (1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
A31_DELTAS = (1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
A31_GRID: dict[str, list[float]] = {
    "W_m": [1.0, 10.0, 100.0, 1000.0],
    "r": [0.5, 1.0, 2.0, 5.0, 10.0],
    "n": [0.02, 0.1],
    "Q_E_m3s": [0.1, 10.0, 1000.0],
    "S": [1e-5, 1e-3],
}
K_LN_A = np.linspace(-12.0, 4.0, 33).tolist()
K_R = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
K_STEPS = (1e-3, 5e-4)


def _cfg(tau: float) -> PhiConfig:
    return PhiConfig(
        tau_l=tau, k_max=400, quad_level=SOLVER.quad_level, quad_trunc=SOLVER.quad_trunc
    )


def _ift_at(ell: Tensor, rows: Rows) -> Tensor:
    """Algorithm Phi backward (Prop 3.3) evaluated at an arbitrary depth ``ell`` (B, 5)."""
    t = section_terms(ell, rows.w, rows.r, level=SOLVER.quad_level, trunc=SOLVER.quad_trunc)
    inv_d = 1.0 / (5.0 / 3.0 - (2.0 / 3.0) * t.eta)
    return torch.stack(
        [
            -(1.0 + (2.0 / 3.0) * t.eta) * inv_d,
            -(2.0 / 3.0) * t.k_r * inv_d,
            inv_d,
            inv_d,
            -0.5 * inv_d,
        ],
        dim=1,
    )


def _k_hessian(h: float) -> dict[str, float]:
    """Max |k_aa|, |k_ar|, |k_rr| over the grid by central FD in (ln a, r) at step ``h``."""
    grid = list(itertools.product(K_LN_A, K_R))
    la, r = (torch.tensor(v, dtype=torch.float64) for v in zip(*grid, strict=True))

    def k(dla: float, dr: float) -> Tensor:
        lv = la + dla
        return section_terms(
            lv, torch.zeros_like(lv), r + dr, level=SOLVER.quad_level, trunc=SOLVER.quad_trunc
        ).k

    k0 = k(0.0, 0.0)
    k_aa = (k(h, 0.0) - 2.0 * k0 + k(-h, 0.0)) / h**2
    k_rr = (k(0.0, h) - 2.0 * k0 + k(0.0, -h)) / h**2
    k_ar = (k(h, h) - k(h, -h) - k(-h, h) + k(-h, -h)) / (4.0 * h**2)
    return {
        name: float(v.abs().max()) for name, v in (("k_aa", k_aa), ("k_ar", k_ar), ("k_rr", k_rr))
    }


def _r6(x: float) -> float:
    return float(f"{x:.6e}")


@pytest.mark.slow
def test_a31_finite_difference_study() -> None:
    """docs/2 section 3.3, A3.1 (P7-D8 (a), P7-B3 (a)): the IFT-gradient error at an unconverged
    root is O(tau_l), and the second derivatives of k are bounded on a compact grid.

    Acceptance (docs/3 section 12.4): least-squares log-log slope of max grad error vs tau_l over
    tau in {1e-4..1e-12} >= 0.95; per row err(tau) <= 1.05 L_g |ell - ell*| + 1e-14 with L_g from
    a delta sweep of the Algorithm Phi backward; tight IFT vs central FD (h = 1e-5) <= 1e-6;
    FD Hessian maxima of k finite and step-consistent (h = 1e-3 vs 5e-4) within 1e-4 relative.
    Writes the deterministic ``reports/a31_fd_study.json``.
    """
    rows = Rows.from_si(list(itertools.product(*A31_GRID.values())))  # type: ignore[arg-type]
    ell_ref, g_ref = ift_grads(rows, _cfg(1e-14))
    base = _ift_at(ell_ref, rows)
    l_g = max(
        float(((_ift_at(ell_ref + sgn * d, rows) - base).abs().amax(dim=1) / d).max())
        for d in A31_DELTAS
        for sgn in (1.0, -1.0)
    )
    sweep, errs = [], []
    for tau in A31_TAUS:
        ell, g = ift_grads(rows, _cfg(tau))
        err_row, off_row = (g - g_ref).abs().amax(dim=1), (ell - ell_ref).abs()
        bound = 1.05 * l_g * off_row + 1e-14
        require(err_row <= bound, err_row - bound, "<= 1.05 L_g |ell - ell*| + 1e-14",
                f"IFT grad error at tau = {tau:.0e}", "docs/2 A3.1", rows)  # fmt: skip
        errs.append(float(err_row.max()))
        sweep.append(
            {
                "tau_l": tau,
                "max_grad_err": _r6(errs[-1]),
                "max_root_offset": _r6(float(off_row.max())),
                "err_over_tau": _r6(errs[-1] / tau),
            }
        )
    slope = float(np.polyfit(np.log10(A31_TAUS), np.log10(errs), 1)[0])
    fd = torch.zeros_like(g_ref)
    h = 1e-5
    for j in range(5):
        up, dn = list(rows.args()), list(rows.args())
        up[j], dn[j] = up[j] + h, dn[j] - h
        fd[:, j] = (phi(*up, _cfg(1e-14)).ell - phi(*dn, _cfg(1e-14)).ell) / (2.0 * h)
    fd_err = (g_ref - fd).abs().amax(dim=1)
    hess = {step: _k_hessian(step) for step in K_STEPS}
    rel = {
        name: abs(hess[K_STEPS[0]][name] - hess[K_STEPS[1]][name]) / hess[K_STEPS[1]][name]
        for name in hess[K_STEPS[1]]
    }
    report: dict[str, Any] = {
        "study": "A3.1 finite-difference study (docs/2 3.3; P7-D8 (a), P7-B3 (a))",
        "params_sha256": load_settings().params_sha256,
        "grid": {**A31_GRID, "rows": rows.n},
        "tau_sweep": sweep,
        "slope_loglog": _r6(slope),
        "L_g": _r6(l_g),
        "C_max_err_over_tau": _r6(max(e / t for e, t in zip(errs, A31_TAUS, strict=True))),
        "fd_check": {"h": h, "max_abs_err": _r6(float(fd_err.max())), "tol": 1e-6},
        "k_second_derivatives": {
            "grid": {"ln_a": [K_LN_A[0], K_LN_A[-1], len(K_LN_A)], "r": K_R},
            "max_abs": {str(st): {k: _r6(v) for k, v in hv.items()} for st, hv in hess.items()},
            "rel_step_diff": {k: _r6(v) for k, v in rel.items()},
        },
        "acceptance": {"slope_min": 0.95, "row_factor": 1.05, "row_abs": 1e-14, "fd_tol": 1e-6,
                       "hessian_rel_tol": 1e-4},
    }  # fmt: skip
    write_json(REPO / "reports" / "a31_fd_study.json", report)
    assert slope >= 0.95, f"A3.1 log-log slope {slope:.4f} < 0.95 (docs/2 3.3); sweep {sweep}"
    require_le(fd_err, 1e-6, "|grad_IFT(tight) - central FD|", "docs/2 A3.1", rows)
    for name, v in rel.items():
        maxima = {st: hv[name] for st, hv in hess.items()}
        assert all(math.isfinite(m) for m in maxima.values()), f"{name} not finite: {maxima}"
        assert v <= 1e-4, f"{name} FD maxima not step-consistent: rel {v:.3e} > 1e-4 ({maxima})"

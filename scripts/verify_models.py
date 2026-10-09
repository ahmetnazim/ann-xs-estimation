"""Phase 6 synthetic sanity script for the scientific core (physics, ladder, losses, artefacts).

Sources: PHASE_06 Step 5 and gate item (4); P6-D3 (a) (quadrature level against an ``mpmath``
reference over ``[r_min, r_max] x beta``); P6-Q1 (a) (E-tier acceptance); proofs 2.3-2.8
(perimeter integrals), 3.1-3.6 and Algorithm Phi, Theorem 4.1 (invariants), Proposition 6.2
(Phi^-1), 8.5 (sigma floor), 9.2-9.3 (mixture moments and exact quantiles) and the hand examples
of proofs 10.2 (E1, E2, E3, E5, E11); protocol 4.1-4.2; P6-D6 (a) (no pickle), P6-D7 (a)
(OpenMP isolation), ADL-051.

**Synthetic data only.** No site table, fold table or outer-test row is read. The end-to-end
section writes a synthetic site/fold table to a temporary directory and trains under a reduced
``params.yaml`` (tiny backbone, 6 epochs); its ``params_sha256`` differs from the frozen file and
is recorded. Every check compares a computed value with a stated limit. The run writes a JSON
report (``runtime.reports_dir/verify_models.json``; not a DVC stage) headed by ``params_sha256`` and
the git state, prints a summary and exits non-zero if any check fails. A section that raises is
recorded as a failed check, and the remaining sections still run.

Usage::

    python scripts/verify_models.py                  # full run (a few minutes, 1 thread)
    python scripts/verify_models.py --quick          # reduced quadrature grid and batch sizes
    python scripts/verify_models.py --only quad,hand --out /tmp/v.json

Owning phase: 6. Kept and committed as the Phase 6 sanity script; the formal tests are Phase 7.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import mpmath as mp
import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor

from src.config import FT_TO_M, ProtocolParams, Settings, load_settings
from src.data.dataset import BLIND_EXCLUDED, SiteDataset
from src.data.normstats import NormStats
from src.models import losses as ls
from src.models import registry as rg
from src.models.backbone import assert_row_wise
from src.models.baselines import M0PowerLaw, MNull, aj_log_area, blind_features, log_targets
from src.models.checkpoint import (
    MODEL_FILE,
    NORMSTATS_FILE,
    RUN_FILE,
    STATE_FILE,
    build_model,
    load_model,
    save_model,
)
from src.models.ensemble import ensemble_roughness, mixture_cdf, mixture_moments, mixture_quantile
from src.models.heads import BoundedTransforms, LadderNet, sigma_from_raw
from src.models.m3_physics import m3_r_bar
from src.models.runner import FitRequest, Tables, run_fit
from src.physics import manning_layer as ml
from src.physics.quadrature import perimeter_integrals
from src.physics.shape import section_terms, t_of_r
from src.utils.logging import run_header

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPORT_NAME: Final[str] = "verify_models.json"

#: (r, beta) grid of the P6-D3 level study: 25 values of r over [r_min, r_max] = [0.5, 10]
#: (with the r = 1 seam) x 23 values of beta at half decades from 1e-8 to 1e3.
R_GRID: Final[tuple[float, ...]] = (
    0.5, 0.501, 0.55, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 1.0, 1.001, 1.01, 1.1,
    1.25, 1.4, 1.49, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0,
)  # fmt: skip
B_GRID: Final[tuple[float, ...]] = tuple(10.0 ** (e / 2) for e in range(-16, 7))
R_QUICK: Final[tuple[float, ...]] = (0.5, 0.99, 1.0, 1.01, 2.0, 10.0)
B_QUICK: Final[tuple[float, ...]] = (1e-8, 1e-4, 1e-2, 1.0, 1e2, 1e3)
#: Candidate rules of the level study (informational; the params rule is the checked one).
LEVELS: Final[tuple[int, ...]] = (4, 5, 6, 7)
TRUNCS: Final[tuple[float, ...]] = (3.0, 3.5, 4.0, 4.5)
MP_DPS: Final[int] = 40
#: The two exact substitutions must agree far below tau_Q for the reference to be trusted.
REF_AGREEMENT: Final[float] = 1e-18
#: Central-difference step and limit of the IFT check (truncation ~ h^2, rounding ~ eps / h).
FD_STEP: Final[float] = 1e-5
FD_LIMIT: Final[float] = 1e-8
#: Optimizer steps of the synthetic loss-decrease check (PHASE_06 Step 5).
LOSS_STEPS: Final[int] = 50
#: Artefact suffixes that would indicate pickle-based serialization (P6-D6 (a), ADL-076).
PICKLE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt"}
)


# ----------------------------------------------------------------------------- report
@dataclass(frozen=True)
class Check:
    """One pass/fail comparison of a computed value with its limit and source."""

    section: str
    name: str
    passed: bool
    value: float | int | bool | str
    limit: str
    source: str


@dataclass
class Report:
    """Accumulates checks, informational values and section timings."""

    checks: list[Check] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)

    def add(
        self,
        section: str,
        name: str,
        passed: bool,
        value: float | int | bool | str,
        limit: str,
        source: str,
    ) -> bool:
        """Record a check; a non-finite float value always fails and is stored as text."""
        ok = bool(passed)
        if isinstance(value, float) and not math.isfinite(value):
            value, ok = repr(value), False
        self.checks.append(Check(section, name, ok, value, limit, source))
        return ok

    def flag(self, section: str, name: str, ok: bool, source: str, limit: str = "true") -> bool:
        """Record a boolean property."""
        return self.add(section, name, ok, bool(ok), limit, source)

    def le(self, section: str, name: str, value: float, bound: float, source: str) -> bool:
        """Record ``value <= bound``; ``value`` must be finite."""
        v = float(value)
        ok = math.isfinite(v) and v <= bound
        return self.add(section, name, ok, v, f"<= {bound:.3g}", source)

    @property
    def failures(self) -> list[Check]:
        """Failed checks in insertion order."""
        return [c for c in self.checks if not c.passed]

    def to_json(self, header: dict[str, Any]) -> dict[str, Any]:
        """Return the report document: header, summary, checks, info and timings."""
        return {
            "header": header,
            "summary": {"n_checks": len(self.checks), "n_failed": len(self.failures)},
            "checks": [asdict(c) for c in self.checks],
            "info": self.info,
            "seconds": self.seconds,
        }


@dataclass(frozen=True)
class Context:
    """Run settings shared by the sections."""

    settings: Settings
    quick: bool
    work: Path

    @property
    def protocol(self) -> ProtocolParams:
        """Return the frozen protocol parameters."""
        return self.settings.protocol


# ----------------------------------------------------------------------------- helpers
def _max_abs(t: Tensor) -> float:
    return float(t.detach().abs().max())


def _t64(*values: float) -> Tensor:
    return torch.tensor(values, dtype=torch.float64)


def _printed(rep: Report, sec: str, name: str, got: float, printed: str, src: str) -> None:
    """Tier-3 rule of P6-Q1 (a): accept within one unit of the last printed digit."""
    unit = 10.0 ** -len(printed.split(".")[1])
    rep.add(sec, name, abs(got - float(printed)) <= unit, got, f"{printed} +/- {unit:g}", src)


def _d9_batch(n: int, seed: int) -> tuple[Tensor, ...]:
    """Draw ``(w, r, nu, q, s)`` in the D9 domain (W 1-500 m, r 0.5-10, n 0.015-0.25)."""
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


def _tight(cfg: ml.PhiConfig) -> ml.PhiConfig:
    """Return the derivative-check solver (the O(tau_l) root error becomes negligible)."""
    return ml.PhiConfig(tau_l=1e-14, k_max=80, quad_level=cfg.quad_level, quad_trunc=cfg.quad_trunc)


def _ift(cfg: ml.PhiConfig, x: Sequence[Tensor]) -> tuple[Tensor, ...]:
    leaves = tuple(t.detach().clone().requires_grad_() for t in x)
    grads = torch.autograd.grad(ml.phi(*leaves, cfg).ell.sum(), leaves)
    return tuple(g.detach() for g in grads)


# ----------------------------------------------------------------------------- mpmath reference
def mp_u_form(beta: float, r: float) -> tuple[Any, Any, Any]:
    """Return ``(F - 1, F_beta, F_r)`` by the u-substitution (proofs 2.3-2.5), split at s = 1."""
    with mp.workdps(MP_DPS):
        b, rr = mp.mpf(beta), mp.mpf(r)

        def s(u: mp.mpf) -> mp.mpf:
            return b**2 * u ** (2 * rr - 2)

        pts: list[Any] = [0, 1]
        if rr != 1:
            us = b ** (-1 / (rr - 1))
            if 0 < us < 1:
                pts = [0, us, 1]
        fm1 = mp.quad(lambda u: s(u) / (mp.sqrt(1 + s(u)) + 1), pts, maxdegree=10)
        fb = mp.quad(lambda u: s(u) / mp.sqrt(1 + s(u)), pts, maxdegree=10) / b
        fr = mp.quad(lambda u: mp.log(u) * s(u) / mp.sqrt(1 + s(u)), pts, maxdegree=10)
        return fm1, fb, fr


def mp_tau_form(beta: float | mp.mpf, r: float) -> tuple[Any, Any, Any]:
    """Return ``(F - 1, F_beta, F_r)`` by the tau-substitution (proofs 2.5), split at s = 1."""
    with mp.workdps(MP_DPS):
        b, rr = mp.mpf(beta), mp.mpf(r)
        hp = 1 / rr - 1

        def g(t: mp.mpf) -> mp.mpf:
            return t**hp

        def h(t: mp.mpf) -> mp.mpf:
            return mp.sqrt(t ** (2 * hp) + b**2)

        pts: list[Any] = [0, 1]
        if rr != 1:
            ts = b ** (1 / hp)
            if 0 < ts < 1:
                pts = [0, ts, 1]
        fm1 = mp.quad(lambda t: b**2 / (h(t) + g(t)), pts, maxdegree=10) / rr
        fb = mp.quad(lambda t: b / h(t), pts, maxdegree=10) / rr
        corr = mp.quad(lambda t: g(t) * mp.log(t) / (h(t) * (h(t) + g(t))), pts, maxdegree=10)
        return fm1, fb, -fm1 / rr + b**2 * corr / rr**3


def mp_phi_ell(w: float, r: float, nu: float, q: float, s: float) -> mp.mpf:
    """Run Algorithm Phi in ``mpmath``: ``ell = ell_0 + (2/5) ln F(2 (r + 1) e^(ell - w), r)``.

    Picard iteration of proofs 3.3 (contraction ``L <= 2/5``) from the wide-channel depth
    ``ell_0 = (3/5)(q + nu - w - s/2)`` (proofs 3.2) until the increment is below ``1e-30``.
    """
    with mp.workdps(MP_DPS):
        ell0 = mp.mpf(3) / 5 * (mp.mpf(q) + mp.mpf(nu) - mp.mpf(w) - mp.mpf(s) / 2)
        ell = ell0
        for _ in range(200):
            beta = 2 * (mp.mpf(r) + 1) * mp.exp(ell - mp.mpf(w))
            new = ell0 + mp.mpf(2) / 5 * mp.log1p(mp_tau_form(beta, r)[0])
            if abs(new - ell) < mp.mpf("1e-30"):
                return new
            ell = new
    raise RuntimeError("mpmath Phi did not converge")


def _rel(got: float, ref: mp.mpf) -> float:
    with mp.workdps(MP_DPS):
        return float(abs(mp.mpf(got) - ref) / abs(ref))


def _rule_error(
    grid: Sequence[tuple[float, float]], refs: Sequence[tuple[Any, ...]], level: int, trunc: float
) -> float:
    """Max relative error of ``(F - 1, F_beta, F_r)`` of one rule over ``grid``."""
    beta = torch.tensor([b for _, b in grid], dtype=torch.float64)
    r = torch.tensor([rr for rr, _ in grid], dtype=torch.float64)
    got = perimeter_integrals(beta, r, level=level, trunc=trunc)
    worst = 0.0
    for j in range(3):
        col = got[j].tolist()
        for i, ref in enumerate(refs):
            worst = max(worst, _rel(col[i], ref[j]))
    return worst


# ----------------------------------------------------------------------------- physics sections
def section_quad(rep: Report, ctx: Context) -> None:
    """Check the params quadrature rule against a 40-digit ``mpmath`` reference (P6-D3 (a)).

    The reference is computed by the u- and the tau-substitution (both exact for every
    ``r > 0``, proofs 2.5); their agreement certifies it. The params rule must reproduce
    ``F - 1``, ``F_beta`` and ``F_r`` to ``tau_Q / 10`` (the margin of the Batch A tests). The
    level study over ``LEVELS x TRUNCS`` is stored as information. The closed forms of
    Proposition 2.4 (``r = 1/2, 1, 2``) are checked at the same tolerance.
    """
    sec, solver = "quad", ctx.protocol.solver
    rs, bs = (R_QUICK, B_QUICK) if ctx.quick else (R_GRID, B_GRID)
    grid = [(r, b) for r in rs for b in bs]
    refs: list[tuple[Any, ...]] = []
    agree = 0.0
    for r, b in grid:
        u_ref, tau_ref = mp_u_form(b, r), mp_tau_form(b, r)
        with mp.workdps(MP_DPS):
            for x, y in zip(u_ref, tau_ref, strict=True):
                agree = max(agree, float(abs(x - y) / abs(y)))
        refs.append(tau_ref)
    rep.info["quad_grid_points"] = len(grid)
    rep.le(sec, "reference: u-form vs tau-form max rel", agree, REF_AGREEMENT, "proofs 2.5")
    tol = solver.tau_q / 10
    err = _rule_error(grid, refs, solver.quad_level, solver.quad_trunc)
    name = f"params rule (level {solver.quad_level}, trunc {solver.quad_trunc}) max rel error"
    rep.le(sec, name, err, tol, "P6-D3 (a); solver.tau_q / 10")
    study = {f"L{lv}_T{tr}": _rule_error(grid, refs, lv, tr) for tr in TRUNCS for lv in LEVELS}
    rep.info["quad_level_study_max_rel"] = study
    passing = [lv for lv in LEVELS if study[f"L{lv}_T{solver.quad_trunc}"] <= tol]
    rep.info["quad_min_passing_level_at_params_trunc"] = min(passing) if passing else None
    closed: dict[float, Callable[[Any], Any]] = {
        1.0: lambda b: mp.sqrt(1 + b**2),
        2.0: lambda b: (mp.sqrt(1 + b**2) + mp.asinh(b) / b) / 2,
        0.5: lambda b: mp.sqrt(1 + b**2) + b**2 * mp.asinh(1 / b),
    }
    betas = [10.0**e for e in (-6, -4, -2, -1, 0, 1, 2)]
    worst = 0.0
    for r, form in closed.items():
        fm1, fb, _ = perimeter_integrals(
            torch.tensor(betas, dtype=torch.float64),
            torch.full((len(betas),), r, dtype=torch.float64),
            level=solver.quad_level,
            trunc=solver.quad_trunc,
        )
        with mp.workdps(MP_DPS):
            for i, b in enumerate(betas):
                worst = max(worst, _rel(float(fm1[i]), form(mp.mpf(b)) - 1))
                worst = max(worst, _rel(float(fb[i]), mp.diff(form, mp.mpf(b))))
    rep.le(sec, "Prop 2.4 closed forms (r = 1/2, 1, 2) max rel", worst, tol, "Prop 2.4")


def section_hand(rep: Report, ctx: Context) -> None:
    """Check the hand examples of proofs 10.2 and an ``mpmath`` Algorithm Phi (E1, E2).

    Printed values are accepted within one unit of their last printed digit (P6-Q1 (a)). The
    ``mpmath`` fixed point must match the params solver to ``tau_l`` (stopping rule of
    Proposition 3.2) and the tight solver to ``1e-13``.
    """
    sec = "hand"
    cfg = ml.PhiConfig.from_solver(ctx.protocol.solver)
    tight = _tight(cfg)
    lv, tr = cfg.quad_level, cfg.quad_trunc

    def logs(w: float, r: float, n: float, q: float, s: float) -> tuple[Tensor, ...]:
        return _t64(math.log(w)), _t64(r), _t64(math.log(n)), _t64(math.log(q)), _t64(math.log(s))

    examples = (
        ("E1", (20.0, 2.0, 0.035, 10.0, 1e-3), ("0.70117", "0.70323", "0.99268")),
        ("E2", (3.0, 1.0, 0.04, 1.0, 0.01), ("0.298513", "0.30796", "0.92505")),
    )
    for tag, raw, printed in examples:
        w, r, nu, q, s = logs(*raw)
        d0 = math.exp(float(ml.wide_channel_depth(w, nu, q, s)[0]))
        _printed(rep, sec, f"{tag} wide-channel depth", d0, printed[0], "proofs 10.2")
        res = ml.phi(w, r, nu, q, s, cfg)
        _printed(rep, sec, f"{tag} dbar*", math.exp(float(res.ell[0])), printed[1], "proofs 10.2")
        kappa = float(section_terms(res.ell, w, r, level=lv, trunc=tr).kappa[0])
        _printed(rep, sec, f"{tag} kappa*", kappa, printed[2], "proofs 10.2")
        ref = mp_phi_ell(*(float(v[0]) for v in (w, r, nu, q, s)))
        ell_tight = float(ml.phi(w, r, nu, q, s, tight).ell[0])
        e_cfg = _abs(float(res.ell[0]), ref)
        rep.le(sec, f"{tag} |ell - ell_mp| (params solver)", e_cfg, cfg.tau_l, "Prop 3.2")
        rep.le(sec, f"{tag} |ell - ell_mp| (tight solver)", _abs(ell_tight, ref), 1e-13, "Prop 3.2")
    w, r, nu, q, s = logs(20.0, 2.0, 0.035, 10.0, 1e-3)
    out = ml.phi(w, r, nu, q, s, cfg).out[0]
    _printed(rep, sec, "E1 d_max", math.exp(float(out[1])), "1.05485", "proofs 10.2")
    _printed(rep, sec, "E1 A", math.exp(float(out[2])), "14.0646", "proofs 10.2")
    _printed(rep, sec, "E1 V", math.exp(float(out[3])), "0.71101", "proofs 10.2")
    for r_e3, printed_f in ((0.5, "1.03497"), (1.0, "1.01980"), (2.0, "1.02606")):
        fm1 = perimeter_integrals(_t64(2.0 * r_e3 * 0.1), _t64(r_e3), level=lv, trunc=tr)[0]
        _printed(rep, sec, f"E3 F(delta = 0.1, r = {r_e3})", 1.0 + float(fm1[0]), printed_f,
                 "proofs 10.2")  # fmt: skip
    with mp.workdps(MP_DPS):
        exact = bool(mp.mpf(repr(FT_TO_M)) ** 3 == mp.mpf("0.028316846592"))
    rep.flag(sec, "E5 lambda_L^3 = 0.028316846592 exactly", exact, "proofs 10.2 E5")
    k_us = FT_TO_M ** (-1.0 / 3.0)
    _printed(rep, sec, "E5 k_US", k_us, "1.48592", "proofs 10.2 E5")
    _printed(rep, sec, "E5 k_US^(3/5)", k_us**0.6, "1.26822", "proofs 10.2 E5")
    chi = -math.log(FT_TO_M)
    d_ft = math.exp(float(ml.wide_channel_depth(w + chi, nu - chi / 3.0, q + 3.0 * chi, s)[0]))
    _printed(rep, sec, "E11 dbar (ft)", d_ft, "2.30042", "proofs 10.2 E11")
    _printed(rep, sec, "E11 dbar (ft -> m)", d_ft * FT_TO_M, "0.701169", "proofs 10.2 E11")


def _abs(got: float, ref: mp.mpf) -> float:
    with mp.workdps(MP_DPS):
        return float(abs(mp.mpf(got) - ref))


def section_grad(rep: Report, ctx: Context) -> None:
    """Check the closed-form IFT backward of Phi (Prop 3.3, Algorithm Phi; P6-D2 (a)).

    IFT gradients against central differences and against the one-step surrogate (P6-D2 (b),
    kept as a cross-check), the Proposition 3.3 ranges (invariant 6 of Theorem 4.1), the
    Proposition 3.6 bound on ``d ell / dr`` and non-convergence raising.
    """
    sec = "grad"
    cfg = ml.PhiConfig.from_solver(ctx.protocol.solver)
    tight = _tight(cfg)
    x = _d9_batch(60 if ctx.quick else 200, seed=5)
    ift = _ift(tight, x)
    leaves = tuple(t.clone().requires_grad_() for t in x)
    sur = torch.autograd.grad(ml.phi_one_step(*leaves, tight).sum(), leaves)
    fd_err, sur_err = 0.0, 0.0
    for j in range(5):
        up, dn = list(x), list(x)
        up[j], dn[j] = x[j] + FD_STEP, x[j] - FD_STEP
        fd = (ml.phi(*up, tight).ell - ml.phi(*dn, tight).ell) / (2 * FD_STEP)
        fd_err = max(fd_err, _max_abs(ift[j] - fd))
        sur_err = max(sur_err, _max_abs(ift[j] - sur[j]))
    rep.le(sec, "max |IFT - central FD| (h = 1e-5, tight solver)", fd_err, FD_LIMIT, "Prop 3.3")
    rep.le(sec, "max |IFT - one-step surrogate|", sur_err, 1e-12, "P6-D2 (b)")
    gw, gr, gnu, gq, gs = _ift(cfg, x)
    r = x[1]
    rep.flag(sec, "d ell/dq in [3/5, 1) (invariant 6)", bool((gq >= 0.6).all() and (gq < 1).all()),
             "Thm 4.1 #6; Prop 3.3")  # fmt: skip
    rep.flag(sec, "d ell/dnu == d ell/dq", bool(torch.equal(gq, gnu)), "Prop 3.3")
    ok_w = bool((gw > -5.0 / 3.0).all() and (gw <= -0.6).all())
    rep.flag(sec, "d ell/dw in (-5/3, -3/5]", ok_w, "Prop 3.3")
    rep.flag(sec, "d ell/ds == -(1/2) d ell/dq", bool(torch.equal(gs, -0.5 * gq)), "Prop 3.3")
    bound = 2.0 * (2.0 * r + 1.0) / (3.0 * r * (r + 1.0))
    rep.flag(sec, "|d ell/dr| <= 2(2r+1)/(3r(r+1))", bool((gr.abs() <= bound).all()), "Prop 3.6")
    e2 = (_t64(math.log(3.0)), _t64(1.0), _t64(math.log(0.04)), _t64(0.0), _t64(math.log(0.01)))
    short = ml.PhiConfig(
        tau_l=cfg.tau_l, k_max=2, quad_level=cfg.quad_level, quad_trunc=cfg.quad_trunc
    )
    try:
        ml.phi(*e2, short)
        raised = False
    except ml.PhiConvergenceError:
        raised = True
    rep.flag(sec, "non-convergence raises (E2, K_max = 2)", raised, "Algorithm Phi")


def section_phi(rep: Report, ctx: Context) -> None:
    """Check Phi on a D9-domain batch: Theorem 4.1, Propositions 3.4-3.5, iterations, timing.

    Invariants 1-4 (``ln A = w + ell``, ``ln V = q - ln A``, ``|rho| <= (5/3) tau_l``,
    ``d_max / dbar = 1 + 1/r``), per-row freezing (iterations in ``1 .. K_max``), bit-identical
    permutation and subset invariance of outputs and IFT gradients (Proposition 3.4) and unit
    equivariance (Proposition 3.5, ft units, tight solver, ``1e-12``). Iteration statistics and
    single-thread timings at B = 256 and 1024 are informational (P6-D9 context).
    """
    sec = "phi"
    cfg = ml.PhiConfig.from_solver(ctx.protocol.solver)
    n = 2000 if ctx.quick else 20000
    x = _d9_batch(n, seed=0)
    w, r, nu, q, s = x
    res = ml.phi(*x, cfg)
    ell, ln_dmax, ln_a, ln_v = res.out.unbind(dim=1)
    rep.flag(sec, "ln A == w + ell (invariant 1)", bool(torch.equal(ln_a, w + ell)), "Thm 4.1 #1")
    ok_v = bool(torch.equal(ln_v, q - w - ell))
    rep.flag(sec, "ln V == q - ln A (invariant 2)", ok_v, "Thm 4.1 #2")
    rho = ml.manning_residual(ell, w, r, nu, q, s, cfg)
    rep.le(sec, "max |rho| (invariant 3)", _max_abs(rho), (5.0 / 3.0) * cfg.tau_l, "Thm 4.1 #3")
    ratio = _max_abs(torch.exp(ln_dmax - ell) / (1.0 + 1.0 / r) - 1.0)
    rep.le(sec, "d_max / dbar vs 1 + 1/r, max rel (invariant 4)", ratio, 1e-14, "Thm 4.1 #4")
    it = res.iterations.double()
    ok_it = int(it.min()) >= 1 and int(it.max()) <= cfg.k_max
    rep.add(sec, "Picard iterations in 1..K_max", ok_it, int(it.max()), f"<= {cfg.k_max}", "3.3")
    rep.info["phi_iterations"] = {
        "rows": n,
        "mean": float(it.mean()),
        "p99": float(it.quantile(0.99)),
        "max": int(it.max()),
    }
    m = 512
    sub_x = tuple(t[:m] for t in x)
    full = ml.phi(*sub_x, cfg)
    perm = torch.randperm(m, generator=torch.Generator().manual_seed(3))
    permuted = ml.phi(*(t[perm] for t in sub_x), cfg)
    ok_p = bool(torch.equal(permuted.out, full.out[perm])) and bool(
        torch.equal(permuted.iterations, full.iterations[perm])
    )
    rep.flag(sec, "permutation invariance, outputs (bit-identical)", ok_p, "Prop 3.4")
    g_full, g_perm = _ift(cfg, sub_x), _ift(cfg, [t[perm] for t in sub_x])
    ok_g = all(bool(torch.equal(a, b[perm])) for a, b in zip(g_perm, g_full, strict=True))
    rep.flag(sec, "permutation invariance, IFT gradients (bit-identical)", ok_g, "Prop 3.4")
    idx = torch.tensor([5, 0, m - 1, 17, 200])
    ok_s = bool(torch.equal(ml.phi(*(t[idx] for t in sub_x), cfg).out, full.out[idx]))
    rep.flag(sec, "subset invariance (bit-identical)", ok_s, "Prop 3.4")
    tight = _tight(cfg)
    chi = -math.log(FT_TO_M)
    ew, er, enu, eq, es = (t[:500] for t in x)
    si = ml.phi(ew, er, enu, eq, es, tight).out
    ft = ml.phi(ew + chi, er, enu - chi / 3.0, eq + 3.0 * chi, es, tight).out
    shift = chi * _t64(1.0, 1.0, 2.0, 1.0)[None, :]
    rep.le(sec, "unit equivariance (ft units) max abs", _max_abs(ft - si - shift), 1e-12, "3.5")
    timing: dict[str, float] = {}
    reps = 5 if ctx.quick else 20
    for b in (256, 1024):
        xb = tuple(t[:b].clone().requires_grad_() for t in x)
        ml.phi(*xb, cfg)  # warm-up
        t0 = time.perf_counter()
        for _ in range(reps):
            ml.phi(*xb, cfg)
        t1 = time.perf_counter()
        for _ in range(reps):
            torch.autograd.backward(ml.phi(*xb, cfg).out.sum())
        t2 = time.perf_counter()
        timing[f"B{b}_forward_ms"] = 1e3 * (t1 - t0) / reps
        timing[f"B{b}_forward_backward_ms"] = 1e3 * (t2 - t1) / reps
    rep.info["phi_timing_ms_1_thread"] = timing


# ----------------------------------------------------------------------------- synthetic tables
def synthetic_site_table(protocol: ProtocolParams, n: int, seed: int) -> pd.DataFrame:
    """Return a primary-variant synthetic site table with every NN, M1, target and physics column.

    Positive targets with ``A = W dbar``; ``S >= S_floor`` (D9); ``V_E`` missing on every 10th
    row (ADL-066), ``F_M1_missing`` on every 11th, ``F_S`` on every 9th row.
    """
    rng = np.random.default_rng(seed)
    pos = np.arange(n)
    da = np.exp(rng.normal(5.0, 1.0, n))
    q = 0.02 * da**0.9 * np.exp(rng.normal(0, 0.2, n))
    s = np.where(pos % 4 == 0, np.exp(rng.normal(-5.0, 0.3, n)), np.exp(rng.normal(-8, 0.8, n)))
    s = np.maximum(s, protocol.domain.s_floor)
    w = 4.0 * q**0.5 * np.exp(rng.normal(0, 0.2, n))
    dbar = 0.3 * q**0.35 * np.exp(rng.normal(0, 0.2, n))
    t = rng.uniform(0.2, 0.9, n)
    df = pd.DataFrame(
        {
            "site_no": [f"{i:08d}" for i in range(n)],
            "variant": "primary",
            "F_DA": False,
            "F_Qsel": False,
            "F_S": pos % 9 == 0,
            "division": np.where(pos % 2 == 0, "A", "B"),
            "DA_km2": da,
            "stream_order": rng.integers(1, 8, n).astype(float),
            "z_min_m": rng.normal(200, 80, n),
            "S": s,
            "Q_E": q,
            "Q_C": q * np.exp(rng.normal(0, 0.1, n)),
            "V_E": np.where(pos % 10 == 3, np.nan, q / (w * dbar)),
            "W": w,
            "dbar": dbar,
            "t_obs": t,
            "dmax": dbar * np.exp(t),
            "A": w * dbar,
            "D50_m": np.exp(rng.normal(math.log(0.01), 1.0, n)),
            "AI": np.exp(rng.normal(0.0, 0.4, n)),
            "damdensws": np.where(pos % 3 == 0, 0.0, rng.uniform(0.01, 0.4, n)),
        }
    )
    for name, kind in protocol.transforms.features.items():
        if name not in df:
            df[name] = np.exp(rng.normal(1.0, 0.5, n)) if kind == "log" else rng.normal(10, 3, n)
    for name in rg.ModelSpec("M1").features(protocol):
        if name not in df:
            df[name] = rng.uniform(0.0, 50.0, n)
    missing = pos % 11 == 5
    df.loc[missing, "CAT_EVI_AMJ_2012"] = np.nan
    df["F_M1_missing"] = missing
    return df


def synthetic_fold_table(sites: pd.Series) -> pd.DataFrame:
    """Return one S-ID block: every 5th site is test, inner fold = position mod 3, ES mod 7."""
    n = len(sites)
    pos = np.arange(n)
    is_test = pos % 5 == 0
    out = pd.DataFrame(
        {
            "family": "S-ID",
            "repeat": np.int8(0),
            "outer_fold": np.int8(0),
            "site_no": sites.to_numpy(),
            "is_test": is_test,
            "inner_fold": np.where(is_test, -1, pos % 3).astype("int8"),
        }
    )
    for m in range(1, 11):
        out[f"es_member_{m}"] = ~is_test & (pos % 7 == m % 7)
    return out


def _view(
    protocol: ProtocolParams, frame: pd.DataFrame, features: Sequence[str]
) -> tuple[SiteDataset, NormStats]:
    names = tuple(features)
    x = frame[list(names)].to_numpy(dtype=np.float64)
    tf = protocol.transforms
    stats = NormStats.fit(x, names, tf.features, tf.log_offset_frac)
    return SiteDataset(frame, np.arange(len(frame)), names, stats, len(frame)), stats


def nn_specs() -> list[rg.ModelSpec]:
    """Return every torch rung: M2-MLP, M3-M6-NN, the M5 ablations and the A4 variants."""
    base = [rg.ModelSpec(m) for m in rg.NN_MODELS]
    m5 = [rg.ModelSpec("M5", a) for a in ("A1", "A2", "A3-SP", "A3-SR", "wA1")]
    a4 = [rg.ModelSpec(m, "A4") for m in rg.NN_MODELS if m in rg.ABLATIONS["A4"]]
    return base + m5 + a4


def _build(
    spec: rg.ModelSpec,
    protocol: ProtocolParams,
    frame: pd.DataFrame,
    seed: int,
) -> tuple[LadderNet, SiteDataset]:
    ds, stats = _view(protocol, frame, spec.features(protocol))
    r_bar = m3_r_bar(frame["t_obs"].to_numpy()) if spec.model == "M3" else None
    hyper = rg.nn_grid(protocol, spec.model)[0]
    torch.manual_seed(seed)
    model = build_model(
        spec, protocol, stats=stats, blind_idx=ds.x_blind_idx, hyper=hyper, r_bar=r_bar
    )
    return model, ds


def _subset(batch: dict[str, Tensor], idx: Tensor, n: int) -> dict[str, Tensor]:
    return {k: (v[idx] if v.dim() > 0 and int(v.shape[0]) == n else v) for k, v in batch.items()}


def _late_epoch(protocol: ProtocolParams) -> int:
    """First (beta-)NLL epoch: every sigma head is in the objective (P6-Q3 (a))."""
    return protocol.uq.warm_start_epochs


# ----------------------------------------------------------------------------- ladder sections
def section_rungs(rep: Report, ctx: Context) -> None:
    """Smoke-test every torch rung at the frozen params (PHASE_06 Step 5, prompt items 3-4).

    Per rung: ``get_num_trainable_params``; no batch-coupled layer (``assert_row_wise``,
    Proposition 3.4 scope); ``mu`` of shape (B, 4), float64 on CPU and finite; sigma
    ``>= sigma_min``; a finite, non-zero gradient on every trainable parameter at the first
    (beta-)NLL epoch; permutation invariance of the whole model (bit-identical) and subset
    invariance to ``1e-12`` (the MLP matrix products pick batch-size-dependent kernels, so a
    4-row subset differs from the full batch by rounding, about 1e-16; Phi itself is checked
    bit-identically in ``section_phi``); the routing zero-gradient test ``d nu / d x_j == 0`` for
    ``j`` in Q_E, S, F_S (M4, M5, M6-NN ``g_nu``; protocol 4.1 ``I_blind``); V_E never a feature;
    Theorem 4.1 spot checks on M5 and M6-NN.
    """
    sec, protocol = "rungs", ctx.protocol
    frame = synthetic_site_table(protocol, 128 if ctx.quick else 256, seed=11)
    n = len(frame)
    tr = BoundedTransforms.from_domain(protocol.domain)
    params_counts: dict[str, int] = {}
    for spec in nn_specs():
        tag = spec.name
        model, ds = _build(spec, protocol, frame, seed=0)
        feats = ds.features
        rep.flag(sec, f"{tag}: V_E not a feature", "V_E" not in feats, "protocol 4.1")
        blind_names = tuple(feats[j] for j in ds.x_blind_idx.tolist())
        rep.flag(sec, f"{tag}: x_blind_idx = I_blind", blind_names == blind_features(feats),
                 "protocol 4.1")  # fmt: skip
        assert_row_wise(model)
        n_params = model.get_num_trainable_params()
        expect = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_counts[tag] = n_params
        rep.flag(sec, f"{tag}: get_num_trainable_params", n_params == expect > 0, "prompt item 3")
        batch = ds[list(range(n))]
        model.train()
        out = model(batch)
        mu = out.mu
        ok_mu = (
            tuple(mu.shape) == (n, 4)
            and mu.dtype == torch.float64
            and mu.device.type == "cpu"
            and bool(torch.isfinite(mu).all())
        )
        rep.flag(sec, f"{tag}: mu (B, 4) float64 CPU finite", ok_mu, "Step 5")
        if out.sigma is not None:
            ok_s = bool(torch.isfinite(out.sigma).all()) and bool(
                (out.sigma >= protocol.uq.sigma_min).all()
            )
            rep.flag(sec, f"{tag}: sigma finite and >= sigma_min", ok_s, "proofs 8.5 (ii)")
        obj = model.objective(batch, out, _late_epoch(protocol))
        rep.flag(sec, f"{tag}: objective finite", bool(torch.isfinite(obj.total)), "Step 5")
        model.zero_grad(set_to_none=True)
        torch.autograd.backward(obj.total)
        bad = [
            name
            for name, p in model.named_parameters()
            if p.requires_grad
            and (p.grad is None or not bool(torch.isfinite(p.grad).all()) or not bool(p.grad.any()))
        ]
        rep.add(sec, f"{tag}: finite non-zero grad on every parameter", not bad,
                ", ".join(bad) or "all", "none missing", "Step 5")  # fmt: skip
        model.eval()
        with torch.no_grad():
            ref = model(batch).mu
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(1))
            ok_p = bool(torch.equal(model(_subset(batch, perm, n)).mu, ref[perm]))
            sub = torch.tensor([3, 0, n - 1, 41])
            d_sub = _max_abs(model(_subset(batch, sub, n)).mu - ref[sub])
        rep.flag(sec, f"{tag}: permutation invariance (bit-identical)", ok_p, "Prop 3.4")
        rep.le(sec, f"{tag}: subset invariance, 4 rows (max abs)", d_sub, 1e-12, "Prop 3.4")
        if spec.model in ("M4", "M5", "M6-NN"):
            _routing(rep, sec, tag, model, batch, feats, "g_nu" if spec.model == "M6-NN" else "nu")
        if spec.model in ("M5", "M6-NN"):
            _theorem_4_1(rep, sec, tag, model, batch, tr)
    rep.info["n_trainable_params"] = params_counts


def _routing(
    rep: Report,
    sec: str,
    tag: str,
    model: LadderNet,
    batch: dict[str, Tensor],
    feats: Sequence[str],
    key: str,
) -> None:
    """Zero-gradient routing test of the nu head (protocol 4.1 ``I_blind``; P6-D4 (a))."""
    out_cols = [j for j, f in enumerate(feats) if f in BLIND_EXCLUDED]
    in_cols = [j for j, f in enumerate(feats) if f not in BLIND_EXCLUDED]
    model.train()
    leaf = batch["x"].clone().requires_grad_()
    out = model({**batch, "x": leaf})
    (g,) = torch.autograd.grad(out.extras[key].sum(), leaf)
    zero = bool((g[:, out_cols] == 0).all())
    live = bool((g[:, in_cols] != 0).any())
    rep.flag(sec, f"{tag}: d {key}/d x == 0 for Q_E, S, F_S", zero and len(out_cols) == 3,
             "prompt item 4; protocol 4.1")  # fmt: skip
    rep.flag(sec, f"{tag}: d {key}/d x != 0 for some I_blind feature", live, "P6-D4 (a)")


def _theorem_4_1(
    rep: Report,
    sec: str,
    tag: str,
    model: LadderNet,
    batch: dict[str, Tensor],
    tr: BoundedTransforms,
) -> None:
    """Theorem 4.1 spot checks on a physics rung: invariants 1, 3, 4 (via t) and 7."""
    with torch.no_grad():
        out = model(batch)
    mu, ex = out.mu, out.extras
    cfg = model.cfg.phi  # type: ignore[union-attr]
    q, s = batch["phys"][:, 0], batch["phys"][:, 1]
    rep.flag(sec, f"{tag}: ln A == w + ell", bool(torch.equal(mu[:, 3], mu[:, 0] + mu[:, 1])),
             "Thm 4.1 #1")  # fmt: skip
    d_t = _max_abs(mu[:, 2] - t_of_r(ex["r"]))  # exact for M5; M6 builds r = r(t)
    rep.le(sec, f"{tag}: t = ln(1 + 1/r) (max abs)", d_t, 1e-14, "Thm 4.1 #4")
    rho = ml.manning_residual(mu[:, 1], mu[:, 0], ex["r"], ex["nu"], q, s, cfg)
    rep.le(sec, f"{tag}: max |rho|", _max_abs(rho), (5.0 / 3.0) * cfg.tau_l, "Thm 4.1 #3")
    nu_max, ln_r_max = tr.nu_min + tr.delta_n, tr.ln_r_min + tr.delta_r
    ok_nu = bool((ex["nu"] >= tr.nu_min).all() and (ex["nu"] <= nu_max).all())
    ln_r = torch.log(ex["r"])
    ok_r = bool((ln_r >= tr.ln_r_min - 1e-15).all() and (ln_r <= ln_r_max + 1e-15).all())
    rep.flag(sec, f"{tag}: nu and r inside the closed bands", ok_nu and ok_r, "Thm 4.1 #7")


def section_losses(rep: Report, ctx: Context) -> None:
    """Check that every rung objective decreases over 50 AdamW steps on a synthetic table.

    Full-batch steps at the params learning rate, without weight decay, at a fixed epoch: the
    MSE warm-start epoch 0 for M2-MLP, M4 and M5 (P6-Q3 (a)) and the first (beta-)NLL epoch for
    every rung. Also the Proposition 8.5 (ii) bounds with the sigma floor and their beta-NLL form
    (stop-gradient factor ``sigma^(2 beta_N)``, direct differentiation, valid for
    ``beta_N <= 1/2``): ``|d/d mu| <= |e| sigma_min^(2 beta_N - 2)`` and
    ``|d/d z_sigma| <= sigma_min^(2 beta_N - 1) + e^2 sigma_min^(2 beta_N - 3)``.
    """
    sec, protocol = "losses", ctx.protocol
    frame = synthetic_site_table(protocol, 128 if ctx.quick else 256, seed=12)
    n = len(frame)
    specs = [rg.ModelSpec(m) for m in rg.NN_MODELS]
    if not ctx.quick:
        specs += [rg.ModelSpec("M5", a) for a in ("A1", "A2", "A3-SP", "A3-SR", "wA1")]
    trace: dict[str, dict[str, float]] = {}
    for spec in specs:
        epochs = [0, _late_epoch(protocol)] if spec.model in rg.WARM_START_MODELS else [0]
        if spec.model in ("M3", "M6-NN"):
            epochs = [0]
        for epoch in epochs:
            model, ds = _build(spec, protocol, frame, seed=1)
            batch = ds[list(range(n))]
            opt = torch.optim.AdamW(model.parameters(), lr=protocol.model.optimizer.lr,
                                    weight_decay=0.0)  # fmt: skip
            model.train()
            values = []
            for _ in range(LOSS_STEPS + 1):
                opt.zero_grad(set_to_none=True)
                total = model.objective(batch, model(batch), epoch).total
                values.append(float(total.detach()))
                torch.autograd.backward(total)
                opt.step()
            first, last = values[0], values[-1]
            tag = f"{spec.name} epoch {epoch}"
            trace[tag] = {"first": first, "last": last}
            rep.add(sec, f"{tag}: objective decreases over {LOSS_STEPS} steps",
                    math.isfinite(last) and last < first, last, f"< {first:.6g}",
                    "Step 5")  # fmt: skip
    rep.info["loss_first_last"] = trace
    s_min, beta_n = protocol.uq.sigma_min, protocol.uq.beta_nll
    g = torch.Generator().manual_seed(2)
    z = (6 * torch.randn(400, 1, dtype=torch.float64, generator=g)).requires_grad_()
    mu = torch.randn(400, 1, dtype=torch.float64, generator=g).requires_grad_()
    y = torch.randn(400, 1, dtype=torch.float64, generator=g)
    one, w1 = torch.ones(400, dtype=torch.float64), torch.ones(1, dtype=torch.float64)
    e = (mu - y).detach().abs()
    for beta in (0.0, beta_n):
        sigma = sigma_from_raw(z, s_min)
        total = ls.gaussian_nll(mu, sigma, y, one, w1, beta_n=beta)
        g_mu, g_z = torch.autograd.grad(total, (mu, z))
        b_mu = e * s_min ** (2 * beta - 2)
        b_z = s_min ** (2 * beta - 1) + e**2 * s_min ** (2 * beta - 3)
        tol = 1 + 1e-12
        ok = bool((g_mu.abs() <= b_mu * tol).all() and (g_z.abs() <= b_z * tol).all())
        rep.flag(sec, f"gradient bounds with sigma floor (beta_N = {beta})", ok, "Prop 8.5 (ii)")
    point = e**2 / (2 * sigma_from_raw(z, s_min).detach() ** 2) + torch.log(
        sigma_from_raw(z, s_min).detach()
    )
    rep.flag(sec, "per-point NLL >= ln sigma_min", bool((point >= math.log(s_min)).all()),
             "Prop 8.5 (ii)")  # fmt: skip


def section_ensemble(rep: Report, ctx: Context) -> None:
    """Check the ensemble: exact mixture quantiles, moments and the ensemble roughness.

    ``Q_p`` lies in ``[min_m(mu_m + z_p sigma_m), max_m(mu_m + z_p sigma_m)]`` and
    ``Psi(Q_p) = p`` (proofs 9.3); the law of total variance (proofs 9.2); a single member reduces
    to ``mu + z_p sigma``; ``nu* = Phi^-1(mu*)`` recovers ``nu`` on Manning-consistent rows to
    ``(5/3) tau_l`` (Proposition 6.2 and the stopping rule of Proposition 3.2).
    """
    sec, uq = "ensemble", ctx.protocol.uq
    g = torch.Generator().manual_seed(11)
    m, b = uq.ensemble_m, 200 if ctx.quick else 1000
    mu = torch.randn(m, b, 4, dtype=torch.float64, generator=g)
    sigma = uq.sigma_min + torch.rand(m, b, 4, dtype=torch.float64, generator=g)
    for p in (uq.alpha / 2, 0.5, 1 - uq.alpha / 2):
        qp = mixture_quantile(p, mu, sigma)
        z = float(torch.special.ndtri(torch.tensor(p, dtype=torch.float64)))
        ends = mu + z * sigma
        inside = bool((qp >= ends.min(0).values).all() and (qp <= ends.max(0).values).all())
        rep.flag(sec, f"Q_{p:g} inside the bracket", inside, "proofs 9.3")
        err = _max_abs(mixture_cdf(qp, mu, sigma) - p)
        rep.le(sec, f"|Psi(Q_{p:g}) - {p:g}|", err, 1e-10, "proofs 9.3")
    mix = mixture_moments(mu, sigma)
    var = (sigma**2).mean(0) + mu.var(0, unbiased=False)
    rep.le(sec, "law of total variance max abs", _max_abs(mix.sigma_star**2 - var), 1e-12, "9.2")
    z95 = float(torch.special.ndtri(torch.tensor(0.95, dtype=torch.float64)))
    one = mixture_quantile(0.95, mu[:1], sigma[:1])
    rep.le(sec, "M = 1: Q_0.95 = mu + z sigma", _max_abs(one - mu[0] - z95 * sigma[0]), 1e-12,
           "proofs 9.3")  # fmt: skip
    cfg = ml.PhiConfig.from_solver(ctx.protocol.solver)
    w, r, nu, q, s = _d9_batch(500, seed=21)
    ell = ml.phi(w, r, nu, q, s, cfg).ell
    mu_star = torch.stack([w, ell, t_of_r(r), w + ell], dim=1)
    nu_back = ensemble_roughness(mu_star, torch.stack([q, s], dim=1), cfg)
    rep.le(sec, "Phi^-1 recovers nu (max abs)", _max_abs(nu_back - nu), (5.0 / 3.0) * cfg.tau_l
           * (1 + 1e-6), "Prop 6.2")  # fmt: skip


# ----------------------------------------------------------------------------- isolation, e2e
def _python(code: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # fixed argv, no shell
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )


def _env() -> dict[str, str]:
    path = os.environ.get("PYTHONPATH", "")
    return {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (str(REPO_ROOT), path) if p)}


def section_isolation(rep: Report, ctx: Context) -> None:
    """Check the import isolation of the XGB path in fresh interpreters (P6-D7 (a), ADL-051).

    ``src.models.baselines``, ``registry``, ``runner`` and the utilities import neither torch nor
    xgboost; the torch rungs never import xgboost; ``src.models`` imports no submodule; the XGB
    worker entry point is torch-free.
    """
    sec = "isolation"
    cases = {
        "baselines/registry/runner/utils import neither torch nor xgboost": (
            "import sys, src.models.baselines, src.models.registry, src.models.runner, "
            "src.evaluation.metrics, src.utils.seeding, src.utils.logging; "
            "print('torch' in sys.modules, 'xgboost' in sys.modules)",
            "False False",
        ),
        "torch rungs do not import xgboost": (
            "import sys, src.models.backbone, src.models.heads, src.models.losses, "
            "src.models.m2_mlp, src.models.m3_physics, src.models.m4_soft, src.models.m5_hard, "
            "src.models.m6_two_stage, src.models.ensemble, src.models.trainer, "
            "src.models.checkpoint; print('xgboost' in sys.modules)",
            "False",
        ),
        "src.models imports no submodule": (
            "import sys, src.models; "
            "print(sorted(m for m in sys.modules if m.startswith('src.models.')), "
            "'torch' in sys.modules)",
            "[] False",
        ),
        "XGB worker entry point is torch-free": (
            "import sys; from src.models.baselines import main; print('torch' in sys.modules)",
            "False",
        ),
    }
    for name, (code, want) in cases.items():
        proc = _python(code, _env())
        got = proc.stdout.strip() if proc.returncode == 0 else f"exit {proc.returncode}"
        rep.add(sec, name, got == want, got, want, "P6-D7 (a); ADL-051")


def _small_params(work: Path) -> Path:
    """Write a reduced ``params.yaml`` (same schema; tiny backbone, 6 epochs, small grids)."""
    raw = yaml.safe_load((REPO_ROOT / "params.yaml").read_text())
    raw["model"]["backbone"].update(hidden_layers=1, hidden_width=8)
    raw["model"]["optimizer"].update(batch_size=16, max_epochs=6, patience=2)
    raw["model"]["inner_grid"].update(weight_decay=[0.0, 1.0e-4], lr=[1.0e-3], m4_lambda_max=[10.0])
    raw["model"]["xgb"].update(
        max_depth=[2, 3], eta=[0.3], min_child_weight=[1], max_rounds=20, early_stopping_rounds=5
    )
    raw["model"]["baselines"]["m1_max_depth"] = [2]
    raw["uq"]["warm_start_epochs"] = 2
    path = work / "params_small.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def section_e2e(rep: Report, ctx: Context) -> None:
    """Run one tiny end-to-end synthetic fit per ladder model, with artefact round trips.

    Under a reduced ``params.yaml``: every torch rung (members, ``--grid 0``) through
    :func:`src.models.runner.run_fit`; M6-XGB in-process with its isolated booster worker; M1 and
    M2-XGB through ``scripts/train.py`` in a torch-free interpreter; M-null, A_J and M0 on the
    synthetic outer-training rows. Checks: outer-test rows blanked before any fit (P6-D1),
    artefact sets (P6-D6 (a)), a no-pickle ``save -> load -> save`` round trip with identical
    state hashes and bit-identical predictions on non-test rows, and no pickle-type file anywhere.
    """
    sec = "e2e"
    small = load_settings(_small_params(ctx.work))
    rep.info["e2e_params_sha256"] = small.params_sha256
    frame = synthetic_site_table(small.protocol, 60, seed=11)
    folds = synthetic_fold_table(frame["site_no"])
    tables = Tables(frame, folds, "a" * 64, "b" * 64)
    root = ctx.work / "checkpoints"
    n_test = int(folds["is_test"].sum())
    nontest = np.flatnonzero(~folds["is_test"].to_numpy())
    torch_specs = [rg.ModelSpec(m) for m in rg.NN_MODELS] + [
        rg.ModelSpec("M5", a) for a in ("A1", "A2", "A3-SP", "A3-SR", "wA1", "A4")
    ]
    for spec in [*torch_specs, rg.ModelSpec("M6-XGB")]:
        req = FitRequest(spec, "primary", "S-ID", 0, 0, "member", 1, 0)
        doc = run_fit(req, tables, small, root)
        tag = spec.name
        rep.add(sec, f"{tag}: outer-test rows blanked", doc["rows"]["test_blanked"] == n_test,
                doc["rows"]["test_blanked"], f"== {n_test}", "P6-D1 (a)")  # fmt: skip
        member = rg.fit_dir(root, spec, "primary", "S-ID", 0, 0, "member", 1)
        files = {p.name for p in member.iterdir()}
        want = (
            {"booster_nu.json", "booster_w.json", "booster_t.json", RUN_FILE}
            if spec.model == "M6-XGB"
            else {STATE_FILE, MODEL_FILE, NORMSTATS_FILE, RUN_FILE}
        )
        rep.add(sec, f"{tag}: artefact set", files == want, ", ".join(sorted(files)),
                ", ".join(sorted(want)), "P6-D6 (a)")  # fmt: skip
        if spec.is_nn:
            hist = doc["fit"]["history"]
            finite = all(math.isfinite(h["train"]) and math.isfinite(h["monitor"]) for h in hist)
            rep.flag(sec, f"{tag}: finite epoch history", finite and len(hist) > 0, "Step 5")
            if spec.model in rg.WARM_START_MODELS:
                ok_best = doc["fit"]["best_epoch"] >= small.protocol.uq.warm_start_epochs
                rep.flag(sec, f"{tag}: best epoch after the warm start", ok_best, "P6-D5 (i)")
            _round_trip(rep, sec, tag, member, small, frame, nontest)
    _xgb_cli(rep, sec, ctx, small, tables)
    _closed_form_baselines(rep, sec, small.protocol, frame, nontest)
    _scan_artefacts(rep, sec, root)


def _round_trip(
    rep: Report,
    sec: str,
    tag: str,
    member: Path,
    settings: Settings,
    frame: pd.DataFrame,
    rows: np.ndarray,
) -> None:
    """``load -> save -> load`` with ``allow_pickle=False``: same state hash, same predictions."""
    sha = settings.params_sha256
    model, stats, doc = load_model(member, settings.protocol, params_sha256=sha)
    again = member.parent / f"{member.name}_roundtrip"
    spec = rg.ModelSpec(**doc["spec"])
    doc2 = save_model(
        again,
        model,
        spec=spec,
        stats=stats,
        blind_idx=doc["blind_idx"],
        hyper=rg.NnHyper(**doc["hyper"]),
        r_bar=doc["r_bar"],
        params_sha256=sha,
    )
    model2, _, _ = load_model(again, settings.protocol, params_sha256=sha)
    ds = SiteDataset(frame, rows, stats.names, stats, len(rows))
    batch = ds[list(range(len(rows)))]
    with torch.no_grad():
        same = bool(torch.equal(model(batch).mu, model2(batch).mu))
    ok = doc2["state"]["sha256"] == doc["state"]["sha256"] and same
    rep.flag(sec, f"{tag}: no-pickle round trip (hash and predictions)", ok, "P6-D6 (a)")
    try:
        load_model(member, settings.protocol, params_sha256="0" * 64)
        refused = False
    except ValueError:
        refused = True
    rep.flag(sec, f"{tag}: load refuses a different params_sha256", refused, "P6-D6 (a)")


def _xgb_cli(rep: Report, sec: str, ctx: Context, small: Settings, tables: Tables) -> None:
    """M1 and M2-XGB members through ``scripts/train.py`` in one torch-free interpreter."""
    site, folds = ctx.work / "site.parquet", ctx.work / "folds.parquet"
    tables.site_table.to_parquet(site)
    tables.folds.to_parquet(folds)
    ck = ctx.work / "checkpoints"
    script = str(REPO_ROOT / "scripts" / "train.py")
    common = [
        "--family", "S-ID", "--fold", "0", "--params", str(small.params_path),
        "--site-table", str(site), "--fold-table", str(folds), "--checkpoint-dir", str(ck),
        "--kind", "member", "--index", "1", "--grid", "0",
    ]  # fmt: skip
    code = (
        "import sys, runpy\n"
        "for model in ('M1', 'M2-XGB'):\n"
        f"    sys.argv = ['train.py', 'fit', '--model', model, *{common!r}]\n"
        "    try:\n"
        f"        runpy.run_path({script!r}, run_name='__main__')\n"
        "    except SystemExit as e:\n"
        "        assert e.code in (0, None), e.code\n"
        "print('torch' in sys.modules, 'xgboost' in sys.modules)\n"
    )
    proc = _python(code, _env())
    got = proc.stdout.strip().splitlines()[-1] if proc.returncode == 0 else proc.stderr[-300:]
    rep.add(sec, "M1, M2-XGB: CLI fits ran without torch", got == "False True", got,
            "False True", "ADL-051; P6-D7 (a)")  # fmt: skip
    for model, targets in (("M1", ("W", "dbar")), ("M2-XGB", ("W", "dbar", "dmax", "A"))):
        member = rg.fit_dir(ck, rg.ModelSpec(model), "primary", "S-ID", 0, 0, "member", 1)
        want = {f"booster_{t}.json" for t in targets} | {RUN_FILE}
        files = {p.name for p in member.iterdir()} if member.exists() else set()
        rep.add(sec, f"{model}: artefact set", files == want, ", ".join(sorted(files)),
                ", ".join(sorted(want)), "P6-D6 (a)")  # fmt: skip


def _closed_form_baselines(
    rep: Report, sec: str, protocol: ProtocolParams, frame: pd.DataFrame, rows: np.ndarray
) -> None:
    """M-null, A_J and M0 fitted and evaluated on the synthetic outer-training rows only."""
    sub = frame.iloc[rows]
    y = log_targets(sub)
    mnull = MNull.fit(sub["DA_km2"].to_numpy(), y).predict(sub["DA_km2"].to_numpy())
    rep.flag(sec, "M-null: finite predictions", all(bool(np.isfinite(v).all()) for v in
             mnull.values()), "protocol 4.1")  # fmt: skip
    ln_a, avail = aj_log_area(sub["Q_E"].to_numpy(), sub["V_E"].to_numpy())
    ok_aj = int(avail.sum()) == int(sub["V_E"].notna().sum()) and bool(
        np.isfinite(ln_a[avail]).all()
    )
    rep.flag(sec, "A_J: available rows = rows with V_E, finite there", ok_aj, "protocol 4.1")
    m0 = M0PowerLaw.fit(
        sub["division"].to_numpy(),
        sub["DA_km2"].to_numpy(),
        y[:, 0],
        y[:, 1],
        protocol.model.baselines.m0_min_division_sites,
    )
    pred, _pooled = m0.predict(sub["division"].to_numpy(), sub["DA_km2"].to_numpy())
    ok_m0 = all(bool(np.isfinite(v).all()) for v in pred.values()) and bool(
        np.array_equal(pred["A"], pred["W"] + pred["dbar"])
    )
    rep.flag(sec, "M0: finite, ln A = ln W + ln dbar", ok_m0, "protocol 4.1")


def _scan_artefacts(rep: Report, sec: str, root: Path) -> None:
    """No pickle-type file; every ``.npz`` / ``.npy`` loads with ``allow_pickle=False``."""
    files = [p for p in root.rglob("*") if p.is_file()]
    pickled = sorted(str(p.relative_to(root)) for p in files if p.suffix in PICKLE_SUFFIXES)
    rep.add(sec, "no pickle-type artefact", not pickled, ", ".join(pickled) or "none", "none",
            "P6-D6 (a); ADL-076")  # fmt: skip
    bad = []
    for p in files:
        if p.suffix in (".npz", ".npy"):
            try:
                with np.load(p, allow_pickle=False):
                    pass
            except (ValueError, OSError):
                bad.append(str(p.relative_to(root)))
    rep.add(sec, "every .npz/.npy loads with allow_pickle=False", not bad,
            ", ".join(bad) or f"{len(files)} files scanned", "all", "P6-D6 (a)")  # fmt: skip


# ----------------------------------------------------------------------------- driver
SECTIONS: Final[dict[str, Callable[[Report, Context], None]]] = {
    "quad": section_quad,
    "hand": section_hand,
    "grad": section_grad,
    "phi": section_phi,
    "rungs": section_rungs,
    "losses": section_losses,
    "ensemble": section_ensemble,
    "isolation": section_isolation,
    "e2e": section_e2e,
}


def run(names: Sequence[str], settings: Settings, *, quick: bool, work: Path) -> Report:
    """Run the named sections in order; an exception becomes a failed check of its section."""
    rep = Report()
    ctx = Context(settings=settings, quick=quick, work=work)
    for name in names:
        t0 = time.perf_counter()
        try:
            SECTIONS[name](rep, ctx)
        except Exception as exc:  # recorded as a failed check, never swallowed
            detail = "".join(traceback.format_exception_only(exc)).strip()
            rep.add(name, "section completed", False, detail[-500:], "no exception", "script")
        rep.seconds[name] = time.perf_counter() - t0
    return rep


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 6 synthetic sanity checks.")
    parser.add_argument("--quick", action="store_true", help="reduced grids and batch sizes")
    parser.add_argument("--only", default=None, help=f"comma list of {', '.join(SECTIONS)}")
    parser.add_argument("--out", type=Path, default=None, help="report path (JSON)")
    parser.add_argument("--params", type=Path, default=None, help="params.yaml (frozen default)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv``, run the checks, write the report; return 0 iff every check passed."""
    args = _parser().parse_args(argv)
    names = list(SECTIONS) if args.only is None else [s.strip() for s in args.only.split(",")]
    unknown = [s for s in names if s not in SECTIONS]
    if unknown:
        raise SystemExit(f"unknown sections {unknown}; choose from {list(SECTIONS)}")
    settings = load_settings(args.params) if args.params else load_settings()
    torch.set_num_threads(settings.runtime.torch_threads_per_worker)
    with tempfile.TemporaryDirectory(prefix="verify_models_") as tmp:
        rep = run(names, settings, quick=args.quick, work=Path(tmp))
    header = run_header(
        settings.params_sha256,
        REPO_ROOT,
        script="scripts/verify_models.py",
        sections=names,
        quick=args.quick,
        torch_threads=torch.get_num_threads(),
    )
    out = args.out or (REPO_ROOT / settings.runtime.reports_dir / REPORT_NAME)
    out.parent.mkdir(parents=True, exist_ok=True)
    rg.write_json(out, rep.to_json(header))
    for name in names:
        checks = [c for c in rep.checks if c.section == name]
        failed = [c for c in checks if not c.passed]
        status = "PASS" if not failed else f"FAIL ({len(failed)})"
        print(f"{name:10s} {status:10s} {len(checks):4d} checks  {rep.seconds[name]:7.1f} s")
    for c in rep.failures:
        print(f"  FAILED [{c.section}] {c.name}: {c.value!r} (limit {c.limit}; {c.source})")
    print(f"report: {out}")
    return 0 if not rep.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

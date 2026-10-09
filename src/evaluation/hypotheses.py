"""M* selection, H1-H4 decision functions and the outcome decision matrix.

Sources: ``0_point_zero.md`` section 3 (frozen H1-H4 wording); proofs section 9.5 (statistic
definitions); protocol section 4.4 (M* rule, Supported-iff / Falsified-iff table, H4 thresholds)
and section 4.7 (outcome decision matrix, no-loss margin +0.05 log units, O-10). Thresholds come
from ``params.yaml`` -> ``evaluation`` (:class:`src.config.EvaluationParams`) and ``domain``.

The functions take already computed intervals and shares; they never read predictions. They
are verified only on hand-built inputs in Phase 7 (P7-D1 (a)); no verdict on real data is
produced here.

Conventions fixed by the protocol text:
- "CI upper bound < 0" is strict; "includes 0" means ``lo <= 0 <= hi``, so ``hi = 0`` falsifies.
- H1: an S-IND CI "entirely above 0" means ``lo > 0``; it blocks support and is reported as a
  contradiction. ``supported`` and ``falsified`` can both be false (S-LRO passes, S-IND
  contradicts).
- H3: a one-split pass is falsified and flagged ``partial``.
- H4: falsified iff ``pi_nu > 0.10`` or ``pi_r > 0.10`` or ``pi_lit < 0.80`` or any of the three
  Spearman CIs has ``lo <= 0`` ("includes 0 or lies below 0").
- Outcome matrix (PI decision 2026-10-09): every matching row is returned without adjudication;
  "H2 supported only" means H2 supported and H1, H3 and H4 all not supported.

Torch-free and xgboost-free. Owning phase: 7 (P7-D1 (a)).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from src.config import DomainParams, EvaluationParams, H4Params

__all__ = [
    "CI",
    "H1Result",
    "H2Result",
    "H3Result",
    "H4Result",
    "boundary_share",
    "decide_h1",
    "decide_h2",
    "decide_h3",
    "decide_h4",
    "literature_share",
    "outcome_rows",
    "select_mstar",
]


@dataclass(frozen=True)
class CI:
    """A finite confidence interval ``[lo, hi]`` with ``lo <= hi``."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        """Validate finiteness and order."""
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)) or self.lo > self.hi:
            raise ValueError(f"invalid CI [{self.lo}, {self.hi}]")

    @property
    def upper_below_zero(self) -> bool:
        """``hi < 0`` (protocol 4.4 'Supported iff ... upper bound < 0')."""
        return self.hi < 0.0

    @property
    def entirely_above_zero(self) -> bool:
        """``lo > 0``."""
        return self.lo > 0.0


@dataclass(frozen=True)
class H1Result:
    """H1 decision (protocol 4.4)."""

    supported: bool
    falsified: bool
    ind_contradiction: bool


@dataclass(frozen=True)
class H2Result:
    """H2 decision (protocol 4.4)."""

    supported: bool
    falsified: bool


@dataclass(frozen=True)
class H3Result:
    """H3 decision (protocol 4.4); ``partial`` = exactly one of LRO and TAIL passes."""

    supported: bool
    falsified: bool
    partial: bool


@dataclass(frozen=True)
class H4Result:
    """H4 decision (protocol 4.4) with the falsification conditions that hold."""

    supported: bool
    falsified: bool
    reasons: tuple[str, ...]


def select_mstar(rmse_by_model: Mapping[str, float], cfg: EvaluationParams) -> str:
    """M*: the candidate with the lowest S-ID OOF ``RMSE_log`` (protocol 4.4), one target.

    The values are the mean over the three repeats. Raises on a missing or non-finite candidate
    and on an exact tie (the protocol gives no tie-break).
    """
    cands = cfg.mstar_candidates
    missing = [m for m in cands if m not in rmse_by_model]
    if missing:
        raise ValueError(f"missing M* candidates: {missing}")
    vals = {m: float(rmse_by_model[m]) for m in cands}
    if not all(math.isfinite(v) for v in vals.values()):
        raise ValueError(f"non-finite RMSE among candidates: {vals}")
    best = min(vals.values())
    winners = [m for m, v in vals.items() if v == best]
    if len(winners) != 1:
        raise ValueError(f"exact tie for M*: {winners}")
    return winners[0]


def decide_h1(ci_lro: CI, ci_ind: CI) -> H1Result:
    """H1 on ``Delta^H1_dbar = RMSE^(M5) - RMSE^(M*)`` (protocol 4.4).

    Supported iff S-LRO ``hi < 0`` and the S-IND CI does not lie entirely above 0. Falsified iff
    the S-LRO CI includes 0 or is positive (``hi >= 0``).
    """
    contradiction = ci_ind.entirely_above_zero
    return H1Result(
        supported=ci_lro.upper_below_zero and not contradiction,
        falsified=not ci_lro.upper_below_zero,
        ind_contradiction=contradiction,
    )


def decide_h2(ci: CI) -> H2Result:
    """H2 on ``|B^(M5)(Q1)| - |B^(M*)(Q1)|``: supported iff ``hi < 0`` (protocol 4.4)."""
    return H2Result(supported=ci.upper_below_zero, falsified=not ci.upper_below_zero)


def decide_h3(ci_lro: CI, ci_tail: CI) -> H3Result:
    """H3: supported iff both H3-LRO and H3-TAIL have ``hi < 0`` (protocol 4.4)."""
    passes = (ci_lro.upper_below_zero, ci_tail.upper_below_zero)
    both = all(passes)
    return H3Result(supported=both, falsified=not both, partial=sum(passes) == 1)


def boundary_share(x: ArrayLike, bounds: tuple[float, float], vartheta: float) -> float:
    """``pi_partial(vartheta)``: share with ``min(x - lo, hi - x) <= vartheta (hi - lo)`` (9.5).

    ``x`` and ``bounds`` are in the transformed (log) space: ``nu = ln n`` with ``Delta_n``, or
    ``ln r`` with ``Delta_r``.
    """
    a = np.asarray(x, dtype=np.float64)
    lo, hi = bounds
    if a.ndim != 1 or a.size == 0 or not bool(np.isfinite(a).all()):
        raise ValueError("x must be a non-empty finite 1-D array")
    if not lo < hi or not 0.0 < vartheta < 0.5:
        raise ValueError("need lo < hi and 0 < vartheta < 0.5")
    dist = np.minimum(a - lo, hi - a)
    return float(np.mean(dist <= vartheta * (hi - lo)))


def literature_share(n_hat: ArrayLike, domain: DomainParams) -> float:
    """``pi_lit``: share of ``n_hat`` in ``N_lit = [0.012, 0.33]`` (protocol 4.4 H4 (a))."""
    a = np.asarray(n_hat, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or not bool(np.isfinite(a).all()):
        raise ValueError("n_hat must be a non-empty finite 1-D array")
    lo, hi = domain.n_lit
    return float(np.mean((a >= lo) & (a <= hi)))


def decide_h4(
    pi_nu: float,
    pi_r: float,
    pi_lit: float,
    rho_cis: Mapping[str, CI],
    cfg: H4Params,
) -> H4Result:
    """H4 (protocol 4.4): falsified iff any pre-registered condition holds.

    ``rho_cis`` must hold the bootstrap CIs of ``rho_S(n, D50)``, ``rho_S(n, S)`` and
    ``rho_S(r, r_obs)`` under the keys ``n_d50``, ``n_s`` and ``r_robs``.
    """
    keys = ("n_d50", "n_s", "r_robs")
    if set(rho_cis) != set(keys):
        raise ValueError(f"rho_cis keys must be {keys}, got {sorted(rho_cis)}")
    for name, v in (("pi_nu", pi_nu), ("pi_r", pi_r), ("pi_lit", pi_lit)):
        if not (math.isfinite(v) and 0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be a share in [0, 1], got {v}")
    reasons: list[str] = []
    if pi_nu > cfg.pi_boundary_max:
        reasons.append("pi_nu")
    if pi_r > cfg.pi_boundary_max:
        reasons.append("pi_r")
    if pi_lit < cfg.pi_lit_min:
        reasons.append("pi_lit")
    reasons.extend(f"rho_{k}" for k in keys if rho_cis[k].lo <= 0.0)
    return H4Result(supported=not reasons, falsified=bool(reasons), reasons=tuple(reasons))


def outcome_rows(
    h1: H1Result,
    h2: H2Result,
    h3: H3Result,
    h4: H4Result,
    ci_h1_lro: CI,
    cfg: EvaluationParams,
) -> tuple[int, ...]:
    """Return every matching row of the protocol 4.7 outcome matrix (1-4), in order.

    1. H1 and H3 supported.
    2. H1 falsified, H4 supported, and S-LRO ``hi < no_loss_margin_log`` (+0.05).
    3. H2 supported, and H1, H3 and H4 all not supported.
    4. M5 worse: S-LRO ``lo > 0``.
    """
    rows: Sequence[tuple[int, bool]] = (
        (1, h1.supported and h3.supported),
        (2, h1.falsified and h4.supported and ci_h1_lro.hi < cfg.no_loss_margin_log),
        (3, h2.supported and not (h1.supported or h3.supported or h4.supported)),
        (4, ci_h1_lro.lo > 0.0),
    )
    return tuple(k for k, hit in rows if hit)

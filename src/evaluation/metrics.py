"""Primary, secondary and descriptive metrics and the physical-consistency analogues.

Sources: protocol sections 4.3.1 and 4.3.2; proofs section 9.5 (site-weighted statistics, S-RND
rows weighted ``1/|I_j|``). :func:`rmse_log` is the Phase 6 inner-selection statistic (P6-D1 (a);
protocol 3.4 item 2); the remaining metrics are implemented under P7-D1 (a) and verified only by
known-answer synthetic tests. Scoring of real predictions stays locked until the Phase 7 gate.

Weights: every mean-based metric accepts optional observation weights (normalized here to sum
to 1; ``None`` means one row per site, uniform). :func:`site_weights` gives the proofs 9.5 weights
``1/(J |I_j|)``. MdAPE is unweighted (a weighted median is not defined in protocol 4.3.1).

Torch-free and xgboost-free (usable in XGB workers, ADL-051). Owning phase: 7 (P7-D1 (a)).

Phase 8 additions (docs/3 13.6 and 13.9, ADL-117; known-answer tests before Event 1):
:func:`weighted_mdape` (the weighted median defined in docs/3 13.6 for S-RND rows weighted
``1/|I_j|``), :func:`v_error_shares` (protocol 4.3.2 V-error variance shares),
:func:`depth_ratio_r` and :func:`back_calc_consistency` (protocol 4.3.2 Manning back-calculation;
proofs Prop 6.2 and Prop 7.4; the back-calculated ``nu`` itself needs the torch quadrature and is
computed by the caller with :func:`src.physics.manning_layer.phi_inverse_nu`) and
:func:`s1_consistency` (protocol 4.6 S1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata

__all__ = [
    "CLOSURE_TOL_LOG",
    "BackCalcResult",
    "apb_pct",
    "back_calc_consistency",
    "bias",
    "bias_log",
    "closure_residual",
    "closure_summary",
    "depth_ratio_r",
    "implied_log_velocity",
    "kge",
    "mdape",
    "multiplicative_bias",
    "nse",
    "nse_log",
    "nse_raw",
    "pearson_r",
    "r2_pearson_raw",
    "rmse",
    "rmse_log",
    "rmse_raw",
    "s1_consistency",
    "shape_violation_share",
    "site_weights",
    "spearman_rho",
    "v_error_shares",
    "weighted_mdape",
]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

CLOSURE_TOL_LOG = math.log(1.05)
"""Protocol 4.3.2: share of predicted closure residuals with ``|eps_A| > ln 1.05``."""

WMEDIAN_TOL = 1e-12
"""Floating-point guard of the weighted median: a cumulative weight within ``1e-12`` of 1/2
counts as reaching it (e.g. ten weights of 0.1 accumulate to 0.5 only up to rounding)."""


def _vec(x: ArrayLike, name: str) -> FloatArray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {a.shape}")
    if not bool(np.isfinite(a).all()):
        raise ValueError(f"{name} contains non-finite values")
    return a


def _pair(pred: ArrayLike, obs: ArrayLike) -> tuple[FloatArray, FloatArray]:
    p, o = _vec(pred, "pred"), _vec(obs, "obs")
    if p.shape != o.shape:
        raise ValueError(f"need equal shapes, got {p.shape} and {o.shape}")
    return p, o


def _positive(p: FloatArray, o: FloatArray) -> None:
    if not (bool((p > 0).all()) and bool((o > 0).all())):
        raise ValueError("raw-space metrics need strictly positive predictions and observations")


def _weights(weights: ArrayLike | None, n: int) -> FloatArray:
    if weights is None:
        return np.full(n, 1.0 / n)
    w = _vec(weights, "weights")
    if w.size != n:
        raise ValueError(f"weights has {w.size} entries, expected {n}")
    if bool((w < 0).any()) or not float(w.sum()) > 0.0:
        raise ValueError("weights must be non-negative with a positive sum")
    return w / float(w.sum())


def site_weights(site_ids: ArrayLike) -> FloatArray:
    """Return ``omega_i = 1/(J |I_j|)`` for observation ``i`` at site ``j`` (proofs 9.5).

    The weights sum to 1, so a weighted mean equals ``(1/J) sum_j (1/|I_j|) sum_{i in I_j}``.
    """
    ids = np.asarray(site_ids)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("site_ids must be a non-empty 1-D array")
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    out: FloatArray = 1.0 / (counts.size * counts[inverse].astype(np.float64))
    return out


def rmse(pred: ArrayLike, obs: ArrayLike, weights: ArrayLike | None = None) -> float:
    """``[sum_j w_j (p_j - o_j)^2]^(1/2)``; on ``t`` inputs this is ``RMSE_t`` (protocol 4.3.1)."""
    p, o = _pair(pred, obs)
    if weights is None:  # Phase 6 arithmetic, kept bit-identical for inner selection
        return math.sqrt(float(np.mean((p - o) ** 2)))
    w = _weights(weights, p.size)
    return math.sqrt(float(np.sum(w * (p - o) ** 2)))


def bias(pred: ArrayLike, obs: ArrayLike, weights: ArrayLike | None = None) -> float:
    """``sum_j w_j (p_j - o_j)``; with ``t`` inputs this is ``B_t`` (protocol 4.3.1)."""
    p, o = _pair(pred, obs)
    if weights is None:
        return float(np.mean(p - o))
    w = _weights(weights, p.size)
    return float(np.sum(w * (p - o)))


def rmse_log(pred_log: ArrayLike, obs_log: ArrayLike, weights: ArrayLike | None = None) -> float:
    """``RMSE_log = [(1/J) sum_j (ln p_j - ln o_j)^2]^(1/2)`` (protocol 4.3.1; one row per site).

    Both inputs are already natural logs. Used for inner selection (protocol 3.4 item 2): the
    inner-fold mean of this statistic for ``dbar`` (NN models) or the model's own targets (XGB).
    Raises on shape mismatch, empty input or non-finite values.
    """
    return rmse(pred_log, obs_log, weights)


def bias_log(pred_log: ArrayLike, obs_log: ArrayLike, weights: ArrayLike | None = None) -> float:
    """``B_log = (1/J) sum_j (ln p_j - ln o_j)`` (protocol 4.3.1; proofs 9.5)."""
    return bias(pred_log, obs_log, weights)


def multiplicative_bias(b_log: float) -> float:
    """``e^B - 1``, the geometric-mean multiplicative bias (proofs 9.5)."""
    if not math.isfinite(b_log):
        raise ValueError("non-finite log bias")
    return math.expm1(b_log)


def nse(pred: ArrayLike, obs: ArrayLike, weights: ArrayLike | None = None) -> float:
    """Nash-Sutcliffe efficiency ``1 - sum w (p - o)^2 / sum w (o - o_bar)^2`` (protocol 4.3.1).

    The benchmark ``o_bar`` is the (weighted) evaluation-set mean. Raises if ``obs`` is constant.
    """
    p, o = _pair(pred, obs)
    w = _weights(weights, p.size)
    den = float(np.sum(w * (o - float(np.sum(w * o))) ** 2))
    if not den > 0.0:
        raise ValueError("NSE undefined: observations have zero variance")
    return 1.0 - float(np.sum(w * (p - o) ** 2)) / den


def nse_log(pred_log: ArrayLike, obs_log: ArrayLike, weights: ArrayLike | None = None) -> float:
    """``R^2_log = NSE_log``: NSE of the logs (protocol 4.3.1)."""
    return nse(pred_log, obs_log, weights)


def nse_raw(pred: ArrayLike, obs: ArrayLike, weights: ArrayLike | None = None) -> float:
    """``R^2_raw = NSE_raw``: NSE in raw space (protocol 4.3.1)."""
    p, o = _pair(pred, obs)
    _positive(p, o)
    return nse(p, o, weights)


def pearson_r(x: ArrayLike, y: ArrayLike, weights: ArrayLike | None = None) -> float:
    """(Weighted) Pearson correlation with population moments; raises on zero variance."""
    a, b = _pair(x, y)
    w = _weights(weights, a.size)
    da, db = a - float(np.sum(w * a)), b - float(np.sum(w * b))
    va, vb = float(np.sum(w * da * da)), float(np.sum(w * db * db))
    if not (va > 0.0 and vb > 0.0):
        raise ValueError("Pearson correlation undefined: zero variance")
    return float(np.sum(w * da * db)) / math.sqrt(va * vb)


def r2_pearson_raw(pred: ArrayLike, obs: ArrayLike, weights: ArrayLike | None = None) -> float:
    """Squared Pearson correlation of ``(p_j, o_j)`` in raw space (comparability only; 4.3.1)."""
    p, o = _pair(pred, obs)
    _positive(p, o)
    return pearson_r(p, o, weights) ** 2


def rmse_raw(pred: ArrayLike, obs: ArrayLike, weights: ArrayLike | None = None) -> float:
    """``RMSE_raw = [(1/J) sum_j (p_j - o_j)^2]^(1/2)`` in raw units (protocol 4.3.1)."""
    p, o = _pair(pred, obs)
    _positive(p, o)
    return rmse(p, o, weights)


def mdape(pred: ArrayLike, obs: ArrayLike) -> float:
    """``MdAPE = 100 median_j |p_j / o_j - 1|`` (protocol 4.3.1; unweighted)."""
    p, o = _pair(pred, obs)
    _positive(p, o)
    return 100.0 * float(np.median(np.abs(p / o - 1.0)))


def apb_pct(pred: ArrayLike, obs: ArrayLike, weights: ArrayLike | None = None) -> float:
    """``APB % = 100 sum_j |p_j - o_j| / sum_j o_j`` (Zarrabi ``abs_pbias``; protocol 4.3.1)."""
    p, o = _pair(pred, obs)
    _positive(p, o)
    w = _weights(weights, p.size)
    return 100.0 * float(np.sum(w * np.abs(p - o))) / float(np.sum(w * o))


def kge(pred: ArrayLike, obs: ArrayLike, weights: ArrayLike | None = None) -> float:
    """``KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)``, raw space (descriptive; 4.3.1).

    ``alpha = sigma_p / sigma_o`` (population standard deviations), ``beta = mu_p / mu_o``.
    """
    p, o = _pair(pred, obs)
    _positive(p, o)
    w = _weights(weights, p.size)
    mp, mo = float(np.sum(w * p)), float(np.sum(w * o))
    sp = math.sqrt(float(np.sum(w * (p - mp) ** 2)))
    so = math.sqrt(float(np.sum(w * (o - mo) ** 2)))
    r = pearson_r(p, o, w)
    return 1.0 - math.sqrt((r - 1.0) ** 2 + (sp / so - 1.0) ** 2 + (mp / mo - 1.0) ** 2)


def spearman_rho(x: ArrayLike, y: ArrayLike) -> float:
    """Spearman ``rho_S`` as the Pearson correlation of mid-ranks (proofs 9.5 H4 (b))."""
    a, b = _pair(x, y)
    return pearson_r(rankdata(a, method="average"), rankdata(b, method="average"))


def closure_residual(ln_a: ArrayLike, ln_w: ArrayLike, ln_dbar: ArrayLike) -> FloatArray:
    """``eps_A = ln A - ln W - ln dbar`` (protocol 4.3.2; observed or predicted)."""
    a, w = _pair(ln_a, ln_w)
    d = _vec(ln_dbar, "ln_dbar")
    if d.shape != a.shape:
        raise ValueError(f"need equal shapes, got {a.shape} and {d.shape}")
    out: FloatArray = a - w - d
    return out


def closure_summary(eps: ArrayLike) -> tuple[float, float]:
    """Return ``(median |eps_A|, share |eps_A| > ln 1.05)`` (protocol 4.3.2)."""
    e = np.abs(_vec(eps, "eps"))
    return float(np.median(e)), float(np.mean(e > CLOSURE_TOL_LOG))


def shape_violation_share(ln_dmax: ArrayLike, ln_dbar: ArrayLike) -> float:
    """Share of rows with ``d_max <= dbar`` (invariant 4 violated; protocol 4.3.2)."""
    m, d = _pair(ln_dmax, ln_dbar)
    return float(np.mean(m <= d))


def implied_log_velocity(ln_q: ArrayLike, ln_a: ArrayLike) -> FloatArray:
    """``ln V = ln Q - ln A`` (continuity by construction; protocol 4.3.2 implied-V skill).

    ``ln_q = q_E`` gives the model's implied velocity; ``ln_q = q_obs`` gives the oracle.
    """
    q, a = _pair(ln_q, ln_a)
    out: FloatArray = q - a
    return out


# ----------------------------------------------------------------------------- Phase 8 additions
def weighted_mdape(pred: ArrayLike, obs: ArrayLike, weights: ArrayLike | None = None) -> float:
    """Weighted MdAPE ``100 x wmedian_j |p_j / o_j - 1|`` (docs/3 13.6; S-RND weights 1/|I_j|).

    The weighted median is the smallest value whose cumulative normalized weight reaches 1/2
    (values sorted ascending; ties keep input order). With uniform weights and an even count it
    is the lower middle value, not the average of the two middle values used by :func:`mdape`.
    """
    p, o = _pair(pred, obs)
    _positive(p, o)
    w = _weights(weights, p.size)
    ape = np.abs(p / o - 1.0)
    order = np.argsort(ape, kind="mergesort")
    cum = np.cumsum(w[order])
    k = int(np.flatnonzero(cum >= 0.5 - WMEDIAN_TOL)[0])
    return 100.0 * float(ape[order[k]])


def v_error_shares(
    dq: ArrayLike, e_a: ArrayLike, weights: ArrayLike | None = None
) -> tuple[float, float]:
    """V-error variance shares ``(s_Q, s_A)`` (protocol 4.3.2; docs/3 13.9).

    ``e_V = ln V_hat - ln V_obs = dq - e_A`` with ``dq = q_E - q_obs`` and ``e_A = ln A_hat -
    ln A_obs``. ``s_Q = Cov(dq, e_V) / Var(e_V)`` and ``s_A = -Cov(e_A, e_V) / Var(e_V)``;
    ``s_Q + s_A = 1`` exactly. Population moments with the given (site) weights.
    """
    q, a = _pair(dq, e_a)
    w = _weights(weights, q.size)
    ev = q - a
    evc = ev - float(np.sum(w * ev))
    var = float(np.sum(w * evc * evc))
    if not var > 0.0:
        raise ValueError("V-error shares undefined: e_V has zero variance")
    qc = q - float(np.sum(w * q))
    ac = a - float(np.sum(w * a))
    return float(np.sum(w * qc * evc)) / var, -float(np.sum(w * ac * evc)) / var


def depth_ratio_r(dbar: ArrayLike, dmax: ArrayLike) -> tuple[FloatArray, BoolArray]:
    """``r = dbar / (dmax - dbar)`` from predicted depths (protocol 4.3.2; M2 back-calculation).

    Returns ``(r, ok)``: rows with ``dmax <= dbar`` violate invariant 4, get ``r = nan`` and
    ``ok = False`` (docs/3 13.9: counted and excluded from the back-calculation statistic).
    """
    d, m = _pair(dbar, dmax)
    if not bool((d > 0).all()):
        raise ValueError("dbar must be > 0")
    ok: BoolArray = m > d
    r: FloatArray = np.full(d.shape, np.nan)
    r[ok] = d[ok] / (m[ok] - d[ok])
    return r, ok


@dataclass(frozen=True)
class BackCalcResult:
    """Manning back-calculation consistency (protocol 4.3.2; docs/3 13.9)."""

    n_rows: int
    n_invariant4: int  # rows with dmax_hat <= dbar_hat, excluded
    n_used: int
    share_outside: float  # weighted share with dist(nu_back, [nu_min, nu_max]) > 0


def back_calc_consistency(
    nu_back: ArrayLike,
    ok: ArrayLike,
    nu_bounds: tuple[float, float],
    weights: ArrayLike | None = None,
) -> BackCalcResult:
    """Share of rows whose back-calculated ``nu`` leaves ``[nu_min, nu_max]`` (proofs Prop 7.4).

    ``nu_back = (5/3) l + w + (2/3) k(e^(l - w), r) + s/2 - q_E`` (proofs Prop 6.2) is computed
    by the caller; rows with ``ok = False`` (invariant 4) may hold ``nan`` and are excluded.
    ``nu_bounds`` are the logs of ``domain.n_bounds``. Weights are renormalized over used rows.
    """
    nu = np.asarray(nu_back, dtype=np.float64)
    use = np.asarray(ok, dtype=bool)
    if nu.ndim != 1 or nu.size == 0 or use.shape != nu.shape:
        raise ValueError("nu_back and ok must be non-empty 1-D arrays of equal shape")
    lo, hi = nu_bounds
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError("nu_bounds must be finite with lo < hi")
    if not bool(use.any()):
        raise ValueError("no row satisfies invariant 4")
    if not bool(np.isfinite(nu[use]).all()):
        raise ValueError("nu_back contains non-finite values on used rows")
    w = _weights(weights, nu.size)[use]
    outside = (nu[use] < lo) | (nu[use] > hi)
    return BackCalcResult(
        n_rows=int(nu.size),
        n_invariant4=int((~use).sum()),
        n_used=int(use.sum()),
        share_outside=float(np.sum(w * outside) / np.sum(w)),
    )


def s1_consistency(n_hat: ArrayLike, n_ref: ArrayLike) -> tuple[float, float]:
    """S1: Spearman ``rho`` and log-RMSE of site-level ``n_hat`` against HyG ``Manning_n``.

    Protocol 4.6 S1 and docs/3 13.9 (descriptive consistency check): both inputs are raw
    Manning n (strictly positive), one row per matched site; the log-RMSE is unweighted.
    """
    p, o = _pair(n_hat, n_ref)
    _positive(p, o)
    return spearman_rho(p, o), rmse_log(np.log(p), np.log(o))

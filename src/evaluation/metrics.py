"""Primary, secondary and descriptive metrics and the physical-consistency analogues.

Sources: protocol sections 4.3.1 and 4.3.2; proofs section 9.5 (site-weighted statistics, S-RND
rows weighted ``1/|I_j|``). :func:`rmse_log` is the Phase 6 inner-selection statistic (P6-D1 (a);
protocol 3.4 item 2); the remaining metrics are implemented under P7-D1 (a) and verified only by
known-answer synthetic tests. Scoring of real predictions stays locked until the Phase 7 gate.

Weights: every mean-based metric accepts optional observation weights (normalized here to sum
to 1; ``None`` means one row per site, uniform). :func:`site_weights` gives the proofs 9.5 weights
``1/(J |I_j|)``. MdAPE is unweighted (a weighted median is not defined in protocol 4.3.1).

Torch-free and xgboost-free (usable in XGB workers, ADL-051). Owning phase: 7 (P7-D1 (a)).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata

__all__ = [
    "CLOSURE_TOL_LOG",
    "apb_pct",
    "bias",
    "bias_log",
    "closure_residual",
    "closure_summary",
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
    "shape_violation_share",
    "site_weights",
    "spearman_rho",
]

FloatArray = NDArray[np.float64]

CLOSURE_TOL_LOG = math.log(1.05)
"""Protocol 4.3.2: share of predicted closure residuals with ``|eps_A| > ln 1.05``."""


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

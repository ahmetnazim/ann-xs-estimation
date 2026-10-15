"""Event 3 statistics: metric cells, paired bootstrap designs, UQ and consistency summaries.

Sources: protocol 4.3.1 (metric definitions and tiers), 4.3.2 (consistency analogues), 4.3.3
(matrix and strata), 4.4 (paired cluster bootstrap, ``B``, seed, division strata, S-ID statistics
per repeat averaged over the three repeats in each draw), 4.5 (S2 at ``1 - 0.05/4``), 4.6 (UQ:
coverage, mean log-width, CRPS, PIT), proofs 9.4-9.5; docs/3 section 13.13 (Event 3 computation
rules) items b (S-ID point statistics = mean over repeats), d (every cell reports ``n``; undefined
metrics are null with a reason), e (bootstrap designs), l (UQ) and m (consistency).

The metric functions are those of :mod:`src.evaluation.metrics`, :mod:`src.evaluation.uq` and
:mod:`src.evaluation.bootstrap` (Phase 7 known-answer tested); this module only selects rows,
averages over repeats and assembles results. Torch-free and xgboost-free. Owning phase: 8.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import BootstrapParams
from src.evaluation import metrics as mt
from src.evaluation import uq
from src.evaluation.bootstrap import (
    bootstrap_matrix,
    percentile_ci,
    two_sample_bootstrap_matrix,
)

__all__ = [
    "DMAX_METRICS",
    "FULL_METRICS",
    "PIT_BINS",
    "STRATA_KINDS",
    "T_METRICS",
    "BootStat",
    "Design",
    "TwoSampleDesign",
    "cell",
    "consistency_summary",
    "log_pair",
    "matrix_records",
    "member_arrays",
    "per_repeat",
    "repeat_cells",
    "run_design",
    "run_two_sample",
    "uq_summary",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

FULL_METRICS: Final[tuple[str, ...]] = (
    "rmse_log",
    "bias_log",
    "mult_bias",
    "nse_log",
    "nse_raw",
    "r2_raw",
    "rmse_raw",
    "mdape",
    "apb_pct",
    "kge",
)
#: Protocol 4.3.3: d_max is primary on RMSE_log and B_log only.
DMAX_METRICS: Final[tuple[str, ...]] = ("rmse_log", "bias_log", "mult_bias")
#: Shape cells: RMSE_t, B_t and rho_S(r-hat, r_obs) computed on t (docs/3 13.13 d).
T_METRICS: Final[tuple[str, ...]] = ("rmse_t", "bias_t", "rho_r")
STRATA_KINDS: Final[tuple[str, ...]] = ("all", "da_q", "stream_order", "division", "far", "nested")
PIT_BINS: Final[int] = 10


# ----------------------------------------------------------------------------- cells
def log_pair(rows: pd.DataFrame, target: str) -> tuple[FloatArray, FloatArray] | None:
    """``(prediction, observation)`` in log space (``t`` as is); ``None`` if not available."""
    pcol, ocol = f"mu_{target}", f"o_{target}"
    if pcol not in rows.columns or ocol not in rows.columns:
        return None
    pred = rows[pcol].to_numpy(dtype=np.float64)
    obs = rows[ocol].to_numpy(dtype=np.float64)
    if target != "t":
        with np.errstate(divide="ignore", invalid="ignore"):
            obs = np.log(obs)
    return pred, obs


def _safe(fn: Callable[[], float]) -> float | None:
    try:
        v = float(fn())
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def cell(
    pred: FloatArray, obs: FloatArray, target: str, weights: FloatArray | None = None
) -> dict[str, float | None]:
    """One metric set (protocol 4.3.1); a metric undefined on these rows is ``None``."""
    w = weights
    if target == "t":
        return {
            "rmse_t": _safe(lambda: mt.rmse(pred, obs, w)),
            "bias_t": _safe(lambda: mt.bias(pred, obs, w)),
            # rho_S(r, r_obs) = rho_S(t, t_obs): r = 1/(e^t - 1) is strictly decreasing (13.13 d)
            "rho_r": _safe(lambda: mt.spearman_rho(pred, obs)),
        }
    out: dict[str, float | None] = {
        "rmse_log": _safe(lambda: mt.rmse_log(pred, obs, w)),
        "bias_log": _safe(lambda: mt.bias_log(pred, obs, w)),
    }
    b = out["bias_log"]
    out["mult_bias"] = None if b is None else math.expm1(b)
    if target == "dmax":
        return out
    p, o = np.exp(pred), np.exp(obs)
    out.update(
        {
            "nse_log": _safe(lambda: mt.nse_log(pred, obs, w)),
            "nse_raw": _safe(lambda: mt.nse_raw(p, o, w)),
            "r2_raw": _safe(lambda: mt.r2_pearson_raw(p, o, w)),
            "rmse_raw": _safe(lambda: mt.rmse_raw(p, o, w)),
            "mdape": _safe(
                (lambda: mt.weighted_mdape(p, o, w)) if w is not None else (lambda: mt.mdape(p, o))
            ),
            "apb_pct": _safe(lambda: mt.apb_pct(p, o, w)),
            "kge": _safe(lambda: mt.kge(p, o, w)),
        }
    )
    return out


def repeat_cells(
    rows: pd.DataFrame, target: str, *, weighted: bool = False
) -> tuple[dict[str, float | None], dict[str, int]] | None:
    """Metric set averaged over repeats (docs/3 13.13 b) and the row counts; ``None`` if absent.

    Rows with a non-finite prediction or observation are excluded and counted. A metric is
    ``None`` if it is undefined in any repeat.
    """
    pair = log_pair(rows, target)
    if pair is None:
        return None
    pred, obs = pair
    finite = np.isfinite(pred) & np.isfinite(obs)
    reps = rows["repeat"].to_numpy(dtype=np.int64)
    w_all = rows["weight"].to_numpy(dtype=np.float64) if weighted else None
    per: list[dict[str, float | None]] = []
    for r in sorted(set(reps.tolist())):
        m = finite & (reps == r)
        if not m.any():
            per.append(dict.fromkeys(_names(target)))
            continue
        per.append(cell(pred[m], obs[m], target, None if w_all is None else w_all[m]))
    if not per:
        per = [dict.fromkeys(_names(target))]
    avg: dict[str, float | None] = {}
    for k in per[0]:
        nums = [v for v in (d[k] for d in per) if v is not None]
        avg[k] = float(np.mean(nums)) if len(nums) == len(per) else None
    counts = {
        "n_rows": int(finite.sum()),
        "n_units": int(rows.loc[finite, "unit_id"].nunique()),
        "n_excluded": int((~finite).sum()),
        "n_repeats": len(per),
    }
    return avg, counts


def _names(target: str) -> tuple[str, ...]:
    return T_METRICS if target == "t" else DMAX_METRICS if target == "dmax" else FULL_METRICS


def _stratum_labels(rows: pd.DataFrame, kind: str) -> pd.Series | None:
    if kind == "all":
        return pd.Series("all", index=rows.index)
    if kind not in rows.columns:
        return None
    col = rows[kind]
    if kind == "far":
        return col.map({True: "far", False: "not_far"}).astype(str)
    return col.astype(str)


def matrix_records(
    rows: pd.DataFrame,
    meta: Mapping[str, Any],
    targets: Sequence[str],
    *,
    weighted: bool = False,
    kinds: Sequence[str] = STRATA_KINDS,
) -> list[dict[str, Any]]:
    """Long-format matrix records (protocol 4.3.3): one per target, stratum and metric."""
    out: list[dict[str, Any]] = []
    for kind in kinds:
        labels = _stratum_labels(rows, kind)
        if labels is None:
            continue
        for label in sorted(set(labels.tolist())):
            sub = rows.loc[labels.eq(label).to_numpy()]
            for target in targets:
                res = repeat_cells(sub, target, weighted=weighted)
                if res is None:
                    continue
                cells, counts = res
                for metric, value in cells.items():
                    out.append(
                        {
                            **meta,
                            "target": target,
                            "stratum_kind": kind,
                            "stratum": label,
                            "metric": metric,
                            "value": value,
                            **counts,
                        }
                    )
    return out


def per_repeat(rows: pd.DataFrame, fn: Callable[[pd.DataFrame], float]) -> float:
    """Mean over repeats of ``fn`` on each repeat's rows (docs/3 13.13 b)."""
    vals = [fn(rows.loc[rows["repeat"].eq(r)]) for r in sorted(rows["repeat"].unique().tolist())]
    if not vals:
        raise ValueError("no rows")
    return float(np.mean(vals))


# ----------------------------------------------------------------------------- bootstrap designs
@dataclass(frozen=True)
class BootStat:
    """One bootstrap statistic: ``fn(per-repeat index lists) -> float`` and its CI level."""

    name: str
    fn: Callable[[Sequence[IntArray]], float]
    level: float


@dataclass(frozen=True)
class Design:
    """Rows of one paired bootstrap design: cluster and stratum labels and repeat codes."""

    cluster: NDArray[np.str_]
    stratum: NDArray[np.str_] | None
    repeat: IntArray

    def split(self, idx: IntArray) -> list[IntArray]:
        """Observation indices of a draw, per repeat (sorted repeat codes)."""
        reps = self.repeat[idx]
        return [idx[reps == r] for r in np.unique(self.repeat)]


def _ci_doc(point: float, reps: FloatArray, level: float) -> dict[str, Any]:
    lo, hi = percentile_ci(reps, level)
    return {"estimate": point, "lo": lo, "hi": hi, "level": level}


def run_design(
    design: Design, stats: Sequence[BootStat], cfg: BootstrapParams, *, n_boot: int | None = None
) -> dict[str, dict[str, Any]]:
    """Point estimates and percentile CIs of several statistics on the same draws (13.13 e)."""
    if not stats:
        raise ValueError("no statistics")
    full = design.split(np.arange(design.repeat.size, dtype=np.int64))
    points = [float(s.fn(full)) for s in stats]

    def vec(idx: IntArray, _groups: IntArray) -> FloatArray:
        parts = design.split(idx)
        return np.array([s.fn(parts) for s in stats], dtype=np.float64)

    reps = bootstrap_matrix(vec, design.cluster, design.stratum, cfg, n_boot=n_boot)
    return {s.name: _ci_doc(points[k], reps[:, k], s.level) for k, s in enumerate(stats)}


@dataclass(frozen=True)
class TwoSampleDesign:
    """H3: set A (ID) and set B (OOD) designs and ``fn(parts_a, parts_b) -> float``."""

    a: Design
    b: Design


def run_two_sample(
    design: TwoSampleDesign,
    stats: Sequence[tuple[str, Callable[[Sequence[IntArray], Sequence[IntArray]], float], float]],
    cfg: BootstrapParams,
    *,
    n_boot: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Two-sample bootstrap (proofs 9.5 H3): A and B resampled independently, paired in each."""
    full_a = design.a.split(np.arange(design.a.repeat.size, dtype=np.int64))
    full_b = design.b.split(np.arange(design.b.repeat.size, dtype=np.int64))
    points = [float(fn(full_a, full_b)) for _, fn, _ in stats]

    def vec(ia: IntArray, _ga: IntArray, ib: IntArray, _gb: IntArray) -> FloatArray:
        pa, pb = design.a.split(ia), design.b.split(ib)
        return np.array([fn(pa, pb) for _, fn, _ in stats], dtype=np.float64)

    reps = two_sample_bootstrap_matrix(
        vec,
        design.a.cluster,
        design.a.stratum,
        design.b.cluster,
        design.b.stratum,
        cfg,
        n_boot=n_boot,
    )
    return {name: _ci_doc(points[k], reps[:, k], lvl) for k, (name, _, lvl) in enumerate(stats)}


# ----------------------------------------------------------------------------- UQ
_MEMBER = re.compile(r"^m(\d{2})_(mu|sigma)_(.+)$")


def member_arrays(rows: pd.DataFrame, target: str) -> tuple[FloatArray, FloatArray] | None:
    """``(mu, sigma)`` member arrays ``(M, n)`` of ``target``; ``None`` without sigma columns."""
    mus: dict[int, str] = {}
    sigmas: dict[int, str] = {}
    for col in rows.columns:
        m = _MEMBER.match(str(col))
        if m and m.group(3) == target:
            (mus if m.group(2) == "mu" else sigmas)[int(m.group(1))] = str(col)
    if not sigmas:
        return None
    if sorted(mus) != sorted(sigmas):
        raise ValueError(f"{target}: member mu and sigma columns differ")
    keys = sorted(mus)
    mu = rows[[mus[k] for k in keys]].to_numpy(dtype=np.float64).T
    sd = rows[[sigmas[k] for k in keys]].to_numpy(dtype=np.float64).T
    return mu, sd


def _uq_one(rows: pd.DataFrame, target: str) -> dict[str, Any] | None:
    ocol = f"o_{target}"
    mem = member_arrays(rows, target)
    if ocol not in rows.columns or mem is None or f"q_lo_{target}" not in rows.columns:
        return None
    obs = rows[ocol].to_numpy(dtype=np.float64)
    if target != "t":
        with np.errstate(divide="ignore", invalid="ignore"):
            obs = np.log(obs)
    lo = rows[f"q_lo_{target}"].to_numpy(dtype=np.float64)
    hi = rows[f"q_hi_{target}"].to_numpy(dtype=np.float64)
    mu, sd = mem
    finite = np.isfinite(obs) & np.isfinite(lo) & np.isfinite(hi)
    reps = rows["repeat"].to_numpy(dtype=np.int64)
    cov, wid, crps = [], [], []
    hist = np.zeros(PIT_BINS, dtype=np.int64)
    for r in sorted(set(reps.tolist())):
        m = finite & (reps == r)
        if not m.any():
            return None
        cov.append(uq.coverage(lo[m], hi[m], obs[m]))
        wid.append(uq.mean_width(lo[m], hi[m]))
        crps.append(float(np.mean(uq.crps_mixture(obs[m], mu[:, m], sd[:, m]))))
        u = uq.pit(obs[m], mu[:, m], sd[:, m])
        hist += np.histogram(u, bins=PIT_BINS, range=(0.0, 1.0))[0]
    return {
        "coverage": float(np.mean(cov)),
        "mean_log_width": float(np.mean(wid)),
        "width_factor": math.exp(float(np.mean(wid))),
        "crps": float(np.mean(crps)),
        "pit_counts": hist.tolist(),
        "n_rows": int(finite.sum()),
        "n_excluded": int((~finite).sum()),
    }


def uq_summary(
    rows: pd.DataFrame,
    targets: Sequence[str],
    band: tuple[float, float],
    *,
    strata: Sequence[str] = ("division", "da_q"),
) -> dict[str, Any]:
    """Coverage, mean log-width, CRPS and PIT per target, overall and by stratum (4.6; 13.13 l)."""
    out: dict[str, Any] = {}
    for target in targets:
        overall = _uq_one(rows, target)
        if overall is None:
            continue
        overall["in_band"] = uq.coverage_in_band(overall["coverage"], band)
        by: dict[str, dict[str, Any]] = {}
        for kind in strata:
            if kind not in rows.columns:
                continue
            labels = rows[kind].astype(str)
            by[kind] = {}
            for lab in sorted(set(labels.tolist())):
                res = _uq_one(rows.loc[labels.eq(lab).to_numpy()], target)
                by[kind][lab] = res if res is not None else {"not_computed": "empty in a repeat"}
        out[target] = {"overall": overall, "by": by}
    return out


# ----------------------------------------------------------------------------- consistency
def _q(a: FloatArray) -> dict[str, float]:
    q = np.quantile(a, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"q05": q[0], "q25": q[1], "median": q[2], "q75": q[3], "q95": q[4]}


def consistency_summary(
    rows: pd.DataFrame, model: str, nu_bounds: tuple[float, float]
) -> dict[str, Any]:
    """Protocol 4.3.2 analogues of one model on one evaluation set (docs/3 13.13 m).

    Statistics are computed per repeat and averaged (docs/3 13.13 b); counts are summed.
    """
    out: dict[str, Any] = {}
    cols = set(rows.columns)
    reps = sorted(rows["repeat"].unique().tolist())

    def each(fn: Callable[[pd.DataFrame], Any]) -> list[Any]:
        return [fn(rows.loc[rows["repeat"].eq(r)]) for r in reps]

    if model in ("M2-MLP", "M2-XGB"):
        eps = each(
            lambda d: mt.closure_summary(mt.closure_residual(d["mu_A"], d["mu_W"], d["mu_dbar"]))
        )
        out["predicted_closure"] = {
            "median_abs": float(np.mean([e[0] for e in eps])),
            "share_gt_ln105": float(np.mean([e[1] for e in eps])),
        }
        out["shape_violation_share"] = float(
            np.mean(each(lambda d: mt.shape_violation_share(d["mu_dmax"], d["mu_dbar"])))
        )
    if "nu_back" in cols:
        res = each(lambda d: _back_calc(d, nu_bounds))
        out["back_calc"] = {
            "share_outside": float(np.mean([r["share_outside"] for r in res])),
            "n_rows": int(sum(r["n_rows"] for r in res)),
            "n_invariant4": int(sum(r["n_invariant4"] for r in res)),
            "n_undefined": int(sum(r["n_undefined"] for r in res)),
            "n_used": int(sum(r["n_used"] for r in res)),
        }
    if model == "M4" and "resid_mean" in cols:
        rm = [d["resid_mean"].to_numpy(dtype=np.float64) for _, d in rows.groupby("repeat")]
        out["manning_residual_M4"] = {
            "rms": float(np.mean([math.sqrt(float(np.mean(a * a))) for a in rm])),
            **{k: float(np.mean([_q(a)[k] for a in rm])) for k in _q(rm[0])},
        }
    elif "resid_maxabs" in cols:
        out["solver_residual_max"] = float(rows["resid_maxabs"].max())
        if "iters_max" in cols:
            out["picard_iterations_max"] = int(rows["iters_max"].max())
    if "mu_A" in cols and "o_V" in cols:
        out["implied_V"] = _implied_v(rows)
    return out


def _back_calc(d: pd.DataFrame, nu_bounds: tuple[float, float]) -> dict[str, Any]:
    nu = d["nu_back"].to_numpy(dtype=np.float64)
    inv4 = (
        d["inv4_violation"].astype(bool).to_numpy()
        if "inv4_violation" in d.columns
        else np.zeros(nu.size, dtype=bool)
    )
    defined = np.isfinite(nu)
    ok = ~inv4 & defined
    res = mt.back_calc_consistency(np.where(ok, nu, 0.0), ok, nu_bounds)
    return {
        "share_outside": res.share_outside,
        "n_rows": res.n_rows,
        "n_invariant4": int(inv4.sum()),
        "n_undefined": int((~inv4 & ~defined).sum()),
        "n_used": res.n_used,
    }


def _implied_v(rows: pd.DataFrame) -> dict[str, Any]:
    """Implied-V skill and the oracle decomposition (protocol 4.3.2; diagnostic only)."""
    per: list[dict[str, float]] = []
    for _, d in rows.groupby("repeat"):
        mu_a = d["mu_A"].to_numpy(dtype=np.float64)
        ok = np.isfinite(mu_a)
        if not ok.any():
            continue
        q_e, q_o = d["ln_QE"].to_numpy()[ok], d["ln_Qobs"].to_numpy()[ok]
        ln_v = np.log(d["o_V"].to_numpy(dtype=np.float64))[ok]
        ln_a = np.log(d["o_A"].to_numpy(dtype=np.float64))[ok]
        v_hat = mt.implied_log_velocity(q_e, mu_a[ok])
        v_orc = mt.implied_log_velocity(q_o, mu_a[ok])
        s_q, s_a = mt.v_error_shares(q_e - q_o, mu_a[ok] - ln_a)
        per.append(
            {
                "rmse_log": mt.rmse_log(v_hat, ln_v),
                "bias_log": mt.bias_log(v_hat, ln_v),
                "oracle_rmse_log": mt.rmse_log(v_orc, ln_v),
                "oracle_bias_log": mt.bias_log(v_orc, ln_v),
                "share_Q": s_q,
                "share_A": s_a,
            }
        )
    if not per:
        return {"not_computed": "no finite A prediction"}
    return {k: float(np.mean([p[k] for p in per])) for k in per[0]}

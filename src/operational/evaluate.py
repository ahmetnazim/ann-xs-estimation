"""OPS-1 evaluation: model stage/width responses, errors, bootstrap contrasts (docs/4 section 3).

Pipeline per family (S-ID primary with 3 repeats, S-LRO secondary):

1. :func:`load_predictions` reads the frozen out-of-fold predictions (natural-log means).
2. :func:`anchor_geometry` forms ``(W0, dbar0, dmax0, r)`` per (repeat, site). Models without
   ``mu_dmax`` take the training-fold median shape ``t`` of the fallback model (M3). Rows with
   ``dmax0 <= dbar0`` are operational failures; ``r`` is clipped to ``domain.r_bounds``.
3. :func:`model_responses` solves OPS-E1 (:mod:`src.operational.rating`) at every observed flow
   and returns the stage error ``e_h`` (m) and the width error ``e_w``.
4. :func:`site_bin_table` reduces to per-(repeat, site, bin) mean squared and mean errors.
5. :func:`score_family` restricts to the common site set, computes site-weighted RMSE/bias per
   bin (averaged over repeats), the paired division-stratified cluster bootstrap of every model's
   RMSE (all contrasts and skills are functions of the same replicates), the geometry-vs-
   operational ranking and the hypothesis verdicts.

Torch is used only inside the rating solver. Owning phase: OPS-1 (post-release extension).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import BootstrapParams
from src.evaluation.bootstrap import bootstrap_matrix, percentile_ci
from src.evaluation.metrics import spearman_rho
from src.operational.config import OperationalParams
from src.operational.rating import solve_lambda_np, stage_change, width_log_change

__all__ = [
    "HIGH_BIN",
    "LEARNED_MODELS",
    "anchor_geometry",
    "bin_labels",
    "load_predictions",
    "model_responses",
    "score_family",
    "site_bin_table",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
HIGH_BIN: Final[str] = "high"
LEARNED_MODELS: Final[tuple[str, ...]] = (
    "M1",
    "M2-MLP",
    "M2-XGB",
    "M3",
    "M4",
    "M5",
    "M6-NN",
    "M6-XGB",
)
_PRED_COLS: Final[tuple[str, ...]] = ("repeat", "site_no", "mu_W", "mu_dbar", "mu_dmax", "mu_t")


def bin_labels(edges: Sequence[float]) -> list[str]:
    """Labels ``"[lo,hi)"`` of the flow-ratio bins followed by :data:`HIGH_BIN`."""
    out = [f"[{lo:g},{hi:g})" for lo, hi in pairwise(edges)]
    return [*out, HIGH_BIN]


def load_predictions(pred_dir: Path, model: str, family: str, analysis: str) -> pd.DataFrame:
    """Return one row per (repeat, site) of a model's frozen predictions for one family.

    Columns: ``repeat``, ``site_no``, ``mu_W``, ``mu_dbar``, ``mu_dmax``, ``mu_t`` (log means;
    ``mu_dmax``/``mu_t`` are NaN when the model does not predict them).
    """
    df = pd.read_parquet(pred_dir / f"{model}.parquet")
    keep = (df["analysis"] == analysis) & (df["family"] == family) & (df["unit"] == "site")
    if "ablation" in df.columns:
        keep &= df["ablation"].fillna("").astype(str).eq("")
    df = df.loc[keep].copy()
    if df.empty:
        raise ValueError(f"{model}: no {analysis}/{family} site predictions")
    for col in ("mu_dmax", "mu_t"):
        if col not in df.columns:
            df[col] = np.nan
    out = df[list(_PRED_COLS)].copy()
    out["site_no"] = out["site_no"].astype(str)
    out["repeat"] = out["repeat"].astype(np.int64)
    if out.duplicated(subset=["repeat", "site_no"]).any():
        raise ValueError(f"{model}/{family}: duplicated (repeat, site_no) predictions")
    return out.sort_values(["repeat", "site_no"], kind="mergesort").reset_index(drop=True)


def anchor_geometry(
    pred: pd.DataFrame,
    fallback: pd.DataFrame | None,
    r_bounds: tuple[float, float],
) -> pd.DataFrame:
    """Anchor section per (repeat, site) (docs/4 sections 2, 3.1, 3.3).

    ``fallback`` (``repeat``, ``site_no``, ``mu_t``) supplies ``t = ln(dmax/dbar)`` where the
    model has no ``mu_dmax``. Output columns: ``repeat``, ``site_no``, ``ln_w0``, ``ln_dbar0``,
    ``ln_dmax0``, ``r``, ``valid``, ``clipped``, ``fallback_shape``.
    """
    g = pred[["repeat", "site_no", "mu_W", "mu_dbar", "mu_dmax"]].copy()
    use_fb = ~np.isfinite(g["mu_dmax"].to_numpy(dtype=np.float64))
    if use_fb.any():
        if fallback is None:
            raise ValueError("model lacks mu_dmax and no fallback shape was given")
        fb = fallback[["repeat", "site_no", "mu_t"]].rename(columns={"mu_t": "_t_fb"})
        g = g.merge(fb, on=["repeat", "site_no"], how="left", validate="1:1")
        g.loc[use_fb, "mu_dmax"] = g.loc[use_fb, "mu_dbar"] + g.loc[use_fb, "_t_fb"]
    ln_w = g["mu_W"].to_numpy(dtype=np.float64)
    ln_d = g["mu_dbar"].to_numpy(dtype=np.float64)
    ln_m = g["mu_dmax"].to_numpy(dtype=np.float64)
    finite = np.isfinite(ln_w) & np.isfinite(ln_d) & np.isfinite(ln_m)
    valid = finite & (ln_m > ln_d)
    r = np.full(ln_d.shape, np.nan)
    d, m = np.exp(ln_d[valid]), np.exp(ln_m[valid])
    r[valid] = d / (m - d)
    lo, hi = r_bounds
    clipped = valid & ((r < lo) | (r > hi))
    r[valid] = np.clip(r[valid], lo, hi)
    return pd.DataFrame(
        {
            "repeat": g["repeat"].to_numpy(dtype=np.int64),
            "site_no": g["site_no"].astype(str).to_numpy(),
            "ln_w0": ln_w,
            "ln_dbar0": ln_d,
            "ln_dmax0": ln_m,
            "r": r,
            "valid": valid,
            "clipped": clipped,
            "fallback_shape": use_fb,
        }
    )


def model_responses(
    obs: pd.DataFrame,
    geom: pd.DataFrame,
    *,
    level: int,
    trunc: float,
    tol: float,
    max_iter: int,
    chunk_rows: int,
) -> tuple[pd.DataFrame, int]:
    """Predicted responses and errors for every (repeat, valid site, observation).

    Returns the frame (``repeat``, ``site_no``, ``q_ratio``, ``e_h``, ``e_w``) and the worst
    iteration count of the rating solver.
    """
    ok = geom.loc[geom["valid"]]
    j = obs.merge(ok, on="site_no", how="inner")
    if j.empty:
        return pd.DataFrame(columns=["repeat", "site_no", "q_ratio", "e_h", "e_w"]), 0
    r = j["r"].to_numpy(dtype=np.float64)
    ln_a0 = (j["ln_dbar0"] - j["ln_w0"]).to_numpy(dtype=np.float64)
    lam, iters = solve_lambda_np(
        j["ln_q_ratio"].to_numpy(dtype=np.float64),
        r,
        ln_a0,
        level=level,
        trunc=trunc,
        tol=tol,
        max_iter=max_iter,
        chunk_rows=chunk_rows,
    )
    dh = stage_change(np.exp(j["ln_dmax0"].to_numpy(dtype=np.float64)), lam)
    dlw = width_log_change(lam, r)
    out = pd.DataFrame(
        {
            "repeat": j["repeat"].to_numpy(dtype=np.int64),
            "site_no": j["site_no"].to_numpy(),
            "q_ratio": j["q_ratio"].to_numpy(dtype=np.float64),
            "e_h": dh - j["dh_obs"].to_numpy(dtype=np.float64),
            "e_w": dlw - j["dlnw_obs"].to_numpy(dtype=np.float64),
        }
    )
    return out, iters


def site_bin_table(resp: pd.DataFrame, edges: Sequence[float], high_ratio: float) -> pd.DataFrame:
    """Per (repeat, site, bin): ``n``, ``mse_h``, ``me_h``, ``n_w``, ``mse_w``, ``me_w``.

    Bins are ``[lo, hi)`` on ``q_ratio`` plus the pooled :data:`HIGH_BIN` (``q_ratio >=
    high_ratio``). Width statistics use rows with a finite ``e_w`` only (NaN when none).
    """
    labels = bin_labels(edges)
    # np.digitize(x, inner) = i  <=>  inner[i-1] <= x < inner[i]  (bins [lo, hi))
    idx = np.digitize(resp["q_ratio"].to_numpy(dtype=np.float64), np.asarray(edges[1:-1]))
    base = resp.assign(bin=np.asarray(labels, dtype=object)[idx])
    high = resp.loc[resp["q_ratio"] >= high_ratio].assign(bin=HIGH_BIN)
    df = pd.concat([base, high], ignore_index=True)
    df["e_h2"] = df["e_h"] ** 2
    df["e_w2"] = df["e_w"] ** 2
    df["has_w"] = np.isfinite(df["e_w"].to_numpy(dtype=np.float64))
    grp = df.groupby(["repeat", "site_no", "bin"], sort=True)
    out = grp.agg(
        n=("e_h", "size"),
        mse_h=("e_h2", "mean"),
        me_h=("e_h", "mean"),
        n_w=("has_w", "sum"),
        mse_w=("e_w2", "mean"),
        me_w=("e_w", "mean"),
    ).reset_index()
    out["n_w"] = out["n_w"].astype(np.int64)
    return out


def _site_weighted(values: pd.Series) -> float:
    v = values.to_numpy(dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _verdict_h1(lo: float, hi: float, margin: float) -> str:
    if hi < 0:
        return "supported (superior)"
    if hi < margin:
        return "supported (non-inferior)"
    return "falsified"


def score_family(
    family: str,
    obs: pd.DataFrame,
    geoms: Mapping[str, pd.DataFrame],
    tables: Mapping[str, pd.DataFrame],
    obs_geometry: pd.DataFrame,
    params: OperationalParams,
    boot: BootstrapParams,
    *,
    n_boot: int | None = None,
) -> dict[str, Any]:
    """Score one family on the common site set (docs/4 sections 3.3-3.5).

    Parameters
    ----------
    family
        ``S-ID`` or ``S-LRO``.
    obs
        Observation frame (for the site -> division map and bin membership).
    geoms
        :func:`anchor_geometry` output per model.
    tables
        :func:`site_bin_table` output per model.
    obs_geometry
        ``site_no``, ``ln_dmax_obs``, ``ln_dbar_obs`` at the anchor (geometry ranking).
    params, boot
        OPS-1 pre-registration and the v1.0.0 bootstrap settings.
    n_boot
        Override of ``boot.b`` (tests only).
    """
    models = list(params.predictions.models)
    edges = params.metrics.flow_bin_edges
    labels = bin_labels(edges)
    ref = params.predictions.reference_model
    obs_sites = set(obs["site_no"].unique())

    failures: dict[str, dict[str, float]] = {}
    common = set(obs_sites)
    repeats: list[int] | None = None
    for m in models:
        g = geoms[m]
        g_obs = g.loc[g["site_no"].isin(obs_sites)]
        failures[m] = {
            "rows": len(g_obs),
            "invalid_share": float((~g_obs["valid"]).mean()) if len(g_obs) else float("nan"),
            "clipped_share": float(g_obs["clipped"].mean()) if len(g_obs) else float("nan"),
            "fallback_shape": bool(g["fallback_shape"].any()),
        }
        rep_m = sorted(int(x) for x in g["repeat"].unique())
        if repeats is None:
            repeats = rep_m
        elif rep_m != repeats:
            raise ValueError(f"{m}: repeats {rep_m} differ from {repeats}")
        n_rep = len(rep_m)
        ok_counts = g_obs.loc[g_obs["valid"]].groupby("site_no").size()
        common &= set(ok_counts.index[ok_counts == n_rep])
    assert repeats is not None
    sites = np.array(sorted(common), dtype=object)
    if sites.size == 0:
        raise ValueError(f"{family}: empty common site set")
    site_pos = {s: i for i, s in enumerate(sites)}
    div_map = obs.drop_duplicates("site_no").set_index("site_no")["division"]
    divisions = div_map.reindex(sites).to_numpy()

    n_m, n_r, n_b, n_s = len(models), len(repeats), len(labels), sites.size
    mse = np.zeros((n_m, n_r, n_b, n_s))
    has = np.zeros((n_b, n_s), dtype=bool)
    rows: list[dict[str, Any]] = []
    for mi, m in enumerate(models):
        t = tables[m]
        t = t.loc[t["site_no"].isin(common)]
        for ri, rep in enumerate(repeats):
            tr = t.loc[t["repeat"] == rep]
            for bi, b in enumerate(labels):
                tb = tr.loc[tr["bin"] == b]
                pos = np.fromiter((site_pos[s] for s in tb["site_no"]), dtype=np.int64)
                mse[mi, ri, bi, pos] = tb["mse_h"].to_numpy(dtype=np.float64)
                has[bi, pos] = True
        for b in labels:
            tb = t.loc[t["bin"] == b]
            per_rep = tb.groupby("repeat")
            rmse_h = float(np.mean([math.sqrt(_site_weighted(x["mse_h"])) for _, x in per_rep]))
            bias_h = float(np.mean([_site_weighted(x["me_h"]) for _, x in per_rep]))
            wt = tb.loc[tb["n_w"] > 0]
            per_rep_w = wt.groupby("repeat")
            rmse_w = float(np.mean([math.sqrt(_site_weighted(x["mse_w"])) for _, x in per_rep_w]))
            bias_w = float(np.mean([_site_weighted(x["me_w"]) for _, x in per_rep_w]))
            first = tb.loc[tb["repeat"] == repeats[0]]
            rows.append(
                {
                    "model": m,
                    "bin": b,
                    "n_sites": int(first["site_no"].nunique()),
                    "n_obs": int(first["n"].sum()),
                    "rmse_h_m": rmse_h,
                    "bias_h_m": bias_h,
                    "rmse_dlnw": rmse_w,
                    "bias_dlnw": bias_w,
                }
            )
    table = pd.DataFrame(rows)
    ref_rmse = table.loc[table["model"] == ref].set_index("bin")["rmse_h_m"]
    table["skill_h_vs_ref"] = 1.0 - table["rmse_h_m"] / table["bin"].map(ref_rmse)

    has_f = has.astype(np.float64)

    def stat(idx: IntArray, groups: IntArray) -> FloatArray:
        del groups
        cnt = has_f[:, idx].sum(axis=1)  # (n_b,)
        sums = mse[:, :, :, idx].sum(axis=3)  # (n_m, n_r, n_b)
        rm = np.sqrt(sums / cnt[None, None, :]).mean(axis=1)  # (n_m, n_b)
        out: FloatArray = rm.ravel()
        return out

    reps = bootstrap_matrix(stat, np.arange(n_s), divisions, boot, n_boot=n_boot)
    reps = reps.reshape(-1, n_m, n_b)
    point = stat(np.arange(n_s, dtype=np.int64), np.arange(n_s, dtype=np.int64)).reshape(n_m, n_b)
    level = 1.0 - boot.alpha_b
    mi_of = {m: i for i, m in enumerate(models)}

    contrasts: list[dict[str, Any]] = []
    pairs = list(params.metrics.contrasts) + [
        (m, ref) for m in LEARNED_MODELS if m in mi_of and (m, ref) not in params.metrics.contrasts
    ]
    for a, c in pairs:
        pre = (a, c) in params.metrics.contrasts
        for bi, b in enumerate(labels):
            diff = reps[:, mi_of[a], bi] - reps[:, mi_of[c], bi]
            lo, hi = percentile_ci(diff, level)
            contrasts.append(
                {
                    "a": a,
                    "b": c,
                    "bin": b,
                    "pre_registered": pre,
                    "delta_rmse_h_m": float(point[mi_of[a], bi] - point[mi_of[c], bi]),
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )

    # geometry vs operational ranking (S-ID-style averaging over repeats, common sites)
    og = obs_geometry.set_index("site_no").reindex(sites)
    geo_rmse: dict[str, float] = {}
    for m in models:
        g = geoms[m]
        g = g.loc[g["site_no"].isin(common)]
        vals = []
        for _, gr in g.groupby("repeat"):
            e = gr.set_index("site_no").reindex(sites)["ln_dmax0"].to_numpy() - og[
                "ln_dmax_obs"
            ].to_numpy(dtype=np.float64)
            vals.append(math.sqrt(float(np.mean(e * e))))
        geo_rmse[m] = float(np.mean(vals))
    hb = labels.index(HIGH_BIN)
    op_rmse = {m: float(point[mi_of[m], hb]) for m in models}
    geo_rank = pd.Series(geo_rmse).rank(method="min").astype(int).to_dict()
    op_rank = pd.Series(op_rmse).rank(method="min").astype(int).to_dict()
    rho = spearman_rho([geo_rmse[m] for m in models], [op_rmse[m] for m in models])
    ranking = [
        {
            "model": m,
            "rmse_log_dmax": geo_rmse[m],
            "rank_geometry": int(geo_rank[m]),
            "rmse_h_high_m": op_rmse[m],
            "rank_operational": int(op_rank[m]),
        }
        for m in models
    ]

    def ci_of(a: str, c: str) -> tuple[float, float, float]:
        row = next(x for x in contrasts if x["a"] == a and x["b"] == c and x["bin"] == HIGH_BIN)
        return float(row["delta_rmse_h_m"]), float(row["ci_lo"]), float(row["ci_hi"])

    hyp: dict[str, Any] = {}
    if "M5" in mi_of and "M2-XGB" in mi_of:
        d, lo, hi = ci_of("M5", "M2-XGB")
        hyp["H-OPS1"] = {
            "delta_rmse_h_m": d,
            "ci": [lo, hi],
            "margin_m": params.hypotheses.noninferiority_margin_m,
            "verdict": _verdict_h1(lo, hi, params.hypotheses.noninferiority_margin_m),
        }
    per_model: dict[str, dict[str, Any]] = {}
    for lm in LEARNED_MODELS:
        if lm not in mi_of:
            continue
        d, lo, hi = ci_of(lm, ref)
        per_model[lm] = {
            "skill": float(1.0 - op_rmse[lm] / op_rmse[ref]),
            "delta_rmse_h_m": d,
            "ci": [lo, hi],
            "beats_reference": bool(hi < 0),
        }
    hyp["H-OPS2"] = {
        "per_model": per_model,
        "verdict": "supported"
        if all(v["beats_reference"] for v in per_model.values())
        else ("falsified"),
    }
    hyp["H-OPS3"] = {
        "spearman_rho": rho,
        "verdict": "rank reversal present" if rho < 1.0 else "no rank reversal",
    }
    return {
        "family": family,
        "repeats": repeats,
        "n_common_sites": int(n_s),
        "failures": failures,
        "table": table.to_dict(orient="records"),
        "contrasts": contrasts,
        "ranking": ranking,
        "hypotheses": hyp,
        "bootstrap": {"b": int(reps.shape[0]), "seed": boot.seed, "level": level},
    }

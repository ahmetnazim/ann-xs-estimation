"""OPS-1 v3 evaluation: all-model regime diagnostics, dynamic-n Oracle, off-anchor states & CCHE1D.

Sources: docs/6_operational_diagnostics_v3.md sections 2-3.

1. :func:`score_all_models_v2` evaluates all 10 models across every flow bin plus ``all_flows`` on
   the OPS-1 v2 ``control`` and ``floodplain`` site-bin tables, stratified by ``all``, ``not_F_S``
   (``S > 1e-5``) and ``F_S`` (``S = 1e-5``), computes all 45 pairwise model contrasts and
   statistical equivalence cliques, and tests ``H-V3-1``.
2. :func:`evaluate_dynamic_roughness_and_oracle` estimates empirical Leopold-Maddock at-a-station
   exponents and leave-one-division-out dynamic roughness ``gamma_{-d} = d ln n / d ln Q``, scores
   ``Oracle`` (true anchor cross-section) + all 10 models + 6 ``M5`` ablations under constant-``n``
   and dynamic-``n``, and tests ``H-V3-2``.
3. :func:`evaluate_off_anchor_states` scores ``Oracle`` + all 10 models on absolute off-anchor
   hydraulic states ``(ln dbar(Q_i), ln dmax(Q_i), dmax(Q_i), ln W(Q_i), ln A(Q_i))`` and the
   anchor-subtraction cancellation decomposition across ADCP measurements, testing ``H-V3-3``.
4. :func:`evaluate_cche1d_readiness` evaluates Dingman continuity reconstruction
   ``A_cons,0 = W_0 dbar_0``, exact hydraulic radius ``R_0 = A_0 / P_0``, 99th-percentile invert
   errors, and the domain-gated ``CCHE1D-Hybrid`` across ``S-ID``, ``S-LRO``, ``S-TAIL-DA`` and
   ``S-TAIL-SO``, testing ``H-V3-4``.

Owning phase: OPS-1 v3 (post-release extension).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from src.config import BootstrapParams
from src.evaluation.bootstrap import bootstrap_matrix, percentile_ci
from src.operational.config import OperationalParams
from src.operational.config_v3 import Operational3Params
from src.operational.evaluate import HIGH_BIN, anchor_geometry, bin_labels
from src.operational.evaluate_v2 import VARIANTS
from src.operational.rating import solve_lambda_np, stage_change, width_log_change
from src.physics.shape import section_terms

__all__ = [
    "ALL_FLOWS_BIN",
    "HYBRID_MODEL",
    "ORACLE_MODEL",
    "REGIMES",
    "add_all_flows_bin",
    "evaluate_cche1d_readiness",
    "evaluate_dynamic_roughness_and_oracle",
    "evaluate_off_anchor_states",
    "load_ablation_predictions",
    "oracle_anchor_geometry",
    "score_all_models_v2",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

ALL_FLOWS_BIN: Final[str] = "all_flows"
ORACLE_MODEL: Final[str] = "Oracle"
HYBRID_MODEL: Final[str] = "CCHE1D-Hybrid"
REGIMES: Final[tuple[str, str, str]] = ("all", "not_F_S", "F_S")
_PRED_COLS: Final[tuple[str, ...]] = ("repeat", "site_no", "mu_W", "mu_dbar", "mu_dmax", "mu_t")


def add_all_flows_bin(bins_df: pd.DataFrame) -> pd.DataFrame:
    """Append the observation-weighted ``all_flows`` bin to a site-bin table (OPS3-E1)."""
    base = bins_df.loc[bins_df["bin"] != HIGH_BIN].copy()
    base["_eh_sum"] = base["n"] * base["mse_h"]
    base["_meh_sum"] = base["n"] * base["me_h"]
    group_cols = [
        c for c in ("variant", "family", "model", "repeat", "site_no") if c in base.columns
    ]
    grp = base.groupby(group_cols, sort=True)
    agg = grp.agg(
        n=("n", "sum"),
        _eh_sum=("_eh_sum", "sum"),
        _meh_sum=("_meh_sum", "sum"),
    ).reset_index()
    agg["mse_h"] = agg["_eh_sum"] / agg["n"]
    agg["me_h"] = agg["_meh_sum"] / agg["n"]
    agg["bin"] = ALL_FLOWS_BIN
    agg["n_w"] = 0
    agg["mse_w"] = np.nan
    agg["me_w"] = np.nan
    keep_cols = list(bins_df.columns)
    return pd.concat([bins_df, agg[keep_cols]], ignore_index=True)


def load_ablation_predictions(
    pred_dir: Path, ablation: str, family: str, analysis: str = "primary"
) -> pd.DataFrame:
    """Load one M5 ablation's site-level predictions from ``{ablation}.parquet``."""
    df = pd.read_parquet(pred_dir / f"{ablation}.parquet")
    keep = (df["analysis"] == analysis) & (df["family"] == family) & (df["unit"] == "site")
    df = df.loc[keep].copy()
    if df.empty:
        raise ValueError(f"M5/{ablation}: no {analysis}/{family} site predictions")
    for col in ("mu_dmax", "mu_t"):
        if col not in df.columns:
            df[col] = np.nan
    out = df[list(_PRED_COLS)].copy()
    out["site_no"] = out["site_no"].astype(str)
    out["repeat"] = out["repeat"].astype(np.int64)
    return out.sort_values(["repeat", "site_no"], kind="mergesort").reset_index(drop=True)


def oracle_anchor_geometry(
    prim: pd.DataFrame, repeats: Sequence[int], r_bounds: tuple[float, float]
) -> pd.DataFrame:
    """Construct the ``Oracle`` anchor geometry table from ground-truth ADCP anchor measurements."""
    sub = prim.loc[(prim["W"] > 0) & (prim["dbar"] > 0) & (prim["dmax"] > prim["dbar"])].copy()
    frames = []
    for rep in repeats:
        frames.append(
            pd.DataFrame(
                {
                    "repeat": int(rep),
                    "site_no": sub["site_no"].astype(str).to_numpy(),
                    "mu_W": np.log(sub["W"].to_numpy(dtype=np.float64)),
                    "mu_dbar": np.log(sub["dbar"].to_numpy(dtype=np.float64)),
                    "mu_dmax": np.log(sub["dmax"].to_numpy(dtype=np.float64)),
                    "mu_t": np.log(
                        sub["dmax"].to_numpy(dtype=np.float64)
                        / sub["dbar"].to_numpy(dtype=np.float64)
                    ),
                }
            )
        )
    pred = pd.concat(frames, ignore_index=True)
    return anchor_geometry(pred, None, r_bounds)


def _verdict_ni(hi: float, margin: float) -> str:
    if hi < 0.0:
        return "supported (superior)"
    if hi < margin:
        return "supported (non-inferior)"
    return "falsified"


def score_all_models_v2(
    family: str,
    bins_df: pd.DataFrame,
    fs_map: Mapping[str, bool],
    divisions: Mapping[str, str],
    models: Sequence[str],
    edges: Sequence[float],
    p3: Operational3Params,
    boot: BootstrapParams,
    *,
    n_boot: int | None = None,
) -> dict[str, Any]:
    """Score all models across flow bins + ``all_flows`` and slope regimes.

    Evaluates all 45 pairwise contrasts per (variant, regime, bin), top-tier equivalence cliques,
    and hypothesis ``H-V3-1``.
    """
    df = add_all_flows_bin(bins_df.loc[bins_df["family"] == family])
    labels = [*bin_labels(edges), ALL_FLOWS_BIN]
    repeats = sorted(int(x) for x in df["repeat"].unique())
    sites = np.array(sorted(df["site_no"].astype(str).unique()), dtype=object)
    if sites.size == 0:
        raise ValueError(f"{family}: empty site set in bins_df")
    pos = {str(s): i for i, s in enumerate(sites)}
    is_fs = np.array([bool(fs_map.get(str(s), False)) for s in sites], dtype=bool)
    reg_masks: dict[str, FloatArray] = {
        "all": np.ones(sites.size, dtype=np.float64),
        "not_F_S": (~is_fs).astype(np.float64),
        "F_S": is_fs.astype(np.float64),
    }
    # If a synthetic test has only one slope regime, avoid zero-denominator NaNs in that stratum
    active_regimes = [r for r in REGIMES if float(reg_masks[r].sum()) > 0]

    keys = [(v, m) for v in VARIANTS for m in models]
    n_k, n_r, n_b, n_s = len(keys), len(repeats), len(labels), sites.size
    mse = np.zeros((n_k, n_r, n_b, n_s), dtype=np.float64)
    me = np.zeros((n_k, n_r, n_b, n_s), dtype=np.float64)
    has = np.zeros((n_b, n_s), dtype=bool)

    for ki, (v, m) in enumerate(keys):
        t = df.loc[(df["variant"] == v) & (df["model"] == m)]
        for ri, rep in enumerate(repeats):
            tr = t.loc[t["repeat"] == rep]
            for bi, b in enumerate(labels):
                tb = tr.loc[tr["bin"] == b]
                idx = np.fromiter((pos[str(s)] for s in tb["site_no"]), dtype=np.int64)
                mse[ki, ri, bi, idx] = tb["mse_h"].to_numpy(dtype=np.float64)
                me[ki, ri, bi, idx] = tb["me_h"].to_numpy(dtype=np.float64)
                has[bi, idx] = True

    has_f = has.astype(np.float64)
    mse_flat = mse.reshape(n_k * n_r * n_b, n_s)
    reg_mat = np.column_stack([reg_masks[r] for r in active_regimes])  # (n_s, n_reg)
    n_reg = len(active_regimes)

    def stat(idx: IntArray, groups: IntArray) -> FloatArray:
        del groups
        counts = np.bincount(idx, minlength=n_s).astype(np.float64)
        w = counts[:, None] * reg_mat  # (n_s, n_reg)
        cnt = has_f @ w  # (n_b, n_reg)
        sums = (mse_flat @ w).reshape(n_k, n_r, n_b, n_reg)
        rm = np.sqrt(sums / np.maximum(cnt[None, None, :, :], 1.0)).mean(axis=1)
        return rm.ravel()

    div = np.array([divisions[str(s)] for s in sites])
    reps = bootstrap_matrix(stat, np.arange(n_s), div, boot, n_boot=n_boot).reshape(
        -1, n_k, n_b, n_reg
    )
    point = stat(np.arange(n_s, dtype=np.int64), np.arange(n_s, dtype=np.int64)).reshape(
        n_k, n_b, n_reg
    )
    level = 1.0 - boot.alpha_b
    kidx = {k: i for i, k in enumerate(keys)}
    bidx = {b: i for i, b in enumerate(labels)}
    ridx = {r: i for i, r in enumerate(active_regimes)}

    # Point table across (variant, regime, model, bin) with median site RMSE and bias
    rows: list[dict[str, Any]] = []
    site_mse_mean_rep = mse.mean(axis=1)  # (n_k, n_b, n_s)
    site_me_mean_rep = me.mean(axis=1)  # (n_k, n_b, n_s)
    for v in VARIANTS:
        for reg in active_regimes:
            rmask = reg_masks[reg] > 0.5
            for bi, b in enumerate(labels):
                bmask = has[bi] & rmask
                n_sites_b = int(bmask.sum())
                if n_sites_b == 0:
                    continue
                ref_rmse = float(point[kidx[(v, "M0")], bi, ridx[reg]])
                for m in models:
                    ki = kidx[(v, m)]
                    rmse_val = float(point[ki, bi, ridx[reg]])
                    bias_val = float(site_me_mean_rep[ki, bi, bmask].mean())
                    med_rmse = float(np.median(np.sqrt(site_mse_mean_rep[ki, bi, bmask])))
                    p90_rmse = float(np.quantile(np.sqrt(site_mse_mean_rep[ki, bi, bmask]), 0.90))
                    rows.append(
                        {
                            "variant": v,
                            "regime": reg,
                            "model": m,
                            "bin": b,
                            "n_sites": n_sites_b,
                            "rmse_h_m": rmse_val,
                            "bias_h_m": bias_val,
                            "median_site_rmse_m": med_rmse,
                            "p90_site_rmse_m": p90_rmse,
                            "skill_vs_M0": float(1.0 - rmse_val / ref_rmse),
                        }
                    )

    # Pairwise contrasts and statistical equivalence cliques for high and all_flows
    pairwise_contrasts: list[dict[str, Any]] = []
    cliques: list[dict[str, Any]] = []
    target_bins = [b for b in ("high", ALL_FLOWS_BIN, "[0,0.5)", "[0.5,2)") if b in bidx]
    for v in VARIANTS:
        for reg in active_regimes:
            ri = ridx[reg]
            for b in target_bins:
                bi = bidx[b]
                ranked = sorted(models, key=lambda m: float(point[kidx[(v, m)], bi, ri]))
                leader = ranked[0]
                clique_members = [leader]
                for m in ranked[1:]:
                    diff = reps[:, kidx[(v, m)], bi, ri] - reps[:, kidx[(v, leader)], bi, ri]
                    lo, hi = percentile_ci(diff, level)
                    if lo <= 0.0:
                        clique_members.append(m)
                cliques.append(
                    {
                        "variant": v,
                        "regime": reg,
                        "bin": b,
                        "leader": leader,
                        "leader_rmse_m": float(point[kidx[(v, leader)], bi, ri]),
                        "top_clique": clique_members,
                        "ranking": [
                            {
                                "rank": rk + 1,
                                "model": m,
                                "rmse_h_m": float(point[kidx[(v, m)], bi, ri]),
                            }
                            for rk, m in enumerate(ranked)
                        ],
                    }
                )
                if b in (HIGH_BIN, ALL_FLOWS_BIN):
                    for a, c in combinations(models, 2):
                        diff = reps[:, kidx[(v, a)], bi, ri] - reps[:, kidx[(v, c)], bi, ri]
                        lo, hi = percentile_ci(diff, level)
                        pairwise_contrasts.append(
                            {
                                "variant": v,
                                "regime": reg,
                                "bin": b,
                                "a": a,
                                "b": c,
                                "delta_rmse_h_m": float(
                                    point[kidx[(v, a)], bi, ri] - point[kidx[(v, c)], bi, ri]
                                ),
                                "ci_lo": lo,
                                "ci_hi": hi,
                                "significant": bool(hi < 0.0 or lo > 0.0),
                            }
                        )

    # Hypothesis H-V3-1: M5 vs M2-XGB on not_F_S vs F_S at high flow under floodplain
    hb = bidx[HIGH_BIN]
    hyp_v3_1: dict[str, Any] = {}
    if "M5" in models and "M2-XGB" in models and "not_F_S" in ridx:
        ri_nf = ridx["not_F_S"]
        diff_nf = (
            reps[:, kidx[("floodplain", "M5")], hb, ri_nf]
            - reps[:, kidx[("floodplain", "M2-XGB")], hb, ri_nf]
        )
        lo_nf, hi_nf = percentile_ci(diff_nf, level)
        pt_nf = float(
            point[kidx[("floodplain", "M5")], hb, ri_nf]
            - point[kidx[("floodplain", "M2-XGB")], hb, ri_nf]
        )
        verdict_nf = _verdict_ni(hi_nf, p3.scoring.noninferiority_margin_m)
        fs_entry: dict[str, Any] = {}
        if "F_S" in ridx:
            ri_fs = ridx["F_S"]
            diff_fs = (
                reps[:, kidx[("floodplain", "M5")], hb, ri_fs]
                - reps[:, kidx[("floodplain", "M2-XGB")], hb, ri_fs]
            )
            lo_fs, hi_fs = percentile_ci(diff_fs, level)
            pt_fs = float(
                point[kidx[("floodplain", "M5")], hb, ri_fs]
                - point[kidx[("floodplain", "M2-XGB")], hb, ri_fs]
            )
            fs_entry = {
                "delta_rmse_fs_m": pt_fs,
                "ci_fs": [lo_fs, hi_fs],
                "exceeds_penalty_threshold": bool(pt_fs >= p3.scoring.fs_penalty_min_m),
            }
        hyp_v3_1 = {
            "delta_rmse_not_fs_m": pt_nf,
            "ci_not_fs": [lo_nf, hi_nf],
            "margin_m": p3.scoring.noninferiority_margin_m,
            "not_fs_verdict": verdict_nf,
            **fs_entry,
            "verdict": verdict_nf
            if (not fs_entry or fs_entry["exceeds_penalty_threshold"])
            else "falsified",
        }

    return {
        "family": family,
        "n_sites_total": int(sites.size),
        "n_sites_not_fs": int((~is_fs).sum()),
        "n_sites_fs": int(is_fs.sum()),
        "table": rows,
        "cliques": cliques,
        "pairwise_contrasts": pairwise_contrasts,
        "H-V3-1": hyp_v3_1,
    }


def evaluate_dynamic_roughness_and_oracle(
    obs: pd.DataFrame,
    unit: pd.DataFrame,
    prim: pd.DataFrame,
    geoms_main: Mapping[str, pd.DataFrame],
    geoms_ablation: Mapping[str, pd.DataFrame],
    p1: OperationalParams,
    p3: Operational3Params,
    r_bounds: tuple[float, float],
    *,
    quad_level: int,
    quad_trunc: float,
    boot: BootstrapParams,
    n_boot: int | None = None,
) -> dict[str, Any]:
    """Estimate Leopold-Maddock dynamic roughness and score Oracle + 10 models + 6 M5 ablations."""
    # 1. Empirical at-a-station exponents and leave-one-division-out gamma_{-d}
    u_cols = ["site_no", "meas_id", "w_m", "dbar_m", "dmax_m", "a_m2", "v_ms"]
    u_sub = unit[u_cols].drop_duplicates(["site_no", "meas_id"]).copy()
    u_sub["site_no"] = u_sub["site_no"].astype(str)
    p_sub = (
        prim[["site_no", "W", "dbar", "dmax", "A", "V"]]
        .drop_duplicates("site_no")
        .rename(
            columns={
                "W": "W_a",
                "dbar": "dbar_a",
                "dmax": "dmax_a",
                "A": "A_a",
                "V": "V_a",
            }
        )
    )
    p_sub["site_no"] = p_sub["site_no"].astype(str)
    m_df = obs.merge(u_sub, on=["site_no", "meas_id"], how="inner").merge(
        p_sub, on="site_no", how="inner"
    )
    valid_n = (
        (m_df["w_m"] > 0)
        & (m_df["dbar_m"] > 0)
        & (m_df["a_m2"] > 0)
        & (m_df["v_ms"] > 0)
        & (m_df["W_a"] > 0)
        & (m_df["dbar_a"] > 0)
        & (m_df["A_a"] > 0)
        & (m_df["V_a"] > 0)
    )
    m_n = m_df.loc[valid_n].copy()
    r_i = m_n["a_m2"] / (m_n["w_m"] + 2.0 * m_n["dbar_m"])
    r_a = m_n["A_a"] / (m_n["W_a"] + 2.0 * m_n["dbar_a"])
    m_n["ln_n_ratio"] = (
        np.log(m_n["a_m2"] / m_n["A_a"]) + (2.0 / 3.0) * np.log(r_i / r_a) - m_n["ln_q_ratio"]
    )
    m_n["n_ratio"] = np.exp(m_n["ln_n_ratio"])

    far = m_n.loc[m_n["ln_q_ratio"].abs() >= p3.dynamic_roughness.min_abs_ln_q_ratio].copy()
    x_q = far["ln_q_ratio"].to_numpy(dtype=np.float64)
    denom = float(np.sum(x_q * x_q))

    def _slope(y: FloatArray, x: FloatArray) -> float:
        return float(np.sum(y * x) / np.sum(x * x))

    far_dmax = far.loc[(far["dmax_m"] > 0) & (far["dmax_a"] > 0)]
    exponents = {
        "n_obs_used": len(far),
        "b_width": _slope(
            np.log(far["w_m"].to_numpy(dtype=np.float64) / far["W_a"].to_numpy(dtype=np.float64)),
            x_q,
        ),
        "f_dbar": _slope(
            np.log(
                far["dbar_m"].to_numpy(dtype=np.float64) / far["dbar_a"].to_numpy(dtype=np.float64)
            ),
            x_q,
        ),
        "f_dmax": _slope(
            np.log(
                far_dmax["dmax_m"].to_numpy(dtype=np.float64)
                / far_dmax["dmax_a"].to_numpy(dtype=np.float64)
            ),
            far_dmax["ln_q_ratio"].to_numpy(dtype=np.float64),
        ),
        "area_b_plus_f": _slope(
            np.log(far["a_m2"].to_numpy(dtype=np.float64) / far["A_a"].to_numpy(dtype=np.float64)),
            x_q,
        ),
        "m_velocity": _slope(
            np.log(far["v_ms"].to_numpy(dtype=np.float64) / far["V_a"].to_numpy(dtype=np.float64)),
            x_q,
        ),
        "gamma_roughness_pooled": float(
            np.sum(far["ln_n_ratio"].to_numpy(dtype=np.float64) * x_q) / denom
        ),
    }

    # Leave-one-division-out gamma_{-d}
    all_divs = sorted(str(d) for d in obs["division"].unique())
    gamma_lodo: dict[str, float] = {}
    for d in all_divs:
        sub_out = far.loc[far["division"].astype(str) != d]
        if len(sub_out) < 10:
            gamma_lodo[d] = exponents["gamma_roughness_pooled"]
        else:
            xd = sub_out["ln_q_ratio"].to_numpy(dtype=np.float64)
            yd = sub_out["ln_n_ratio"].to_numpy(dtype=np.float64)
            gamma_lodo[d] = float(np.sum(yd * xd) / np.sum(xd * xd))

    # Median n_i / n_a by flow bin
    edges = p1.metrics.flow_bin_edges
    labels = bin_labels(edges)
    idx_bin = np.digitize(m_n["q_ratio"].to_numpy(dtype=np.float64), np.asarray(edges[1:-1]))
    m_n["bin"] = np.asarray(labels[:-1], dtype=object)[idx_bin]
    n_ratio_by_bin: dict[str, float] = {}
    for b in labels[:-1]:
        sub_b = m_n.loc[m_n["bin"] == b, "n_ratio"]
        if len(sub_b):
            n_ratio_by_bin[b] = float(np.median(sub_b))
    sub_hi = m_n.loc[m_n["q_ratio"] >= p1.metrics.high_flow_ratio, "n_ratio"]
    if len(sub_hi):
        n_ratio_by_bin[HIGH_BIN] = float(np.median(sub_hi))

    # 2. Score Oracle + 10 models + 6 M5 ablations on high-flow observations
    any_geom = next(iter(geoms_main.values()))
    repeats = sorted(int(x) for x in any_geom["repeat"].unique())
    oracle_geom = oracle_anchor_geometry(prim, repeats, r_bounds)
    all_geoms: dict[str, pd.DataFrame] = {
        ORACLE_MODEL: oracle_geom,
        **geoms_main,
        **geoms_ablation,
    }
    model_names = list(all_geoms.keys())
    obs_hi = obs.loc[obs["q_ratio"] >= p1.metrics.high_flow_ratio].copy()
    obs_hi["gamma_lodo"] = obs_hi["division"].astype(str).map(gamma_lodo)

    common = set(obs_hi["site_no"].astype(str).unique())
    n_rep = len(repeats)
    for g in all_geoms.values():
        ok = g.loc[g["valid"]].groupby("site_no").size()
        common &= set(ok.index[ok == n_rep])
    sites = np.array(sorted(common), dtype=object)
    if sites.size == 0:
        raise ValueError("empty common site set for dynamic roughness evaluation")
    pos = {str(s): i for i, s in enumerate(sites)}
    obs_hi = obs_hi.loc[obs_hi["site_no"].isin(common)].reset_index(drop=True)

    modes = ("constant_n", "dynamic_n_lodo")
    n_m, n_modes, n_s = len(model_names), len(modes), sites.size
    mse = np.zeros((n_m, n_modes, n_rep, n_s), dtype=np.float64)
    me = np.zeros((n_m, n_modes, n_rep, n_s), dtype=np.float64)
    anchor_dmax_mean: dict[str, float] = {}

    for mi, m in enumerate(model_names):
        g = all_geoms[m]
        g_ok = g.loc[g["valid"] & g["site_no"].isin(common)].copy()
        anchor_dmax_mean[m] = float(np.exp(g_ok["ln_dmax0"].to_numpy(dtype=np.float64)).mean())
        j = obs_hi.merge(g_ok, on="site_no", how="inner")
        r_arr = j["r"].to_numpy(dtype=np.float64)
        ln_a0 = (j["ln_dbar0"] - j["ln_w0"]).to_numpy(dtype=np.float64)
        dmax0 = np.exp(j["ln_dmax0"].to_numpy(dtype=np.float64))
        dh_obs = j["dh_obs"].to_numpy(dtype=np.float64)
        ln_q_const = j["ln_q_ratio"].to_numpy(dtype=np.float64)
        ln_q_dyn = (1.0 + j["gamma_lodo"].to_numpy(dtype=np.float64)) * ln_q_const

        for mode_i, ln_q_in in enumerate((ln_q_const, ln_q_dyn)):
            lam, _ = solve_lambda_np(
                ln_q_in,
                r_arr,
                ln_a0,
                level=quad_level,
                trunc=quad_trunc,
                tol=p1.rating.newton_tol,
                max_iter=p1.rating.max_iter,
                chunk_rows=p1.rating.chunk_rows,
            )
            err = stage_change(dmax0, lam) - dh_obs
            tmp = pd.DataFrame(
                {
                    "repeat": j["repeat"].to_numpy(dtype=np.int64),
                    "site_no": j["site_no"].astype(str).to_numpy(),
                    "e2": err * err,
                    "e": err,
                }
            )
            agg = tmp.groupby(["repeat", "site_no"], sort=True).mean().reset_index()
            for ri, rep in enumerate(repeats):
                ar = agg.loc[agg["repeat"] == rep]
                idx = np.fromiter((pos[str(s)] for s in ar["site_no"]), dtype=np.int64)
                mse[mi, mode_i, ri, idx] = ar["e2"].to_numpy(dtype=np.float64)
                me[mi, mode_i, ri, idx] = ar["e"].to_numpy(dtype=np.float64)

    mse_flat = mse.reshape(n_m * n_modes * n_rep, n_s)

    def stat(idx: IntArray, groups: IntArray) -> FloatArray:
        del groups
        w = np.bincount(idx, minlength=n_s).astype(np.float64)
        sums = (mse_flat @ w).reshape(n_m, n_modes, n_rep)
        rm = np.sqrt(sums / float(idx.size)).mean(axis=2)  # (n_m, n_modes)
        return rm.ravel()

    div_map = obs_hi.drop_duplicates("site_no").set_index("site_no")["division"].to_dict()
    div = np.array([str(div_map[str(s)]) for s in sites])
    reps = bootstrap_matrix(stat, np.arange(n_s), div, boot, n_boot=n_boot).reshape(
        -1, n_m, n_modes
    )
    point = stat(np.arange(n_s, dtype=np.int64), np.arange(n_s, dtype=np.int64)).reshape(
        n_m, n_modes
    )
    bias_pt = me.mean(axis=(2, 3))  # (n_m, n_modes)
    level = 1.0 - boot.alpha_b
    midx = {m: i for i, m in enumerate(model_names)}

    table: list[dict[str, Any]] = []
    for mi, m in enumerate(model_names):
        table.append(
            {
                "model": m,
                "mean_anchor_dmax0_m": anchor_dmax_mean[m],
                "rmse_const_n_m": float(point[mi, 0]),
                "bias_const_n_m": float(bias_pt[mi, 0]),
                "rmse_dyn_n_m": float(point[mi, 1]),
                "bias_dyn_n_m": float(bias_pt[mi, 1]),
                "delta_rmse_dyn_minus_const_m": float(point[mi, 1] - point[mi, 0]),
            }
        )

    # Hypothesis H-V3-2: Oracle bias reduction and M2-XGB vs M3 rank reversal restoration
    or_const_bias = float(bias_pt[midx[ORACLE_MODEL], 0])
    or_dyn_bias = float(bias_pt[midx[ORACLE_MODEL], 1])
    diff_const = reps[:, midx["M2-XGB"], 0] - reps[:, midx["M3"], 0]
    lo_c, hi_c = percentile_ci(diff_const, level)
    pt_c = float(point[midx["M2-XGB"], 0] - point[midx["M3"], 0])

    diff_dyn = reps[:, midx["M2-XGB"], 1] - reps[:, midx["M3"], 1]
    lo_d, hi_d = percentile_ci(diff_dyn, level)
    pt_d = float(point[midx["M2-XGB"], 1] - point[midx["M3"], 1])

    h_v3_2_ok = (
        or_const_bias > p3.dynamic_roughness.oracle_const_bias_min_m
        and abs(or_dyn_bias) < p3.dynamic_roughness.oracle_dyn_abs_bias_max_m
        and pt_c > 0.0
        and hi_d < 0.0
    )
    hyp_v3_2 = {
        "oracle_bias_const_n_m": or_const_bias,
        "oracle_bias_dyn_n_m": or_dyn_bias,
        "m2xgb_minus_m3_const_n_m": pt_c,
        "m2xgb_minus_m3_const_n_ci": [lo_c, hi_c],
        "m2xgb_minus_m3_dyn_n_m": pt_d,
        "m2xgb_minus_m3_dyn_n_ci": [lo_d, hi_d],
        "verdict": "supported" if h_v3_2_ok else "falsified",
    }

    return {
        "n_common_high_flow_sites": int(n_s),
        "n_high_flow_obs": len(obs_hi),
        "leopold_maddock_exponents": exponents,
        "gamma_lodo_by_division": gamma_lodo,
        "median_n_ratio_by_bin": n_ratio_by_bin,
        "table": table,
        "H-V3-2": hyp_v3_2,
    }


def evaluate_off_anchor_states(
    obs: pd.DataFrame,
    unit: pd.DataFrame,
    prim: pd.DataFrame,
    geoms_main: Mapping[str, pd.DataFrame],
    p1: OperationalParams,
    r_bounds: tuple[float, float],
    *,
    quad_level: int,
    quad_trunc: float,
    boot: BootstrapParams,
    n_boot: int | None = None,
) -> dict[str, Any]:
    """Score Oracle + 10 models on off-anchor absolute hydraulic states and test H-V3-3."""
    u_cols = ["site_no", "meas_id", "w_m", "dbar_m", "dmax_m", "a_m2"]
    u_sub = unit[u_cols].drop_duplicates(["site_no", "meas_id"]).copy()
    u_sub["site_no"] = u_sub["site_no"].astype(str)
    p_sub = (
        prim[["site_no", "dmax"]].drop_duplicates("site_no").rename(columns={"dmax": "dmax_a_obs"})
    )
    p_sub["site_no"] = p_sub["site_no"].astype(str)

    m_df = obs.merge(u_sub, on=["site_no", "meas_id"], how="inner").merge(
        p_sub, on="site_no", how="inner"
    )
    m_df = m_df.loc[
        (m_df["w_m"] > 0) & (m_df["dbar_m"] > 0) & (m_df["a_m2"] > 0) & (m_df["dmax_m"] > 0)
    ].reset_index(drop=True)

    any_geom = next(iter(geoms_main.values()))
    repeats = sorted(int(x) for x in any_geom["repeat"].unique())
    oracle_geom = oracle_anchor_geometry(prim, repeats, r_bounds)
    all_geoms: dict[str, pd.DataFrame] = {ORACLE_MODEL: oracle_geom, **geoms_main}
    model_names = list(all_geoms.keys())
    n_rep = len(repeats)

    common = set(m_df["site_no"].astype(str).unique())
    for g in all_geoms.values():
        ok = g.loc[g["valid"]].groupby("site_no").size()
        common &= set(ok.index[ok == n_rep])
    m_df = m_df.loc[m_df["site_no"].isin(common)].reset_index(drop=True)

    hi_df = m_df.loc[m_df["q_ratio"] >= p1.metrics.high_flow_ratio].copy()
    hi_sites = np.array(sorted(hi_df["site_no"].astype(str).unique()), dtype=object)
    if hi_sites.size == 0:
        raise ValueError("empty high-flow site set for off-anchor state evaluation")
    hi_pos = {str(s): i for i, s in enumerate(hi_sites)}
    n_m, n_s_hi = len(model_names), hi_sites.size
    mse_ln_dmax_hi = np.zeros((n_m, n_rep, n_s_hi), dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for mi, m in enumerate(model_names):
        g = all_geoms[m]
        g_ok = g.loc[g["valid"] & g["site_no"].isin(common)].copy()
        j = m_df.merge(g_ok, on="site_no", how="inner")
        r_arr = j["r"].to_numpy(dtype=np.float64)
        ln_a0 = (j["ln_dbar0"] - j["ln_w0"]).to_numpy(dtype=np.float64)
        lam, _ = solve_lambda_np(
            j["ln_q_ratio"].to_numpy(dtype=np.float64),
            r_arr,
            ln_a0,
            level=quad_level,
            trunc=quad_trunc,
            tol=p1.rating.newton_tol,
            max_iter=p1.rating.max_iter,
            chunk_rows=p1.rating.chunk_rows,
        )
        ln_dbar_pred = j["ln_dbar0"].to_numpy(dtype=np.float64) + lam
        ln_dmax_pred = j["ln_dmax0"].to_numpy(dtype=np.float64) + lam
        ln_w_pred = j["ln_w0"].to_numpy(dtype=np.float64) + width_log_change(lam, r_arr)
        ln_a_pred = ln_w_pred + ln_dbar_pred

        e_dbar = ln_dbar_pred - np.log(j["dbar_m"].to_numpy(dtype=np.float64))
        e_dmax = ln_dmax_pred - np.log(j["dmax_m"].to_numpy(dtype=np.float64))
        e_w = ln_w_pred - np.log(j["w_m"].to_numpy(dtype=np.float64))
        e_a = ln_a_pred - np.log(j["a_m2"].to_numpy(dtype=np.float64))

        dmax0_pred = np.exp(j["ln_dmax0"].to_numpy(dtype=np.float64))
        dmax_i_pred = np.exp(ln_dmax_pred)
        e_dmax_raw = dmax_i_pred - j["dmax_m"].to_numpy(dtype=np.float64)
        e_anchor_raw = dmax0_pred - j["dmax_a_obs"].to_numpy(dtype=np.float64)
        e_dh_raw = (dmax_i_pred - dmax0_pred) - j["dh_obs"].to_numpy(dtype=np.float64)

        is_hi = j["q_ratio"].to_numpy(dtype=np.float64) >= p1.metrics.high_flow_ratio
        rows.append(
            {
                "model": m,
                "rmse_ln_dbar_all": math.sqrt(float(np.mean(e_dbar * e_dbar))),
                "rmse_ln_dmax_all": math.sqrt(float(np.mean(e_dmax * e_dmax))),
                "rmse_ln_w_all": math.sqrt(float(np.mean(e_w * e_w))),
                "rmse_ln_a_all": math.sqrt(float(np.mean(e_a * e_a))),
                "rmse_ln_dmax_high": math.sqrt(float(np.mean(e_dmax[is_hi] ** 2))),
                "rmse_raw_dmax_high_m": math.sqrt(float(np.mean(e_dmax_raw[is_hi] ** 2))),
                "bias_anchor_dmax0_high_m": float(np.mean(e_anchor_raw[is_hi])),
                "bias_flood_dmax_high_m": float(np.mean(e_dmax_raw[is_hi])),
                "bias_stage_rise_high_m": float(np.mean(e_dh_raw[is_hi])),
            }
        )

        # Site-weighted high-flow ln dmax MSE for paired cluster bootstrap (H-V3-3)
        tmp_hi = pd.DataFrame(
            {
                "repeat": j["repeat"].to_numpy(dtype=np.int64)[is_hi],
                "site_no": j["site_no"].astype(str).to_numpy()[is_hi],
                "e2": (e_dmax[is_hi]) ** 2,
            }
        )
        agg_hi = tmp_hi.groupby(["repeat", "site_no"], sort=True).mean().reset_index()
        for ri, rep in enumerate(repeats):
            ar = agg_hi.loc[agg_hi["repeat"] == rep]
            idx = np.fromiter((hi_pos[str(s)] for s in ar["site_no"]), dtype=np.int64)
            mse_ln_dmax_hi[mi, ri, idx] = ar["e2"].to_numpy(dtype=np.float64)

    mse_flat = mse_ln_dmax_hi.reshape(n_m * n_rep, n_s_hi)

    def stat(idx: IntArray, groups: IntArray) -> FloatArray:
        del groups
        w = np.bincount(idx, minlength=n_s_hi).astype(np.float64)
        sums = (mse_flat @ w).reshape(n_m, n_rep)
        return np.sqrt(sums / float(idx.size)).mean(axis=1)

    div_map = hi_df.drop_duplicates("site_no").set_index("site_no")["division"].to_dict()
    div = np.array([str(div_map[str(s)]) for s in hi_sites])
    reps = bootstrap_matrix(stat, np.arange(n_s_hi), div, boot, n_boot=n_boot).reshape(-1, n_m)
    point = stat(np.arange(n_s_hi, dtype=np.int64), np.arange(n_s_hi, dtype=np.int64))
    level = 1.0 - boot.alpha_b
    midx = {m: i for i, m in enumerate(model_names)}

    for r in rows:
        r["site_weighted_rmse_ln_dmax_high"] = float(point[midx[r["model"]]])

    diff_m3 = reps[:, midx["M2-XGB"]] - reps[:, midx["M3"]]
    lo_m3, hi_m3 = percentile_ci(diff_m3, level)
    pt_m3 = float(point[midx["M2-XGB"]] - point[midx["M3"]])

    diff_m5 = reps[:, midx["M2-XGB"]] - reps[:, midx["M5"]]
    lo_m5, hi_m5 = percentile_ci(diff_m5, level)
    pt_m5 = float(point[midx["M2-XGB"]] - point[midx["M5"]])

    hyp_v3_3 = {
        "m2xgb_minus_m3_rmse_ln_dmax_high": pt_m3,
        "m2xgb_minus_m3_ci": [lo_m3, hi_m3],
        "m2xgb_minus_m5_rmse_ln_dmax_high": pt_m5,
        "m2xgb_minus_m5_ci": [lo_m5, hi_m5],
        "verdict": "supported" if (hi_m3 < 0.0 and hi_m5 < 0.0) else "falsified",
    }

    return {
        "n_off_anchor_obs": len(m_df),
        "n_off_anchor_high_flow_obs": len(hi_df),
        "n_high_flow_sites": int(n_s_hi),
        "table": rows,
        "H-V3-3": hyp_v3_3,
    }


def _compute_ln_r0(
    ln_w: FloatArray,
    ln_dbar: FloatArray,
    ln_dmax: FloatArray,
    shape_floor: float,
    r_bounds: tuple[float, float],
    quad_level: int,
    quad_trunc: float,
) -> tuple[FloatArray, FloatArray]:
    """Return ``(ln_dmax_floored, ln_R0)`` using 513-node tanh-sinh quadrature."""
    ln_floor = ln_dbar + math.log(shape_floor)
    ln_m = np.maximum(ln_dmax, ln_floor)
    d = np.exp(ln_dbar)
    m = np.exp(ln_m)
    r = np.clip(d / (m - d), r_bounds[0], r_bounds[1])
    ln_a = ln_dbar - ln_w
    t_lna = torch.from_numpy(ln_a.copy())
    t_r = torch.from_numpy(r.copy())
    with torch.no_grad():
        terms = section_terms(
            t_lna, torch.zeros_like(t_lna), t_r, level=quad_level, trunc=quad_trunc
        )
    ln_r0 = ln_dbar - terms.log_f.numpy()
    return ln_m, ln_r0


def evaluate_cche1d_readiness(
    pred_dir: Path,
    prim: pd.DataFrame,
    p1: OperationalParams,
    p3: Operational3Params,
    r_bounds: tuple[float, float],
    *,
    quad_level: int,
    quad_trunc: float,
    boot: BootstrapParams,
    n_boot: int | None = None,
) -> dict[str, Any]:
    """Evaluate CCHE1D section continuity, hydraulic radius R0, and CCHE1D-Hybrid across splits."""
    models = list(p1.predictions.models)
    splits = list(p3.cche1d.splits)
    tol_ln = math.log(1.0 + p3.cche1d.continuity_rel_tol)

    p_obs = prim.loc[
        (prim["W"] > 0) & (prim["dbar"] > 0) & (prim["A"] > 0) & (prim["dmax"] > 0)
    ].copy()
    p_obs["site_no"] = p_obs["site_no"].astype(str)
    _, ln_r_obs_arr = _compute_ln_r0(
        np.log(p_obs["W"].to_numpy(dtype=np.float64)),
        np.log(p_obs["dbar"].to_numpy(dtype=np.float64)),
        np.log(p_obs["dmax"].to_numpy(dtype=np.float64)),
        p3.cche1d.shape_ratio_floor,
        r_bounds,
        quad_level,
        quad_trunc,
    )
    p_obs["ln_A_obs"] = np.log(p_obs["A"].to_numpy(dtype=np.float64))
    p_obs["ln_dmax_obs"] = np.log(p_obs["dmax"].to_numpy(dtype=np.float64))
    p_obs["ln_R_obs"] = ln_r_obs_arr
    p_obs["R_obs"] = np.exp(ln_r_obs_arr)
    obs_lookup = p_obs.set_index("site_no")

    split_tables: dict[str, list[dict[str, Any]]] = {}
    tail_contrasts: dict[str, dict[str, Any]] = {}
    m2_sid_cont: dict[str, float] = {}

    for sp in splits:
        fb_raw = pd.read_parquet(pred_dir / f"{p1.predictions.fallback_shape_model}.parquet")
        fb = fb_raw.loc[
            (fb_raw["analysis"] == "primary")
            & (fb_raw["family"] == sp)
            & (fb_raw["unit"] == "site")
        ][["repeat", "site_no", "mu_t"]].copy()
        fb["site_no"] = fb["site_no"].astype(str)
        fb["repeat"] = fb["repeat"].astype(np.int64)

        sp_rows: list[dict[str, Any]] = []
        per_model_err_r0: dict[str, pd.DataFrame] = {}

        for m in models:
            df = pd.read_parquet(pred_dir / f"{m}.parquet")
            keep = (df["analysis"] == "primary") & (df["family"] == sp) & (df["unit"] == "site")
            if "ablation" in df.columns:
                keep &= df["ablation"].fillna("").astype(str).eq("")
            df = df.loc[keep].copy()
            df["site_no"] = df["site_no"].astype(str)
            df["repeat"] = df["repeat"].astype(np.int64)
            df = df.loc[df["site_no"].isin(obs_lookup.index)].copy()
            if "mu_dmax" not in df.columns:
                df["mu_dmax"] = np.nan
            if "mu_A" not in df.columns:
                df["mu_A"] = np.nan

            df = df.merge(
                fb.rename(columns={"mu_t": "_t_fb"}),
                on=["repeat", "site_no"],
                how="left",
            )
            no_dmax = ~np.isfinite(df["mu_dmax"].to_numpy(dtype=np.float64))
            df.loc[no_dmax, "mu_dmax"] = df.loc[no_dmax, "mu_dbar"] + df.loc[no_dmax, "_t_fb"]

            ln_w = df["mu_W"].to_numpy(dtype=np.float64)
            ln_dbar = df["mu_dbar"].to_numpy(dtype=np.float64)
            ln_dmax_raw = df["mu_dmax"].to_numpy(dtype=np.float64)
            ln_a_cons = ln_w + ln_dbar
            has_a = np.isfinite(df["mu_A"].to_numpy(dtype=np.float64))
            ln_a_dir = np.where(has_a, df["mu_A"].to_numpy(dtype=np.float64), ln_a_cons)

            ln_m, ln_r0 = _compute_ln_r0(
                ln_w,
                ln_dbar,
                ln_dmax_raw,
                p3.cche1d.shape_ratio_floor,
                r_bounds,
                quad_level,
                quad_trunc,
            )

            ob = obs_lookup.reindex(df["site_no"])
            e_a_dir = ln_a_dir - ob["ln_A_obs"].to_numpy(dtype=np.float64)
            e_a_cons = ln_a_cons - ob["ln_A_obs"].to_numpy(dtype=np.float64)
            e_r0_ln = ln_r0 - ob["ln_R_obs"].to_numpy(dtype=np.float64)
            e_r0_raw = np.exp(ln_r0) - ob["R_obs"].to_numpy(dtype=np.float64)
            e_dmax_ln = ln_m - ob["ln_dmax_obs"].to_numpy(dtype=np.float64)
            e_dmax_raw = np.exp(ln_m) - ob["dmax"].to_numpy(dtype=np.float64)
            is_fs = ob["F_S"].to_numpy(dtype=bool)

            cont_viol_dir = float(np.mean(np.abs(ln_a_dir - ln_a_cons) > tol_ln))
            rmse_a_dir = math.sqrt(float(np.mean(e_a_dir * e_a_dir)))
            rmse_a_cons = math.sqrt(float(np.mean(e_a_cons * e_a_cons)))
            row_dict: dict[str, Any] = {
                "split": sp,
                "model": m,
                "n_sites": int(df["site_no"].nunique()),
                "rmse_log_A_direct": rmse_a_dir,
                "rmse_log_A_cons": rmse_a_cons,
                "continuity_violation_direct": cont_viol_dir,
                "continuity_violation_cons": 0.0,
                "rmse_log_R0": math.sqrt(float(np.mean(e_r0_ln * e_r0_ln))),
                "rmse_raw_R0_m": math.sqrt(float(np.mean(e_r0_raw * e_r0_raw))),
                "rmse_log_R0_not_FS": math.sqrt(float(np.mean(e_r0_ln[~is_fs] ** 2)))
                if (~is_fs).any()
                else float("nan"),
                "rmse_log_R0_FS": math.sqrt(float(np.mean(e_r0_ln[is_fs] ** 2)))
                if is_fs.any()
                else float("nan"),
                "rmse_log_dmax0": math.sqrt(float(np.mean(e_dmax_ln * e_dmax_ln))),
                "rmse_raw_dmax0_m": math.sqrt(float(np.mean(e_dmax_raw * e_dmax_raw))),
                "p99_abs_err_dmax0_m": float(np.quantile(np.abs(e_dmax_raw), 0.99)),
                "max_pred_dmax0_m": float(np.max(np.exp(ln_m))),
            }
            sp_rows.append(row_dict)
            if sp == "S-ID" and m == p3.cche1d.in_domain_model:
                m2_sid_cont = {
                    "continuity_violation_direct": cont_viol_dir,
                    "continuity_violation_cons": 0.0,
                    "rmse_log_A_direct": rmse_a_dir,
                    "rmse_log_A_cons": rmse_a_cons,
                }

            per_model_err_r0[m] = pd.DataFrame(
                {
                    "repeat": df["repeat"].to_numpy(dtype=np.int64),
                    "site_no": df["site_no"].to_numpy(),
                    "division": ob["division"].astype(str).to_numpy(),
                    "e2_r0": e_r0_ln * e_r0_ln,
                }
            )

        # Add CCHE1D-Hybrid row: uses in_domain_model on S-ID/S-LRO, tail_model on S-TAIL-*
        chosen = p3.cche1d.in_domain_model if sp in ("S-ID", "S-LRO") else p3.cche1d.tail_model
        base_row = next(r for r in sp_rows if r["model"] == chosen)
        sp_rows.append({**base_row, "model": HYBRID_MODEL, "selected_model": chosen})
        split_tables[sp] = sp_rows

        # Paired bootstrap contrast on tail splits: CCHE1D-Hybrid (tail_model) vs in_domain_model
        if sp in ("S-TAIL-DA", "S-TAIL-SO"):
            df_tm = per_model_err_r0[p3.cche1d.tail_model]
            df_id = per_model_err_r0[p3.cche1d.in_domain_model]
            m_tail = df_tm.merge(
                df_id, on=["repeat", "site_no", "division"], suffixes=("_tail", "_in")
            )
            e2_t = m_tail["e2_r0_tail"].to_numpy(dtype=np.float64)
            e2_i = m_tail["e2_r0_in"].to_numpy(dtype=np.float64)
            n_rows = len(m_tail)

            def stat_tail(
                idx: IntArray,
                groups: IntArray,
                _et: FloatArray = e2_t,
                _ei: FloatArray = e2_i,
            ) -> FloatArray:
                del groups
                return np.array(
                    [
                        math.sqrt(float(np.mean(_et[idx]))),
                        math.sqrt(float(np.mean(_ei[idx]))),
                    ],
                    dtype=np.float64,
                )

            reps_t = bootstrap_matrix(
                stat_tail,
                np.arange(n_rows),
                m_tail["division"].to_numpy(),
                boot,
                n_boot=n_boot,
            )
            pt_t = stat_tail(np.arange(n_rows, dtype=np.int64), np.arange(n_rows, dtype=np.int64))
            diff_t = reps_t[:, 0] - reps_t[:, 1]
            lo_t, hi_t = percentile_ci(diff_t, 1.0 - boot.alpha_b)
            tail_contrasts[sp] = {
                "hybrid_model": p3.cche1d.tail_model,
                "in_domain_model": p3.cche1d.in_domain_model,
                "delta_rmse_log_R0": float(pt_t[0] - pt_t[1]),
                "ci": [lo_t, hi_t],
            }

    cont_ok = (
        m2_sid_cont.get("continuity_violation_direct", 0.0) > 0.45
        and m2_sid_cont.get("continuity_violation_cons", 1.0) == 0.0
        and m2_sid_cont.get("rmse_log_A_cons", 1.0) <= m2_sid_cont.get("rmse_log_A_direct", 0.0)
    )
    tails_ok = all(v["delta_rmse_log_R0"] < 0.0 for v in tail_contrasts.values())
    hyp_v3_4 = {
        "m2xgb_sid_continuity": m2_sid_cont,
        "tail_hybrid_vs_m2xgb_R0": tail_contrasts,
        "verdict": "supported" if (cont_ok and tails_ok) else "falsified",
    }

    return {
        "splits": split_tables,
        "H-V3-4": hyp_v3_4,
    }

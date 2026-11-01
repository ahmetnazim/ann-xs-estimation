"""OPS-1 v2 evaluation: control vs floodplain stage responses and H-FP1..4 (docs/5 section 3).

1. :func:`load_terrain` turns the cached acquisition records into :class:`Terrain` objects in
   NAVD88 and applies the site exclusions (status, no-data share, anchor/DEM mismatch).
2. :func:`filter_truncated` drops observations whose observed WSE is above the lower transect
   end (model-independent).
3. :func:`site_responses` computes, for one site, the control (closed form) and floodplain
   (strip conveyance) stage changes for every (family, model, repeat) section;
   :func:`all_responses` runs it over sites in a process pool.
4. :func:`score_v2` scores both variants on one common site set with a single paired,
   division-stratified cluster bootstrap and evaluates the pre-registered hypotheses.

Owning phase: OPS-1 v2 (post-release extension).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import BootstrapParams
from src.evaluation.bootstrap import bootstrap_matrix, percentile_ci
from src.evaluation.metrics import spearman_rho
from src.operational.config_v2 import Operational2Params
from src.operational.evaluate import HIGH_BIN, bin_labels
from src.operational.floodplain import Terrain, control_stage_change, floodplain_stage_change

__all__ = [
    "VARIANTS",
    "SiteTask",
    "all_responses",
    "filter_truncated",
    "load_terrain",
    "score_v2",
    "site_responses",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
VARIANTS: Final[tuple[str, str]] = ("control", "floodplain")


def load_terrain(
    cache_dir: Path, anchors: pd.DataFrame, p: Operational2Params
) -> tuple[dict[str, tuple[Terrain, float]], dict[str, int]]:
    """``site_no -> (terrain, WSE_a in NAVD88)`` for usable sites, and exclusion counts.

    ``anchors`` needs ``site_no``, ``w_a``, ``wse_a``.
    """
    from pyproj import Transformer

    from src.operational.floodplain import prepare_terrain

    to_m = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    counts: dict[str, int] = {}
    out: dict[str, tuple[Terrain, float]] = {}
    t = p.transect
    for row in anchors.to_dict(orient="records"):
        path = cache_dir / f"{row['site_no']}.json"
        if not path.exists():
            counts["excluded_not_acquired"] = counts.get("excluded_not_acquired", 0) + 1
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        status = str(rec["status"])
        if status != "ok":
            counts[status] = counts.get(status, 0) + 1
            continue
        g = rec["geometry"]
        s = rec["samples"]
        z = np.array([np.nan if v is None else float(v) for v in s["z"]], dtype=np.float64)
        px, py = to_m.transform(np.asarray(s["lon"]), np.asarray(s["lat"]))
        h = g["length"] / 2.0
        sx, sy = g["cx"] - h * g["nx"], g["cy"] - h * g["ny"]
        x = (np.asarray(px) - sx) * g["nx"] + (np.asarray(py) - sy) * g["ny"]
        if float(np.mean(~np.isfinite(z))) > t.max_nodata_share:
            counts["excluded_nodata"] = counts.get("excluded_nodata", 0) + 1
            continue
        w_a = float(row["w_a"])
        try:
            ter = prepare_terrain(
                x,
                z,
                h,
                max(t.channel_search_factor_w * w_a, t.channel_search_min_m),
                t.flat_tol_m,
            )
        except ValueError:
            counts["excluded_nodata"] = counts.get("excluded_nodata", 0) + 1
            continue
        wse = float(row["wse_a"]) + float(rec["datum_shift_m"])
        if abs(ter.z_ch - wse) > p.datum.anchor_dem_max_offset_m:
            counts["excluded_anchor_dem"] = counts.get("excluded_anchor_dem", 0) + 1
            continue
        out[str(row["site_no"])] = (ter, wse)
    counts["sites_usable"] = len(out)
    return out, counts


def filter_truncated(
    obs: pd.DataFrame, terrain: Mapping[str, tuple[Terrain, float]]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep observations at usable sites whose observed WSE is below the lower transect end."""
    o = obs.loc[obs["site_no"].isin(list(terrain))].copy()
    n0 = len(o)
    z_end = o["site_no"].map({k: v[0].z_end_low for k, v in terrain.items()})
    wse = o["site_no"].map({k: v[1] for k, v in terrain.items()})
    keep = (wse + o["dh_obs"]) <= z_end
    o = o.loc[keep]
    counts = {
        "rows_at_usable_sites": n0,
        "rows_excluded_truncated": int((~keep).sum()),
        "rows_kept": len(o),
        "sites_with_rows": int(o["site_no"].nunique()),
    }
    return o.reset_index(drop=True), counts


@dataclass(frozen=True)
class SiteTask:
    """One site's terrain, anchor WSE, observed flow ratios and the model sections to evaluate."""

    site_no: str
    terrain: Terrain
    wse: float
    ln_q: FloatArray
    keys: tuple[tuple[str, str, int], ...]  # (family, model, repeat)
    w0: FloatArray
    dmax0: FloatArray
    r: FloatArray
    channel_points: int
    z_levels: int


def site_responses(task: SiteTask) -> tuple[str, FloatArray, FloatArray, FloatArray]:
    """``(site_no, dh_control, dh_floodplain, truncated)``, each ``(n_keys, n_obs)``."""
    n_k, n_o = len(task.keys), task.ln_q.size
    ctrl = np.empty((n_k, n_o))
    fp = np.empty((n_k, n_o))
    tr = np.zeros((n_k, n_o))
    for k in range(n_k):
        ctrl[k] = control_stage_change(task.dmax0[k], task.r[k], task.ln_q)
        dh, t = floodplain_stage_change(
            task.terrain,
            wse_a=task.wse,
            w0=float(task.w0[k]),
            dmax0=float(task.dmax0[k]),
            r=float(task.r[k]),
            ln_q_ratio=task.ln_q,
            channel_points=task.channel_points,
            z_levels=task.z_levels,
        )
        fp[k], tr[k] = dh, t
    return task.site_no, ctrl, fp, tr


def all_responses(tasks: Sequence[SiteTask], workers: int) -> list[tuple[Any, ...]]:
    """Run :func:`site_responses` over ``tasks`` (process pool when ``workers > 1``)."""
    if workers <= 1:
        return [site_responses(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(site_responses, tasks, chunksize=8))


def _verdict_ni(hi: float, margin: float) -> str:
    if hi < 0:
        return "supported (superior)"
    if hi < margin:
        return "supported (non-inferior)"
    return "falsified"


def score_v2(
    family: str,
    tables: Mapping[tuple[str, str], pd.DataFrame],
    common: set[str],
    divisions: Mapping[str, str],
    geo_err: Mapping[str, FloatArray],
    models: Sequence[str],
    repeats: Sequence[int],
    edges: Sequence[float],
    p: Operational2Params,
    boot: BootstrapParams,
    *,
    n_boot: int | None = None,
) -> dict[str, Any]:
    """Score both variants of one family on ``common`` sites with one bootstrap.

    ``tables[(variant, model)]`` are :func:`src.operational.evaluate.site_bin_table` outputs;
    ``geo_err[model]`` holds per-site anchor ``ln dmax`` errors averaged over repeats (ranking).
    """
    labels = bin_labels(edges)
    sites = np.array(sorted(common), dtype=object)
    pos = {s: i for i, s in enumerate(sites)}
    keys = [(v, m) for v in VARIANTS for m in models]
    n_k, n_r, n_b, n_s = len(keys), len(repeats), len(labels), sites.size
    mse = np.zeros((n_k, n_r, n_b, n_s))
    has = np.zeros((n_b, n_s), dtype=bool)
    rows: list[dict[str, Any]] = []
    for ki, key in enumerate(keys):
        t = tables[key]
        t = t.loc[t["site_no"].isin(common)]
        for ri, rep in enumerate(repeats):
            tr_ = t.loc[t["repeat"] == rep]
            for bi, b in enumerate(labels):
                tb = tr_.loc[tr_["bin"] == b]
                idx = np.fromiter((pos[s] for s in tb["site_no"]), dtype=np.int64)
                mse[ki, ri, bi, idx] = tb["mse_h"].to_numpy(dtype=np.float64)
                has[bi, idx] = True
        for b in labels:
            tb = t.loc[t["bin"] == b]
            per = tb.groupby("repeat")
            rows.append(
                {
                    "variant": key[0],
                    "model": key[1],
                    "bin": b,
                    "n_sites": int(tb.loc[tb["repeat"] == repeats[0], "site_no"].nunique()),
                    "n_obs": int(tb.loc[tb["repeat"] == repeats[0], "n"].sum()),
                    "rmse_h_m": float(np.mean([math.sqrt(x["mse_h"].mean()) for _, x in per])),
                    "bias_h_m": float(np.mean([x["me_h"].mean() for _, x in per])),
                }
            )
    table = pd.DataFrame(rows)
    ref = table.loc[table["model"] == "M0"].set_index(["variant", "bin"])["rmse_h_m"]
    table["skill_h_vs_M0"] = 1.0 - table["rmse_h_m"] / [
        ref[(v, b)] for v, b in zip(table["variant"], table["bin"], strict=True)
    ]
    has_f = has.astype(np.float64)

    def stat(idx: IntArray, groups: IntArray) -> FloatArray:
        del groups
        cnt = has_f[:, idx].sum(axis=1)
        sums = mse[:, :, :, idx].sum(axis=3)
        out: FloatArray = np.sqrt(sums / cnt[None, None, :]).mean(axis=1).ravel()
        return out

    div = np.array([divisions[s] for s in sites])
    reps = bootstrap_matrix(stat, np.arange(n_s), div, boot, n_boot=n_boot).reshape(-1, n_k, n_b)
    point = stat(np.arange(n_s, dtype=np.int64), np.arange(n_s, dtype=np.int64)).reshape(n_k, n_b)
    level = 1.0 - boot.alpha_b
    kidx = {k: i for i, k in enumerate(keys)}
    hb = labels.index(HIGH_BIN)

    def contrast(a: tuple[str, str], c: tuple[str, str], bi: int) -> dict[str, Any]:
        diff = reps[:, kidx[a], bi] - reps[:, kidx[c], bi]
        lo, hi = percentile_ci(diff, level)
        return {
            "a": f"{a[1]}@{a[0]}",
            "b": f"{c[1]}@{c[0]}",
            "bin": labels[bi],
            "delta_rmse_h_m": float(point[kidx[a], bi] - point[kidx[c], bi]),
            "ci_lo": lo,
            "ci_hi": hi,
        }

    contrasts = []
    for bi in range(n_b):
        contrasts.append(contrast(("floodplain", "M0"), ("control", "M0"), bi))
        contrasts.append(contrast(("floodplain", "M5"), ("floodplain", "M2-XGB"), bi))
        contrasts.append(contrast(("floodplain", "M3"), ("floodplain", "M0"), bi))
        for m in models:
            if m != "M0":
                contrasts.append(contrast(("floodplain", m), ("control", m), bi))
    hfp1 = next(
        c
        for c in contrasts
        if c["a"] == "M0@floodplain" and c["b"] == "M0@control" and c["bin"] == HIGH_BIN
    )
    hfp2 = next(
        c
        for c in contrasts
        if c["a"] == "M5@floodplain" and c["b"] == "M2-XGB@floodplain" and c["bin"] == HIGH_BIN
    )
    hfp3 = next(
        c
        for c in contrasts
        if c["a"] == "M3@floodplain" and c["b"] == "M0@floodplain" and c["bin"] == HIGH_BIN
    )
    geo = {m: float(np.sqrt(np.mean(geo_err[m] ** 2))) for m in models}
    ranking: dict[str, list[dict[str, Any]]] = {}
    rho: dict[str, float] = {}
    for v in VARIANTS:
        op = {m: float(point[kidx[(v, m)], hb]) for m in models}
        g_rank = pd.Series(geo).rank(method="min").astype(int)
        o_rank = pd.Series(op).rank(method="min").astype(int)
        rho[v] = spearman_rho([geo[m] for m in models], [op[m] for m in models])
        ranking[v] = [
            {
                "model": m,
                "rmse_log_dmax": geo[m],
                "rank_geometry": int(g_rank[m]),
                "rmse_h_high_m": op[m],
                "rank_operational": int(o_rank[m]),
            }
            for m in models
        ]
    hyp = {
        "H-FP1": {**hfp1, "verdict": "supported" if hfp1["ci_hi"] < 0 else "falsified"},
        "H-FP2": {
            **hfp2,
            "margin_m": p.scoring.noninferiority_margin_m,
            "verdict": _verdict_ni(hfp2["ci_hi"], p.scoring.noninferiority_margin_m),
        },
        "H-FP3": {
            **hfp3,
            "verdict": "compensating-error diagnosis supported"
            if hfp3["ci_hi"] >= 0
            else "not supported (M3 still significantly better)",
        },
        "H-FP4": {"spearman_rho": rho},
    }
    return {
        "family": family,
        "repeats": list(repeats),
        "n_common_sites": int(n_s),
        "table": table.to_dict(orient="records"),
        "contrasts": contrasts,
        "ranking": ranking,
        "hypotheses": hyp,
        "bootstrap": {"b": int(reps.shape[0]), "seed": boot.seed, "level": level},
    }

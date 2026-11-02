"""OPS-1 v3 runner: all-model regime diagnostics, dynamic-n Oracle, off-anchor states & CCHE1D.

Usage::

    uv run python -m scripts.ops3 [--out reports/operational] [--n-boot N]

Reads ``reports/operational/ops2_site_bins.parquet``,
``reports/operational/ops1_observations.parquet``,
``data/processed/unit_mapped/hydroswot.parquet``, ``data/processed/site_table.parquet``, and the
frozen ``data/processed/predictions/*.parquet``. Writes ``ops3_results.json`` and
``ops3_summary.md``. Refuses a dirty git tree unless ``--allow-dirty`` (development only).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import REPO_ROOT, load_settings
from src.data.io import sha256_file
from src.operational.config_v3 import load_operational_v3
from src.operational.evaluate import HIGH_BIN, anchor_geometry, load_predictions
from src.operational.evaluate_v3 import (
    ALL_FLOWS_BIN,
    evaluate_cche1d_readiness,
    evaluate_dynamic_roughness_and_oracle,
    evaluate_off_anchor_states,
    load_ablation_predictions,
    score_all_models_v2,
)
from src.utils.logging import git_state

LOG = logging.getLogger("ops3")
PRED_DIR = REPO_ROOT / "data/processed/predictions"
SITE_TABLE = REPO_ROOT / "data/processed/site_table.parquet"
UNIT = REPO_ROOT / "data/processed/unit_mapped/hydroswot.parquet"
OPS1_OBS = REPO_ROOT / "reports/operational/ops1_observations.parquet"
OPS2_BINS = REPO_ROOT / "reports/operational/ops2_site_bins.parquet"


def _clean(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, str):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _summary(b: dict[str, Any]) -> str:
    pv = b["provenance"]
    lines = [
        (
            "# OPS-1 v3 Operational Diagnostics, Dynamic-Resistance & "
            "CCHE1D Readiness Benchmark — results"
        ),
        "",
        f"- git commit: `{pv['git']['commit']}` (dirty: {pv['git']['dirty']})",
        f"- params_operational_v3.yaml SHA-256: `{pv['params_operational_v3_sha256']}`",
        f"- inherited params_operational_v2.yaml SHA-256: `{pv['params_operational_v2_sha256']}`",
        f"- inherited params_operational.yaml SHA-256: `{pv['params_operational_sha256']}`",
        "",
    ]

    for fam, res in b["v2_all_models"].items():
        lines += [
            f"## 1. {fam} All-Model OPS-1 v2 Diagnostics (total = {res['n_sites_total']:,}, "
            f"not_F_S = {res['n_sites_not_fs']:,}, F_S = {res['n_sites_fs']:,})",
            "",
            (
                "| Variant | Regime | Bin | Model | Sites | RMSE Δh (m) | Bias Δh (m) | "
                "Median Site RMSE (m) | Skill vs M0 |"
            ),
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for r in res["table"]:
            if r["bin"] in (HIGH_BIN, ALL_FLOWS_BIN) and (
                r["variant"] == "floodplain" or r["regime"] == "all"
            ):
                lines.append(
                    f"| {r['variant']} | {r['regime']} | {r['bin']} | {r['model']} | "
                    f"{r['n_sites']} | {r['rmse_h_m']:.3f} | {r['bias_h_m']:+.3f} | "
                    f"{r['median_site_rmse_m']:.3f} | {r['skill_vs_M0']:+.3f} |"
                )
        lines += ["", "Statistical equivalence cliques (95% paired cluster bootstrap):", ""]
        for cl in res["cliques"]:
            if cl["variant"] == "floodplain":
                members = ", ".join(cl["top_clique"])
                lines.append(
                    f"- `{cl['variant']} / {cl['regime']} / {cl['bin']}`: leader `{cl['leader']}` "
                    f"({cl['leader_rmse_m']:.3f} m), top clique `[{members}]`"
                )
        h1 = res["H-V3-1"]
        if h1:
            lines += [
                "",
                f"- **H-V3-1 ({fam})**: {h1['verdict']} — `not_F_S` M5 - M2-XGB = "
                f"{h1['delta_rmse_not_fs_m']:+.4f} m [{h1['ci_not_fs'][0]:+.4f}, "
                f"{h1['ci_not_fs'][1]:+.4f}]; `F_S` M5 - M2-XGB = "
                f"{h1.get('delta_rmse_fs_m', 0.0):+.4f} m "
                f"[{h1.get('ci_fs', [0.0, 0.0])[0]:+.4f}, {h1.get('ci_fs', [0.0, 0.0])[1]:+.4f}]",
                "",
            ]

    dyn = b["dynamic_roughness"]
    ex = dyn["leopold_maddock_exponents"]
    lines += [
        "## 2. Leopold-Maddock Dynamic Resistance & Oracle Benchmark (S-ID high flow)",
        "",
        f"Off-anchor empirical exponents (n = {ex['n_obs_used']:,} rows with "
        f"|ln(Q_i/Q_a)| >= 0.2): width `b = {ex['b_width']:+.4f}`, "
        f"mean depth `f = {ex['f_dbar']:+.4f}`, max depth `f_max = {ex['f_dmax']:+.4f}`, "
        f"area `b+f = {ex['area_b_plus_f']:+.4f}`, velocity `m = {ex['m_velocity']:+.4f}`, "
        f"roughness `gamma = {ex['gamma_roughness_pooled']:+.4f}`.",
        "",
        (
            "| Model | Mean Anchor dmax0 (m) | RMSE Const-n (m) | Bias Const-n (m) | "
            "RMSE Dyn-n LODO (m) | Bias Dyn-n LODO (m) | ΔRMSE (Dyn - Const) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in dyn["table"]:
        lines.append(
            f"| {r['model']} | {r['mean_anchor_dmax0_m']:.3f} | {r['rmse_const_n_m']:.3f} | "
            f"{r['bias_const_n_m']:+.3f} | {r['rmse_dyn_n_m']:.3f} | {r['bias_dyn_n_m']:+.3f} | "
            f"{r['delta_rmse_dyn_minus_const_m']:+.3f} |"
        )
    h2 = dyn["H-V3-2"]
    lines += [
        "",
        f"- **H-V3-2**: {h2['verdict']} — Oracle bias const-n "
        f"`{h2['oracle_bias_const_n_m']:+.3f} m` -> dyn-n `{h2['oracle_bias_dyn_n_m']:+.3f} m`; "
        f"M2-XGB - M3 const-n = `{h2['m2xgb_minus_m3_const_n_m']:+.4f} m` "
        f"[{h2['m2xgb_minus_m3_const_n_ci'][0]:+.4f}, "
        f"{h2['m2xgb_minus_m3_const_n_ci'][1]:+.4f}] -> dyn-n = "
        f"`{h2['m2xgb_minus_m3_dyn_n_m']:+.4f} m` [{h2['m2xgb_minus_m3_dyn_n_ci'][0]:+.4f}, "
        f"{h2['m2xgb_minus_m3_dyn_n_ci'][1]:+.4f}]",
        "",
    ]

    off = b["off_anchor_states"]
    lines += [
        f"## 3. Off-Anchor Absolute Hydraulic States (n = {off['n_off_anchor_obs']:,} obs, "
        f"high flow n = {off['n_off_anchor_high_flow_obs']:,})",
        "",
        (
            "| Model | RMSE ln dbar | RMSE ln dmax | RMSE ln W | RMSE ln A | "
            "High RMSE ln dmax | High Raw RMSE dmax (m) | Anchor Bias (m) | "
            "Flood Bias (m) | Stage-Rise Bias (m) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in off["table"]:
        lines.append(
            f"| {r['model']} | {r['rmse_ln_dbar_all']:.4f} | {r['rmse_ln_dmax_all']:.4f} | "
            f"{r['rmse_ln_w_all']:.4f} | {r['rmse_ln_a_all']:.4f} | "
            f"{r['site_weighted_rmse_ln_dmax_high']:.4f} | {r['rmse_raw_dmax_high_m']:.3f} | "
            f"{r['bias_anchor_dmax0_high_m']:+.3f} | {r['bias_flood_dmax_high_m']:+.3f} | "
            f"{r['bias_stage_rise_high_m']:+.3f} |"
        )
    h3 = off["H-V3-3"]
    lines += [
        "",
        f"- **H-V3-3**: {h3['verdict']} — high-flow site-weighted RMSE(ln dmax(Q_i)): "
        f"M2-XGB - M3 = `{h3['m2xgb_minus_m3_rmse_ln_dmax_high']:+.4f}` "
        f"[{h3['m2xgb_minus_m3_ci'][0]:+.4f}, {h3['m2xgb_minus_m3_ci'][1]:+.4f}]; "
        f"M2-XGB - M5 = `{h3['m2xgb_minus_m5_rmse_ln_dmax_high']:+.4f}` "
        f"[{h3['m2xgb_minus_m5_ci'][0]:+.4f}, {h3['m2xgb_minus_m5_ci'][1]:+.4f}]",
        "",
    ]

    cche = b["cche1d_readiness"]
    lines += [
        "## 4. CCHE1D 1D Hydraulic & Sediment-Transport Readiness Benchmark",
        "",
        (
            "| Split | Model | Sites | RMSE_log A_dir | RMSE_log A_cons | Cont Viol Dir | "
            "RMSE_log R0 | RMSE_raw R0 (m) | RMSE_log dmax0 | P99 |err dmax0| (m) |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sp, rows_sp in cche["splits"].items():
        for r in rows_sp:
            lines.append(
                f"| {sp} | {r['model']} | {r['n_sites']} | {r['rmse_log_A_direct']:.4f} | "
                f"{r['rmse_log_A_cons']:.4f} | {r['continuity_violation_direct']:.3f} | "
                f"{r['rmse_log_R0']:.4f} | {r['rmse_raw_R0_m']:.3f} | "
                f"{r['rmse_log_dmax0']:.4f} | {r['p99_abs_err_dmax0_m']:.3f} |"
            )
    h4 = cche["H-V3-4"]
    m2c = h4["m2xgb_sid_continuity"]
    lines += [
        "",
        f"- **H-V3-4**: {h4['verdict']} — M2-XGB S-ID continuity violation "
        f"`{m2c['continuity_violation_direct']:.3f}` -> `{m2c['continuity_violation_cons']:.3f}` "
        f"(RMSE_log A `{m2c['rmse_log_A_direct']:.4f}` -> `{m2c['rmse_log_A_cons']:.4f}`); "
        "tail R0 contrasts (CCHE1D-Hybrid - M2-XGB): "
        + ", ".join(
            f"{sp} `{v['delta_rmse_log_R0']:+.4f}` [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]"
            for sp, v in h4["tail_hybrid_vs_m2xgb_R0"].items()
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run OPS-1 v3; return the exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "reports/operational")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    git = git_state(REPO_ROOT)
    if git.get("dirty") and not args.allow_dirty:
        LOG.error("git tree is dirty; commit first (or --allow-dirty for development)")
        return 2

    t0 = time.time()
    settings = load_settings()
    s3 = load_operational_v3()
    p3, p1 = s3.params, s3.v1.params
    proto = settings.protocol
    r_bounds = (float(proto.domain.r_bounds[0]), float(proto.domain.r_bounds[1]))

    site = pd.read_parquet(SITE_TABLE)
    site["site_no"] = site["site_no"].astype(str)
    prim = site.loc[site["variant"] == "primary"].copy()
    fs_map = dict(zip(prim["site_no"], prim["F_S"].astype(bool), strict=True))
    div_map = dict(zip(prim["site_no"], prim["division"].astype(str), strict=True))

    obs = pd.read_parquet(OPS1_OBS)
    obs["site_no"] = obs["site_no"].astype(str)
    unit = pd.read_parquet(UNIT)
    unit["site_no"] = unit["site_no"].astype(str)
    bins_v2 = pd.read_parquet(OPS2_BINS)
    bins_v2["site_no"] = bins_v2["site_no"].astype(str)

    models = list(p1.predictions.models)
    edges = p1.metrics.flow_bin_edges

    # 1. All-model OPS-1 v2 stratified evaluation (S-ID & S-LRO)
    v2_all_out: dict[str, Any] = {}
    for fam in p3.scoring.families:
        v2_all_out[fam] = score_all_models_v2(
            fam,
            bins_v2,
            fs_map,
            div_map,
            models,
            edges,
            p3,
            proto.evaluation.bootstrap,
            n_boot=args.n_boot,
        )
    LOG.info("completed v2 all-model stratified scoring in %.1fs", time.time() - t0)

    # 2. Load S-ID anchor geometries for main models and M5 ablations
    fb_sid = load_predictions(PRED_DIR, p1.predictions.fallback_shape_model, "S-ID", "primary")
    geoms_sid: dict[str, pd.DataFrame] = {
        str(m): anchor_geometry(
            load_predictions(PRED_DIR, m, "S-ID", p1.predictions.analysis),
            fb_sid,
            r_bounds,
        )
        for m in models
    }
    geoms_abl: dict[str, pd.DataFrame] = {
        str(ab): anchor_geometry(
            load_ablation_predictions(PRED_DIR, ab, "S-ID", p1.predictions.analysis),
            fb_sid,
            r_bounds,
        )
        for ab in p3.ablation_models
    }

    dyn_out = evaluate_dynamic_roughness_and_oracle(
        obs,
        unit,
        prim,
        geoms_sid,
        geoms_abl,
        p1,
        p3,
        r_bounds,
        quad_level=proto.solver.quad_level,
        quad_trunc=proto.solver.quad_trunc,
        boot=proto.evaluation.bootstrap,
        n_boot=args.n_boot,
    )
    LOG.info("completed dynamic-roughness & Oracle scoring in %.1fs", time.time() - t0)

    # 3. Off-anchor absolute hydraulic state evaluation
    off_out = evaluate_off_anchor_states(
        obs,
        unit,
        prim,
        geoms_sid,
        p1,
        r_bounds,
        quad_level=proto.solver.quad_level,
        quad_trunc=proto.solver.quad_trunc,
        boot=proto.evaluation.bootstrap,
        n_boot=args.n_boot,
    )
    LOG.info("completed off-anchor state scoring in %.1fs", time.time() - t0)

    # 4. CCHE1D readiness evaluation across S-ID, S-LRO, S-TAIL-DA, S-TAIL-SO
    cche_out = evaluate_cche1d_readiness(
        PRED_DIR,
        prim,
        p1,
        p3,
        r_bounds,
        quad_level=proto.solver.quad_level,
        quad_trunc=proto.solver.quad_trunc,
        boot=proto.evaluation.bootstrap,
        n_boot=args.n_boot,
    )
    LOG.info("completed CCHE1D readiness scoring in %.1fs", time.time() - t0)

    inputs = {
        str(p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p): sha256_file(p)
        for p in [
            SITE_TABLE,
            UNIT,
            OPS1_OBS,
            OPS2_BINS,
            *[PRED_DIR / f"{m}.parquet" for m in models],
        ]
    }
    bundle = {
        "study": (
            "OPS-1 v3 Operational Diagnostics, Dynamic-Resistance & CCHE1D Readiness Benchmark"
        ),
        "provenance": {
            "git": git,
            "params_sha256": settings.params_sha256,
            "params_operational_sha256": s3.v1.sha256,
            "params_operational_v2_sha256": s3.v2.sha256,
            "params_operational_v3_sha256": s3.sha256,
            "inputs_sha256": inputs,
            "n_boot_override": args.n_boot,
        },
        "v2_all_models": v2_all_out,
        "dynamic_roughness": dyn_out,
        "off_anchor_states": off_out,
        "cche1d_readiness": cche_out,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    clean = _clean(bundle)
    assert isinstance(clean, dict)
    (args.out / "ops3_results.json").write_text(
        json.dumps(clean, indent=1, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.out / "ops3_summary.md").write_text(_summary(clean), encoding="utf-8")
    LOG.info("wrote %s in %.1fs", args.out, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""OPS-1 v2 runner: control vs floodplain stage responses, H-FP1..4 and the result bundle.

Usage::

    uv run python -m scripts.ops2 [--out reports/operational] [--n-boot N] [--workers K]

Requires ``data/raw/ops_transects/`` from ``scripts.ops2_acquire``. Writes ``ops2_results.json``,
``ops2_summary.md`` and ``ops2_site_bins.parquet``. Refuses a dirty git tree unless
``--allow-dirty`` (development only; recorded in the bundle).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import REPO_ROOT, load_settings
from src.data.io import sha256_file
from src.operational.config_v2 import load_operational_v2
from src.operational.evaluate import HIGH_BIN, anchor_geometry, load_predictions, site_bin_table
from src.operational.evaluate_v2 import (
    VARIANTS,
    SiteTask,
    all_responses,
    filter_truncated,
    load_terrain,
    score_v2,
)
from src.operational.observations import build_observations, read_stage_fields
from src.operational.transects import anchor_frame
from src.utils.logging import git_state

LOG = logging.getLogger("ops2")
PRED_DIR = REPO_ROOT / "data/processed/predictions"
SITE_TABLE = REPO_ROOT / "data/processed/site_table.parquet"
UNIT = REPO_ROOT / "data/processed/unit_mapped/hydroswot.parquet"
RAW = REPO_ROOT / "data/raw/hydroswot/SWOT_ADCP_Dataset.txt"
CACHE = REPO_ROOT / "data/raw/ops_transects"


def _clean(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, str):
        obj = obj.item()  # type: ignore[union-attr]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _summary(b: dict[str, Any]) -> str:
    pv = b["provenance"]
    lines = [
        "# OPS-1 v2 Floodplain-augmented Operational Stage-Response Test — results",
        "",
        f"- git commit: `{pv['git']['commit']}` (dirty: {pv['git']['dirty']})",
        f"- params_operational_v2.yaml SHA-256: `{pv['params_operational_v2_sha256']}`",
        f"- inherited params_operational.yaml SHA-256: `{pv['params_operational_sha256']}`",
        "",
        "## Sites and observations",
        "",
        "Terrain: " + ", ".join(f"{k} {v}" for k, v in b["terrain_counts"].items()),
        "",
        "Observations: " + ", ".join(f"{k} {v}" for k, v in b["observation_counts"].items()),
        "",
    ]
    for fam, res in b["families"].items():
        lines += [
            f"## {fam} — common sites n = {res['n_common_sites']:,}, repeats {res['repeats']}",
            "",
            "| Variant | Model | bin | sites | obs | RMSE Δh (m) | bias Δh (m) | skill vs M0 |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for r in res["table"]:
            lines.append(
                f"| {r['variant']} | {r['model']} | {r['bin']} | {r['n_sites']} | {r['n_obs']} | "
                f"{r['rmse_h_m']:.3f} | {r['bias_h_m']:+.3f} | {r['skill_h_vs_M0']:+.3f} |"
            )
        lines += ["", "High-flow contrasts (ΔRMSE Δh, 95% CI):", ""]
        for c in res["contrasts"]:
            if c["bin"] == HIGH_BIN:
                lines.append(
                    f"- {c['a']} - {c['b']}: {c['delta_rmse_h_m']:+.4f} m "
                    f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]"
                )
        for v, rk in res["ranking"].items():
            lines += [
                "",
                f"Ranking ({v}):",
                "",
                "| Model | RMSE_log dmax | g-rank | RMSE Δh high | op-rank |",
                "|---|---:|---:|---:|---:|",
            ]
            for x in rk:
                lines.append(
                    f"| {x['model']} | {x['rmse_log_dmax']:.4f} | {x['rank_geometry']} | "
                    f"{x['rmse_h_high_m']:.4f} | {x['rank_operational']} |"
                )
        h = res["hypotheses"]
        lines += [
            "",
            f"- **H-FP1**: {h['H-FP1']['verdict']} ({h['H-FP1']['delta_rmse_h_m']:+.4f} m "
            f"[{h['H-FP1']['ci_lo']:+.4f}, {h['H-FP1']['ci_hi']:+.4f}])",
            f"- **H-FP2**: {h['H-FP2']['verdict']} ({h['H-FP2']['delta_rmse_h_m']:+.4f} m "
            f"[{h['H-FP2']['ci_lo']:+.4f}, {h['H-FP2']['ci_hi']:+.4f}])",
            f"- **H-FP3**: {h['H-FP3']['verdict']} ({h['H-FP3']['delta_rmse_h_m']:+.4f} m "
            f"[{h['H-FP3']['ci_lo']:+.4f}, {h['H-FP3']['ci_hi']:+.4f}])",
            "- **H-FP4**: Spearman rho "
            + ", ".join(f"{k} {v:.3f}" for k, v in h["H-FP4"]["spearman_rho"].items()),
            "",
            "Floodplain predictions in truncated terrain (share): "
            + ", ".join(f"{m} {s:.3f}" for m, s in res["truncated_share"].items()),
            "",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run OPS-1 v2; return the exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "reports/operational")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 2, 1))
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    git = git_state(REPO_ROOT)
    if git.get("dirty") and not args.allow_dirty:
        LOG.error("git tree is dirty; commit first (or --allow-dirty for development)")
        return 2
    settings = load_settings()
    s2 = load_operational_v2()
    p2, p1 = s2.params, s2.v1.params
    proto = settings.protocol

    site = pd.read_parquet(SITE_TABLE)
    site["site_no"] = site["site_no"].astype(str)
    prim = site.loc[site["variant"] == "primary"].copy()
    stage = read_stage_fields(RAW)
    unit = pd.read_parquet(UNIT)
    obs, _ = build_observations(
        unit, stage, prim, p1.observations, window_years=p1.observations.window_years
    )
    alt = pd.read_csv(
        RAW,
        sep="\t",
        dtype=str,
        usecols=["site_no", "alt_va", "alt_datum_cd"],
        encoding="latin-1",
        keep_default_na=False,
    )
    anchors = anchor_frame(site, stage, alt)
    terrain, t_counts = load_terrain(CACHE, anchors, p2)
    obs2, o_counts = filter_truncated(obs, terrain)
    LOG.info("terrain %s; observations %s", t_counts, o_counts)
    ln_dmax_obs = prim.set_index("site_no")["dmax"].map(math.log)
    divisions = obs2.drop_duplicates("site_no").set_index("site_no")["division"].to_dict()
    obs_sites = set(obs2["site_no"])
    models = list(p1.predictions.models)

    geoms: dict[tuple[str, str], pd.DataFrame] = {}
    common: dict[str, set[str]] = {}
    repeats: dict[str, list[int]] = {}
    for fam in p2.scoring.families:
        fb = load_predictions(PRED_DIR, p1.predictions.fallback_shape_model, fam, "primary")
        cset = set(obs_sites)
        for m in models:
            g = anchor_geometry(
                load_predictions(PRED_DIR, m, fam, p1.predictions.analysis),
                fb,
                tuple(proto.domain.r_bounds),
            )
            geoms[(fam, m)] = g
            n_rep = g["repeat"].nunique()
            ok = g.loc[g["valid"]].groupby("site_no").size()
            cset &= set(ok.index[ok == n_rep])
            repeats[fam] = sorted(int(x) for x in g["repeat"].unique())
        common[fam] = cset

    tasks: list[SiteTask] = []
    obs_by_site = dict(iter(obs2.groupby("site_no")))
    for s in sorted(set().union(*common.values())):
        keys, w0, d0, rr = [], [], [], []
        for fam in p2.scoring.families:
            if s not in common[fam]:
                continue
            for m in models:
                g = geoms[(fam, m)]
                for row in g.loc[g["site_no"] == s].itertuples(index=False):
                    keys.append((fam, m, int(row.repeat)))
                    w0.append(math.exp(row.ln_w0))
                    d0.append(math.exp(row.ln_dmax0))
                    rr.append(float(row.r))
        ter, wse = terrain[s]
        tasks.append(
            SiteTask(
                site_no=s,
                terrain=ter,
                wse=wse,
                ln_q=obs_by_site[s]["ln_q_ratio"].to_numpy(dtype=np.float64),
                keys=tuple(keys),
                w0=np.array(w0),
                dmax0=np.array(d0),
                r=np.array(rr),
                channel_points=p2.section.channel_points,
                z_levels=p2.section.z_levels,
            )
        )
    t0 = time.time()
    results = all_responses(tasks, args.workers)
    LOG.info("%d sites solved in %.1fs with %d workers", len(tasks), time.time() - t0, args.workers)

    parts: dict[tuple[str, str, str], list[pd.DataFrame]] = {}
    trunc_n: dict[tuple[str, str], list[float]] = {}
    for task, (s, ctrl, fp, tr) in zip(tasks, results, strict=True):
        o = obs_by_site[s]
        for k, (fam, m, rep) in enumerate(task.keys):
            for v, dh in (("control", ctrl[k]), ("floodplain", fp[k])):
                parts.setdefault((v, fam, m), []).append(
                    pd.DataFrame(
                        {
                            "repeat": rep,
                            "site_no": s,
                            "q_ratio": o["q_ratio"].to_numpy(),
                            "e_h": dh - o["dh_obs"].to_numpy(),
                            "e_w": np.nan,
                        }
                    )
                )
            trunc_n.setdefault((fam, m), []).extend(tr[k].tolist())
    edges = p1.metrics.flow_bin_edges
    fam_out: dict[str, Any] = {}
    bins_frames = []
    for fam in p2.scoring.families:
        tables = {}
        for v in VARIANTS:
            for m in models:
                resp = pd.concat(parts[(v, fam, m)], ignore_index=True)
                tables[(v, m)] = site_bin_table(resp, edges, p1.metrics.high_flow_ratio)
                bins_frames.append(tables[(v, m)].assign(variant=v, model=m, family=fam))
        cs = common[fam] & set(obs_by_site)
        geo_err = {}
        for m in models:
            g = geoms[(fam, m)]
            g = g.loc[g["site_no"].isin(cs)]
            geo_err[m] = (g["ln_dmax0"] - g["site_no"].map(ln_dmax_obs)).to_numpy(dtype=float)
        res = score_v2(
            fam,
            tables,
            cs,
            divisions,
            geo_err,
            models,
            repeats[fam],
            edges,
            p2,
            proto.evaluation.bootstrap,
            n_boot=args.n_boot,
        )
        res["truncated_share"] = {m: float(np.mean(trunc_n[(fam, m)])) for m in models}
        fam_out[fam] = res
    inputs = {
        str(p.relative_to(REPO_ROOT)): sha256_file(p)
        for p in [
            SITE_TABLE,
            UNIT,
            RAW,
            CACHE / "MANIFEST.json",
            *[PRED_DIR / f"{m}.parquet" for m in models],
        ]
    }
    bundle = {
        "study": "OPS-1 v2 Floodplain-augmented Operational Stage-Response Test",
        "provenance": {
            "git": git,
            "params_sha256": settings.params_sha256,
            "params_operational_sha256": s2.v1.sha256,
            "params_operational_v2_sha256": s2.sha256,
            "inputs_sha256": inputs,
            "n_boot_override": args.n_boot,
        },
        "terrain_counts": t_counts,
        "observation_counts": o_counts,
        "families": fam_out,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    clean = _clean(bundle)
    assert isinstance(clean, dict)
    (args.out / "ops2_results.json").write_text(
        json.dumps(clean, indent=1, allow_nan=False) + "\n", encoding="utf-8"
    )
    pd.concat(bins_frames, ignore_index=True).to_parquet(
        args.out / "ops2_site_bins.parquet", index=False
    )
    (args.out / "ops2_summary.md").write_text(_summary(clean), encoding="utf-8")
    LOG.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

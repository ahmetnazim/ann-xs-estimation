"""OPS-1 runner: score every model's operational stage response and write the result bundle.

Usage::

    uv run python -m scripts.operational [--out reports/operational] [--n-boot N]

Reads the frozen v1.0.0 predictions, ``site_table.parquet``, the unit-mapped and raw HYDROSWOT
tables, ``params.yaml`` (quadrature, ``domain.r_bounds``, bootstrap) and
``params_operational.yaml`` (pre-registration). Writes ``ops1_results.json``,
``ops1_observations.parquet``, ``ops1_site_bins.parquet`` and ``ops1_summary.md``. Refuses to
run on a dirty git tree unless ``--allow-dirty`` (development only; recorded in the bundle).
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
from src.operational.config import load_operational
from src.operational.evaluate import (
    HIGH_BIN,
    anchor_geometry,
    load_predictions,
    model_responses,
    score_family,
    site_bin_table,
)
from src.operational.observations import build_observations, read_stage_fields
from src.utils.logging import git_state

ROOT = REPO_ROOT
LOG = logging.getLogger("ops1")
PRED_DIR = ROOT / "data/processed/predictions"
SITE_TABLE = ROOT / "data/processed/site_table.parquet"
UNIT_HYDROSWOT = ROOT / "data/processed/unit_mapped/hydroswot.parquet"
RAW_HYDROSWOT = ROOT / "data/raw/hydroswot/SWOT_ADCP_Dataset.txt"


def _clean(obj: object) -> object:
    """JSON-safe copy: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, str):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _run_variant(
    name: str,
    families: list[str],
    window: float | None,
    unit: pd.DataFrame,
    stage: pd.DataFrame,
    anchors: pd.DataFrame,
    obs_geometry: pd.DataFrame,
    n_boot: int | None,
) -> tuple[dict[str, Any], pd.DataFrame, list[pd.DataFrame]]:
    settings = load_settings()
    ops = load_operational().params
    proto = settings.protocol
    obs, counts = build_observations(unit, stage, anchors, ops.observations, window_years=window)
    LOG.info("%s: %d observations at %d sites", name, len(obs), obs["site_no"].nunique())
    out: dict[str, Any] = {"window_years": window, "observation_counts": counts, "families": {}}
    site_bins: list[pd.DataFrame] = []
    for fam in families:
        t0 = time.time()
        fb = load_predictions(PRED_DIR, ops.predictions.fallback_shape_model, fam, "primary")
        geoms: dict[str, pd.DataFrame] = {}
        tables: dict[str, pd.DataFrame] = {}
        worst = 0
        for m in ops.predictions.models:
            pred = load_predictions(PRED_DIR, m, fam, ops.predictions.analysis)
            geoms[m] = anchor_geometry(pred, fb, tuple(proto.domain.r_bounds))
            resp, it = model_responses(
                obs,
                geoms[m],
                level=proto.solver.quad_level,
                trunc=proto.solver.quad_trunc,
                tol=ops.rating.newton_tol,
                max_iter=ops.rating.max_iter,
                chunk_rows=ops.rating.chunk_rows,
            )
            worst = max(worst, it)
            tables[m] = site_bin_table(
                resp, ops.metrics.flow_bin_edges, ops.metrics.high_flow_ratio
            )
            site_bins.append(tables[m].assign(model=m, family=fam, variant=name))
        res = score_family(
            fam, obs, geoms, tables, obs_geometry, ops, proto.evaluation.bootstrap, n_boot=n_boot
        )
        res["solver_max_iterations"] = worst
        out["families"][fam] = res
        LOG.info("%s/%s scored in %.1fs", name, fam, time.time() - t0)
    return out, obs.assign(variant=name), site_bins


def _summary_md(bundle: dict[str, Any]) -> str:
    lines = [
        "# OPS-1 Operational Stage-Response Test — results",
        "",
        f"- git commit: `{bundle['provenance']['git']['commit']}` "
        f"(dirty: {bundle['provenance']['git']['dirty']})",
        f"- params_operational.yaml SHA-256: `{bundle['provenance']['params_operational_sha256']}`",
        f"- params.yaml SHA-256: `{bundle['provenance']['params_sha256']}`",
        "",
    ]
    for vname, var in bundle["variants"].items():
        lines += [f"## Variant `{vname}` (window = {var['window_years']} y)", ""]
        c = var["observation_counts"]
        lines += [
            f"Observations kept: {c['rows_kept']:,} at {c['sites_with_rows']:,} sites "
            f"(eligible sites {c['sites_eligible']:,}).",
            "",
        ]
        for fam, res in var["families"].items():
            lines += [
                f"### {fam} — common sites n = {res['n_common_sites']:,}, repeats {res['repeats']}",
                "",
                "| Model | bin | sites | obs | RMSE Δh (m) | bias Δh (m) | skill vs M0 | "
                "RMSE ΔlnW |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
            for row in res["table"]:
                lines.append(
                    f"| {row['model']} | {row['bin']} | {row['n_sites']} | {row['n_obs']} | "
                    f"{row['rmse_h_m']:.3f} | {row['bias_h_m']:+.3f} | "
                    f"{row['skill_h_vs_ref']:+.3f} | {row['rmse_dlnw']:.3f} |"
                )
            lines += ["", "Contrasts (high flow, ΔRMSE Δh, 95% CI):", ""]
            for ct in res["contrasts"]:
                if ct["bin"] == HIGH_BIN:
                    tag = "pre-registered" if ct["pre_registered"] else "descriptive"
                    lines.append(
                        f"- {ct['a']} - {ct['b']}: {ct['delta_rmse_h_m']:+.4f} m "
                        f"[{ct['ci_lo']:+.4f}, {ct['ci_hi']:+.4f}] ({tag})"
                    )
            lines += ["", "| Model | RMSE_log dmax | geometry rank | RMSE Δh high | op. rank |"]
            lines += ["|---|---:|---:|---:|---:|"]
            for rk in res["ranking"]:
                lines.append(
                    f"| {rk['model']} | {rk['rmse_log_dmax']:.4f} | {rk['rank_geometry']} | "
                    f"{rk['rmse_h_high_m']:.4f} | {rk['rank_operational']} |"
                )
            h = res["hypotheses"]
            lines += [
                "",
                f"- **H-OPS1**: {h['H-OPS1']['verdict']} (Δ = {h['H-OPS1']['delta_rmse_h_m']:+.4f}"
                f" m, CI {h['H-OPS1']['ci'][0]:+.4f}, {h['H-OPS1']['ci'][1]:+.4f})",
                f"- **H-OPS2**: {h['H-OPS2']['verdict']}",
                f"- **H-OPS3**: {h['H-OPS3']['verdict']} (Spearman rho = "
                f"{h['H-OPS3']['spearman_rho']:.3f})",
                "",
                "Failure / clip shares: "
                + ", ".join(
                    f"{m} {v['invalid_share']:.3f}/{v['clipped_share']:.3f}"
                    for m, v in res["failures"].items()
                ),
                "",
            ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run OPS-1 and write the bundle; return the process exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "reports/operational")
    ap.add_argument("--n-boot", type=int, default=None, help="override B (development only)")
    ap.add_argument("--allow-dirty", action="store_true", help="development only")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    git = git_state(ROOT)
    if git.get("dirty") and not args.allow_dirty:
        LOG.error("git tree is dirty; commit first (or --allow-dirty for development)")
        return 2
    settings = load_settings()
    ops_settings = load_operational()
    ops = ops_settings.params

    site = pd.read_parquet(SITE_TABLE)
    anchors = site.loc[site["variant"] == "primary"].copy()
    anchors["site_no"] = anchors["site_no"].astype(str)
    obs_geometry = pd.DataFrame(
        {
            "site_no": anchors["site_no"],
            "ln_dmax_obs": anchors["dmax"].map(math.log),
            "ln_dbar_obs": anchors["dbar"].map(math.log),
        }
    )
    unit = pd.read_parquet(UNIT_HYDROSWOT)
    stage = read_stage_fields(RAW_HYDROSWOT)

    variants: dict[str, Any] = {}
    obs_frames: list[pd.DataFrame] = []
    bins: list[pd.DataFrame] = []
    primary, o, b = _run_variant(
        "primary",
        list(ops.predictions.families),
        ops.observations.window_years,
        unit,
        stage,
        anchors,
        obs_geometry,
        args.n_boot,
    )
    variants["primary"] = primary
    obs_frames.append(o)
    bins += b
    if ops.sensitivity.window_years_none:
        sens, o, b = _run_variant(
            "no_window",
            [ops.predictions.primary_family],
            None,
            unit,
            stage,
            anchors,
            obs_geometry,
            args.n_boot,
        )
        variants["no_window"] = sens
        obs_frames.append(o)
        bins += b

    inputs = {
        str(p.relative_to(ROOT)): sha256_file(p)
        for p in [
            SITE_TABLE,
            UNIT_HYDROSWOT,
            RAW_HYDROSWOT,
            *[PRED_DIR / f"{m}.parquet" for m in ops.predictions.models],
        ]
    }
    bundle = {
        "study": "OPS-1 Operational Stage-Response Test",
        "provenance": {
            "git": git,
            "params_sha256": settings.params_sha256,
            "params_operational_sha256": ops_settings.sha256,
            "inputs_sha256": inputs,
            "n_boot_override": args.n_boot,
        },
        "variants": variants,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    clean = _clean(bundle)
    (args.out / "ops1_results.json").write_text(
        json.dumps(clean, indent=1, sort_keys=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    pd.concat(obs_frames, ignore_index=True).to_parquet(
        args.out / "ops1_observations.parquet", index=False
    )
    pd.concat(bins, ignore_index=True).to_parquet(args.out / "ops1_site_bins.parquet", index=False)
    (args.out / "ops1_summary.md").write_text(_summary_md(clean), encoding="utf-8")
    LOG.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

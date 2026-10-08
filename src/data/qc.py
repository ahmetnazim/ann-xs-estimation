"""Stage qc: rules R1-R11 on the unit-mapped HYDROSWOT table (protocol section 2.4).

Sources: protocol section 2.4 (rule order fixed; every rule logs rows and sites removed or
flagged; rules are row-wise and deterministic, so QC cannot leak across folds, L5), section 2.2
(gross-error bounds, epsilon_A tolerance, t band), section 2.3 (physiography polygons).

Rules as implemented (SI units; ``params.qc`` and ``params.domain.r_bounds``):

* R1 keep ``site_tp_cd == "ST"``.
* R2 keep sites whose NAD83 point intersects F&J polygons of exactly one ``DIVISION``; points
  in no polygon, in polygons without a division, or on a boundary between two divisions are
  dropped. Invalid polygons are repaired with ``shapely.make_valid`` first (13 as delivered).
* R3 drop every row of a measurement (``site_no``, ``q_meas_no``) that has any part with
  ``q_meas_chan_nu >= 2`` (multi-channel). Channel 0 and 1 rows of single-channel measurements
  are kept.
* R4 drop null or non-positive ``DA_site_km2``, ``Q_obs``, ``Q_meas``, ``W``, ``A``, ``dbar``,
  ``dmax``, ``V``, ``Vmax``.
* R5 drop rows outside the a priori bounds (``W``, ``dbar``, ``dmax``, ``V``); counted as F_bnd.
* R6 drop ``dbar > dmax`` or ``V > Vmax``.
* R7 drop if any pair is outside +/- ``q_tol`` relative to its reference:
  ``Q_meas`` vs ``Q_obs``, ``A V`` vs ``Q_meas``, ``A V`` vs ``Q_obs``.
* R8 ``eps_A = ln(A / (W dbar))``; flag ``F_eps`` if ``|eps_A| > ln(1 + epsA_tol)`` (kept;
  primary analyses drop, SA3 keeps).
* R9 flag ``F_t`` if ``dmax == dbar`` (kept; primary analyses drop, SA3 keeps).
* R10 flag ``F_qual`` if ``q_meas_qual_cd == "POOR"``, ``q_meas_used_fg == "N"`` or
  ``data_aging_cd`` is not approved (``P`` provisional or ``W`` working); kept.
* R11 keep the first row (lowest ``row_id``) of each duplicated (``site_no``, ``q_meas_no``).
* ``t_obs = ln(dmax / dbar)``; flag ``F_tout`` if outside ``domain.t_band`` (kept).

Owning phase: 5.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from src.config import Settings, load_settings
from src.data.io import read_parquet, sha256_file, unit_manifest_sha256, write_json, write_parquet
from src.data.schemas import HydroswotInit

if TYPE_CHECKING:
    from geopandas import GeoDataFrame

__all__ = [
    "APPROVED_AGING",
    "QcLog",
    "apply_qc",
    "assign_division",
    "load_divisions",
    "main",
]

LOG = logging.getLogger(__name__)

#: ``data_aging_cd`` value of approved data; any other value is provisional or working (R10).
APPROVED_AGING: Final[str] = "A"
_R4_FIELDS: Final[tuple[str, ...]] = (
    "DA_site_km2",
    "Q_obs",
    "Q_meas",
    "W",
    "A",
    "dbar",
    "dmax",
    "V",
    "Vmax",
)


@dataclass
class QcLog:
    """Per-rule counts of rows and sites (protocol 2.4: every rule logs its effect)."""

    rules: dict[str, dict[str, Any]] = field(default_factory=dict)

    def drop(
        self, rule: str, frame: pd.DataFrame, keep: pd.Series, **extra: object
    ) -> pd.DataFrame:
        """Apply a dropping rule and record rows and sites before and after."""
        out = frame.loc[keep]
        sites_in = frame["site_no"].nunique()
        sites_out = out["site_no"].nunique()
        self.rules[rule] = {
            "action": "drop",
            "rows_in": len(frame),
            "rows_removed": len(frame) - len(out),
            "sites_in": int(sites_in),
            "sites_removed": int(sites_in - sites_out),
            "rows_out": len(out),
            "sites_out": int(sites_out),
            **extra,
        }
        return out

    def flag(self, rule: str, frame: pd.DataFrame, flag: pd.Series, **extra: object) -> None:
        """Record a flagging rule (rows kept)."""
        self.rules[rule] = {
            "action": "flag",
            "rows_in": len(frame),
            "rows_flagged": int(flag.sum()),
            "sites_with_flag": int(frame.loc[flag, "site_no"].nunique()),
            **extra,
        }


def load_divisions(physio_dir: Path) -> GeoDataFrame:
    """Load the F&J polygons (``DIVISION`` stripped; empty -> missing; geometry repaired)."""
    import geopandas as gpd
    import shapely

    shp = next(iter(sorted(physio_dir.rglob("physio.shp"))))
    gdf = gpd.read_file(shp)[["DIVISION", "geometry"]]
    gdf["DIVISION"] = gdf["DIVISION"].astype("string").str.strip().replace("", pd.NA)
    gdf["geometry"] = shapely.make_valid(gdf.geometry.to_numpy())
    return gdf


def assign_division(lat: pd.Series, lon: pd.Series, polygons: GeoDataFrame) -> pd.Series:
    """Return the single F&J division of each point, or missing (R2).

    Work is done on unique coordinates. A point is assigned only if every polygon it
    intersects has the same, non-missing division.
    """
    import geopandas as gpd

    coords = pd.DataFrame({"lat": lat, "lon": lon}).dropna().drop_duplicates()
    points = gpd.GeoDataFrame(
        coords, geometry=gpd.points_from_xy(coords["lon"], coords["lat"]), crs="EPSG:4269"
    )
    polys = polygons.to_crs("EPSG:4269")
    hits = gpd.sjoin(points, polys, how="left", predicate="intersects")
    grouped = hits.groupby(level=0)["DIVISION"].agg(
        lambda s: s.iloc[0] if s.notna().all() and s.nunique() == 1 else pd.NA
    )
    coords["division"] = grouped.reindex(coords.index)
    merged = pd.DataFrame({"lat": lat, "lon": lon}).merge(coords, on=["lat", "lon"], how="left")
    return pd.Series(merged["division"].to_numpy(), index=lat.index, dtype="string")


def _ratio_outside(a: pd.Series, ref: pd.Series, tol: float) -> pd.Series:
    return ((a / ref) - 1.0).abs() > tol


def apply_qc(
    frame: pd.DataFrame, division: pd.Series, settings: Settings
) -> tuple[pd.DataFrame, QcLog]:
    """Apply R1-R11 in protocol order and add the QC columns of :class:`HydroswotInit`.

    Parameters
    ----------
    frame
        ``unit_mapped/hydroswot.parquet`` (:class:`HydroswotSI`).
    division
        F&J division per row of ``frame`` (:func:`assign_division`), aligned on the index.
    settings
        Validated settings (``qc`` and ``domain`` groups).
    """
    qc, domain = settings.protocol.qc, settings.protocol.domain
    log = QcLog()
    df = frame.assign(division=division)
    df = log.drop("R1", df, df["site_tp_cd"].eq("ST").fillna(False))
    df = log.drop("R2", df, df["division"].notna())
    multi = df["q_meas_chan_nu"].ge(2).fillna(False)
    multi_meas = multi.groupby([df["site_no"], df["q_meas_no"]]).transform("any")
    df = log.drop(
        "R3",
        df,
        ~multi_meas,
        rows_chan_ge_2=int(multi.sum()),
        rows_chan_0_kept=int((df["q_meas_chan_nu"].eq(0).fillna(False) & ~multi_meas).sum()),
    )
    positive = pd.concat([df[c].gt(0) for c in _R4_FIELDS], axis=1).fillna(False).all(axis=1)
    df = log.drop("R4", df, positive)
    within = (
        df["W"].le(qc.w_max_m)
        & df["dbar"].le(qc.dbar_max_m)
        & df["dmax"].le(qc.dmax_max_m)
        & df["V"].le(qc.v_max_ms)
    )
    df = log.drop("R5", df, within)
    r6_fail = df["dbar"].gt(df["dmax"]) | df["V"].gt(df["Vmax"])
    q5_fail = df["q5_Mean_GT_Max"].eq(-1).fillna(False)
    df = log.drop("R6", df, ~r6_fail, agree_with_q5=int((r6_fail == q5_fail).sum()))
    av = df["A"] * df["V"]
    pairs = {
        "meas_vs_q": _ratio_outside(df["Q_meas"], df["Q_obs"], qc.q_tol),
        "av_vs_meas": _ratio_outside(av, df["Q_meas"], qc.q_tol),
        "av_vs_q": _ratio_outside(av, df["Q_obs"], qc.q_tol),
    }
    r7_fail = pairs["meas_vs_q"] | pairs["av_vs_meas"] | pairs["av_vs_q"]
    builtin = (
        df["q1_Outside5Percent"].eq(-1)
        | df["q2_q_va_OutsideBounds"].eq(-1)
        | df["q2_meas_q_va_OutsideBounds"].eq(-1)
    ).fillna(False)
    df = log.drop(
        "R7",
        df,
        ~r7_fail,
        per_pair={k: int(v.sum()) for k, v in pairs.items()},
        agree_with_q1_q2=int((r7_fail == builtin).sum()),
    )
    df = df.copy()
    df["eps_A"] = np.log(df["A"] / (df["W"] * df["dbar"]))
    df["t_obs"] = np.log(df["dmax"] / df["dbar"])
    df["F_eps"] = df["eps_A"].abs() > qc.eps_a_log_tol
    log.flag("R8", df, df["F_eps"], tolerance=qc.eps_a_log_tol)
    df["F_t"] = df["dmax"].eq(df["dbar"])
    log.flag("R9", df, df["F_t"])
    df["F_qual"] = (
        df["q_meas_qual_cd"].eq("POOR").fillna(False)
        | df["q_meas_used_fg"].eq("N").fillna(False)
        | df["data_aging_cd"].ne(APPROVED_AGING).fillna(True)
    )
    log.flag("R10", df, df["F_qual"])
    df = df.sort_values("row_id", kind="mergesort")
    df = log.drop("R11", df, ~df.duplicated(["site_no", "q_meas_no"], keep="first"))
    t_lo, t_hi = domain.t_band
    df = df.copy()
    df["F_tout"] = (df["t_obs"] < t_lo) | (df["t_obs"] > t_hi)
    log.flag("F_tout", df, df["F_tout"], band=[t_lo, t_hi])
    df["division"] = df["division"].astype(str)
    out = HydroswotInit.validate(df.reset_index(drop=True))
    return out, log


def run(settings: Settings) -> None:
    """Read the unit-mapped table and the polygons, apply QC, write the table and the counts."""
    rt = settings.runtime
    src = rt.processed_dir / "unit_mapped" / "hydroswot.parquet"
    physio = rt.raw_dir / "physiography"
    frame = read_parquet(src)
    division = assign_division(frame["lat"], frame["lon"], load_divisions(physio))
    out, log = apply_qc(frame, division, settings)
    LOG.info("hydroswot_init: %d rows at %d sites", len(out), out["site_no"].nunique())
    write_parquet(
        out,
        rt.processed_dir / "qc" / "hydroswot_init.parquet",
        stage="qc",
        params_sha256=settings.params_sha256,
        inputs={
            "unit_mapped/hydroswot.parquet": sha256_file(src),
            "physiography/MANIFEST.sha256": unit_manifest_sha256(physio),
        },
        sort_by=["row_id"],
    )
    write_json(
        rt.reports_dir / "qc_counts.json",
        {
            "params_sha256": settings.params_sha256,
            "rules": log.rules,
            "rows_out": len(out),
            "sites_out": int(out["site_no"].nunique()),
            "divisions": {k: int(v) for k, v in sorted(out["division"].value_counts().items())},
            "eps_a_log_tol": settings.protocol.qc.eps_a_log_tol,
            "ln_1_plus_q_tol": math.log1p(settings.protocol.qc.q_tol),
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run ``python -m src.data.qc`` (the dvc.yaml ``qc`` stage)."""
    argparse.ArgumentParser(description="QC rules R1-R11 (protocol 2.4).").parse_args(argv)
    settings = load_settings()
    logging.basicConfig(
        level=settings.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    LOG.info("params_sha256=%s", settings.params_sha256)
    run(settings)


if __name__ == "__main__":
    main()

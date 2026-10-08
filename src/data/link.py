"""Stage link: gage-to-reach link (protocol section 2.5).

Sources: protocol section 2.5 (GageLoc primary; 500 m snap fallback in EPSG:5070 with the
drainage-area check; F_snap, F_DA; k-NN diagnostic, ADL-011), section 2.1 (``drain_area_va``
in km2 via 2.589988110336, already applied by unit_map).

Algorithm (one row per ``hydroswot_init`` site):

1. **Primary.** ``GageLoc.FLComID`` where ``GageLoc.Source_Fea == site_no`` (exact string
   match). A GageLoc COMID absent from the network (not in the VAA or a non-network flowline)
   counts as "no GageLoc record".
2. **Fallback.** Among network flowlines within ``link.snap_max_m`` of the site (EPSG:5070),
   the nearest one whose drainage area passes check 3. Flag ``F_snap``. Ties in distance go to
   the smaller COMID. Sites without such a flowline are left unlinked (counted, not output).
3. **Drainage-area check.** ``|ln(TotDASqKm / DA_site_km2)| <= ln(link.da_ratio_max)``. A
   GageLoc link that fails is kept with ``F_DA = True`` (excluded from the primary set later,
   included in SA6).
4. **Diagnostic.** The nearest network flowline (no radius, no DA check) of every site; the
   GageLoc vs k-NN disagreement rate is written to ``reports/link_counts.json``.

The site coordinate and ``DA_site_km2`` are those of the site's first ``hydroswot_init`` row
(lowest ``row_id``); sites whose rows disagree are counted in the report.

Owning phase: 5.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from src.config import Settings, load_settings
from src.data.io import read_parquet, sha256_file, unit_manifest_sha256, write_json, write_parquet
from src.data.nhdplus import CRS_ALBERS, VPUS, read_layer

if TYPE_CHECKING:
    from geopandas import GeoDataFrame

__all__ = ["LINK_METHODS", "da_log_ratio", "link_sites", "main", "site_points"]

LOG = logging.getLogger(__name__)

LINK_METHODS: Final[tuple[str, ...]] = ("gageloc", "snap")


def site_points(init: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per site: first-row coordinates and drainage area, projected to EPSG:5070."""
    from pyproj import Transformer

    ordered = init.sort_values("row_id", kind="mergesort")
    first = ordered.groupby("site_no", sort=True).first()
    n_coord = ordered.groupby("site_no")[["lat", "lon"]].nunique().max(axis=1)
    n_da = ordered.groupby("site_no")["DA_site_km2"].nunique()
    tr = Transformer.from_crs("EPSG:4269", CRS_ALBERS, always_xy=True)
    x, y = tr.transform(first["lon"].to_numpy(), first["lat"].to_numpy())
    sites = pd.DataFrame(
        {
            "site_no": first.index.astype(str),
            "lat": first["lat"].to_numpy(),
            "lon": first["lon"].to_numpy(),
            "x5070": x,
            "y5070": y,
            "DA_site_km2": first["DA_site_km2"].to_numpy(),
        }
    )
    counts = {
        "sites": len(sites),
        "sites_with_varying_coordinates": int((n_coord > 1).sum()),
        "sites_with_varying_drain_area": int((n_da > 1).sum()),
    }
    return sites, counts


def da_log_ratio(da_nhd: pd.Series | np.ndarray, da_site: pd.Series | np.ndarray) -> np.ndarray:
    """Return ``ln(TotDASqKm / DA_site_km2)``; non-positive areas give ``+inf`` (check fails)."""
    nhd = np.asarray(da_nhd, dtype="float64")
    site = np.asarray(da_site, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log(nhd / site)
    return np.where((nhd > 0) & (site > 0), ratio, np.inf)


def _candidates(
    sites: pd.DataFrame, flowlines: GeoDataFrame, radius: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pairs within ``radius``, nearest flowline per site) for one VPU."""
    import shapely

    pts = shapely.points(sites["x5070"].to_numpy(), sites["y5070"].to_numpy())
    geoms = flowlines.geometry.to_numpy()
    tree = shapely.STRtree(geoms)
    i_pt, i_fl = tree.query(pts, predicate="dwithin", distance=radius)
    pairs = pd.DataFrame(
        {
            "site_no": sites["site_no"].to_numpy()[i_pt],
            "comid": flowlines["comid"].to_numpy()[i_fl],
            "dist_m": shapely.distance(pts[i_pt], geoms[i_fl]),
        }
    )
    j_pt, j_fl = tree.query_nearest(pts, all_matches=True)
    near = pd.DataFrame(
        {
            "site_no": sites["site_no"].to_numpy()[j_pt],
            "comid": flowlines["comid"].to_numpy()[j_fl],
            "dist_m": shapely.distance(pts[j_pt], geoms[j_fl]),
        }
    )
    return pairs, near


def link_sites(
    sites: pd.DataFrame,
    gageloc: pd.DataFrame,
    vaa: pd.DataFrame,
    flowlines: GeoDataFrame,
    *,
    snap_max_m: float,
    da_ratio_max: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Link sites to COMIDs (steps 1-4 of the module docstring).

    Parameters
    ----------
    sites
        Output of :func:`site_points`.
    gageloc
        ``unit_mapped/nhd_gageloc.parquet``.
    vaa
        ``unit_mapped/nhd_vaa.parquet`` (``comid``, ``DA_km2``, ``non_network``).
    flowlines
        ``GeoDataFrame[comid, geometry]`` in EPSG:5070 (all VPUs, or the relevant subset).
    snap_max_m, da_ratio_max
        ``params.link`` values.
    """
    network = vaa.loc[~vaa["non_network"], ["comid", "DA_km2"]]
    da_by_comid = network.set_index("comid")["DA_km2"]
    tol = math.log(da_ratio_max)
    fl = flowlines.loc[flowlines["comid"].isin(da_by_comid.index)].reset_index(drop=True)
    pairs, near = _candidates(sites, fl, snap_max_m)

    # Step 4 diagnostic: nearest network flowline; distance ties -> smaller COMID.
    knn = (
        near.sort_values(["site_no", "dist_m", "comid"], kind="mergesort")
        .drop_duplicates("site_no")
        .set_index("site_no")
    )
    site_da = sites.set_index("site_no")["DA_site_km2"]

    # Step 1: GageLoc, restricted to network COMIDs.
    gl = gageloc.loc[gageloc["site_no"].isin(sites["site_no"])].set_index("site_no")["comid"]
    gl_in_net = gl[gl.isin(da_by_comid.index)]
    primary = pd.DataFrame({"site_no": gl_in_net.index, "comid": gl_in_net.to_numpy()})
    primary["link_method"] = "gageloc"
    primary["snap_dist_m"] = np.nan

    # Step 2: snap fallback for the other sites, DA check used to choose the candidate.
    rest = pairs.loc[~pairs["site_no"].isin(primary["site_no"])].copy()
    rest["da_log_ratio"] = da_log_ratio(
        da_by_comid.reindex(rest["comid"]).to_numpy(), site_da.reindex(rest["site_no"]).to_numpy()
    )
    rest = rest.loc[np.abs(rest["da_log_ratio"]) <= tol]
    snapped = (
        rest.sort_values(["site_no", "dist_m", "comid"], kind="mergesort")
        .drop_duplicates("site_no")
        .rename(columns={"dist_m": "snap_dist_m"})[["site_no", "comid", "snap_dist_m"]]
    )
    snapped["link_method"] = "snap"
    linked = pd.concat([primary, snapped], ignore_index=True)

    # Step 3: drainage-area check on every link.
    ratio = da_log_ratio(
        da_by_comid.reindex(linked["comid"]).to_numpy(),
        site_da.reindex(linked["site_no"]).to_numpy(),
    )
    linked["da_log_ratio"] = ratio
    linked["F_snap"] = linked["link_method"].eq("snap")
    linked["F_DA"] = np.abs(ratio) > tol
    linked["knn_comid"] = knn["comid"].reindex(linked["site_no"]).to_numpy()
    linked["knn_dist_m"] = knn["dist_m"].reindex(linked["site_no"]).to_numpy()
    out = sites.merge(linked, on="site_no", how="inner")
    out = out.sort_values("site_no", kind="mergesort").reset_index(drop=True)

    gl_sites = out["link_method"].eq("gageloc")
    disagree = out.loc[gl_sites, "knn_comid"] != out.loc[gl_sites, "comid"]
    counts: dict[str, Any] = {
        "sites_in": len(sites),
        "gageloc_record": len(gl),
        "gageloc_comid_not_in_network": int(len(gl) - len(gl_in_net)),
        "linked_gageloc": int(gl_sites.sum()),
        "linked_snap": int((~gl_sites).sum()),
        "unlinked": int(len(sites) - len(out)),
        "F_DA": int(out["F_DA"].sum()),
        "F_DA_gageloc": int((out["F_DA"] & gl_sites).sum()),
        "knn_disagree_gageloc": int(disagree.sum()),
        "knn_disagree_rate_gageloc": float(disagree.mean()) if gl_sites.any() else None,
        "snap_dist_m_quantiles": {
            str(q): float(v) for q, v in out["snap_dist_m"].quantile([0.5, 0.9, 1.0]).items()
        }
        if (~gl_sites).any()
        else None,
        "da_ratio_tolerance_log": tol,
    }
    return out, counts


def run(settings: Settings) -> None:
    """Link every ``hydroswot_init`` site and write ``site_comid.parquet`` and the counts."""
    import geopandas as gpd

    rt = settings.runtime
    init_path = rt.processed_dir / "qc" / "hydroswot_init.parquet"
    um = rt.processed_dir / "unit_mapped"
    nhd_dir = rt.raw_dir / "nhdplusv21"
    sites, site_counts = site_points(read_parquet(init_path))
    vaa = read_parquet(um / "nhd_vaa.parquet")
    flowlines = gpd.GeoDataFrame(
        pd.concat([read_layer(nhd_dir, "flowline", vpu) for vpu in VPUS], ignore_index=True),
        crs=CRS_ALBERS,
    )
    out, counts = link_sites(
        sites,
        read_parquet(um / "nhd_gageloc.parquet"),
        vaa,
        flowlines,
        snap_max_m=settings.protocol.link.snap_max_m,
        da_ratio_max=settings.protocol.link.da_ratio_max,
    )
    LOG.info("linked %d of %d sites", len(out), len(sites))
    write_parquet(
        out,
        rt.processed_dir / "link" / "site_comid.parquet",
        stage="link",
        params_sha256=settings.params_sha256,
        inputs={
            "qc/hydroswot_init.parquet": sha256_file(init_path),
            "unit_mapped/nhd_gageloc.parquet": sha256_file(um / "nhd_gageloc.parquet"),
            "unit_mapped/nhd_vaa.parquet": sha256_file(um / "nhd_vaa.parquet"),
            "nhdplusv21/MANIFEST.sha256": unit_manifest_sha256(nhd_dir),
        },
        sort_by=["site_no"],
    )
    write_json(
        rt.reports_dir / "link_counts.json",
        {"params_sha256": settings.params_sha256, **site_counts, **counts},
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run ``python -m src.data.link`` (the dvc.yaml ``link`` stage)."""
    argparse.ArgumentParser(description="Gage-reach link (protocol 2.5).").parse_args(argv)
    settings = load_settings()
    logging.basicConfig(
        level=settings.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    LOG.info("params_sha256=%s", settings.params_sha256)
    run(settings)


if __name__ == "__main__":
    main()

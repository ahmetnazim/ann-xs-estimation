"""Stage ind_prepare: eHydro independent set (S-IND), deduplicated against training reaches.

Sources: protocol sections 1.1 (ehydro, O-5 (c)), 2.3 (``ehydro_ind``), 3.2 (S-IND), 3.5 (L8),
11.3 (rules E-R1 to E-R12, pre-registered at ``07e00d6``) and 11.4 (survey reader E-R5a to
E-R5e, ``708f832``); Zarrabi et al. (2025) section 2.5 for the reach formulas. Reads params
``ehydro`` and ``splits.ind_dedupe_m``.

Order of work (:func:`run`):

1. The selected surveys are read from ``reports/ehydro_selection.json`` (E-R3), whose URL order
   must equal ``acquire.ehydro.urls``.
2. Every survey is classed from its metadata only (E-R5c). If more than half are excluded, the
   report is written and the stage STOPs before any depth value is read (E-R5 (i)).
3. The ``water`` surveys are read (E-R5a/b/d) and turned into wetted cells (E-R5 (ii)-(iv)),
   pooled oldest to newest by ``surveydatestart`` then ``surveyjobidpk`` (E-R5e, E-R6).
4. :func:`derive_locations`: reach values (E-R6), coverage (E-R7), locations (E-R8) and L8
   (E-R9, :func:`l8_status`).
5. ``data/processed/ind/ehydro_ind.parquet`` (Pandera ``EhydroInd``) and
   ``reports/ind_prepare.json`` (E-R12; counts and DA only, no W or d̄ summary and no model
   prediction) are written; the E-R10 floor verdict is recorded.

L8 rule (docs/3 11.3 E-R9): an eHydro location is dropped if any of its member COMIDs is a
training COMID, or if any member flowline lies within ``splits.ind_dedupe_m`` (EPSG:5070) of a
training site. S-IND trains on **all** post-QC sites, so every site in U is a training site.

Torch-free. Owning phase: 5 (L8); 6 (P6-D10).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from geopandas import GeoDataFrame
    from shapely import Geometry

    from src.config import EhydroParams, Settings
    from src.data.ehydro import Cells, PlaneDecision

__all__ = [
    "EXPECTED_PLANE_COUNTS",
    "SurveyMeta",
    "dedupe_l8",
    "derive_locations",
    "l8_status",
    "main",
    "plane_summary",
    "run",
    "selected_surveys",
]

LOG = logging.getLogger(__name__)

STAGE: Final[str] = "ind_prepare"
#: docs/3 11.4: counts expected from the metadata inspection (reported against, never tuned).
EXPECTED_PLANE_COUNTS: Final[Mapping[str, int]] = {"water": 430, "geodetic": 30, "unresolved": 23}
#: E-R5 (i): STOP if more than this share of the selected surveys is excluded.
MAX_EXCLUDED_SHARE: Final[float] = 0.5
_SELECTION_REPORT: Final[str] = "ehydro_selection.json"
_REPORT: Final[str] = "ind_prepare.json"
_OUTPUT: Final[str] = "ind/ehydro_ind.parquet"
_DA_QUANTILES: Final[tuple[float, ...]] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


# ----------------------------------------------------------------------------- L8 (E-R9)
def l8_status(locations: pd.DataFrame, train_sites: pd.DataFrame, radius_m: float) -> pd.Series:
    """L8 outcome per location (docs/3 3.5 L8, 11.3 E-R9).

    Parameters
    ----------
    locations
        One row per (eHydro location, member COMID): ``loc_id``, ``comid``, ``geometry`` (the
        member flowline as a shapely geometry, EPSG:5070).
    train_sites
        ``site_no``, ``comid``, ``x5070``, ``y5070`` of every training site.
    radius_m
        ``splits.ind_dedupe_m``.

    Returns
    -------
    pandas.Series
        Indexed by ``loc_id`` (sorted): ``"dropped_comid"`` if a member COMID is a training
        COMID, else ``"dropped_distance"`` if a member flowline lies within ``radius_m`` of a
        training site, else ``"kept"``.
    """
    import shapely

    train_comids = set(train_sites["comid"].astype("int64"))
    by_comid = locations.groupby("loc_id")["comid"].agg(
        lambda s: bool(set(s.astype("int64")) & train_comids)
    )
    tree = shapely.STRtree(shapely.points(train_sites["x5070"], train_sites["y5070"]))
    hit, _ = tree.query(
        np.asarray(locations["geometry"].to_numpy(), dtype=object),
        predicate="dwithin",
        distance=radius_m,
    )
    near_ids = set(locations["loc_id"].to_numpy()[np.unique(hit)])
    near = pd.Series(by_comid.index.isin(list(near_ids)), index=by_comid.index)
    status = pd.Series("kept", index=by_comid.index, name="dedupe_status", dtype=object)
    status[near.to_numpy()] = "dropped_distance"
    status[by_comid.to_numpy()] = "dropped_comid"
    return status


def dedupe_l8(
    locations: pd.DataFrame, train_sites: pd.DataFrame, radius_m: float
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the L8 overlap rule (docs/3 3.5 L8, 11.3 E-R9); see :func:`l8_status`.

    Returns
    -------
    tuple
        Kept rows of ``locations`` and the counts (dropped by COMID, by distance, kept).
    """
    status = l8_status(locations, train_sites, radius_m)
    kept_ids = status.index[status.eq("kept").to_numpy()]
    kept = locations.loc[locations["loc_id"].isin(kept_ids)].reset_index(drop=True)
    counts = {
        "locations": len(status),
        "dropped_comid": int(status.eq("dropped_comid").sum()),
        "dropped_distance_only": int(status.eq("dropped_distance").sum()),
        "kept": int(status.eq("kept").sum()),
    }
    return kept, counts


# ----------------------------------------------------------------------------- surveys (E-R3, E-R5)
@dataclass(frozen=True)
class SurveyMeta:
    """One selected survey (E-R3), as listed in ``reports/ehydro_selection.json``."""

    name: str
    surveyjobidpk: str
    surveydatestart: pd.Timestamp
    district: str
    rivers: tuple[str, ...]
    vpus: tuple[str, ...]


def _survey_name(url: str) -> str:
    """Folder name of an extracted survey: the zip file name without its extension."""
    return Path(unquote(urlparse(url).path)).stem


def selected_surveys(selection: Mapping[str, object], urls: Sequence[str]) -> list[SurveyMeta]:
    """Selected surveys in report order; the report URLs must equal ``acquire.ehydro.urls``.

    Raises
    ------
    EhydroError
        If the URL lists differ (order included) or two surveys share a folder name.
    """
    from src.data.ehydro import EhydroError

    records = selection["selected"]
    if not isinstance(records, list):
        raise EhydroError("selection report: 'selected' is not a list")
    report_urls = [str(r["url"]) for r in records]
    if report_urls != [str(u) for u in urls]:
        raise EhydroError("selection report URLs differ from params acquire.ehydro.urls")
    out = [
        SurveyMeta(
            name=_survey_name(str(r["url"])),
            surveyjobidpk=str(r["surveyjobidpk"]),
            surveydatestart=pd.Timestamp(str(r["surveydatestart"])),
            district=str(r["district"]),
            rivers=tuple(str(r["rivers"]).split(";")),
            vpus=tuple(str(r["vpus"]).split(";")),
        )
        for r in records
    ]
    if len({s.name for s in out}) != len(out):
        raise EhydroError("selection report: two surveys share a folder name")
    return out


def plane_summary(
    surveys: Sequence[SurveyMeta], decisions: Mapping[str, PlaneDecision], rivers: Sequence[str]
) -> dict[str, object]:
    """E-R5c / E-R12 counts: classes, sources, districts, per-river surveys kept, exclusions."""
    from src.data.ehydro import PLANE_CLASSES

    classes = {c: 0 for c in PLANE_CLASSES}
    by_source: dict[str, int] = {}
    by_district: dict[str, dict[str, int]] = {}
    per_river = {r: {"selected": 0, "water": 0} for r in rivers}
    excluded: list[dict[str, object]] = []
    for s in surveys:
        d = decisions[s.name]
        classes[d.cls] += 1
        key = f"{d.source}:{d.cls}"
        by_source[key] = by_source.get(key, 0) + 1
        if d.cls != "water":
            dist = by_district.setdefault(s.district, {})
            dist[d.cls] = dist.get(d.cls, 0) + 1
            excluded.append(
                {
                    "survey": s.name,
                    "district": s.district,
                    "class": d.cls,
                    "source": d.source,
                    "statement": d.statement,
                    "edomv": list(d.edomv),
                    "field_values": list(d.field_values),
                }
            )
        for r in s.rivers:
            per_river[r]["selected"] += 1
            per_river[r]["water"] += int(d.cls == "water")
    n = len(surveys)
    n_excluded = n - classes["water"]
    return {
        "classes": classes,
        "expected": dict(EXPECTED_PLANE_COUNTS),
        "matches_expected": classes == dict(EXPECTED_PLANE_COUNTS),
        "by_source": dict(sorted(by_source.items())),
        "excluded_by_district": dict(sorted(by_district.items())),
        "excluded_share": n_excluded / n if n else 0.0,
        "stop": n == 0 or n_excluded / n > MAX_EXCLUDED_SHARE,
        "surveys_per_river": per_river,
        "rivers_with_water_survey": sum(1 for v in per_river.values() if v["water"] > 0),
        "excluded": excluded,
    }


# ----------------------------------------------------------------------------- E-R6 to E-R9
def _da_distribution(values: pd.Series) -> dict[str, object]:
    present = values.dropna().to_numpy(dtype=np.float64)
    out: dict[str, object] = {"n": int(values.size), "n_missing": int(values.isna().sum())}
    if present.size:
        qs = np.quantile(present, list(_DA_QUANTILES))
        out["quantiles_km2"] = {
            f"q{int(q * 100):03d}": float(v) for q, v in zip(_DA_QUANTILES, qs, strict=True)
        }
    return out


def derive_locations(
    cells: Cells,
    footprints: Sequence[Geometry],
    flowlines: GeoDataFrame,
    catchments: GeoDataFrame,
    vaa: pd.DataFrame,
    edges: pd.DataFrame,
    train_sites: pd.DataFrame,
    params: EhydroParams,
    radius_m: float,
    coverage: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """E-R6 to E-R9 on pooled cells; returns the ``EhydroInd`` table and the counts.

    Parameters
    ----------
    cells
        Pooled wetted cells (E-R6, :func:`src.data.ehydro.pool_cells`).
    footprints
        E-R5d footprints (EPSG:5070) of the surveys whose depths were used.
    flowlines
        Frame flowlines ``comid, river, vpu, lengthkm, geometry`` (E-R1, EPSG:5070).
    catchments
        ``comid, geometry`` (EPSG:5070); only frame COMIDs are used (E-R1).
    vaa
        ``comid, hydroseq, DA_km2`` (PlusFlowlineVAA ``HydroSeq``, ``TotDASqKm``).
    edges
        PlusFlow ``from_comid, to_comid``.
    train_sites
        ``site_no, comid, x5070, y5070`` of every site in U.
    params
        ``ehydro``.
    radius_m
        ``splits.ind_dedupe_m``.
    coverage
        E-R7 coverage per frame COMID, if already computed with the same footprints.
    """
    from src.data.ehydro import (
        EhydroError,
        build_locations,
        coverage_fraction,
        reach_values,
    )
    from src.data.schemas import EhydroInd

    if flowlines["comid"].duplicated().any():
        raise EhydroError("E-R1: a frame COMID appears twice")
    frame_ids = set(flowlines["comid"].astype("int64"))
    catch = catchments.loc[catchments["comid"].isin(list(frame_ids))].reset_index(drop=True)
    lengthkm = dict(zip(flowlines["comid"].astype("int64"), flowlines["lengthkm"], strict=True))
    reach = reach_values(cells, catch, lengthkm, params.cell_m)  # E-R6
    cov = (
        coverage_fraction(flowlines, list(footprints), params.buffer_m)
        if coverage is None
        else coverage
    )  # E-R7
    covered = set(cov.index[cov.to_numpy() >= params.coverage_min].astype("int64"))
    with_cells = set(reach["comid"].astype("int64"))
    retained = reach.loc[reach["comid"].isin(list(covered))].copy()
    counts: dict[str, object] = {
        "frame_comids": len(frame_ids),
        "frame_catchments": len(catch),
        "cells_pooled": len(cells),
        "comids_with_cells": len(with_cells),
        "comids_coverage_pass": len(covered),
        "comids_coverage_pass_without_catchment": len(
            covered - set(catch["comid"].astype("int64"))
        ),
        "comids_coverage_pass_without_cells": len(covered - with_cells),
        "comids_dropped_by_coverage": len(with_cells - covered),
        "comids_retained": len(retained),
    }
    v = vaa.set_index("comid")
    missing = sorted(set(retained["comid"].astype("int64")) - set(v.index))
    if missing:
        raise EhydroError(f"E-R8: retained COMIDs without VAA HydroSeq: {missing[:5]}")
    attrs = flowlines.set_index("comid")
    retained["river"] = attrs.loc[retained["comid"], "river"].to_numpy()
    retained["lengthkm"] = attrs.loc[retained["comid"], "lengthkm"].to_numpy(dtype=np.float64)
    retained["hydroseq"] = v.loc[retained["comid"], "hydroseq"].to_numpy()
    cols = ["comid", "river", "lengthkm", "hydroseq", "W", "dbar"]
    locations, members = build_locations(retained[cols], edges, params.loc_length_km)  # E-R8
    counts["locations"] = len(locations)
    if locations.empty:
        table = pd.DataFrame({c: pd.Series(dtype=object) for c in EhydroInd.to_schema().columns})
        counts["l8"] = {"locations": 0, "dropped_comid": 0, "dropped_distance": 0, "kept": 0}
        return table, counts
    members = members.assign(
        geometry=attrs.loc[members["comid"], "geometry"].to_numpy(),
        vpu=attrs.loc[members["comid"], "vpu"].to_numpy(),
        hydroseq=v.loc[members["comid"], "hydroseq"].to_numpy(),
        da_km2=v.loc[members["comid"], "DA_km2"].to_numpy(dtype=np.float64),
    )
    status = l8_status(members[["loc_id", "comid", "geometry"]], train_sites, radius_m)  # E-R9
    rows = []
    for loc_id, part in members.groupby("loc_id", sort=False):
        i_out = int(np.argmin(part["hydroseq"].to_numpy()))
        outlet = int(part["comid"].to_numpy()[i_out])
        rows.append(
            {
                "loc_id": str(loc_id),
                "vpus": ";".join(sorted(set(part["vpu"].astype(str)))),
                "comids": ";".join(str(int(c)) for c in part["comid"]),
                "comid_outlet": outlet,
                "DA_km2": float(part["da_km2"].to_numpy()[i_out]),
            }
        )
    table = locations.merge(pd.DataFrame(rows), on="loc_id", how="left", validate="one_to_one")
    table["dedupe_status"] = status.reindex(table["loc_id"]).to_numpy()
    table["comid_outlet"] = table["comid_outlet"].astype(np.int64)
    table["n_comids"] = table["n_comids"].astype(np.int64)
    table = table[list(EhydroInd.to_schema().columns)]
    table = EhydroInd.validate(table)
    counts["l8"] = {
        "locations": len(table),
        "dropped_comid": int(table["dedupe_status"].eq("dropped_comid").sum()),
        "dropped_distance": int(table["dedupe_status"].eq("dropped_distance").sum()),
        "kept": int(table["dedupe_status"].eq("kept").sum()),
    }
    return table, counts


def _river_summary(table: pd.DataFrame, rivers: Sequence[str]) -> dict[str, object]:
    """Per-river locations and the E-R1 homonym check (VPUs of the river's locations)."""
    out: dict[str, object] = {}
    for river in rivers:
        part = table.loc[table["river"].eq(river)]
        kept = part.loc[part["dedupe_status"].eq("kept")]
        out[river] = {
            "locations": len(part),
            "kept": len(kept),
            "vpus_all": sorted({x for s in part["vpus"] for x in str(s).split(";")}),
            "vpus_kept": sorted({x for s in kept["vpus"] for x in str(s).split(";")}),
        }
    return out


# ----------------------------------------------------------------------------- stage
def _read_inputs_nhd(
    nhd_dir: Path, vpus: Sequence[str], frame_comids: set[int]
) -> tuple[GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """Frame catchments of ``vpus`` and the national VAA and PlusFlow tables (raw NHDPlusV2.1)."""
    import geopandas as gpd

    from src.data.nhdplus import CRS_ALBERS, read_layer, read_nhd_table

    parts = []
    for vpu in vpus:
        layer = read_layer(nhd_dir, "catchment", vpu)
        parts.append(layer.loc[layer["comid"].isin(list(frame_comids))])
    catchments = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=CRS_ALBERS)
    vaa_raw, _ = read_nhd_table(nhd_dir, "vaa")
    vaa = vaa_raw.rename(columns={"ComID": "comid", "HydroSeq": "hydroseq", "TotDASqKm": "DA_km2"})
    plusflow, _ = read_nhd_table(nhd_dir, "plusflow")
    edges = plusflow.rename(columns={"FromComID": "from_comid", "ToComID": "to_comid"})
    return catchments, vaa[["comid", "hydroseq", "DA_km2"]], edges[["from_comid", "to_comid"]]


def _train_sites(site_table: pd.DataFrame) -> pd.DataFrame:
    """One row per site of U: ``site_no, comid, x5070, y5070`` (equal across variants)."""
    from src.data.ehydro import EhydroError

    sites = site_table[["site_no", "comid", "x5070", "y5070"]].drop_duplicates()
    if sites["site_no"].duplicated().any():
        raise EhydroError("site_table: comid or coordinates differ between variants of a site")
    return sites.sort_values("site_no", kind="mergesort").reset_index(drop=True)


def run(settings: Settings) -> dict[str, object]:
    """Run the stage; writes ``ehydro_ind.parquet`` and ``reports/ind_prepare.json``.

    Raises
    ------
    EhydroStop
        After the report is written, if more than half of the selected surveys are excluded
        by E-R5c (no depth value is read in that case).
    """
    from src.data.ehydro import (
        Cells,
        EhydroStop,
        classify_survey,
        coverage_fraction,
        depth_surface,
        pool_cells,
        read_frame_flowlines,
        read_survey,
    )
    from src.data.io import (
        read_parquet,
        sha256_file,
        unit_manifest_sha256,
        write_json,
        write_parquet,
    )

    rt = settings.runtime
    p = settings.protocol.ehydro
    raw = Path(rt.raw_dir)
    survey_root = raw / "ehydro"
    nhd_dir = raw / "nhdplusv21"
    sel_path = Path(rt.reports_dir) / _SELECTION_REPORT
    report_path = Path(rt.reports_dir) / _REPORT
    site_path = Path(rt.processed_dir) / "site_table.parquet"
    selection = json.loads(sel_path.read_text(encoding="utf-8"))
    urls = [str(u) for u in settings.protocol.acquire.ehydro.urls]
    surveys = selected_surveys(selection, urls)
    decisions = {s.name: classify_survey(survey_root / s.name) for s in surveys}
    planes = plane_summary(surveys, decisions, p.rivers)
    inputs: dict[str, str] = {
        "ehydro": unit_manifest_sha256(survey_root),
        "nhdplusv21": unit_manifest_sha256(nhd_dir),
        "ehydro_selection.json": sha256_file(sel_path),
        "site_table.parquet": sha256_file(site_path),
    }
    report: dict[str, object] = {
        "rules": "docs/3 11.3 E-R1 to E-R12 (07e00d6); 11.4 E-R5a to E-R5e (708f832)",
        "params_sha256": settings.params_sha256,
        "inputs": inputs,
        "selection": {
            "counts": selection["counts"],
            "bytes_total_known": selection["bytes_total_known"],
            "condition_codes": selection["condition_codes"],
        },
        "reference_plane": planes,
    }
    if planes["stop"]:
        write_json(report_path, report)
        share = float(planes["excluded_share"])  # type: ignore[arg-type]
        raise EhydroStop(f"E-R5 (i): {share:.1%} of surveys excluded; see {report_path}")

    water = sorted(
        (s for s in surveys if decisions[s.name].cls == "water"),
        key=lambda s: (s.surveydatestart, s.surveyjobidpk),
    )
    surfaces: list[Cells] = []
    footprints: list[Geometry] = []
    rows: list[dict[str, object]] = []
    for s in water:  # E-R5e: oldest first, so the later survey wins in pool_cells
        pts = read_survey(survey_root / s.name)
        surface = depth_surface(pts.x, pts.y, pts.depth_m, pts.footprint, p.cell_m)
        surfaces.append(surface)
        footprints.append(pts.footprint)
        rows.append(
            {
                "survey": s.name,
                "points_read": pts.n_read,
                "null_depth": pts.n_null_depth,
                "empty_geometry": pts.n_empty_geometry,
                "units": ";".join(pts.units),
                "footprint_repaired": pts.footprint_repaired,
                "wetted_cells": len(surface),
            }
        )
        LOG.info("%s: %d points, %d wetted cells", s.name, pts.n_read, len(surface))
    cells = pool_cells(surfaces)
    per_survey = pd.DataFrame(rows)
    units = per_survey["units"].str.split(";").explode().value_counts().sort_index()
    report["surveys_read"] = {
        "n": len(per_survey),
        "points_read": int(per_survey["points_read"].sum()),
        "null_depth_dropped": int(per_survey["null_depth"].sum()),
        "empty_geometry_dropped": int(per_survey["empty_geometry"].sum()),
        "surveys_by_unit": {str(k): int(v) for k, v in units.items()},
        "footprints_repaired": int(per_survey["footprint_repaired"].sum()),
        "without_wetted_cells": per_survey.loc[per_survey["wetted_cells"].eq(0), "survey"].tolist(),
        "per_survey": per_survey.to_dict("records"),
    }

    flowlines = read_frame_flowlines(nhd_dir, p.rivers)
    coverage = coverage_fraction(flowlines, footprints, p.buffer_m)  # E-R7, computed once
    touched = flowlines.loc[coverage.to_numpy() > 0.0, "vpu"].astype(str)
    catchments, vaa, edges = _read_inputs_nhd(
        nhd_dir, sorted(set(touched)), set(flowlines["comid"].astype("int64"))
    )
    sites = _train_sites(read_parquet(site_path))
    table, counts = derive_locations(
        cells,
        footprints,
        flowlines,
        catchments,
        vaa,
        edges,
        sites,
        p,
        settings.protocol.splits.ind_dedupe_m,
        coverage=coverage,
    )
    kept = table.loc[table["dedupe_status"].eq("kept")]
    n_rivers = int(kept["river"].nunique())
    report["reaches_and_locations"] = counts
    report["catchment_vpus"] = sorted(set(touched))
    report["training_sites"] = len(sites)
    report["per_river"] = _river_summary(table, p.rivers)
    report["da_distribution_kept"] = _da_distribution(kept["DA_km2"])
    report["e_r10"] = {
        "locations_kept": len(kept),
        "rivers_kept": n_rivers,
        "min_locations": p.min_locations,
        "min_rivers": p.min_rivers,
        "descriptive_only": len(kept) < p.min_locations or n_rivers < p.min_rivers,
        "note": "rivers are counted by GNIS name; homonyms are listed under per_river",
    }
    write_parquet(
        table,
        Path(rt.processed_dir) / _OUTPUT,
        stage=STAGE,
        params_sha256=settings.params_sha256,
        inputs=inputs,
        sort_by=["loc_id"],
    )
    write_json(report_path, report)
    LOG.info("ind_prepare: %d locations, %d kept on %d rivers", len(table), len(kept), n_rivers)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Run ``python -m src.data.ind_prepare``."""
    from src.config import load_settings

    argparse.ArgumentParser(description="eHydro independent set (S-IND, L8).").parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(load_settings())


if __name__ == "__main__":
    main()

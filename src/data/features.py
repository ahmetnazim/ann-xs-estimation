"""Stage features: join predictors to ``site_core`` and apply R12-R14 (protocol 2.1-2.4, 2.7).

Sources: protocol section 2.1 (units: D50 mm x 1e-3; Global-AI value x readme scale; slope and
TotDASqKm factor 1; MinElevSmo cm x 0.01 already applied by unit_map), section 2.3 (raw
contracts; AI zonal mean per NHDPlusV2.1 catchment, nodata excluded), section 2.4 (R12 slope
missing, R14 missing predictor: site excluded, never imputed), section 3.5 L4 (banned
predictors), ADL-005 (``V_E`` only in ``A_J``), O-2 (M1 uses the USGS release; M2, M4, M5 and
M6 use the StreamCat set), P5-D4 (in-house zonal mean, ``all_touched=False``), P5-D5
(``params.features``).

Output ``data/processed/site_table.parquet``: ``site_core`` (long; one row per site and
selection variant) with the per-site predictors appended. Sites failing R12 or R14 on the core
predictor set are removed from every variant (they are outside the domain); missing M1-only
predictors set ``F_M1_missing`` (M1 only, protocol 2.3). ``reports/feature_missing.json``
counts missing values per predictor and per division (protocol 2.7).

Owning phase: 5.
"""

from __future__ import annotations

import argparse
import json
import math
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from src.config import Settings, load_settings
from src.data.io import read_parquet, sha256_file, unit_manifest_sha256, write_json, write_parquet
from src.data.readers import mask_sentinels, read_attr_table, read_dbf
from src.data.schemas import BANNED_PREDICTORS, SENTINELS, assert_no_banned
from src.data.validate_raw import D50_FIELD, USGS_ATTR_TABLES, streamcat_fields

if TYPE_CHECKING:
    from geopandas import GeoDataFrame

__all__ = [
    "CORE_NHD_PREDICTORS",
    "M1_PREDICTORS",
    "aridity_zonal_mean",
    "build_site_table",
    "landcover_sums",
    "main",
    "read_streamcat",
    "read_usgs_attrs",
]

LOG = logging.getLogger(__name__)

#: NHDPlusV2.1 predictors shared by every model (VAA, ElevSlope; P3-D6). ``Q_E``, ``S`` and
#: ``F_S`` are physics inputs (routed away from the n-head under I_blind, protocol 4.1).
CORE_NHD_PREDICTORS: Final[tuple[str, ...]] = (
    "DA_km2",
    "stream_order",
    "z_min_m",
    "S",
    "F_S",
    "Q_E",
)
#: USGS release attributes for M1 (Zarrabi Table 1 classes; NLCD grouped like StreamCat).
M1_PREDICTORS: Final[tuple[str, ...]] = (
    "CAT_EVI_AMJ_2012",
    "CAT_EVI_JAS_2012",
    "CAT_EVI_JFM_2012",
    "CAT_EVI_OND_2011",
    "ACC_NDAMS2010",
    "CAT_POPDENS10",
    "CAT_SILTAVE",
    "CAT_CLAYAVE",
    "CAT_SANDAVE",
    "m1_urban",
    "m1_forest",
    "m1_agriculture",
)
_M1_LANDCOVER: Final[Mapping[str, tuple[str, ...]]] = {
    "m1_urban": ("CAT_NLCD16_21", "CAT_NLCD16_22", "CAT_NLCD16_23", "CAT_NLCD16_24"),
    "m1_forest": ("CAT_NLCD16_41", "CAT_NLCD16_42", "CAT_NLCD16_43"),
    "m1_agriculture": ("CAT_NLCD16_81", "CAT_NLCD16_82"),
}


def landcover_sums(frame: pd.DataFrame, groups: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Return one column per group: the sum of its classes (missing if any class is missing)."""
    return pd.DataFrame(
        {name: frame[list(cols)].sum(axis=1, min_count=len(cols)) for name, cols in groups.items()},
        index=frame.index,
    )


def read_streamcat(unit_dir: Path, metrics: Sequence[str], comids: set[int]) -> pd.DataFrame:
    """Read the StreamCat responses for ``comids`` (``comid`` + metrics, float64)."""
    parts = []
    for path in sorted(unit_dir.glob("streamcat_*.json")):
        frame = pd.DataFrame.from_records(json.loads(path.read_text(encoding="utf-8"))["items"])
        frame = frame.loc[frame["comid"].isin(comids), ["comid", *metrics]]
        parts.append(frame.astype({"comid": "int64", **dict.fromkeys(metrics, "float64")}))
    out = pd.concat(parts, ignore_index=True)
    for col in metrics:
        out[col], _ = mask_sentinels(out[col], SENTINELS)
    return out


def read_usgs_attrs(unit_dir: Path, comids: set[int]) -> pd.DataFrame:
    """Read the USGS release tables for ``comids``; sentinels -> missing; NLCD grouped."""
    merged: pd.DataFrame | None = None
    for table, columns in USGS_ATTR_TABLES.items():
        path = next(p for p in (unit_dir / table).glob("*") if p.suffix.lower() == ".txt")
        frame, _ = read_attr_table(path, columns)
        frame = frame.loc[frame["COMID"].isin(comids)]
        for col in columns:
            frame[col], _ = mask_sentinels(frame[col], SENTINELS)
        merged = frame if merged is None else merged.merge(frame, on="COMID", how="outer")
    if merged is None:
        raise ValueError("no usgs_attrs table configured")
    merged = pd.concat([merged, landcover_sums(merged, _M1_LANDCOVER)], axis=1)
    return merged.rename(columns={"COMID": "comid"})[["comid", *M1_PREDICTORS]]


def aridity_zonal_mean(
    catchments: GeoDataFrame, raster: Path, scale: float, fill: Sequence[int]
) -> pd.DataFrame:
    """Mean Global-AI over each catchment: pixel centres inside the polygon, fill excluded.

    Parameters
    ----------
    catchments
        ``GeoDataFrame[comid, geometry]`` (any CRS; reprojected to the raster CRS).
    raster
        Global-AI v3.1 annual GeoTIFF.
    scale
        Readme scale factor (AI = value x scale).
    fill
        Raw values excluded from the mean (the file declares no nodata).

    Returns
    -------
    pandas.DataFrame
        ``comid``, ``AI``, ``ai_pixels`` and ``F_AI_touched`` (no pixel centre inside, so the
        touched pixels were used; ``AI`` is missing only if those are all fill).
    """
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import Window, from_bounds

    rows = []
    with rasterio.open(raster) as src:
        polys = catchments.to_crs(src.crs)
        for comid, geom in zip(polys["comid"].to_numpy(), polys.geometry, strict=True):
            raw = from_bounds(*geom.bounds, transform=src.transform)
            col0, row0 = math.floor(raw.col_off), math.floor(raw.row_off)
            col1 = math.ceil(raw.col_off + raw.width)
            row1 = math.ceil(raw.row_off + raw.height)
            window = Window(col0, row0, max(col1 - col0, 1), max(row1 - row0, 1))
            band = src.read(1, window=window, boundless=True, fill_value=fill[0])
            inside = ~geometry_mask(
                [geom],
                out_shape=band.shape,
                transform=src.window_transform(window),
                all_touched=False,
            )
            valid = inside & ~np.isin(band, list(fill))
            touched = False
            if not valid.any():
                # Catchment smaller than a pixel: no pixel centre inside. Fall back to the
                # pixels the polygon touches (flagged; PI decision pending, Batch C Q).
                touched = True
                inside = ~geometry_mask(
                    [geom],
                    out_shape=band.shape,
                    transform=src.window_transform(window),
                    all_touched=True,
                )
                valid = inside & ~np.isin(band, list(fill))
            n = int(valid.sum())
            ai = float(band[valid].astype("float64").mean() * scale) if n else np.nan
            rows.append((int(comid), ai, n, touched))
    return pd.DataFrame(rows, columns=["comid", "AI", "ai_pixels", "F_AI_touched"])


def build_site_table(
    core: pd.DataFrame,
    per_site: pd.DataFrame,
    core_predictors: Sequence[str],
    m1_predictors: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join per-site predictors to ``site_core`` and apply R12 and R14 (core set).

    Parameters
    ----------
    core
        ``site_core.parquet`` (long; ``site_no``, ``variant``, ``comid``, ``division``, ...).
    per_site
        One row per ``comid`` with every predictor column.
    core_predictors
        Predictors used by M2, M4, M5 and M6 (missing -> site excluded, R14).
    m1_predictors
        M1-only predictors (missing -> ``F_M1_missing``).
    """
    assert_no_banned([*core_predictors, *m1_predictors])
    sites = core.drop_duplicates("site_no")[["site_no", "comid", "division", "Q_E"]]
    joined = sites.merge(per_site, on="comid", how="left")
    r12 = joined["S"].isna()
    missing_core = joined[list(core_predictors)].isna()
    r14 = missing_core.drop(columns=["S"], errors="ignore").any(axis=1) & ~r12
    keep = joined.loc[~r12 & ~r14, "site_no"]
    counts: dict[str, Any] = {
        "sites_in": len(sites),
        "R12_slope_missing": int(r12.sum()),
        "R14_core_missing": int(r14.sum()),
        "sites_out": len(keep),
        "missing_per_predictor": {c: int(v) for c, v in missing_core.sum().items()},
        "missing_per_division": {
            str(k): int(v) for k, v in joined.loc[r12 | r14, "division"].value_counts().items()
        },
        "m1_missing_per_predictor": {
            c: int(v) for c, v in joined[list(m1_predictors)].isna().sum().items()
        },
    }
    joined["F_M1_missing"] = joined[list(m1_predictors)].isna().any(axis=1)
    counts["F_M1_missing_sites_out"] = int(joined.loc[~r12 & ~r14, "F_M1_missing"].sum())
    feats = joined.drop(columns=["comid", "division", "Q_E"])
    table = core.loc[core["site_no"].isin(keep)].merge(feats, on="site_no", how="left")
    table["A_J"] = table["Q_E"] / table["V_E"]
    counts["A_J_missing_rows"] = int(table["A_J"].isna().sum())
    table = table.sort_values(["site_no", "variant"], kind="mergesort").reset_index(drop=True)
    return table, counts


def _per_site_predictors(settings: Settings, comids: set[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assemble every predictor for the linked COMIDs (one row per COMID)."""
    import geopandas as gpd

    from src.data.nhdplus import VPUS, read_layer

    rt = settings.runtime
    feat = settings.protocol.features
    um = rt.processed_dir / "unit_mapped"
    vaa = read_parquet(um / "nhd_vaa.parquet")[["comid", "vpu", "DA_km2", "stream_order"]]
    elev = read_parquet(um / "nhd_elevslope.parquet")
    frame = vaa.loc[vaa["comid"].isin(comids)].merge(elev, on="comid", how="left")
    metrics = streamcat_fields(settings)
    sc = read_streamcat(rt.raw_dir / "streamcat", metrics, comids)
    sc = pd.concat(
        [sc.drop(columns=[c for g in feat.landcover_groups.values() for c in g]),
         landcover_sums(sc, feat.landcover_groups)],
        axis=1,
    )  # fmt: skip
    frame = frame.merge(sc, on="comid", how="left")
    frame = frame.merge(read_usgs_attrs(rt.raw_dir / "usgs_attrs", comids), on="comid", how="left")
    d50_path = next(p for p in (rt.raw_dir / "d50").glob("*") if p.suffix.lower() == ".dbf")
    d50 = read_dbf(d50_path, ["COMID", D50_FIELD]).rename(columns={"COMID": "comid"})
    d50 = d50.astype({"comid": "int64"})
    d50["D50_m"] = d50[D50_FIELD].astype("float64") * feat.d50_mm_to_m
    frame = frame.merge(d50.loc[d50["comid"].isin(comids), ["comid", "D50_m"]], on="comid", how="left")
    raster = next(iter(sorted((rt.raw_dir / "aridity").rglob("ai_v31_yr.tif"))))
    ai_parts = []
    for vpu in VPUS:
        wanted = set(frame.loc[frame["vpu"].eq(vpu), "comid"])
        if not wanted:
            continue
        cats = read_layer(rt.raw_dir / "nhdplusv21", "catchment", vpu)
        cats = gpd.GeoDataFrame(cats.loc[cats["comid"].isin(wanted)], crs=cats.crs)
        ai_parts.append(aridity_zonal_mean(cats, raster, feat.aridity_scale, feat.aridity_fill))
    ai = pd.concat(ai_parts, ignore_index=True)
    frame = frame.merge(ai, on="comid", how="left")
    counts = {
        "comids": len(comids),
        "comids_without_catchment": int(frame["ai_pixels"].isna().sum()),
        "catchments_ai_touched_fallback": int(frame["F_AI_touched"].fillna(False).sum()),
        "catchments_without_valid_pixel": int(frame["ai_pixels"].eq(0).sum()),
    }
    sc_set = [c for c in metrics if c not in {c for g in feat.landcover_groups.values() for c in g}]
    frame.attrs["streamcat_predictors"] = [*sc_set, *feat.landcover_groups]
    return frame.drop(columns=["vpu"]), counts


def run(settings: Settings) -> None:
    """Build ``site_table.parquet`` and ``feature_missing.json``."""
    rt = settings.runtime
    p_core = rt.processed_dir / "site_core.parquet"
    core = read_parquet(p_core)
    per_site, ai_counts = _per_site_predictors(settings, set(core["comid"]))
    core_predictors = [
        *CORE_NHD_PREDICTORS,
        *per_site.attrs["streamcat_predictors"],
        "D50_m",
        "AI",
    ]
    table, counts = build_site_table(core, per_site, core_predictors, M1_PREDICTORS)
    LOG.info("site_table: %d rows, %d sites", len(table), table["site_no"].nunique())
    write_parquet(
        table,
        rt.processed_dir / "site_table.parquet",
        stage="features",
        params_sha256=settings.params_sha256,
        inputs={
            "site_core.parquet": sha256_file(p_core),
            **{
                f"unit_mapped/{n}": sha256_file(rt.processed_dir / "unit_mapped" / n)
                for n in ("nhd_vaa.parquet", "nhd_elevslope.parquet")
            },
            **{
                f"{u}/MANIFEST.sha256": unit_manifest_sha256(rt.raw_dir / u)
                for u in ("streamcat", "usgs_attrs", "d50", "aridity", "nhdplusv21")
            },
        },
        sort_by=["site_no", "variant"],
    )
    write_json(
        rt.reports_dir / "feature_missing.json",
        {
            "params_sha256": settings.params_sha256,
            "core_predictors": core_predictors,
            "m1_predictors": list(M1_PREDICTORS),
            "banned": sorted(BANNED_PREDICTORS),
            **counts,
            **ai_counts,
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run ``python -m src.data.features`` (the dvc.yaml ``features`` stage)."""
    argparse.ArgumentParser(description="Feature join, R12-R14 (protocol 2.4).").parse_args(argv)
    settings = load_settings()
    logging.basicConfig(
        level=settings.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    LOG.info("params_sha256=%s", settings.params_sha256)
    run(settings)


if __name__ == "__main__":
    main()


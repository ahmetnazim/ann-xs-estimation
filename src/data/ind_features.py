"""Stage ind_features (docs/3 13.3 stage 0a): S-IND predictors per member COMID and E-R8 helpers.

Sources: docs/3 section 13.5 (locations = the ``ehydro_ind`` rows with ``dedupe_status`` kept;
predictors built per member COMID from the raw units with the functions of
``src/data/features.py``, the same columns as ``site_table``, R12/R13/R14 applied, no imputation;
a location is excluded for every model if any member COMID lacks a predictor of the NN set (22)
or the M1 set (19), or lies outside the domain D; M0 takes the F&J division containing the
``point_on_surface`` of the member COMID's catchment (EPSG:5070) with the protocol 4.1 pooled
fallback; A_J follows the primary masking rule; aggregation mirrors E-R8: length-weighted Ŵ and
d̂ with L_c = ``LENGTHKM``, Â = Ŵ·d̂, A_J as the length-weighted mean of A_J,c), section 2.4
(R12 slope missing, R13 ``Q_E <= 0`` or missing: outside D; R14 missing predictor), section 2.7
(no imputation), ADL-005 (``V_E`` only in A_J), section 11.3 E-R8 (the targets were built as
LENGTHKM-weighted means of per-COMID values, Â = Ŵ·d̂).

Target isolation: only ``loc_id``, ``river``, ``comids``, ``n_comids`` and ``dedupe_status`` are
read from ``ehydro_ind``; W, d̄ and A are never read here (:func:`member_table` refuses them).

Exclusion reasons are exclusive per COMID, in this order: ``R13`` (``Q_E`` not > 0), ``R12``
(slope missing), ``R14_nn`` (an NN predictor missing), ``R14_m1`` (an M1 predictor missing).
Rows of excluded locations stay in the table with ``loc_in_support = False`` so the counts can be
traced; predictions use the supported rows only. ``reports/ind_features.json`` holds counts only.

Torch-free and xgboost-free (ADL-051). Owning phase: 8.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from src.config import Settings, load_settings
from src.data.io import sha256_file, unit_manifest_sha256, write_json, write_parquet
from src.data.schemas import assert_no_banned
from src.data.srnd_frame import M1_SET

if TYPE_CHECKING:
    from geopandas import GeoDataFrame

__all__ = [
    "EXCLUSION_ORDER",
    "LOCATION_COLUMNS",
    "aggregate_e_r8",
    "build_ind_features",
    "catchment_divisions",
    "ind_features_path",
    "length_weighted_mean",
    "main",
    "member_table",
]

LOG = logging.getLogger(__name__)

#: The only ``ehydro_ind`` columns this stage reads (targets W, d̄, A are never read).
LOCATION_COLUMNS: Final[tuple[str, ...]] = (
    "loc_id",
    "river",
    "comids",
    "n_comids",
    "dedupe_status",
)
#: Exclusive exclusion reasons per member COMID, in precedence order (docs/3 13.5, protocol 2.4).
EXCLUSION_ORDER: Final[tuple[str, ...]] = ("R13", "R12", "R14_nn", "R14_m1")
_TARGETS: Final[frozenset[str]] = frozenset({"W", "dbar", "A", "dmax"})
_EROM: Final[tuple[str, ...]] = ("Q_E", "Q_C", "V_E")


def ind_features_path(processed_dir: Path) -> Path:
    """``data/processed/ind/ehydro_ind_features.parquet`` (docs/3 13.3 stage 0a)."""
    return processed_dir / "ind" / "ehydro_ind_features.parquet"


# ----------------------------------------------------------------------------- members
def member_table(ind: pd.DataFrame) -> pd.DataFrame:
    """One row per member COMID of every kept location (``loc_id, river, n_comids, member, comid``).

    ``member`` is the 0-based position in ``comids`` (upstream to downstream). Raises if a target
    column is present: S-IND targets are read only inside a scoring event (docs/3 13.4).
    """
    leaked = sorted(_TARGETS & set(ind.columns))
    if leaked:
        raise ValueError(f"ind_features must not read the S-IND targets {leaked}")
    missing = [c for c in LOCATION_COLUMNS if c not in ind.columns]
    if missing:
        raise KeyError(f"ehydro_ind lacks {missing}")
    kept = ind.loc[ind["dedupe_status"].eq("kept")].sort_values("loc_id", kind="mergesort")
    rows = []
    for loc_id, river, comids, n in kept[["loc_id", "river", "comids", "n_comids"]].itertuples(
        index=False
    ):
        ids = [int(c) for c in str(comids).split(";")]
        if len(ids) != int(n):
            raise ValueError(f"{loc_id}: n_comids differs from the member list")
        rows += [(str(loc_id), str(river), int(n), i, c) for i, c in enumerate(ids)]
    out = pd.DataFrame(rows, columns=["loc_id", "river", "n_comids", "member", "comid"])
    return out.astype({"n_comids": "int64", "member": "int64", "comid": "int64"})


def catchment_divisions(catchments: GeoDataFrame, polygons: GeoDataFrame) -> pd.Series:
    """F&J division of each catchment's ``point_on_surface`` (EPSG:5070) or missing (docs/3 13.5).

    The point is taken in EPSG:5070 and assigned with the R2 rule of
    :func:`src.data.qc.assign_division` (exactly one non-missing division, boundaries missing).
    Returns a string Series indexed by ``comid``.
    """
    import geopandas as gpd

    from src.data.qc import assign_division

    # GeoPandas exposes shapely's point_on_surface as representative_point.
    pts = gpd.GeoSeries(catchments.to_crs("EPSG:5070").geometry.representative_point())
    geo = pts.to_crs("EPSG:4269")
    lat = pd.Series(geo.y.to_numpy(), index=catchments.index)
    lon = pd.Series(geo.x.to_numpy(), index=catchments.index)
    division = assign_division(lat, lon, polygons)
    return pd.Series(
        division.to_numpy(), index=catchments["comid"].astype("int64").to_numpy(), dtype="string"
    )


# ----------------------------------------------------------------------------- builder
def _reasons(table: pd.DataFrame, nn: Sequence[str], m1: Sequence[str]) -> pd.Series:
    r13 = ~(table["Q_E"] > 0)
    r12 = table["S"].isna()
    r14_nn = table[list(nn)].isna().any(axis=1)
    r14_m1 = table[list(m1)].isna().any(axis=1)
    reason = np.select([r13, r12, r14_nn, r14_m1], list(EXCLUSION_ORDER), default="")
    return pd.Series(reason, index=table.index, dtype="string")


def build_ind_features(
    members: pd.DataFrame,
    per_comid: pd.DataFrame,
    erom: pd.DataFrame,
    lengths: Mapping[int, float],
    divisions: pd.Series,
    nn_predictors: Sequence[str],
    m1_predictors: Sequence[str],
    log_features: Sequence[str] = (),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join predictors to every member COMID and apply the common-support rule (docs/3 13.5).

    Parameters
    ----------
    members
        :func:`member_table` output.
    per_comid
        One row per ``comid`` with every predictor column (``features._per_site_predictors``).
    erom
        ``unit_mapped/nhd_erom.parquet`` (``comid``, ``Q_E``, ``Q_C``, ``V_E``).
    lengths
        ``LENGTHKM`` per COMID (NHDFlowline; must be > 0 for every member).
    divisions
        F&J division per COMID (:func:`catchment_divisions`); missing -> M0 pooled fallback.
    nn_predictors, m1_predictors
        The NN set (22) and the M1 set (19).
    log_features
        NN predictors with a ``log`` transform; supported rows with a value <= 0 are counted
        (they would be refused by ``NormStats`` at prediction time).
    """
    assert_no_banned([*nn_predictors, *m1_predictors])
    if per_comid["comid"].duplicated().any() or erom["comid"].duplicated().any():
        raise ValueError("per-COMID inputs must have one row per comid")
    table = members.merge(per_comid, on="comid", how="left", validate="many_to_one")
    table = table.merge(erom[["comid", *_EROM]], on="comid", how="left", validate="many_to_one")
    length = table["comid"].map(lambda c: lengths.get(int(c), np.nan)).astype("float64")
    if not bool((length > 0).all()):
        raise ValueError("every member COMID needs LENGTHKM > 0 (E-R8)")
    table["lengthkm"] = length
    table["division"] = (
        pd.Series(divisions, dtype="string").reindex(table["comid"].to_numpy()).to_numpy()
    )
    table["division"] = table["division"].astype("string")
    table["A_J"] = table["Q_E"] / table["V_E"]  # ADL-005; missing V_E -> unavailable
    table["F_M1_missing"] = table[list(m1_predictors)].isna().any(axis=1)
    table["exclusion"] = _reasons(table, nn_predictors, m1_predictors)
    bad_loc = table.groupby("loc_id")["exclusion"].transform(lambda s: bool(s.ne("").any()))
    table["loc_in_support"] = ~bad_loc.astype(bool)
    table = table.sort_values(["loc_id", "member"], kind="mergesort").reset_index(drop=True)
    sup = table.loc[table["loc_in_support"]]
    locs = table.drop_duplicates("loc_id")
    sup_locs = sup.drop_duplicates("loc_id")
    any_reason = {
        r: int(table.loc[table["exclusion"].eq(r), "loc_id"].nunique()) for r in EXCLUSION_ORDER
    }
    log_bad = (
        sup[list(log_features)].le(0).any(axis=1) if log_features else pd.Series(False, sup.index)
    )
    aj_missing = sup.groupby("loc_id")["A_J"].apply(lambda s: bool(s.isna().any()))
    counts: dict[str, Any] = {
        "locations_kept": int(locs.shape[0]),
        "member_rows": len(table),
        "member_comids_unique": int(table["comid"].nunique()),
        "comids_by_reason": {r: int(table["exclusion"].eq(r).sum()) for r in EXCLUSION_ORDER},
        "locations_excluded": int((~locs["loc_in_support"]).sum()),
        "locations_excluded_with_reason": any_reason,
        "locations_in_support": int(sup_locs.shape[0]),
        "locations_in_support_single_comid": int(sup_locs["n_comids"].eq(1).sum()),
        "locations_in_support_multi_comid": int(sup_locs["n_comids"].gt(1).sum()),
        "rivers_in_support": sorted(sup_locs["river"].unique().tolist()),
        "member_rows_in_support": len(sup),
        "division_missing_in_support": int(sup["division"].isna().sum()),
        "aj_unavailable_locations_in_support": int(aj_missing.sum()),
        "nn_log_nonpositive_rows_in_support": int(log_bad.sum()),
        "ai_touched_fallback_in_support": int(sup["F_AI_touched"].fillna(False).astype(bool).sum())
        if "F_AI_touched" in sup
        else 0,
    }
    return table, counts


# ----------------------------------------------------------------------------- E-R8 aggregation
def length_weighted_mean(
    loc_id: Sequence[str] | pd.Series, lengthkm: np.ndarray, values: np.ndarray
) -> pd.Series:
    """``Σ_c L_c v_c / Σ_c L_c`` per location (E-R8); missing if any member value is missing.

    Locations are returned in sorted ``loc_id`` order. ``lengthkm`` must be finite and > 0.
    """
    length = np.asarray(lengthkm, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    ids = pd.Series(np.asarray(loc_id), dtype="string")
    if not (length.shape == vals.shape == (len(ids),)):
        raise ValueError("loc_id, lengthkm and values must be 1-D of equal length")
    if not (bool(np.isfinite(length).all()) and bool((length > 0).all())):
        raise ValueError("LENGTHKM must be finite and > 0")
    if bool(np.isinf(vals).any()):
        raise ValueError("values must be finite or missing")
    frame = pd.DataFrame({"loc_id": ids, "lv": length * vals, "l": length, "na": np.isnan(vals)})
    grouped = frame.groupby("loc_id", sort=True)
    out = grouped["lv"].sum() / grouped["l"].sum()
    out[grouped["na"].any()] = np.nan
    out.index = out.index.astype(str)
    return out.astype("float64")


def aggregate_e_r8(
    loc_id: Sequence[str] | pd.Series, lengthkm: np.ndarray, w_hat: np.ndarray, d_hat: np.ndarray
) -> pd.DataFrame:
    """Location-level Ŵ, d̂ (length-weighted per-COMID point predictions) and Â = Ŵ·d̂ (E-R8).

    ``w_hat`` and ``d_hat`` are per-COMID point predictions in metres (exp(μ*_c)); they must be
    finite and > 0. Returns ``loc_id, W, dbar, A`` in sorted ``loc_id`` order.
    """
    for name, v in (("w_hat", w_hat), ("d_hat", d_hat)):
        arr = np.asarray(v, dtype=np.float64)
        if not (bool(np.isfinite(arr).all()) and bool((arr > 0).all())):
            raise ValueError(f"{name} must be finite and > 0")
    w = length_weighted_mean(loc_id, lengthkm, w_hat)
    d = length_weighted_mean(loc_id, lengthkm, d_hat)
    return pd.DataFrame(
        {
            "loc_id": w.index.to_numpy(),
            "W": w.to_numpy(),
            "dbar": d.to_numpy(),
            "A": (w * d).to_numpy(),
        }
    )


# ----------------------------------------------------------------------------- stage
def _flowline_lengths(nhd_dir: Path, vpus: Sequence[str], comids: set[int]) -> dict[int, float]:
    import warnings

    from src.data.nhdplus import NhdTable, find_table_files
    from src.data.readers import read_dbf

    files = find_table_files(nhd_dir, NhdTable("NHDFlowline.dbf", ("COMID",), ("COMID",), ()))
    out: dict[int, float] = {}
    for vpu in vpus:
        with warnings.catch_warnings():
            # GDAL reports that measured (M) geometries are read without M; expected and harmless.
            warnings.filterwarnings("ignore", message="Measured \\(M\\) geometry types")
            frame = read_dbf(files[vpu], ["COMID", "LENGTHKM"])
        frame = frame.loc[frame["COMID"].astype("int64").isin(comids)]
        for c, length in zip(frame["COMID"].astype("int64"), frame["LENGTHKM"], strict=True):
            out[int(c)] = float(length)
    return out


def _divisions(settings: Settings, vpus: Sequence[str], comids: set[int]) -> pd.Series:
    import geopandas as gpd

    from src.data.nhdplus import CRS_ALBERS, read_layer
    from src.data.qc import load_divisions

    rt = settings.runtime
    parts = []
    for vpu in vpus:
        layer = read_layer(rt.raw_dir / "nhdplusv21", "catchment", vpu)
        parts.append(layer.loc[layer["comid"].isin(list(comids))])
    cats = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=CRS_ALBERS)
    if cats["comid"].duplicated().any():
        raise ValueError("a member COMID has more than one catchment polygon")
    return catchment_divisions(cats, load_divisions(rt.raw_dir / "physiography"))


def run(settings: Settings) -> dict[str, Any]:
    """Build ``ehydro_ind_features.parquet`` and ``reports/ind_features.json``."""
    import pyarrow.parquet as pq

    from src.data.features import _per_site_predictors
    from src.data.io import read_parquet

    rt = settings.runtime
    p_ind = rt.processed_dir / "ind" / "ehydro_ind.parquet"
    p_erom = rt.processed_dir / "unit_mapped" / "nhd_erom.parquet"
    ind = pq.read_table(p_ind, columns=[*LOCATION_COLUMNS, "vpus"]).to_pandas()
    members = member_table(ind.drop(columns=["vpus"]))
    kept = ind.loc[ind["dedupe_status"].eq("kept")]
    vpus = sorted({v for s in kept["vpus"] for v in str(s).split(";")})
    comids = set(members["comid"].tolist())
    per_comid, ai_counts = _per_site_predictors(settings, comids)
    nn = list(settings.protocol.transforms.features)
    log_feats = [f for f, k in settings.protocol.transforms.features.items() if k == "log"]
    table, counts = build_ind_features(
        members,
        per_comid,
        read_parquet(p_erom),
        _flowline_lengths(rt.raw_dir / "nhdplusv21", vpus, comids),
        _divisions(settings, vpus, comids),
        nn,
        list(M1_SET),
        log_feats,
    )
    write_parquet(
        table,
        ind_features_path(rt.processed_dir),
        stage="ind_features",
        params_sha256=settings.params_sha256,
        inputs={
            "ind/ehydro_ind.parquet": sha256_file(p_ind),
            **{
                f"unit_mapped/{n}": sha256_file(rt.processed_dir / "unit_mapped" / n)
                for n in ("nhd_vaa.parquet", "nhd_elevslope.parquet", "nhd_erom.parquet")
            },
            **{
                f"{u}/MANIFEST.sha256": unit_manifest_sha256(rt.raw_dir / u)
                for u in ("streamcat", "usgs_attrs", "d50", "aridity", "nhdplusv21", "physiography")
            },
        },
        sort_by=["loc_id", "member"],
    )
    report = {
        "params_sha256": settings.params_sha256,
        "predictors_nn": nn,
        "predictors_m1": list(M1_SET),
        **counts,
        "ai": ai_counts,
    }
    write_json(rt.reports_dir / "ind_features.json", report)
    LOG.info(
        "ind_features: %d of %d kept locations in support",
        counts["locations_in_support"],
        counts["locations_kept"],
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Run ``python -m src.data.ind_features`` (the dvc.yaml ``ind_features`` stage)."""
    argparse.ArgumentParser(description="S-IND predictors per COMID (docs/3 13.5).").parse_args(
        argv
    )
    settings = load_settings()
    logging.basicConfig(
        level=settings.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    LOG.info("params_sha256=%s", settings.params_sha256)
    run(settings)


if __name__ == "__main__":
    main()

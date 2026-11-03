"""Build deterministic static JSON/GeoJSON data bundle for the `/webgis` application.

Reads frozen repository artifacts (`data/processed/*`, `reports/results/*`,
`reports/operational/*`, `docs/operational/best_model_map/tables/*`, and
`data/raw/{physiography,nhdplusv21,ops_transects}`) without modifying any frozen
v1.0.0 files, and writes compact static bundles into `webgis/data/`:

- `webgis/data/physio_divisions.geojson`: 8 Fenneman & Johnson physiographic
  divisions in WGS84 (`EPSG:4326`) with division-level winners and bootstrap CIs.
- `webgis/data/sites_conus.json`: Dataset of all 3,988 primary CONUS USGS
  `HYDROSWOT` gauge reaches with physical covariates, data-availability flags,
  observed cross-section geometry, out-of-fold (`S-ID` and `S-LRO`) model
  predictions (`M2-XGB`, `M2-MLP`, `M1`, `M5`, `M4`, `M3`, `M0`), 90%
  deep-ensemble UQ bounds (`M5`, `M2-MLP`), 3DEP lidar floodplain profiles, and
  per-reach RMSEs across `geom`, `op_fp`, and `op_dyn`.
- `webgis/data/sites_ehydro.json`: All 131 independent USACE `eHydro` (`S-IND`)
  surveyed locations across 18 rivers with WGS84 coordinates, observed geometry,
  and model predictions + 90% UQ bounds.
- `webgis/data/benchmarks.json`: Complete multi-regime benchmark tables
  (`Event 1-3`, `H1-H4`, `OPS-1 v1-v3`, `CCHE1D` readiness, QA/QC cascade, and
  the 7-record SHA-256 `scoring_ledger.json`).
- `webgis/data/manifest.json`: SHA-256 inventory of generated WebGIS assets.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any, Final

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
OUT_DIR: Final[Path] = REPO_ROOT / "webgis" / "data"
CRS_WGS84: Final[str] = "EPSG:4326"
CRS_ALBERS: Final[str] = "EPSG:5070"
_VPU_RE: Final[re.Pattern[str]] = re.compile(r"/NHDPlus(\d\d[A-Z]?)/")

HEADLINE_MODELS: Final[tuple[str, ...]] = (
    "M2-XGB",
    "M2-MLP",
    "M1",
    "M5",
    "M4",
    "M3",
    "M0",
)
ALL_TEN_MODELS: Final[tuple[str, ...]] = (
    "M-null",
    "M0",
    "M1",
    "M2-MLP",
    "M2-XGB",
    "M3",
    "M4",
    "M5",
    "M6-NN",
    "M6-XGB",
)
UQ_MODELS: Final[tuple[str, ...]] = ("M5", "M2-MLP")
FP_SAMPLE_INDICES: Final[tuple[int, ...]] = (
    0,
    100,
    200,
    300,
    375,
    425,
    460,
    480,
    490,
    495,
    500,
    505,
    510,
    520,
    540,
    575,
    625,
    700,
    800,
    900,
    1000,
)


def _r(val: object, digits: int = 4) -> float | None:
    """Round finite float to `digits` decimals, or return None for NaN/None."""
    if val is None or isinstance(val, bool):
        return None
    if not isinstance(val, (int, float, str, np.number)):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, digits)


def _exp_r(val: object, digits: int = 4) -> float | None:
    """Exponentiate finite log-space float and round to `digits` decimals."""
    f = _r(val, 12)
    if f is None:
        return None
    return round(math.exp(f), digits)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    path.write_text(text, encoding="utf-8")
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _round_coords(coords: object, digits: int = 4) -> object:
    if isinstance(coords, (list, tuple)):
        if coords and isinstance(coords[0], (int, float)):
            return [round(float(c), digits) for c in coords[:2]]
        return [_round_coords(c, digits) for c in coords]
    return coords


def build_physio_geojson(gamma_by_div: dict[str, float]) -> dict[str, Any]:
    """Dissolve and simplify the 8 Fenneman & Johnson divisions into EPSG:4326 GeoJSON."""
    physio_dir = REPO_ROOT / "data" / "raw" / "physiography"
    shp = next(iter(sorted(physio_dir.rglob("physio.shp"))))
    gdf = gpd.read_file(shp)[["DIVISION", "geometry"]]
    gdf["DIVISION"] = gdf["DIVISION"].astype("string").str.strip().replace("", pd.NA)
    gdf["geometry"] = shapely.make_valid(gdf.geometry.to_numpy())
    gdf = gdf.dropna(subset=["DIVISION"]).copy()
    gdf["DIVISION"] = gdf["DIVISION"].astype(str)
    gdf_5070 = gdf.to_crs(CRS_ALBERS)
    dissolved = gdf_5070.dissolve(by="DIVISION").reset_index()
    dissolved["geometry"] = shapely.simplify(
        dissolved.geometry.to_numpy(), tolerance=1800.0, preserve_topology=True
    )
    dissolved_4326 = dissolved.to_crs(CRS_WGS84)

    div_table_path = (
        REPO_ROOT
        / "docs"
        / "operational"
        / "best_model_map"
        / "tables"
        / "best_model_divisions.parquet"
    )
    div_stats: dict[str, dict[str, Any]] = {}
    if div_table_path.is_file():
        df_div = pd.read_parquet(div_table_path)
        for _, row in df_div.iterrows():
            dname = str(row["division"])
            crit = str(row["criterion"])
            div_stats.setdefault(dname, {})[crit] = {
                "n_sites": int(row["n_sites"]),
                "best": str(row["best"]),
                "best_rmse": _r(row["best_rmse"], 4),
                "second": str(row["second"]),
                "margin": _r(row["margin"], 4),
                "margin_ci_lo": _r(row["margin_ci_lo"], 4),
                "margin_ci_hi": _r(row["margin_ci_hi"], 4),
                "significant": bool(row["significant"]),
                "p_best": _r(row["p_best"], 3),
                "rmse_by_model": {
                    m: _r(row[f"rmse_{m}"], 4) for m in ALL_TEN_MODELS if f"rmse_{m}" in row
                },
            }

    features: list[dict[str, Any]] = []
    for _, row in dissolved_4326.iterrows():
        dname = str(row["DIVISION"])
        stats = div_stats.get(dname, {})
        if not stats:
            continue
        geom_json = shapely.to_geojson(row.geometry)
        geom_obj = json.loads(geom_json)
        geom_obj["coordinates"] = _round_coords(geom_obj.get("coordinates", []), 4)
        rep = row.geometry.representative_point()
        props: dict[str, Any] = {
            "division": dname,
            "label_lon": _r(rep.x, 4),
            "label_lat": _r(rep.y, 4),
            "gamma_lodo": _r(gamma_by_div.get(dname), 4),
            "n_sites": stats.get("geom", {}).get("n_sites", 0),
            "criteria": stats,
        }
        for crit in ("geom", "op_fp", "op_dyn"):
            if crit in stats:
                props[f"{crit}_best"] = stats[crit]["best"]
                props[f"{crit}_second"] = stats[crit]["second"]
                props[f"{crit}_best_rmse"] = stats[crit]["best_rmse"]
                props[f"{crit}_margin"] = stats[crit]["margin"]
                props[f"{crit}_significant"] = stats[crit]["significant"]
        features.append({"type": "Feature", "properties": props, "geometry": geom_obj})

    return {"type": "FeatureCollection", "features": features}


def _load_model_site_predictions(
    pred_dir: Path, model: str, family: str, fallback_t: pd.Series[Any]
) -> pd.DataFrame:
    """Average out-of-fold log predictions across repeats per `site_no` for `family`."""
    df = pd.read_parquet(pred_dir / f"{model}.parquet")
    keep = (df["analysis"] == "primary") & (df["family"] == family) & (df["unit"] == "site")
    if "ablation" in df.columns:
        keep &= df["ablation"].fillna("").astype(str).eq("")
    sub = df.loc[keep].copy()
    sub["site_no"] = sub["site_no"].astype(str)

    num_cols = [
        c
        for c in (
            "mu_W",
            "mu_dbar",
            "mu_dmax",
            "mu_A",
            "mu_t",
            "q_lo_W",
            "q_hi_W",
            "q_lo_dbar",
            "q_hi_dbar",
            "q_lo_A",
            "q_hi_A",
            "nu_mean",
            "nu_back",
            "r_mean",
            "r_hat",
        )
        if c in sub.columns
    ]
    agg = sub.groupby("site_no", sort=False)[num_cols].mean()

    if "mu_dmax" not in agg.columns:
        agg["mu_dmax"] = agg["mu_dbar"] + fallback_t.reindex(agg.index)
    else:
        missing_dmax = agg["mu_dmax"].isna()
        if missing_dmax.any():
            agg.loc[missing_dmax, "mu_dmax"] = (
                agg.loc[missing_dmax, "mu_dbar"] + fallback_t.reindex(agg.index)[missing_dmax]
            )

    if "r_mean" in agg.columns:
        agg["r_val"] = agg["r_mean"]
    elif "r_hat" in agg.columns:
        agg["r_val"] = agg["r_hat"]
    else:
        diff = np.exp(agg["mu_dmax"] - agg["mu_dbar"]) - 1.0
        agg["r_val"] = np.clip(1.0 / np.maximum(diff, 1e-3), 0.5, 10.0)

    if "nu_mean" in agg.columns:
        agg["n_val"] = np.exp(agg["nu_mean"])
    elif "nu_back" in agg.columns:
        agg["n_val"] = np.exp(agg["nu_back"])
    else:
        agg["n_val"] = np.nan

    return agg


def _load_3dep_transect_preview(site_no: str) -> list[Any] | None:
    """Return `[length_m, z0_m, dz_21_points]` for `site_no` if a valid 3DEP transect exists."""
    p = REPO_ROOT / "data" / "raw" / "ops_transects" / f"{site_no}.json"
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if raw.get("status") != "ok":
        return None
    samples = raw.get("samples", {})
    z_list = samples.get("z", [])
    geom = raw.get("geometry", {})
    if not isinstance(z_list, list) or len(z_list) < 1001:
        return None
    z_arr = np.asarray(z_list, dtype=np.float64)
    if not np.all(np.isfinite(z_arr)):
        return None
    z_center = float(z_arr[500])
    dz_sub = [round(float(z_arr[idx] - z_center), 1) for idx in FP_SAMPLE_INDICES]
    return [_r(geom.get("length"), 0), _r(z_center, 1), dz_sub]


def _win_entry(bm_df: pd.DataFrame, sid: str, prefix: str, digits: int) -> list[Any]:
    if sid not in bm_df.index:
        return [None, None, None, None, False]
    best_raw = bm_df.at[sid, f"{prefix}_best"]
    sec_raw = bm_df.at[sid, f"{prefix}_second"]
    rmse_raw = bm_df.at[sid, f"{prefix}_best_rmse"]
    mar_raw = bm_df.at[sid, f"{prefix}_margin"]
    dec_raw = bm_df.at[sid, f"{prefix}_decisive"]
    best_val = str(best_raw) if pd.notna(best_raw) else None
    sec_val = str(sec_raw) if pd.notna(sec_raw) else None
    dec_val = bool(dec_raw) if pd.notna(dec_raw) else False
    return [best_val, sec_val, _r(rmse_raw, digits), _r(mar_raw, digits), dec_val]


def build_conus_sites() -> dict[str, Any]:
    """Build compact dataset for all 3,988 CONUS USGS `HYDROSWOT` gauge reaches."""
    pred_dir = REPO_ROOT / "data" / "processed" / "predictions"
    site_df = pd.read_parquet(REPO_ROOT / "data" / "processed" / "site_table.parquet")
    site_df["site_no"] = site_df["site_no"].astype(str)
    prim = site_df.loc[(site_df["variant"] == "primary") & (~site_df["F_DA"])].copy()
    prim = prim.sort_values("site_no", kind="mergesort").reset_index(drop=True)

    # Fallback shape from M3 S-ID and S-LRO
    m3_df = pd.read_parquet(pred_dir / "M3.parquet")
    m3_sid = m3_df.loc[
        (m3_df["analysis"] == "primary") & (m3_df["family"] == "S-ID") & (m3_df["unit"] == "site")
    ]
    fb_sid = m3_sid.groupby("site_no")["mu_t"].mean()
    m3_slro = m3_df.loc[
        (m3_df["analysis"] == "primary") & (m3_df["family"] == "S-LRO") & (m3_df["unit"] == "site")
    ]
    fb_slro = m3_slro.groupby("site_no")["mu_t"].mean()

    preds_sid = {
        m: _load_model_site_predictions(pred_dir, m, "S-ID", fb_sid) for m in HEADLINE_MODELS
    }
    preds_slro = {
        m: _load_model_site_predictions(pred_dir, m, "S-LRO", fb_slro) for m in HEADLINE_MODELS
    }

    # Best-model tables
    bm_dir = REPO_ROOT / "docs" / "operational" / "best_model_map" / "tables"

    def _read_indexed(p: Path) -> pd.DataFrame:
        df = pd.read_parquet(p)
        if "site_no" in df.columns:
            df = df.set_index("site_no")
        df.index = df.index.astype(str)
        return df

    bm_sites = _read_indexed(bm_dir / "best_model_sites.parquet")
    wide_geom = _read_indexed(bm_dir / "best_model_wide_geom.parquet")
    wide_op_fp = _read_indexed(bm_dir / "best_model_wide_op_fp.parquet")
    wide_op_dyn = _read_indexed(bm_dir / "best_model_wide_op_dyn.parquet")

    # Off-anchor observation counts per site from ops1_observations
    ops1_obs = pd.read_parquet(REPO_ROOT / "reports" / "operational" / "ops1_observations.parquet")
    ops1_prim = ops1_obs.loc[ops1_obs["variant"] == "primary"]
    obs_counts = ops1_prim.groupby(ops1_prim["site_no"].astype(str)).size().to_dict()
    hf_counts = (
        ops1_prim.loc[ops1_prim["q_ratio"] >= 2.0]
        .groupby(ops1_prim["site_no"].astype(str))
        .size()
        .to_dict()
    )

    # S-TAIL test flags from fold_assignment
    fa = pd.read_parquet(REPO_ROOT / "data" / "processed" / "splits" / "fold_assignment.parquet")
    fa["site_no"] = fa["site_no"].astype(str)
    tail_da_sites = set(fa.loc[(fa["family"] == "S-TAIL-DA") & (fa["is_test"]), "site_no"].tolist())
    tail_so_sites = set(fa.loc[(fa["family"] == "S-TAIL-SO") & (fa["is_test"]), "site_no"].tolist())

    # Event 1 spatial audit strata (median across repeats per site)
    audit_strata = pd.read_parquet(REPO_ROOT / "reports" / "results" / "audit_strata.parquet")
    audit_strata["site_no"] = audit_strata["site_no"].astype(str)
    audit_site = audit_strata.groupby("site_no")[["nn_km", "psi_max"]].median()

    records: list[dict[str, Any]] = []
    for _, row in prim.iterrows():
        sid = str(row["site_no"])

        w_obs = float(row["W"])
        dbar_obs = float(row["dbar"])
        dmax_obs = float(row["dmax"])
        a_obs = float(row["A"])
        r_obs = float(np.clip(dbar_obs / max(dmax_obs - dbar_obs, 1e-4), 0.5, 10.0))

        model_pack: dict[str, Any] = {}
        for fam_key, pred_map in (("sid", preds_sid), ("slro", preds_slro)):
            fam_list: list[list[float | None] | None] = []
            for m in HEADLINE_MODELS:
                mdf = pred_map[m]
                if sid not in mdf.index:
                    fam_list.append(None)
                    continue
                entry: list[float | None] = [
                    _exp_r(mdf.at[sid, "mu_W"], 2),
                    _exp_r(mdf.at[sid, "mu_dbar"], 3),
                    _exp_r(mdf.at[sid, "mu_dmax"], 3),
                    _exp_r(mdf.at[sid, "mu_A"], 2),
                    _r(mdf.at[sid, "r_val"], 2),
                    _r(mdf.at[sid, "n_val"], 4),
                ]
                if m in UQ_MODELS and "q_lo_dbar" in mdf.columns and "q_hi_dbar" in mdf.columns:
                    entry.extend(
                        [
                            _exp_r(mdf.at[sid, "q_lo_W"], 2),
                            _exp_r(mdf.at[sid, "q_hi_W"], 2),
                            _exp_r(mdf.at[sid, "q_lo_dbar"], 3),
                            _exp_r(mdf.at[sid, "q_hi_dbar"], 3),
                        ]
                    )
                fam_list.append(entry)
            model_pack[fam_key] = fam_list

        rmse_geom = (
            [_r(wide_geom.at[sid, m], 3) for m in ALL_TEN_MODELS]
            if sid in wide_geom.index
            else None
        )
        rmse_fp = (
            [_r(wide_op_fp.at[sid, m], 3) for m in ALL_TEN_MODELS]
            if sid in wide_op_fp.index
            else None
        )
        rmse_dyn = (
            [_r(wide_op_dyn.at[sid, m], 3) for m in ALL_TEN_MODELS]
            if sid in wide_op_dyn.index
            else None
        )

        fp_preview = _load_3dep_transect_preview(sid) if rmse_fp is not None else None
        nn_km = _r(audit_site.at[sid, "nn_km"], 1) if sid in audit_site.index else None
        psi_max = _r(audit_site.at[sid, "psi_max"], 2) if sid in audit_site.index else None

        rec: dict[str, Any] = {
            "id": sid,
            "c": int(row["comid"]),
            "d": str(row["division"]),
            "la": _r(row["lat"], 4),
            "lo": _r(row["lon"], 4),
            "da": _r(row["DA_km2"], 1),
            "so": int(row["stream_order"]),
            "s": _r(row["S"], 6),
            "fs": 1 if bool(row["F_S"]) else 0,
            "qe": _r(row["Q_E"], 2),
            "ve": _r(row["V_E"], 2),
            "qo": _r(row["Q_obs"], 2),
            "d50": _r(float(row["D50_m"]) * 1000.0, 2),
            "ai": _r(row["AI"], 2),
            "ns": int(row["n_rows_site"]),
            "no": int(obs_counts.get(sid, 0)),
            "nh": int(hf_counts.get(sid, 0)),
            "tda": 1 if sid in tail_da_sites else 0,
            "tso": 1 if sid in tail_so_sites else 0,
            "nn": nn_km,
            "psi": psi_max,
            "obs": [
                _r(w_obs, 2),
                _r(dbar_obs, 3),
                _r(dmax_obs, 3),
                _r(a_obs, 2),
                _r(float(row["V"]), 2),
                _r(r_obs, 2),
            ],
            "w": [
                _win_entry(bm_sites, sid, "geom", 3),
                _win_entry(bm_sites, sid, "op_fp", 3),
                _win_entry(bm_sites, sid, "op_dyn", 3),
            ],
            "rm": [rmse_geom, rmse_fp, rmse_dyn],
            "p": model_pack,
            "fp": fp_preview,
        }
        records.append(rec)

    return {
        "schema": {
            "obs_fields": ["W_m", "dbar_m", "dmax_m", "A_m2", "V_ms", "r_obs"],
            "pred_fields_base": ["W_m", "dbar_m", "dmax_m", "A_m2", "r_hat", "n_hat"],
            "pred_fields_uq_extra": ["W_lo90_m", "W_hi90_m", "dbar_lo90_m", "dbar_hi90_m"],
            "win_order": ["geom", "op_fp", "op_dyn"],
            "win_fields": ["best", "second", "best_rmse", "margin", "decisive"],
            "rmse_order": ["geom", "op_fp", "op_dyn"],
            "rmse_model_order": list(ALL_TEN_MODELS),
            "headline_models": list(HEADLINE_MODELS),
            "fp_fields": ["length_m", "z_center_navd88_m", "dz_profile_m"],
            "fp_xi_coords": [round((idx - 500) / 500.0, 3) for idx in FP_SAMPLE_INDICES],
        },
        "n_sites": len(records),
        "sites": records,
    }


def build_ehydro_sites() -> dict[str, Any]:
    """Build JSON dataset for the 131 USACE `eHydro` (`S-IND`) locations across 18 rivers."""
    eh_df = pd.read_parquet(REPO_ROOT / "data" / "processed" / "ind" / "ehydro_ind.parquet")
    kept = eh_df.loc[eh_df["dedupe_status"].eq("kept")].copy()
    kept = kept.sort_values("loc_id", kind="mergesort").reset_index(drop=True)

    eh_feat = pd.read_parquet(
        REPO_ROOT / "data" / "processed" / "ind" / "ehydro_ind_features.parquet"
    )
    eh_feat_agg = eh_feat.groupby("loc_id", sort=False)[
        ["S", "F_S", "Q_E", "V_E", "D50_m", "AI", "division"]
    ].first()

    # Resolve outlet COMID centroids in EPSG:4326 from NHDPlusV2.1
    nhd_dir = REPO_ROOT / "data" / "raw" / "nhdplusv21"
    shp_by_vpu: dict[str, Path] = {}
    for p in sorted(nhd_dir.rglob("*")):
        if p.name.lower() == "nhdflowline.shp":
            vpu_match = _VPU_RE.search(p.as_posix())
            if vpu_match is not None:
                shp_by_vpu[vpu_match.group(1)] = p

    vpu_to_comids: dict[str, set[int]] = {}
    for _, r in kept.iterrows():
        c_out = int(r["comid_outlet"])
        for vpu in str(r["vpus"]).split(";"):
            vpu_to_comids.setdefault(vpu.strip(), set()).add(c_out)

    coords_by_comid: dict[int, tuple[float, float]] = {}
    for vpu, comid_set in sorted(vpu_to_comids.items()):
        shp_path = shp_by_vpu[vpu]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Measured \\(M\\) geometry types")
            fields = [str(f) for f in pyogrio.read_info(shp_path)["fields"]]
            comid_col = next(f for f in fields if f.upper() == "COMID")
            id_list = ",".join(str(c) for c in sorted(comid_set))
            sub_gdf = pyogrio.read_dataframe(
                shp_path,
                columns=[comid_col],
                where=f"{comid_col} IN ({id_list})",
            )
        if sub_gdf.empty:
            continue
        sub_gdf = sub_gdf.rename(columns={comid_col: "comid"})
        sub_gdf["comid"] = sub_gdf["comid"].astype("int64")
        sub_gdf["geometry"] = shapely.force_2d(sub_gdf.geometry.to_numpy())
        sub_4326 = sub_gdf.to_crs(CRS_WGS84)
        centroids = shapely.centroid(sub_4326.geometry.to_numpy())
        lons = shapely.get_x(centroids)
        lats = shapely.get_y(centroids)
        for cid, lon_val, lat_val in zip(sub_4326["comid"].to_numpy(), lons, lats, strict=True):
            coords_by_comid[int(cid)] = (float(lon_val), float(lat_val))

    pred_dir = REPO_ROOT / "data" / "processed" / "predictions"
    m3_df = pd.read_parquet(pred_dir / "M3.parquet")
    m3_sind = m3_df.loc[
        (m3_df["analysis"] == "primary")
        & (m3_df["family"] == "S-IND")
        & (m3_df["unit"] == "location")
    ]
    fb_sind = m3_sind.groupby("loc_id")["mu_t"].mean()

    sind_preds: dict[str, pd.DataFrame] = {}
    for m in HEADLINE_MODELS:
        df = pd.read_parquet(pred_dir / f"{m}.parquet")
        keep = (
            (df["analysis"] == "primary") & (df["family"] == "S-IND") & (df["unit"] == "location")
        )
        if "ablation" in df.columns:
            keep &= df["ablation"].fillna("").astype(str).eq("")
        sub = df.loc[keep].copy()
        sub["loc_id"] = sub["loc_id"].astype(str)
        num_cols = [
            c
            for c in (
                "mu_W",
                "mu_dbar",
                "mu_dmax",
                "mu_A",
                "mu_t",
                "q_lo_W",
                "q_hi_W",
                "q_lo_dbar",
                "q_hi_dbar",
                "nu_mean",
                "nu_back",
                "r_mean",
                "r_hat",
            )
            if c in sub.columns
        ]
        agg = sub.groupby("loc_id", sort=False)[num_cols].mean()
        if "mu_dmax" not in agg.columns:
            agg["mu_dmax"] = agg["mu_dbar"] + fb_sind.reindex(agg.index)
        else:
            miss = agg["mu_dmax"].isna()
            if miss.any():
                agg.loc[miss, "mu_dmax"] = (
                    agg.loc[miss, "mu_dbar"] + fb_sind.reindex(agg.index)[miss]
                )
        if "r_mean" in agg.columns:
            agg["r_val"] = agg["r_mean"]
        elif "r_hat" in agg.columns:
            agg["r_val"] = agg["r_hat"]
        else:
            agg["r_val"] = 2.0
        if "nu_mean" in agg.columns:
            agg["n_val"] = np.exp(agg["nu_mean"])
        elif "nu_back" in agg.columns:
            agg["n_val"] = np.exp(agg["nu_back"])
        else:
            agg["n_val"] = np.nan
        sind_preds[m] = agg

    valid_loc_ids = set(sind_preds["M5"].index)
    locations: list[dict[str, Any]] = []
    for _, row in kept.iterrows():
        lid = str(row["loc_id"])
        if lid not in valid_loc_ids:
            continue
        c_out = int(row["comid_outlet"])
        lon_val, lat_val = coords_by_comid.get(c_out, (0.0, 0.0))
        if lid in eh_feat_agg.index:
            div_str = str(eh_feat_agg.at[lid, "division"])
            s_val = _r(eh_feat_agg.at[lid, "S"], 6)
            fs_val = bool(eh_feat_agg.at[lid, "F_S"])
            qe_val = _r(eh_feat_agg.at[lid, "Q_E"], 2)
            ve_val = _r(eh_feat_agg.at[lid, "V_E"], 3)
            d50_raw = _r(eh_feat_agg.at[lid, "D50_m"], 8)
            d50_val = round(d50_raw * 1000.0, 2) if d50_raw is not None else None
            ai_val = _r(eh_feat_agg.at[lid, "AI"], 3)
        else:
            div_str = "USACE Navigation"
            s_val = None
            fs_val = False
            qe_val = None
            ve_val = None
            d50_val = None
            ai_val = None

        w_obs = float(row["W"])
        dbar_obs = float(row["dbar"])
        a_obs = float(row["A"])
        dmax_obs = dbar_obs * 1.5

        pred_dict: dict[str, Any] = {}
        best_a_model = None
        best_a_err = float("inf")
        for m in HEADLINE_MODELS:
            mdf = sind_preds[m]
            if lid not in mdf.index:
                continue
            mu_a_val = _r(mdf.at[lid, "mu_A"], 8)
            if mu_a_val is not None:
                a_err = abs(mu_a_val - math.log(a_obs))
                if a_err < best_a_err:
                    best_a_err = a_err
                    best_a_model = m
            entry: list[float | None] = [
                _exp_r(mdf.at[lid, "mu_W"], 2),
                _exp_r(mdf.at[lid, "mu_dbar"], 3),
                _exp_r(mdf.at[lid, "mu_dmax"], 3),
                _exp_r(mdf.at[lid, "mu_A"], 2),
                _r(mdf.at[lid, "r_val"], 2),
                _r(mdf.at[lid, "n_val"], 4),
            ]
            if m in UQ_MODELS and "q_lo_dbar" in mdf.columns and "q_hi_dbar" in mdf.columns:
                entry.extend(
                    [
                        _exp_r(mdf.at[lid, "q_lo_W"], 2),
                        _exp_r(mdf.at[lid, "q_hi_W"], 2),
                        _exp_r(mdf.at[lid, "q_lo_dbar"], 3),
                        _exp_r(mdf.at[lid, "q_hi_dbar"], 3),
                    ]
                )
            pred_dict[m] = entry

        locations.append(
            {
                "id": lid,
                "river": str(row["river"]),
                "comid": c_out,
                "n_comids": int(row["n_comids"]),
                "length_km": _r(row["length_km"], 2),
                "div": div_str,
                "lat": _r(lat_val, 4),
                "lon": _r(lon_val, 4),
                "da": _r(row["DA_km2"], 1),
                "s": s_val,
                "fs": fs_val,
                "qe": qe_val,
                "ve": ve_val,
                "d50": d50_val,
                "ai": ai_val,
                "single_comid_uq": int(row["n_comids"]) == 1,
                "best_area_model": best_a_model,
                "obs": [_r(w_obs, 2), _r(dbar_obs, 3), _r(dmax_obs, 3), _r(a_obs, 2), None, 2.0],
                "pred": pred_dict,
            }
        )

    return {
        "n_locations": len(locations),
        "n_rivers": len({loc["river"] for loc in locations}),
        "n_single_comid_uq": sum(1 for loc in locations if loc["single_comid_uq"]),
        "locations": locations,
    }


def _slim_ops_family(fam_obj: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(fam_obj, dict):
        return {}
    return {
        k: v
        for k, v in fam_obj.items()
        if k
        in (
            "n_common_sites",
            "n_observations",
            "stage_rmse_m",
            "stage_bias_m",
            "width_rmse_log",
            "ranking",
            "hypotheses",
            "control",
            "floodplain",
        )
    }


def build_benchmarks_bundle() -> tuple[dict[str, Any], dict[str, float]]:
    """Compile comprehensive benchmark, operational, QA/QC, and ledger bundle."""
    em = pd.read_parquet(REPO_ROOT / "reports" / "results" / "event3_matrix.parquet")
    em_prim = em.loc[
        (em["analysis"] == "primary")
        & (em["ablation"].fillna("").astype(str) == "")
        & (em["stratum_kind"] == "all")
    ].copy()

    metrics_matrix: list[dict[str, Any]] = []
    for (fam, mod, tgt), grp in em_prim.groupby(["family", "model", "target"], sort=True):
        mvals = {str(r["metric"]): _r(r["value"], 4) for _, r in grp.iterrows()}
        first = grp.iloc[0]
        metrics_matrix.append(
            {
                "family": str(fam),
                "model": str(mod),
                "target": str(tgt),
                "n_units": int(first["n_units"]),
                "n_rows": int(first["n_rows"]),
                "metrics": mvals,
            }
        )

    # Ablations and sensitivity analyses on S-ID and S-LRO for dbar, A, W
    em_abl = em.loc[
        (em["stratum_kind"] == "all")
        & (em["family"].isin(["S-ID", "S-LRO", "S-IND"]))
        & (em["target"].isin(["dbar", "A", "W"]))
        & (em["metric"].isin(["rmse_log", "mdape", "bias_log"]))
    ].copy()
    keep_mods = ("M5", "M2-XGB", "M2-MLP", "M4", "M6-NN")
    ablations_list: list[dict[str, Any]] = []
    for (spec, mod, abl, ana, fam, tgt), grp in em_abl.groupby(
        ["spec", "model", "ablation", "analysis", "family", "target"], sort=True
    ):
        if str(ana) == "primary" and str(abl) == "" and str(mod) not in keep_mods:
            continue
        mvals = {str(r["metric"]): _r(r["value"], 4) for _, r in grp.iterrows()}
        ablations_list.append(
            {
                "spec": str(spec),
                "model": str(mod),
                "ablation": str(abl) if str(abl) else "none",
                "analysis": str(ana),
                "family": str(fam),
                "target": str(tgt),
                "metrics": mvals,
            }
        )

    ev3 = json.loads(
        (REPO_ROOT / "reports" / "results" / "event3.json").read_text(encoding="utf-8")
    )
    ev3_hyp = json.loads(
        (REPO_ROOT / "reports" / "results" / "event3_hypotheses.json").read_text(encoding="utf-8")
    )
    ev3_uq = json.loads(
        (REPO_ROOT / "reports" / "results" / "event3_uq.json").read_text(encoding="utf-8")
    )
    ev3_cons = json.loads(
        (REPO_ROOT / "reports" / "results" / "event3_consistency.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (REPO_ROOT / "reports" / "results" / "audit.json").read_text(encoding="utf-8")
    )
    ledger = json.loads(
        (REPO_ROOT / "reports" / "results" / "scoring_ledger.json").read_text(encoding="utf-8")
    )
    mstar = json.loads(
        (REPO_ROOT / "reports" / "results" / "mstar.json").read_text(encoding="utf-8")
    )

    ops1 = json.loads(
        (REPO_ROOT / "reports" / "operational" / "ops1_results.json").read_text(encoding="utf-8")
    )
    ops2 = json.loads(
        (REPO_ROOT / "reports" / "operational" / "ops2_results.json").read_text(encoding="utf-8")
    )
    ops3 = json.loads(
        (REPO_ROOT / "reports" / "operational" / "ops3_results.json").read_text(encoding="utf-8")
    )

    qc_counts = json.loads((REPO_ROOT / "reports" / "qc_counts.json").read_text(encoding="utf-8"))
    link_counts = json.loads(
        (REPO_ROOT / "reports" / "link_counts.json").read_text(encoding="utf-8")
    )
    cond_counts = json.loads(
        (REPO_ROOT / "reports" / "condition_counts.json").read_text(encoding="utf-8")
    )
    feat_missing = json.loads(
        (REPO_ROOT / "reports" / "feature_missing.json").read_text(encoding="utf-8")
    )
    split_sum = json.loads(
        (REPO_ROOT / "reports" / "split_summary.json").read_text(encoding="utf-8")
    )

    shares_path = (
        REPO_ROOT
        / "docs"
        / "operational"
        / "best_model_map"
        / "tables"
        / "best_model_shares.parquet"
    )
    best_shares: list[dict[str, Any]] = []
    if shares_path.is_file():
        sh_df = pd.read_parquet(shares_path)
        for _, r in sh_df.iterrows():
            best_shares.append(
                {
                    "criterion": str(r["criterion"]),
                    "model": str(r["model"]),
                    "win_share": _r(r["win_share"], 4),
                    "near_best_share": _r(r["near_best_share"], 4),
                    "mean_regret": _r(r["mean_regret"], 4),
                    "pooled_rmse": _r(r["pooled_rmse"], 4),
                }
            )

    uq_summary: dict[str, Any] = {}
    for k in ("M5|primary", "M2-MLP|primary", "M4|primary"):
        if k in ev3_uq:
            uq_summary[k.split("|")[0]] = ev3_uq[k]

    gamma_by_div: dict[str, float] = {
        str(k): float(v)
        for k, v in ops3.get("dynamic_roughness", {}).get("gamma_lodo_by_division", {}).items()
    }

    v2_all = ops3.get("v2_all_models", {})
    v2_slim: dict[str, Any] = {}
    for split_k in ("S-ID", "S-LRO"):
        sp_obj = v2_all.get(split_k, {})
        v2_slim[split_k] = {
            "slope_stratification": sp_obj.get("slope_stratification"),
            "H-V3-1": sp_obj.get("H-V3-1"),
            "floodplain_high_cliques": (
                sp_obj.get("variants", {})
                .get("floodplain", {})
                .get("bins", {})
                .get("high", {})
                .get("equivalence_cliques")
            ),
            "floodplain_high_table": (
                sp_obj.get("variants", {})
                .get("floodplain", {})
                .get("bins", {})
                .get("high", {})
                .get("models")
            ),
            "control_high_table": (
                sp_obj.get("variants", {})
                .get("control", {})
                .get("bins", {})
                .get("high", {})
                .get("models")
            ),
        }

    bundle: dict[str, Any] = {
        "meta": {
            "package": "ann-xs-estimation",
            "version": "1.0.0+ops1.v3",
            "run_id": ev3.get("run_id"),
            "params_sha256": ev3.get("params_sha256"),
            "plan_sha256": ev3.get("plan_sha256"),
            "audit_label": ev3.get("audit_label"),
            "outcome_4_7": ev3.get("outcome_4_7"),
            "mstar": mstar,
        },
        "event1_audit": {
            "n_sites": audit.get("n_sites"),
            "d_nn_median_km": _r(audit.get("d_nn_median_km"), 2),
            "rho_hat_km": _r(audit.get("rho_hat_km"), 2),
            "rho_star_km": _r(audit.get("rho_star_km"), 2),
            "label": audit.get("label"),
            "far": audit.get("far"),
            "nested": audit.get("nested"),
            "series": audit.get("series"),
        },
        "metrics_matrix": metrics_matrix,
        "ablations": ablations_list,
        "hypotheses": {
            "H1": ev3_hyp.get("H1"),
            "H2": ev3_hyp.get("H2"),
            "H3": ev3_hyp.get("H3"),
            "H4": ev3_hyp.get("H4"),
            "S2": ev3_hyp.get("S2"),
            "outcome_table_4_7": ev3.get("outcome_table_4_7"),
        },
        "uq": {
            "alpha": ev3_uq.get("alpha"),
            "band": ev3_uq.get("band"),
            "models": uq_summary,
        },
        "consistency": {
            "continuity": ev3_cons.get("continuity"),
            "models": ev3_cons.get("models"),
        },
        "operational": {
            "ops1_v1": {
                "primary_s_id": _slim_ops_family(
                    ops1.get("variants", {}).get("primary", {}).get("S-ID")
                ),
                "primary_s_lro": _slim_ops_family(
                    ops1.get("variants", {}).get("primary", {}).get("S-LRO")
                ),
            },
            "ops2_floodplain": {
                "terrain_counts": ops2.get("terrain_counts"),
                "observation_counts": ops2.get("observation_counts"),
                "s_id": _slim_ops_family(ops2.get("families", {}).get("S-ID")),
                "s_lro": _slim_ops_family(ops2.get("families", {}).get("S-LRO")),
            },
            "ops3_diagnostics": {
                "v2_all_models": v2_slim,
                "dynamic_roughness": ops3.get("dynamic_roughness"),
                "off_anchor_states": ops3.get("off_anchor_states"),
                "cche1d_readiness": ops3.get("cche1d_readiness"),
            },
            "best_model_shares": best_shares,
        },
        "data_provenance": {
            "qc_counts": qc_counts,
            "link_counts": link_counts,
            "condition_counts": cond_counts,
            "feature_missing": feat_missing,
            "split_summary": split_sum,
            "scoring_ledger": ledger,
        },
    }
    return bundle, gamma_by_div


def main() -> None:
    """Entry point for generating the static WebGIS data bundle."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmarks, gamma_by_div = build_benchmarks_bundle()
    physio = build_physio_geojson(gamma_by_div)
    conus = build_conus_sites()
    ehydro = build_ehydro_sites()

    files_meta = [
        _write_json(OUT_DIR / "benchmarks.json", benchmarks),
        _write_json(OUT_DIR / "physio_divisions.geojson", physio),
        _write_json(OUT_DIR / "sites_conus.json", conus),
        _write_json(OUT_DIR / "sites_ehydro.json", ehydro),
    ]
    manifest = {
        "schema_version": 1,
        "package": "ann-xs-estimation-webgis",
        "n_conus_sites": conus["n_sites"],
        "n_ehydro_locations": ehydro["n_locations"],
        "n_physio_divisions": len(physio["features"]),
        "files": files_meta,
    }
    _write_json(OUT_DIR / "manifest.json", manifest)
    total_kb = sum(f["size_bytes"] for f in files_meta) / 1024.0
    print(
        f"WebGIS bundle built: {conus['n_sites']} CONUS sites, "
        f"{ehydro['n_locations']} eHydro locations, "
        f"{len(physio['features'])} divisions ({total_kb:.1f} KB total)"
    )


if __name__ == "__main__":
    main()

"""Stage unit_map: sentinels to missing, then the exact unit map U to SI (protocol 2.1).

Sources: protocol section 2.1 (exact factors; sentinels become missing **before** U and are
never scaled; NAD27 rows are transformed, never mixed), section 2.3 (delivered field names),
section 2.7 (no imputation); proofs section 1.2 (lambda_L = 0.3048 m/ft exactly).

Outputs (``data/processed/unit_mapped/``, one Parquet file per table, rows sorted by key):

* ``hydroswot.parquet``: every delivered HYDROSWOT row in SI units (:class:`HydroswotSI`). No
  row is removed here; QC (protocol 2.4) is the next stage.
* ``nhd_vaa.parquet``, ``nhd_erom.parquet``, ``nhd_elevslope.parquet``, ``nhd_gageloc.parquet``,
  ``nhd_plusflow.parquet``: the NHDPlusV2.1 tables concatenated over the 21 VPUs.

Each file carries the stage name, ``params_sha256`` and the SHA-256 of the input unit
manifests in its Parquet metadata (protocol 1.2). Torch-free (ADL-051).

Owning phase: 5.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.config import FT_TO_M, Settings, load_settings
from src.data.io import unit_manifest_sha256, write_parquet
from src.data.nhdplus import NON_NETWORK_DIVERGENCE, NON_NETWORK_ORDER, read_nhd_table
from src.data.readers import (
    HYDROSWOT_DT_FORMAT,
    HYDROSWOT_FILE,
    cast_datetime,
    cast_float,
    cast_int,
    mask_sentinels,
    read_hydroswot_text,
)
from src.data.schemas import (
    HYDROSWOT_FLAG_FIELDS,
    SENTINELS,
    HydroswotSI,
    NhdElevSlopeSI,
    NhdEromSI,
    NhdGageLocSI,
    NhdPlusFlowSI,
    NhdVaaSI,
)

__all__ = [
    "CFS_TO_CMS",
    "CM_TO_M",
    "FT2_TO_M2",
    "MI2_TO_KM2",
    "OUT_DIR_NAME",
    "main",
    "map_elevslope",
    "map_erom",
    "map_gageloc",
    "map_hydroswot",
    "map_plusflow",
    "map_vaa",
]

LOG = logging.getLogger(__name__)

#: Exact factors of protocol 2.1 (international foot and mile; all derived from 0.3048).
FT2_TO_M2: Final[float] = 0.09290304  # ft2 -> m2 = 0.3048**2
CFS_TO_CMS: Final[float] = 0.028316846592  # ft3/s -> m3/s = 0.3048**3
MI2_TO_KM2: Final[float] = 2.589988110336  # mi2 -> km2 = 1.609344**2
CM_TO_M: Final[float] = 0.01  # MinElevSmo is in centimetres (NHDPlusV2 User Guide)

OUT_DIR_NAME: Final[str] = "unit_mapped"
_DATUM_NAD83: Final[str] = "NAD83"
_DATUM_NAD27: Final[str] = "NAD27"

#: HYDROSWOT field -> (SI column, factor). Order fixes the output column order.
_HYDROSWOT_SCALE: Final[tuple[tuple[str, str, float], ...]] = (
    ("drain_area_va", "DA_site_km2", MI2_TO_KM2),
    ("q_va", "Q_obs", CFS_TO_CMS),
    ("meas_q_va", "Q_meas", CFS_TO_CMS),
    ("q_adj_va", "Q_adj", CFS_TO_CMS),
    ("stream_wdth_va", "W", FT_TO_M),
    ("xsec_area_va", "A", FT2_TO_M2),  # delivered name (protocol: xsecond_area_va; ADL)
    ("mean_depth_va", "dbar", FT_TO_M),
    ("max_depth_va", "dmax", FT_TO_M),
    ("mean_vel_va", "V", FT_TO_M),
    ("max_vel_va", "Vmax", FT_TO_M),
)
_HYDROSWOT_CODES: Final[tuple[str, ...]] = (
    "q_meas_td",
    "q_meas_qual_cd",
    "q_meas_used_fg",
    "q_meas_type_cd",
    "data_aging_cd",
    "chan_mat_cd",
    "chan_stability_cd",
    "base_flw_cd",
)


def _si(values: pd.Series, factor: float) -> tuple[pd.Series, int]:
    """Mask sentinels (V4) and only then scale by an exact factor (protocol 2.1)."""
    masked, n_sentinel = mask_sentinels(values, SENTINELS)
    return masked * factor, n_sentinel


def _to_nad83(lat: pd.Series, lon: pd.Series, datum: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return NAD83 coordinates: NAD27 rows are transformed, unknown datums become missing."""
    lat83 = lat.where(datum.eq(_DATUM_NAD83))
    lon83 = lon.where(datum.eq(_DATUM_NAD83))
    nad27 = datum.eq(_DATUM_NAD27) & lat.notna() & lon.notna()
    if nad27.any():
        from pyproj import Transformer

        tr = Transformer.from_crs("EPSG:4267", "EPSG:4269", always_xy=True)
        x, y = tr.transform(lon[nad27].to_numpy(), lat[nad27].to_numpy())
        lon83.loc[nad27] = x
        lat83.loc[nad27] = y
    return lat83, lon83


def map_hydroswot(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map the delivered HYDROSWOT text table to :class:`HydroswotSI` (no row removed).

    Parameters
    ----------
    raw
        Output of :func:`readers.read_hydroswot_text` (stripped text columns).

    Returns
    -------
    tuple
        The SI frame (validated) and counts of masked sentinels and datum handling.
    """
    counts: dict[str, int] = {}
    out = pd.DataFrame({"row_id": np.arange(len(raw), dtype="int64")}, index=raw.index)
    out["site_no"] = raw["site_no"].astype("str")
    out["q_meas_no"] = raw["q_meas_no"].astype("str")
    out["q_meas_dt"] = cast_datetime(raw["q_meas_dt"], HYDROSWOT_DT_FORMAT).values
    datum = raw["coord_datum_cd"].replace("", pd.NA).astype("string")
    out["site_tp_cd"] = raw["site_tp_cd"].replace("", pd.NA).astype("str")
    out["coord_datum_cd"] = datum.astype("str")
    lat, counts["sentinel_dec_lat_va"] = mask_sentinels(
        cast_float(raw["dec_lat_va"]).values, SENTINELS
    )
    lon, counts["sentinel_dec_long_va"] = mask_sentinels(
        cast_float(raw["dec_long_va"]).values, SENTINELS
    )
    out["lat"], out["lon"] = _to_nad83(lat, lon, datum.fillna(""))
    counts["datum_nad27_transformed"] = int(datum.eq(_DATUM_NAD27).sum())
    counts["datum_unknown_set_missing"] = int((~datum.isin([_DATUM_NAD83, _DATUM_NAD27])).sum())
    chan = cast_int(raw["q_meas_chan_nu"], "Int16").values
    out["q_meas_chan_nu"] = chan
    for field, column, factor in _HYDROSWOT_SCALE:
        out[column], counts[f"sentinel_{field}"] = _si(cast_float(raw[field]).values, factor)
    for code in _HYDROSWOT_CODES:
        out[code] = raw[code].replace("", pd.NA).astype("str")
    for flag in HYDROSWOT_FLAG_FIELDS:
        out[flag] = cast_int(raw[flag], "Int8").values
    out = out.reset_index(drop=True)[list(HydroswotSI.to_schema().columns)]
    return HydroswotSI.validate(out), counts


def map_vaa(vaa: pd.DataFrame) -> pd.DataFrame:
    """Map PlusFlowlineVAA (lengths already km, areas km2; non-network order -> missing)."""
    non_network = vaa["Divergence"].eq(NON_NETWORK_DIVERGENCE) & vaa["StreamOrde"].eq(
        NON_NETWORK_ORDER
    )
    out = pd.DataFrame(
        {
            "comid": vaa["ComID"],
            "vpu": vaa["vpu"].astype(str),
            "hydroseq": vaa["HydroSeq"],
            "up_hydroseq": vaa["UpHydroSeq"],
            "dn_hydroseq": vaa["DnHydroSeq"],
            "levelpath": vaa["LevelPathI"],
            "terminalpath": vaa["TerminalPa"],
            "startflag": vaa["StartFlag"],
            "terminalfl": vaa["TerminalFl"],
            "divergence": vaa["Divergence"],
            "non_network": non_network,
            "stream_order": vaa["StreamOrde"].where(~non_network).astype("Int8"),
            "pathlength_km": vaa["PathLength"],
            "arbolate_km": vaa["ArbolateSu"],
            "DA_km2": vaa["TotDASqKm"],
        }
    )
    return NhdVaaSI.validate(out.sort_values("comid", kind="mergesort").reset_index(drop=True))


def map_erom(erom: pd.DataFrame) -> pd.DataFrame:
    """Map EROM_MA0001: ``Q_E`` and ``Q_C`` cfs -> m3/s, ``V_E`` ft/s -> m/s."""
    out = pd.DataFrame(
        {
            "comid": erom["ComID"],
            "Q_E": erom["Q0001E"] * CFS_TO_CMS,
            "Q_C": erom["Q0001C"] * CFS_TO_CMS,
            "V_E": erom["V0001E"] * FT_TO_M,
        }
    )
    return NhdEromSI.validate(out.sort_values("comid", kind="mergesort").reset_index(drop=True))


def map_elevslope(elev: pd.DataFrame, s_floor: float) -> pd.DataFrame:
    """Map ElevSlope: ``S`` (m/m, factor 1), ``z_min_m`` = MinElevSmo (cm) x 0.01, flag ``F_S``."""
    out = pd.DataFrame(
        {
            "comid": elev["ComID"],
            "S": elev["Slope"],
            "z_min_m": elev["MinElevSmo"] * CM_TO_M,
            "F_S": elev["Slope"].eq(s_floor),
        }
    )
    return NhdElevSlopeSI.validate(
        out.sort_values("comid", kind="mergesort").reset_index(drop=True)
    )


def map_gageloc(gage: pd.DataFrame) -> pd.DataFrame:
    """Map GageLoc: ``site_no`` = ``Source_Fea`` (string, as delivered), ``comid`` = ``FLComID``."""
    out = pd.DataFrame({"site_no": gage["Source_Fea"].astype(str), "comid": gage["FLComID"]})
    return NhdGageLocSI.validate(
        out.sort_values("site_no", kind="mergesort").reset_index(drop=True)
    )


def map_plusflow(flow: pd.DataFrame) -> pd.DataFrame:
    """Map PlusFlow to an edge list (rows sorted by ``from_comid``, ``to_comid``)."""
    out = pd.DataFrame({"from_comid": flow["FromComID"], "to_comid": flow["ToComID"]})
    out = out.sort_values(["from_comid", "to_comid"], kind="mergesort").reset_index(drop=True)
    return NhdPlusFlowSI.validate(out)


def _require_valid(reports_dir: Path, units: Sequence[str]) -> None:
    report = json.loads((reports_dir / "validate_raw.json").read_text(encoding="utf-8"))
    failed = [u for u in units if report["units"].get(u, {}).get("status") != "pass"]
    if failed:
        raise SystemExit(f"validate_raw has not passed for {failed}")


def run(settings: Settings) -> None:
    """Write every ``unit_mapped`` table."""
    rt = settings.runtime
    _require_valid(rt.reports_dir, ("hydroswot", "nhdplusv21"))
    out_dir = rt.processed_dir / OUT_DIR_NAME
    hs_dir, nhd_dir = rt.raw_dir / "hydroswot", rt.raw_dir / "nhdplusv21"
    sha = settings.params_sha256
    hydro, counts = map_hydroswot(read_hydroswot_text(hs_dir / HYDROSWOT_FILE))
    LOG.info("hydroswot: %d rows; %s", len(hydro), counts)
    write_parquet(
        hydro,
        out_dir / "hydroswot.parquet",
        stage="unit_map",
        params_sha256=sha,
        inputs={"hydroswot/MANIFEST.sha256": unit_manifest_sha256(hs_dir)},
        sort_by=["row_id"],
    )
    nhd_inputs = {"nhdplusv21/MANIFEST.sha256": unit_manifest_sha256(nhd_dir)}
    s_floor = settings.protocol.domain.s_floor
    for name in ("vaa", "erom", "elevslope", "gageloc", "plusflow"):
        raw, _ = read_nhd_table(nhd_dir, name)
        if name == "vaa":
            frame = map_vaa(raw)
        elif name == "erom":
            frame = map_erom(raw)
        elif name == "elevslope":
            frame = map_elevslope(raw, s_floor)
        elif name == "gageloc":
            frame = map_gageloc(raw)
        else:
            frame = map_plusflow(raw)
        LOG.info("nhd_%s: %d rows", name, len(frame))
        write_parquet(
            frame,
            out_dir / f"nhd_{name}.parquet",
            stage="unit_map",
            params_sha256=sha,
            inputs=nhd_inputs,
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Run ``python -m src.data.unit_map`` (the dvc.yaml ``unit_map`` stage)."""
    argparse.ArgumentParser(description="Unit map to SI (protocol 2.1).").parse_args(argv)
    settings = load_settings()
    logging.basicConfig(
        level=settings.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    LOG.info("params_sha256=%s", settings.params_sha256)
    run(settings)


if __name__ == "__main__":
    main()

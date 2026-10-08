"""Stage validate_raw: rules V1-V5 on every raw unit (protocol section 2.3).

Sources: protocol section 1.1 (manifest and publisher checksums), section 2.3 (raw schemas and
rules V1-V5), section 2.7 (missing data are never imputed); ADL-058 (``MANIFEST.json`` and
``MANIFEST.sha256`` written by ``acquire``).

Rules (common to every unit):

* **V1** every downloaded file is present with the size and SHA-256 recorded at acquisition
  (and the publisher md5 where one is pinned); every extracted member is present with its
  recorded size.
* **V2** every required field is present and casts losslessly to its stated type.
* **V3** the join key is unique, or its multiplicity is logged where the protocol allows it.
* **V4** sentinel codes become missing **before** any value rule is counted.
* **V5** counts of missing and out-of-rule values are written to ``reports/validate_raw.json``.

A unit fails only on V1, V2 or V3. The report is always written; the stage then exits with
status 1 if any unit failed, so that no downstream stage runs on an invalid unit. Nothing is
written to ``data/``.

Owning phase: 5.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd

from src.config import Settings, load_settings
from src.data.io import sha256_file, write_json
from src.data.nhdplus import validate_nhdplus
from src.data.readers import (
    HYDROSWOT_DT_FORMAT,
    HYDROSWOT_FILE,
    RawSchemaError,
    cast_datetime,
    cast_float,
    cast_int,
    mask_sentinels,
    read_attr_table,
    read_dbf,
    read_hydroswot_text,
)
from src.data.schemas import (
    HYDROSWOT_CODE_FIELDS,
    HYDROSWOT_FLAG_FIELDS,
    HYDROSWOT_NUMERIC_FIELDS,
    HYDROSWOT_REQUIRED,
    SENTINELS,
)

__all__ = [
    "ARIDITY_FILL",
    "ARIDITY_SCALE",
    "D50_FIELD",
    "REPORT_NAME",
    "USGS_ATTR_TABLES",
    "UnitReport",
    "check_manifest",
    "main",
    "streamcat_fields",
    "validate_all",
]

LOG = logging.getLogger(__name__)

REPORT_NAME: Final[str] = "validate_raw.json"

#: Raw units validated by this stage (dvc.yaml ``validate_raw`` deps); hyg and zarrabi_product
#: are validated at first use (protocol 2.3), the eHydro units at ``ind_prepare``.
VALIDATED_UNITS: Final[tuple[str, ...]] = (
    "hydroswot",
    "nhdplusv21",
    "streamcat",
    "usgs_attrs",
    "d50",
    "aridity",
    "physiography",
)

#: Global-AI v3.1 readme: "Global-AI values need to be multiplied by 0.0001".
ARIDITY_SCALE: Final[float] = 1.0e-4
#: The GeoTIFF declares no nodata value. 0 is the ocean fill (AI must be > 0, protocol 2.3) and
#: 65535 is the uint16 fill; both are excluded from the zonal mean (Batch C finding).
ARIDITY_FILL: Final[tuple[int, ...]] = (0, 65535)
#: Bounding box (lon/lat, degrees) used for the CONUS statistics of the aridity raster.
CONUS_BBOX: Final[tuple[float, float, float, float]] = (-125.0, 24.0, -66.0, 50.0)

#: D50 value field as delivered; the unit is in the field name (mm; converted to m in features).
D50_FIELD: Final[str] = "D50[mm]"

#: usgs_attrs tables (ScienceBase child items) -> required metric columns (Zarrabi Table 1 for M1:
#: dams, 2010 population density, seasonal EVI x 4, soil texture, developed/forest/agriculture).
USGS_ATTR_TABLES: Final[Mapping[str, tuple[str, ...]]] = {
    "EVI_AMJ_2012_CONUS": ("CAT_EVI_AMJ_2012",),
    "EVI_JAS_2012_CONUS": ("CAT_EVI_JAS_2012",),
    "EVI_JFM_2012_CONUS": ("CAT_EVI_JFM_2012",),
    "EVI_OND_2011_CONUS": ("CAT_EVI_OND_2011",),
    "NID_2010_CONUS": ("ACC_NDAMS2010",),
    "POPDENS10_CONUS": ("CAT_POPDENS10",),
    "STATSGO_TEXT_CAT_CONUS": ("CAT_SILTAVE", "CAT_CLAYAVE", "CAT_SANDAVE"),
    "NLCD16_CAT_CONUS": tuple(
        f"CAT_NLCD16_{c}" for c in ("21", "22", "23", "24", "41", "42", "43", "81", "82")
    ),
}

_SITE_NO_RE: Final[str] = r"^\d{8,15}$"
_HYDROSWOT_POSITIVE: Final[tuple[str, ...]] = (
    "q_va",
    "meas_q_va",
    "stream_wdth_va",
    "xsec_area_va",
    "mean_depth_va",
    "max_depth_va",
    "mean_vel_va",
    "max_vel_va",
)


@dataclass
class UnitReport:
    """Result of V1-V5 for one raw unit."""

    unit: str
    failures: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def fail(self, rule: str, message: str) -> None:
        """Record a V1, V2 or V3 failure."""
        if rule not in ("V1", "V2", "V3"):
            raise ValueError(f"only V1-V3 fail a unit, got {rule}")
        self.failures.append(f"{rule}: {message}")

    @property
    def passed(self) -> bool:
        """``True`` when no V1-V3 failure was recorded."""
        return not self.failures

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready report entry."""
        return {"status": "pass" if self.passed else "fail", "failures": self.failures, **self.info}


# ----------------------------------------------------------------------------- V1
def check_manifest(unit_dir: Path, report: UnitReport) -> None:
    """V1: re-hash every downloaded file and check every extracted member's size."""
    json_path = unit_dir / "MANIFEST.json"
    sha_path = unit_dir / "MANIFEST.sha256"
    if not json_path.is_file() or not sha_path.is_file():
        report.fail("V1", f"{unit_dir.name}: MANIFEST.json or MANIFEST.sha256 missing")
        return
    records: list[dict[str, Any]] = json.loads(json_path.read_text(encoding="utf-8"))["files"]
    listed = {}
    for line in sha_path.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        listed[name] = digest
    if listed != {str(rec["name"]): str(rec["sha256"]) for rec in records}:
        report.fail("V1", "MANIFEST.sha256 and MANIFEST.json disagree")
    n_bytes = n_members = 0
    for rec in records:
        path = unit_dir / str(rec["name"])
        if not path.is_file():
            report.fail("V1", f"missing file {rec['name']}")
            continue
        if path.stat().st_size != rec["size"]:
            report.fail("V1", f"{rec['name']}: size {path.stat().st_size} != {rec['size']}")
        elif sha256_file(path) != rec["sha256"]:
            report.fail("V1", f"{rec['name']}: SHA-256 differs from acquisition")
        if rec.get("md5_publisher") and rec.get("md5") != rec["md5_publisher"]:
            report.fail("V1", f"{rec['name']}: md5 differs from the publisher value")
        n_bytes += int(rec["size"])
        for member in rec.get("extracted", []):
            n_members += 1
            mpath = unit_dir / str(member["path"])
            if not mpath.is_file() or mpath.stat().st_size != member["size"]:
                report.fail("V1", f"extracted member missing or resized: {member['path']}")
    report.info["v1"] = {"files": len(records), "bytes": n_bytes, "extracted": n_members}


# ----------------------------------------------------------------------------- helpers
def _missing_counts(frame: pd.DataFrame, columns: Iterable[str]) -> dict[str, int]:
    return {c: int(frame[c].isna().sum()) for c in columns}


def _value_counts(values: pd.Series) -> dict[str, int]:
    counts = values.fillna("<NA>").astype(str).value_counts()
    return {str(k): int(v) for k, v in sorted(counts.items())}


def _key_multiplicity(frame: pd.DataFrame, key: Sequence[str]) -> dict[str, int]:
    dup = frame.duplicated(list(key), keep=False)
    return {
        "rows_in_duplicated_keys": int(dup.sum()),
        "duplicated_keys": int(frame.loc[dup, list(key)].drop_duplicates().shape[0]),
    }


# ----------------------------------------------------------------------------- hydroswot
def validate_hydroswot(unit_dir: Path, settings: Settings, report: UnitReport) -> None:
    """V2-V5 for ``SWOT_ADCP_Dataset.txt``.

    V3: ``(site_no, q_meas_no)`` is not unique as delivered (multi-channel parts share it), so
    the multiplicity is logged; QC R3 and R11 resolve it (protocol 2.4).
    """
    del settings
    raw = read_hydroswot_text(unit_dir / HYDROSWOT_FILE)
    absent = [c for c in HYDROSWOT_REQUIRED if c not in raw.columns]
    if absent:
        report.fail("V2", f"required fields absent: {absent}")
        return
    bad: dict[str, int] = {}
    frame = pd.DataFrame(index=raw.index)
    for col in HYDROSWOT_NUMERIC_FIELDS:
        res = cast_int(raw[col], "Int16") if col == "q_meas_chan_nu" else cast_float(raw[col])
        frame[col], bad[col] = res.values, res.n_bad
    for col in HYDROSWOT_FLAG_FIELDS:
        res = cast_int(raw[col], "Int8")
        frame[col], bad[col] = res.values, res.n_bad
    dt = cast_datetime(raw["q_meas_dt"], HYDROSWOT_DT_FORMAT)
    frame["q_meas_dt"], bad["q_meas_dt"] = dt.values, dt.n_bad
    site_ok = raw["site_no"].str.fullmatch(_SITE_NO_RE)
    bad["site_no"] = int((~site_ok).sum())
    bad["q_meas_no"] = int(raw["q_meas_no"].eq("").sum())
    lossy = {k: v for k, v in sorted(bad.items()) if v}
    if lossy:
        report.fail("V2", f"lossy casts: {lossy}")

    # V4: sentinels -> missing before any value rule.
    sentinels: dict[str, int] = {}
    for col in HYDROSWOT_NUMERIC_FIELDS:
        frame[col], sentinels[col] = mask_sentinels(frame[col], SENTINELS)

    # V3: logged multiplicity (allowed by the protocol for this unit).
    keyed = pd.DataFrame({"site_no": raw["site_no"], "q_meas_no": raw["q_meas_no"]})
    multiplicity = _key_multiplicity(keyed, ["site_no", "q_meas_no"])
    chan = frame["q_meas_chan_nu"]
    dup = keyed.duplicated(keep=False)
    multiplicity["duplicated_rows_with_chan_ge_1"] = int((dup & chan.ge(1).fillna(False)).sum())

    # V5: missing and out-of-rule counts (diagnostics; never a failure).
    retrieved = _retrieved_utc(unit_dir, HYDROSWOT_FILE)
    out_of_rule = {
        f"{c}_le_0": int(frame[c].le(0).fillna(False).sum()) for c in _HYDROSWOT_POSITIVE
    }
    out_of_rule["lat_out_of_range"] = int((frame["dec_lat_va"].abs() > 90).sum())
    out_of_rule["lon_out_of_range"] = int((frame["dec_long_va"].abs() > 180).sum())
    out_of_rule["flags_not_in_0_-1"] = int(
        sum((~frame[c].isin([0, -1]) & frame[c].notna()).sum() for c in HYDROSWOT_FLAG_FIELDS)
    )
    out_of_rule["chan_nu_ge_2"] = int(chan.ge(2).fillna(False).sum())
    out_of_rule["q_meas_dt_after_retrieval"] = int(
        (frame["q_meas_dt"] > pd.Timestamp(retrieved).tz_localize(None)).sum()
    )
    report.info.update(
        {
            "rows": len(raw),
            "columns": int(raw.shape[1]),
            "sites": int(raw["site_no"].nunique()),
            "v3_multiplicity": multiplicity,
            "v4_sentinels": sentinels,
            "v5_missing": _missing_counts(frame, [*HYDROSWOT_NUMERIC_FIELDS, "q_meas_dt"])
            | {c: int(raw[c].eq("").sum()) for c in HYDROSWOT_CODE_FIELDS},
            "v5_out_of_rule": out_of_rule,
            "q_meas_dt_range": [str(frame["q_meas_dt"].min()), str(frame["q_meas_dt"].max())],
            "codes": {
                c: _value_counts(raw[c].replace("", pd.NA))
                for c in (
                    "coord_datum_cd",
                    "site_tp_cd",
                    "q_meas_qual_cd",
                    "q_meas_used_fg",
                    "data_aging_cd",
                )
            },
        }
    )


def _retrieved_utc(unit_dir: Path, name: str) -> str:
    records = json.loads((unit_dir / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    return str(next(r["retrieved_utc"] for r in records if r["name"] == name))


# ----------------------------------------------------------------------------- attribute tables
def _validate_attr_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
    key: str,
    report: UnitReport,
    label: str,
    *,
    sentinels: Iterable[float] = SENTINELS,
) -> dict[str, Any]:
    if frame[key].duplicated().any():
        report.fail("V3", f"{label}: {key} not unique ({int(frame[key].duplicated().sum())})")
    masked: dict[str, int] = {}
    for col in columns:
        frame[col], masked[col] = mask_sentinels(frame[col], sentinels)
    return {
        "rows": len(frame),
        "v4_sentinels": masked,
        "v5_missing": _missing_counts(frame, columns),
        "v5_negative": {c: int(frame[c].lt(0).sum()) for c in columns},
        "key_le_0": int(frame[key].le(0).sum()),
    }


def validate_usgs_attrs(unit_dir: Path, settings: Settings, report: UnitReport) -> None:
    """V2-V5 for the ScienceBase NHDPlusV2 attribute tables (key ``COMID``)."""
    del settings
    tables: dict[str, Any] = {}
    for table, columns in USGS_ATTR_TABLES.items():
        hits = sorted(p for p in (unit_dir / table).glob("*") if p.suffix.lower() == ".txt")
        if len(hits) != 1:
            report.fail("V2", f"{table}: expected one .txt table, found {len(hits)}")
            continue
        try:
            frame, _ = read_attr_table(hits[0], columns)
        except RawSchemaError as exc:
            report.fail("V2", f"{table}: {exc}")
            continue
        tables[table] = _validate_attr_frame(frame, columns, "COMID", report, table)
    report.info["tables"] = tables


def streamcat_fields(settings: Settings) -> tuple[str, ...]:
    """StreamCat metric fields as delivered: each ``name=`` metric with the ``ws`` suffix.

    The metric list has one source, the ``acquire.streamcat`` URLs in ``params.yaml``.
    """
    names: set[tuple[str, ...]] = set()
    for url in settings.protocol.acquire.streamcat.urls:
        query = parse_qs(urlsplit(str(url)).query)
        names.add(tuple(query["name"][0].split(",")))
    if len(names) != 1:
        raise ValueError("acquire.streamcat URLs disagree on the metric list")
    return tuple(f"{name}ws" for name in next(iter(names)))


def validate_streamcat(unit_dir: Path, settings: Settings, report: UnitReport) -> None:
    """V2-V5 for the StreamCat regional API responses (key ``comid``, unique nationally)."""
    metrics = streamcat_fields(settings)
    parts: list[pd.DataFrame] = []
    per_region: dict[str, int] = {}
    for path in sorted(unit_dir.glob("streamcat_*.json")):
        items = json.loads(path.read_text(encoding="utf-8"))["items"]
        frame = pd.DataFrame.from_records(items)
        absent = [c for c in ("comid", *metrics) if c not in frame.columns]
        if absent:
            report.fail("V2", f"{path.name}: fields absent: {absent}")
            continue
        try:
            comid = frame["comid"].astype("int64")
            values = frame[list(metrics)].astype("float64")
        except (TypeError, ValueError) as exc:
            report.fail("V2", f"{path.name}: lossy cast ({exc})")
            continue
        parts.append(pd.concat([comid, values], axis=1))
        per_region[path.stem.removeprefix("streamcat_")] = len(frame)
    if not parts:
        report.fail("V2", "no StreamCat response could be read")
        return
    allrows = pd.concat(parts, ignore_index=True)
    summary = _validate_attr_frame(allrows, metrics, "comid", report, "streamcat")
    report.info.update({"regions": per_region, **summary})


def validate_d50(unit_dir: Path, settings: Settings, report: UnitReport) -> None:
    """V2-V5 for the D50 table (key ``COMID``; value ``D50[mm]``, millimetres as delivered)."""
    del settings
    hits = sorted(p for p in unit_dir.glob("*") if p.suffix.lower() == ".dbf")
    if len(hits) != 1:
        report.fail("V2", f"expected one .dbf, found {len(hits)}")
        return
    try:
        frame = read_dbf(hits[0], ["COMID", D50_FIELD])
        frame["COMID"] = frame["COMID"].astype("int64")
        frame[D50_FIELD] = frame[D50_FIELD].astype("float64")
    except (RawSchemaError, TypeError, ValueError) as exc:
        report.fail("V2", str(exc))
        return
    summary = _validate_attr_frame(frame, [D50_FIELD], "COMID", report, "d50")
    summary["v5_le_0"] = int(frame[D50_FIELD].le(0).sum())
    summary["range_mm"] = [float(frame[D50_FIELD].min()), float(frame[D50_FIELD].max())]
    report.info.update(summary)


# ----------------------------------------------------------------------------- rasters, polygons
def validate_aridity(unit_dir: Path, settings: Settings, report: UnitReport) -> None:
    """V2-V5 for the Global-AI v3.1 annual raster (single band, scaled integer)."""
    del settings
    import rasterio
    from rasterio.windows import from_bounds

    hits = sorted(unit_dir.rglob("ai_v31_yr.tif"))
    if len(hits) != 1:
        report.fail("V2", f"expected one ai_v31_yr.tif, found {len(hits)}")
        return
    with rasterio.open(hits[0]) as src:
        if src.count != 1 or not np.issubdtype(np.dtype(src.dtypes[0]), np.integer):
            report.fail("V2", f"expected one integer band, got {src.count} {src.dtypes}")
            return
        if src.crs is None:
            report.fail("V2", "raster has no CRS")
            return
        window = from_bounds(*CONUS_BBOX, transform=src.transform)
        band = src.read(1, window=window)
        report.info.update(
            {
                "crs": src.crs.to_string(),
                "dtype": src.dtypes[0],
                "declared_nodata": src.nodata,
                "res_deg": list(src.res),
                "scale": ARIDITY_SCALE,
            }
        )
    fill = np.isin(band, ARIDITY_FILL)
    valid = band[~fill].astype("float64") * ARIDITY_SCALE
    report.info["conus_window"] = {
        "pixels": int(band.size),
        "fill_pixels": {str(v): int((band == v).sum()) for v in ARIDITY_FILL},
        "ai_min": float(valid.min()) if valid.size else None,
        "ai_max": float(valid.max()) if valid.size else None,
        "ai_mean": float(valid.mean()) if valid.size else None,
    }


def validate_physiography(unit_dir: Path, settings: Settings, report: UnitReport) -> None:
    """V2-V5 for the Fenneman & Johnson polygons (``DIVISION`` field; valid geometry)."""
    del settings
    import geopandas as gpd

    hits = sorted(unit_dir.rglob("physio.shp"))
    if len(hits) != 1:
        report.fail("V2", f"expected one physio.shp, found {len(hits)}")
        return
    gdf = gpd.read_file(hits[0])
    if "DIVISION" not in gdf.columns or gdf.crs is None:
        report.fail("V2", "DIVISION field or CRS missing")
        return
    division = gdf["DIVISION"].astype("string").str.strip().replace("", pd.NA)
    invalid = ~gdf.geometry.is_valid
    report.info.update(
        {
            "polygons": len(gdf),
            "crs": gdf.crs.to_string(),
            "v3_multiplicity": _key_multiplicity(gdf, ["PHYSIODD_I"])
            if "PHYSIODD_I" in gdf.columns
            else None,
            "v5_invalid_geometry": int(invalid.sum()),
            "v5_missing_division": int(division.isna().sum()),
            "divisions": _value_counts(division.dropna()),
        }
    )


Checker = Callable[[Path, Settings, UnitReport], None]


CHECKERS: Final[Mapping[str, Checker]] = {
    "hydroswot": validate_hydroswot,
    "nhdplusv21": validate_nhdplus,
    "streamcat": validate_streamcat,
    "usgs_attrs": validate_usgs_attrs,
    "d50": validate_d50,
    "aridity": validate_aridity,
    "physiography": validate_physiography,
}


# ----------------------------------------------------------------------------- driver
def validate_all(
    settings: Settings, units: Sequence[str] = VALIDATED_UNITS
) -> dict[str, UnitReport]:
    """Run V1 then V2-V5 on each unit; a V1 failure skips the content checks of that unit."""
    reports: dict[str, UnitReport] = {}
    for unit in units:
        unit_dir = settings.runtime.raw_dir / unit
        report = UnitReport(unit)
        LOG.info("validate_raw: %s", unit)
        check_manifest(unit_dir, report)
        if report.passed:
            CHECKERS[unit](unit_dir, settings, report)
        reports[unit] = report
    return reports


def main(argv: Sequence[str] | None = None) -> None:
    """Run ``python -m src.data.validate_raw`` (the dvc.yaml ``validate_raw`` stage)."""
    parser = argparse.ArgumentParser(description="Validate raw units (protocol 2.3, V1-V5).")
    parser.add_argument("--unit", action="append", choices=VALIDATED_UNITS)
    args = parser.parse_args(argv)
    settings = load_settings()
    logging.basicConfig(
        level=settings.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    LOG.info("params_sha256=%s", settings.params_sha256)
    reports = validate_all(settings, tuple(args.unit) if args.unit else VALIDATED_UNITS)
    payload = {
        "params_sha256": settings.params_sha256,
        "units": {name: rep.as_dict() for name, rep in reports.items()},
        "failed": sorted(name for name, rep in reports.items() if not rep.passed),
    }
    write_json(settings.runtime.reports_dir / REPORT_NAME, payload)
    if payload["failed"]:
        LOG.error("validate_raw failed: %s", payload["failed"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""eHydro (T5): catalogue, survey selection, depth surface, reach values and locations.

Sources: docs/3 section 11.3, rules E-R1 to E-R12, pre-registered at commit ``07e00d6`` before
any eHydro record was read; section 1.1 (ehydro, O-5 (c)), 3.2 (S-IND), 3.5 (L8). Zarrabi et al.
(2025) section 2.5: d̄ is the mean of the depth pixels in the NHDPlusV2.1 catchment,
A = Σ(depth × pixel area) / reach length and W = A / d̄. Reads params ``ehydro``.

This module holds:

- ``python -m src.data.ehydro select``: the catalogue query (attributes and footprints only),
  eligibility (E-R2), selection (E-R3) and the HEAD size pilot (E-R4). It writes
  ``reports/ehydro_selection.json``, and the raw catalogue pages (with SHA-256) under
  ``<dvc_store>/ehydro_catalogue/<UTC>/``, outside the repository and Google Drive.
- The survey reader (E-R5 (i), docs/3 section 11.4 E-R5a-E-R5d, commit ``708f832``): the
  reference-plane class from the metadata files, the ``SurveyPoint`` depths and the
  ``SurveyJob`` footprint of the delivered geodatabase.
- The depth surface on a fixed grid (E-R5 (ii)-(iv)), pooling (E-R6), reach values (E-R6),
  coverage (E-R7) and locations (E-R8).

The stage itself is :mod:`src.data.ind_prepare`. Reach values are computed for frame flowlines
only (E-R1): cells in the catchment of a tributary with another name are not used.

No clamping: the two masks applied (outside the interpolation hull, depth ≤ 0) are the
pre-registered E-R5 (iii) and (iv). Torch-free.

Owning phase: 6 (P6-D10).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from geopandas import GeoDataFrame
    from shapely import Geometry

    from src.config import EhydroParams, Settings

__all__ = [
    "CATALOGUE_FIELDS",
    "FOOT_UNITS",
    "GEO_RE",
    "PLANE_CLASSES",
    "STATEMENT_RE",
    "WATER_RE",
    "Cells",
    "EhydroError",
    "EhydroStop",
    "PlaneDecision",
    "SurveyPoints",
    "build_locations",
    "catalogue_frame",
    "catalogue_where",
    "classify_plane_text",
    "classify_survey",
    "condition_codes",
    "coverage_fraction",
    "decide_plane",
    "depth_surface",
    "eligible_surveys",
    "head_sizes",
    "main",
    "plane_statement",
    "pool_cells",
    "query_catalogue",
    "reach_values",
    "read_frame_flowlines",
    "read_survey",
    "requests_get",
    "requests_head",
    "run_select",
    "select_latest",
    "survey_field_datums",
    "survey_gdb",
]

LOG = logging.getLogger(__name__)

#: SurveyJob attributes requested (field list read from the layer schema; no records).
CATALOGUE_FIELDS: Final[tuple[str, ...]] = (
    "surveyjobidpk",
    "sdsfeaturename",
    "surveytype",
    "channelareaidfk",
    "usacedistrictcode",
    "surveydatestart",
    "surveydateend",
    "sourcedatalocation",
    "sourceprojection",
    "projectedarea",
)
#: ``maxRecordCount`` of the SurveyJob layer (layer schema).
PAGE_SIZE: Final[int] = 2000
TIMEOUT_S: Final[int] = 120
_RETRIES: Final[int] = 4
_RETRY_STATUS: Final[frozenset[int]] = frozenset({408, 429, 500, 502, 503, 504})
#: E-R2 (ii): a condition-survey code is "CS" or contains "condition" (case-insensitive).
_CONDITION_RE: Final[re.Pattern[str]] = re.compile(r"^\s*(cs|.*condition.*)\s*$", re.IGNORECASE)
_ZIP_URL_RE: Final[re.Pattern[str]] = re.compile(r"^https://\S+\.zip$", re.IGNORECASE)
#: Surface tiles bound memory for long footprints (cells per side).
_TILE: Final[int] = 4096
_SELECTION_REPORT: Final[str] = "ehydro_selection.json"
#: Selected-survey columns written to the report, and their report names.
_SELECTED_COLUMNS: Final[dict[str, str]] = {
    "surveyjobidpk": "surveyjobidpk",
    "channelareaidfk": "channelareaidfk",
    "usacedistrictcode": "district",
    "surveytype": "surveytype",
    "surveydatestart": "surveydatestart",
    "projectedarea": "projectedarea",
    "rivers": "rivers",
    "vpus": "vpus",
    "sourcedatalocation": "url",
    "content_length": "content_length",
}

Get = Callable[[str, Mapping[str, str]], bytes]
Head = Callable[[str], int | None]


class EhydroError(RuntimeError):
    """The catalogue or a derivation step returned something the rules do not cover."""


class EhydroStop(RuntimeError):
    """A pre-registered STOP condition of docs/3 section 11.3 was met; the PI decides."""


# ----------------------------------------------------------------------------- HTTP
def _retrying(call: Callable[[], Any], what: str, sleep: Callable[[float], None]) -> Any:
    """Run ``call`` with capped exponential backoff on connection errors and retryable codes."""
    import requests

    for attempt in range(_RETRIES):
        try:
            resp = call()
        except (requests.ConnectionError, requests.Timeout):
            if attempt == _RETRIES - 1:
                raise
            sleep(2.0**attempt)
            continue
        if resp.status_code in _RETRY_STATUS and attempt < _RETRIES - 1:
            sleep(2.0**attempt)
            continue
        resp.raise_for_status()
        return resp
    raise EhydroError(f"{what}: retries exhausted")  # pragma: no cover - loop always returns


def requests_get(url: str, params: Mapping[str, str]) -> bytes:
    """GET ``url`` with ``params`` and return the body."""
    import requests

    from src.data.acquire import USER_AGENT

    resp = _retrying(
        lambda: requests.get(
            url, params=dict(params), headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S
        ),
        url,
        time.sleep,
    )
    body: bytes = resp.content
    return body


def requests_head(url: str) -> int | None:
    """Return the ``Content-Length`` of ``url`` after redirects, or ``None`` if absent."""
    import requests

    from src.data.acquire import USER_AGENT

    resp = _retrying(
        lambda: requests.head(
            url, headers={"User-Agent": USER_AGENT}, allow_redirects=True, timeout=TIMEOUT_S
        ),
        url,
        time.sleep,
    )
    value = resp.headers.get("Content-Length")
    return int(value) if value is not None and value.isdigit() else None


# ----------------------------------------------------------------------------- E-R1 frame
def read_frame_flowlines(nhd_dir: Path, rivers: Sequence[str]) -> GeoDataFrame:
    """Read the frame flowlines (E-R1): exact ``GNIS_NAME`` match, all 21 VPUs, EPSG:5070.

    Returns ``GeoDataFrame[comid, river, vpu, lengthkm, geometry]`` (2-D lines).

    Raises
    ------
    EhydroStop
        If a frame river has no flowline (E-R1 was checked on the acquired NHDSnapshot).
    """
    import warnings

    import geopandas as gpd
    import pyogrio
    import shapely

    from src.data.nhdplus import CRS_ALBERS, VPUS, NhdTable, find_table_files
    from src.data.readers import resolve_columns

    table = NhdTable("NHDFlowline.dbf", key=("COMID",), ints=("COMID",), floats=())
    files = find_table_files(nhd_dir, table)
    quoted = ", ".join("'" + name.replace("'", "''") + "'" for name in rivers)
    parts: list[GeoDataFrame] = []
    for vpu in VPUS:
        path = files[vpu].with_suffix(".shp")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Measured \\(M\\) geometry types")
            fields = [str(f) for f in pyogrio.read_info(path)["fields"]]
            cols = resolve_columns(fields, ["COMID", "GNIS_NAME", "LENGTHKM"])
            gdf = pyogrio.read_dataframe(
                path, columns=list(cols.values()), where=f"{cols['GNIS_NAME']} IN ({quoted})"
            )
        gdf = gdf.rename(columns={v: k.lower() for k, v in cols.items()})
        gdf = gdf.rename(columns={"gnis_name": "river"})
        gdf["vpu"] = vpu
        parts.append(gdf)
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    out["comid"] = out["comid"].astype("int64")
    out["lengthkm"] = out["lengthkm"].astype("float64")
    out["geometry"] = shapely.force_2d(out.geometry.to_numpy())
    missing = sorted(set(rivers) - set(out["river"]))
    if missing:
        raise EhydroStop(f"E-R1: no NHDPlusV2.1 flowline named {missing}")
    frame: GeoDataFrame = out.to_crs(CRS_ALBERS)[["comid", "river", "vpu", "lengthkm", "geometry"]]
    return frame


# ----------------------------------------------------------------------------- E-R2 catalogue
def catalogue_where(start: date, end: date) -> str:
    """SQL ``where`` for ``surveydatestart`` in [start, end] (``end`` inclusive, E-R2 (iii))."""
    stop = end + timedelta(days=1)
    return (
        f"surveydatestart >= DATE '{start.isoformat()}' "
        f"AND surveydatestart < DATE '{stop.isoformat()}'"
    )


def query_catalogue(
    url: str, where: str, get: Get = requests_get
) -> tuple[list[dict[str, Any]], list[bytes]]:
    """Page through the SurveyJob layer (attributes and footprints only; GeoJSON, EPSG:4326).

    Returns the features and the raw page bodies. The feature count must equal the server's
    ``returnCountOnly`` answer.
    """
    count_doc = json.loads(
        get(f"{url}/query", {"where": where, "returnCountOnly": "true", "f": "json"})
    )
    if "count" not in count_doc:
        raise EhydroError(f"catalogue count query failed: {count_doc}")
    expected = int(count_doc["count"])
    features: list[dict[str, Any]] = []
    pages: list[bytes] = []
    while True:
        body = get(
            f"{url}/query",
            {
                "where": where,
                "outFields": ",".join(CATALOGUE_FIELDS),
                "returnGeometry": "true",
                "outSR": "4326",
                "orderByFields": "OBJECTID ASC",
                "resultOffset": str(len(features)),
                "resultRecordCount": str(PAGE_SIZE),
                "f": "geojson",
            },
        )
        doc = json.loads(body)
        if "error" in doc:
            raise EhydroError(f"catalogue page query failed: {doc['error']}")
        page = doc.get("features", [])
        if not page:
            break
        pages.append(body)
        features.extend(page)
    if len(features) != expected:
        raise EhydroError(f"catalogue returned {len(features)} features, server count {expected}")
    return features, pages


def catalogue_frame(features: Sequence[Mapping[str, Any]]) -> tuple[GeoDataFrame, int]:
    """Build the catalogue frame in EPSG:5070; returns it and the number of repaired footprints.

    Dates arrive as epoch milliseconds (UTC). Invalid footprints are repaired with
    ``make_valid`` before the intersection test, as for the physiography polygons (docs/3 10.5).
    """
    import geopandas as gpd
    import shapely

    from src.data.nhdplus import CRS_ALBERS

    gdf = gpd.GeoDataFrame.from_features(list(features), crs="EPSG:4326")
    for field in CATALOGUE_FIELDS:
        if field not in gdf.columns:
            gdf[field] = pd.NA
    for field in ("surveydatestart", "surveydateend"):
        gdf[field] = pd.to_datetime(gdf[field], unit="ms", utc=True)
    gdf["projectedarea"] = pd.to_numeric(gdf["projectedarea"], errors="raise").astype("float64")
    geoms = gdf.geometry.to_numpy()
    present = ~(shapely.is_missing(geoms) | shapely.is_empty(geoms))
    invalid = present & ~shapely.is_valid(geoms)
    geoms[invalid] = shapely.make_valid(geoms[invalid])
    gdf = gdf.set_geometry(geoms)
    out: GeoDataFrame = gdf.to_crs(CRS_ALBERS)
    return out, int(invalid.sum())


def condition_codes(types: pd.Series) -> tuple[tuple[str, ...], dict[str, int]]:
    """E-R2 (ii): the condition-survey code(s) among the catalogue's distinct ``surveytype`` values.

    Raises
    ------
    EhydroStop
        If no value denotes a condition survey.
    """
    counts = {str(k): int(v) for k, v in types.fillna("<null>").value_counts().sort_index().items()}
    codes = tuple(sorted(c for c in counts if c != "<null>" and _CONDITION_RE.match(c)))
    if not codes:
        raise EhydroStop(f"E-R2: no condition-survey code among surveytype values {counts}")
    return codes, counts


def eligible_surveys(
    catalogue: GeoDataFrame,
    flowlines: GeoDataFrame,
    codes: Sequence[str],
    start: date,
    end: date,
) -> tuple[GeoDataFrame, dict[str, int]]:
    """Apply E-R2 (i)-(iv); adds ``rivers`` and ``vpus`` (sorted, ``;``-joined) per survey."""
    import geopandas as gpd

    counts = {"catalogue": len(catalogue)}
    cat = catalogue.loc[catalogue["surveytype"].isin(list(codes))]
    counts["condition_survey"] = len(cat)
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end + timedelta(days=1), tz="UTC")
    cat = cat.loc[cat["surveydatestart"].ge(lo) & cat["surveydatestart"].lt(hi)]
    counts["in_window"] = len(cat)
    urls = cat["sourcedatalocation"].astype("string").fillna("")
    cat = cat.loc[urls.str.match(_ZIP_URL_RE.pattern, case=False)]
    counts["https_zip"] = len(cat)
    cat = cat.loc[cat.geometry.notna() & ~cat.geometry.is_empty]
    counts["with_footprint"] = len(cat)
    hits = gpd.sjoin(
        cat[["surveyjobidpk", "geometry"]],
        flowlines[["river", "vpu", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    agg = hits.groupby("surveyjobidpk").agg(
        rivers=("river", lambda s: ";".join(sorted(set(s)))),
        vpus=("vpu", lambda s: ";".join(sorted(set(s)))),
    )
    out = cat.merge(agg, left_on="surveyjobidpk", right_index=True, how="inner")
    counts["intersects_frame"] = len(out)
    return out, counts


def select_latest(eligible: pd.DataFrame) -> pd.DataFrame:
    """E-R3: the latest survey per ``channelareaidfk``; ties by larger area, then smaller ID.

    A survey without a channel area forms its own group (counted in the report).
    """
    frame = eligible.copy()
    area = frame["channelareaidfk"].astype("string")
    frame["_group"] = area.where(area.notna() & area.str.len().gt(0)).fillna(
        "survey:" + frame["surveyjobidpk"].astype("string")
    )
    frame = frame.sort_values(
        ["_group", "surveydatestart", "projectedarea", "surveyjobidpk"],
        ascending=[True, False, False, True],
        na_position="last",
        kind="mergesort",
    )
    out = frame.drop_duplicates("_group", keep="first").drop(columns="_group")
    return out.sort_values("surveyjobidpk", kind="mergesort").reset_index(drop=True)


def head_sizes(urls: Sequence[str], head: Head = requests_head) -> list[int | None]:
    """E-R4: ``Content-Length`` of every selected zip (``None`` if the server sends none)."""
    return [head(url) for url in urls]


# ----------------------------------------------------------------------------- E-R5 (i) reader
#: docs/3 11.4 E-R5c, water list (tested first). Case-insensitive.
WATER_RE: Final[re.Pattern[str]] = re.compile(
    r"low\s*water\s*reference\s*plane|\bLWRP|mean\s+lower\s+low\s+water|\bMLLW\b|"
    r"mean\s+low\s+gulf|\bMLG\b|columbia\s+river\s+datum|\bCRD\b|low\s+water\s+datum|\bLWD\b|"
    r"\bpool\b|minimum\s*pool|flat\s*pool|normal\s*pool|"
    r"\b(?:ohio|monongahela|allegheny|kanawha|mississippi|missouri|river)\s+datum|"
    r"\b(?:ORD|MRD|BRD|MND)\b|"
    r"water\s+surface\s+(?:at|on)\s+(?:the\s+)?(?:time|date)|"
    r"depths?\s+on\s+the\s+date\s+of\s+(?:the\s+)?survey|low\s+steady\s+flow|\bLSF\b",
    re.IGNORECASE,
)
#: docs/3 11.4 E-R5c, geodetic list. Case-insensitive.
GEO_RE: Final[re.Pattern[str]] = re.compile(
    r"\bNAVD|\bNGVD|national\s+geodetic|north\s+american\s+vertical|\bMSL\s*1912|"
    r"mean\s+sea\s+level\s+(?:datum\s+)?(?:of\s+)?1912|\bIGLD|great\s+lakes\s+datum|"
    r"bottom\s+elevation",
    re.IGNORECASE,
)
#: E-R5c S1: "Vertical [Reference] Datum [Name]:" up to ". " or " - " before a capital letter
#: (case-sensitive lookahead), or up to a tag end.
STATEMENT_RE: Final[re.Pattern[str]] = re.compile(
    r"vertical\s+(?:reference\s+)?datum(?:\s+name)?\s*:\s*(.+?)"
    r"(?:\.\s+(?=(?-i:[A-Z]))|\s+-\s+(?=(?-i:[A-Z]))|</|$)",
    re.IGNORECASE | re.DOTALL,
)
#: E-R5c S1 metadata files, in reading order (top level of the survey folder).
METADATA_SUFFIXES: Final[tuple[str, ...]] = (".XML", ".MET", ".GEN")
#: E-R5b: accepted ``elevationUOM`` values (all feet; the misspelling is read as US survey).
FOOT_UNITS: Final[frozenset[str]] = frozenset({"usSurveyFoot", "internationalFoot", "usSuveryFoot"})
#: E-R5c classes; only ``water`` surveys are used.
PLANE_CLASSES: Final[tuple[str, ...]] = ("water", "geodetic", "unresolved")


def classify_plane_text(text: str | None) -> str | None:
    """E-R5c: ``"water"`` if the water list matches, else ``"geodetic"``, else ``None``."""
    if text is None or not text.strip():
        return None
    if WATER_RE.search(text):
        return "water"
    if GEO_RE.search(text):
        return "geodetic"
    return None


@dataclass(frozen=True)
class PlaneDecision:
    """E-R5c outcome for one survey: class, deciding source and the evidence read."""

    cls: str
    source: str  # "S1", "S2", "S3", "S2+S3", "conflict" or "none"
    statement: str | None
    edomv: tuple[str, ...]
    field_values: tuple[str, ...]


def _metadata_files(survey_dir: Path) -> list[Path]:
    files = [
        p for p in survey_dir.iterdir() if p.is_file() and p.suffix.upper() in METADATA_SUFFIXES
    ]
    return sorted(files, key=lambda p: (METADATA_SUFFIXES.index(p.suffix.upper()), p.name))


def _read_text(path: Path) -> str:
    import html

    return html.unescape(path.read_text(encoding="utf-8", errors="replace"))


def plane_statement(survey_dir: Path) -> tuple[str | None, tuple[str, ...]]:
    """E-R5c S1 and S2 from the survey's metadata files.

    Returns the S1 text (the first "Vertical ... Datum ...:" statement in ``.XML``, ``.MET``,
    ``.GEN``; otherwise the first XML ``<depthdn>``, then ``<altdatum>``, prefixed by the tag) and
    the S2 ``<edomv>`` values of the XML attribute ``elevationDatum``.
    """
    files = _metadata_files(survey_dir)
    statement: str | None = None
    for path in files:
        match = STATEMENT_RE.search(re.sub(r"\s+", " ", _read_text(path)))
        if match:
            statement = match.group(1).strip()
            break
    xmls = [p for p in files if p.suffix.upper() == ".XML"]
    xml = _read_text(xmls[0]) if xmls else ""
    if statement is None:
        for tag in ("depthdn", "altdatum"):
            found = re.search(rf"(?is)<{tag}>\s*(.*?)\s*</{tag}>", xml)
            if found and found.group(1).strip():
                statement = f"<{tag}> " + re.sub(r"\s+", " ", found.group(1)).strip()
                break
    edomv: list[str] = []
    for attr in re.finditer(r"(?is)<attr>(.*?)</attr>", xml):
        body = attr.group(1)
        label = re.search(r"(?is)<attrlabl>\s*(.*?)\s*</attrlabl>", body)
        if label and label.group(1).strip().lower() == "elevationdatum":
            edomv += [
                re.sub(r"\s+", " ", v).strip()
                for v in re.findall(r"(?is)<edomv>(.*?)</edomv>", body)
            ]
    return statement, tuple(edomv)


def decide_plane(
    statement: str | None, edomv: Sequence[str], field_values: Sequence[str]
) -> PlaneDecision:
    """E-R5c precedence: S1 decides alone; else one class over S2 ∪ S3, two → unresolved."""
    evidence = (statement, tuple(edomv), tuple(field_values))
    s1 = classify_plane_text(statement)
    if s1 is not None:
        return PlaneDecision(s1, "S1", *evidence)
    s2 = {c for c in map(classify_plane_text, edomv) if c is not None}
    s3 = {c for c in map(classify_plane_text, field_values) if c is not None}
    both = s2 | s3
    if len(both) == 1:
        source = "S2+S3" if s2 and s3 else ("S2" if s2 else "S3")
        return PlaneDecision(next(iter(both)), source, *evidence)
    return PlaneDecision("unresolved", "conflict" if both else "none", *evidence)


def survey_gdb(survey_dir: Path) -> Path:
    """The single file geodatabase of a delivered survey (E-R5a)."""
    found = sorted(survey_dir.rglob("*.gdb"))
    if len(found) != 1:
        raise EhydroError(f"E-R5a: {survey_dir.name} has {len(found)} geodatabases, expected 1")
    return found[0]


def _read_layer_quiet(gdb: Path, layer: str, columns: Sequence[str] | None) -> GeoDataFrame:
    import warnings

    import pyogrio

    with warnings.catch_warnings():
        # Measured (M) points/lines are read without M (2-D use only); expected and harmless.
        warnings.filterwarnings("ignore", message="Measured \\(M\\) geometry types")
        out: GeoDataFrame = pyogrio.read_dataframe(
            gdb, layer=layer, columns=None if columns is None else list(columns)
        )
    return out


def survey_field_datums(gdb: Path) -> tuple[str, ...]:
    """E-R5c S3: distinct non-blank ``SurveyPoint.elevationDatum`` values (sorted)."""
    import warnings

    import pyogrio

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Measured \\(M\\) geometry types")
        frame = pyogrio.read_dataframe(
            gdb, layer="SurveyPoint", columns=["elevationDatum"], read_geometry=False
        )
    values = frame["elevationDatum"].dropna().astype(str).str.strip()
    return tuple(sorted(set(values[values.str.len() > 0])))


def classify_survey(survey_dir: Path) -> PlaneDecision:
    """E-R5c for one delivered survey (metadata only; no depth value is read)."""
    statement, edomv = plane_statement(survey_dir)
    return decide_plane(statement, edomv, survey_field_datums(survey_gdb(survey_dir)))


@dataclass(frozen=True)
class SurveyPoints:
    """E-R5a/b sounding points of one survey in EPSG:5070 and its E-R5d footprint."""

    x: np.ndarray
    y: np.ndarray
    depth_m: np.ndarray
    footprint: Geometry
    n_read: int
    n_null_depth: int
    n_empty_geometry: int
    units: tuple[str, ...]
    footprint_repaired: bool


def read_survey(survey_dir: Path) -> SurveyPoints:
    """E-R5a/b/d: ``SurveyPoint`` depths in metres (positive down) and the ``SurveyJob`` polygon.

    depth = ``Z_depth`` × ``FT_TO_M`` (E-R5b; the 2 ppm US-survey-foot difference is not
    corrected, as E-R5 (i) prescribes ``FT_TO_M``). Points with a null ``Z_depth`` or an empty
    geometry are dropped and counted. No other filter: non-positive depths are handled by
    E-R5 (iv) after interpolation.

    Raises
    ------
    EhydroStop
        If a non-null ``elevationUOM`` is not a foot, or no unit is given (E-R5b).
    """
    import shapely

    from src.config import FT_TO_M
    from src.data.nhdplus import CRS_ALBERS

    gdb = survey_gdb(survey_dir)
    points = _read_layer_quiet(gdb, "SurveyPoint", ["Z_depth", "elevationUOM"])
    units = tuple(sorted(set(points["elevationUOM"].dropna().astype(str).str.strip())))
    if not units or not set(units) <= FOOT_UNITS:
        raise EhydroStop(f"E-R5b: {survey_dir.name} elevationUOM {units} is not a foot")
    geoms = points.geometry.to_numpy()
    empty = shapely.is_missing(geoms) | shapely.is_empty(geoms)
    z = pd.to_numeric(points["Z_depth"], errors="raise").to_numpy(dtype=np.float64)
    null_z = np.isnan(z)
    keep = ~(empty | null_z)
    kept = points.loc[keep].to_crs(CRS_ALBERS)
    xy = shapely.get_coordinates(shapely.force_2d(kept.geometry.to_numpy()))
    job = _read_layer_quiet(gdb, "SurveyJob", []).to_crs(CRS_ALBERS)
    polygons = job.geometry.to_numpy()
    invalid = ~shapely.is_valid(polygons)
    polygons[invalid] = shapely.make_valid(polygons[invalid])
    footprint = shapely.union_all(polygons)
    repaired = bool(invalid.any())
    return SurveyPoints(
        x=xy[:, 0].astype(np.float64),
        y=xy[:, 1].astype(np.float64),
        depth_m=z[keep] * FT_TO_M,
        footprint=footprint,
        n_read=len(points),
        n_null_depth=int(null_z.sum()),
        n_empty_geometry=int((empty & ~null_z).sum()),
        units=units,
        footprint_repaired=repaired,
    )


# ----------------------------------------------------------------------------- E-R5/E-R6 cells
@dataclass(frozen=True)
class Cells:
    """Wetted cells of a depth surface on the global grid (E-R5).

    Cell ``(ix, iy)`` has its centre at ``((ix + 0.5) c, (iy + 0.5) c)`` in EPSG:5070, where
    ``c`` is ``ehydro.cell_m``; ``d`` is the depth in metres (> 0).
    """

    ix: np.ndarray
    iy: np.ndarray
    d: np.ndarray

    def __len__(self) -> int:
        """Number of cells."""
        return int(self.d.size)

    @staticmethod
    def empty() -> Cells:
        """A surface without wetted cells."""
        return Cells(np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float64))


def depth_surface(
    x: np.ndarray, y: np.ndarray, depth: np.ndarray, footprint: Geometry, cell_m: float
) -> Cells:
    """E-R5 (ii)-(iv): bin, interpolate inside the footprint and keep wetted cells.

    (ii) Points (EPSG:5070, depth in m, positive down) are binned into ``cell_m`` cells; the
    cell value is the mean. (iii) Cell values are interpolated linearly (Delaunay) onto the
    centres of the cells whose centre lies in ``footprint``; centres outside the convex hull of
    the binned cells get no value. (iv) Cells with depth ≤ 0 are not wetted. Fewer than three
    binned cells, or collinear ones, give no surface.
    """
    import shapely
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import QhullError

    if not (x.shape == y.shape == depth.shape):
        raise ValueError("x, y and depth must have the same shape")
    binned = (
        pd.DataFrame(
            {
                "ix": np.floor(x / cell_m).astype(np.int64),
                "iy": np.floor(y / cell_m).astype(np.int64),
                "d": depth.astype(np.float64),
            }
        )
        .groupby(["ix", "iy"], sort=True)["d"]
        .mean()
    )
    if len(binned) < 3:
        return Cells.empty()
    bix = binned.index.get_level_values("ix").to_numpy()
    biy = binned.index.get_level_values("iy").to_numpy()
    centres = np.column_stack([(bix + 0.5) * cell_m, (biy + 0.5) * cell_m])
    try:
        interp = LinearNDInterpolator(centres, binned.to_numpy())
    except QhullError:
        return Cells.empty()
    minx, miny, maxx, maxy = footprint.bounds
    ix0, ix1 = int(np.floor(minx / cell_m)), int(np.ceil(maxx / cell_m))
    iy0, iy1 = int(np.floor(miny / cell_m)), int(np.ceil(maxy / cell_m))
    out_ix: list[np.ndarray] = []
    out_iy: list[np.ndarray] = []
    out_d: list[np.ndarray] = []
    for tx in range(ix0, ix1, _TILE):
        for ty in range(iy0, iy1, _TILE):
            width, height = min(_TILE, ix1 - tx), min(_TILE, iy1 - ty)
            transform = from_origin(tx * cell_m, (ty + height) * cell_m, cell_m, cell_m)
            # all_touched gives a superset of the cells whose centre is inside; the exact
            # centre test (contains_xy) then makes the result independent of the tiling.
            touched = geometry_mask(
                [footprint],
                out_shape=(height, width),
                transform=transform,
                invert=True,
                all_touched=True,
            )
            rows, cols = np.nonzero(touched)
            cix = tx + cols.astype(np.int64)
            ciy = ty + height - 1 - rows.astype(np.int64)
            centre = shapely.contains_xy(footprint, (cix + 0.5) * cell_m, (ciy + 0.5) * cell_m)
            cix, ciy = cix[centre], ciy[centre]
            if cix.size == 0:
                continue
            values = interp((cix + 0.5) * cell_m, (ciy + 0.5) * cell_m)
            keep = ~np.isnan(values) & (values > 0)  # E-R5 (iii) hull, (iv) wetted
            out_ix.append(cix[keep])
            out_iy.append(ciy[keep])
            out_d.append(values[keep])
    if not out_d:
        return Cells.empty()
    return Cells(np.concatenate(out_ix), np.concatenate(out_iy), np.concatenate(out_d))


def pool_cells(surfaces: Sequence[Cells]) -> Cells:
    """E-R6: pool surfaces ordered oldest to newest; where cells overlap the later one wins."""
    if not surfaces:
        return Cells.empty()
    frame = pd.DataFrame(
        {
            "ix": np.concatenate([s.ix for s in surfaces]),
            "iy": np.concatenate([s.iy for s in surfaces]),
            "d": np.concatenate([s.d for s in surfaces]),
        }
    ).drop_duplicates(["ix", "iy"], keep="last")
    return Cells(frame["ix"].to_numpy(), frame["iy"].to_numpy(), frame["d"].to_numpy())


def reach_values(
    cells: Cells, catchments: GeoDataFrame, lengthkm: Mapping[int, float], cell_m: float
) -> pd.DataFrame:
    """E-R6 (Zarrabi 2.5): per COMID, d̄ = mean depth, A = Σ d c² / L, W = A / d̄.

    ``catchments`` is ``GeoDataFrame[comid, geometry]`` (EPSG:5070); a cell belongs to the
    catchment that contains its centre (a centre in two overlapping catchments goes to the
    first). L = ``LENGTHKM`` × 1000. Returns ``comid, n_cells, dbar, A, W``.
    """
    import shapely

    cols = ["comid", "n_cells", "dbar", "A", "W"]
    if len(cells) == 0 or catchments.empty:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})
    points = shapely.points((cells.ix + 0.5) * cell_m, (cells.iy + 0.5) * cell_m)
    tree = shapely.STRtree(catchments.geometry.to_numpy())
    p_idx, c_idx = tree.query(points, predicate="within")
    pairs = pd.DataFrame({"p": p_idx, "c": c_idx}).sort_values(["p", "c"], kind="mergesort")
    pairs = pairs.drop_duplicates("p", keep="first")
    frame = pd.DataFrame(
        {
            "comid": catchments["comid"].to_numpy()[pairs["c"].to_numpy()].astype("int64"),
            "d": cells.d[pairs["p"].to_numpy()],
        }
    )
    agg = frame.groupby("comid", sort=True)["d"].agg(n_cells="size", dbar="mean", d_sum="sum")
    length_m = agg.index.map(lambda c: float(lengthkm[int(c)]) * 1000.0).to_numpy()
    if np.any(length_m <= 0):
        raise EhydroError("E-R6: a COMID with cells has a non-positive LENGTHKM")
    agg["A"] = agg["d_sum"].to_numpy() * cell_m**2 / length_m
    agg["W"] = agg["A"] / agg["dbar"]
    out = agg.drop(columns="d_sum").reset_index()
    out["n_cells"] = out["n_cells"].astype("int64")
    return out[cols]


# ----------------------------------------------------------------------------- E-R7 coverage
def coverage_fraction(
    flowlines: GeoDataFrame, footprints: Sequence[Geometry], buffer_m: float
) -> pd.Series:
    """E-R7: share of each flowline's length inside the union of buffered footprints."""
    import shapely

    lines = flowlines.geometry.to_numpy()
    length = shapely.length(lines)
    if np.any(length <= 0):
        raise EhydroError("E-R7: zero-length flowline")
    if not footprints:
        return pd.Series(0.0, index=flowlines["comid"].to_numpy(), name="coverage")
    zone = shapely.union_all(shapely.buffer(np.asarray(footprints, dtype=object), buffer_m))
    inside = shapely.length(shapely.intersection(lines, zone))
    return pd.Series(inside / length, index=flowlines["comid"].to_numpy(), name="coverage")


# ----------------------------------------------------------------------------- E-R8 locations
def build_locations(
    reaches: pd.DataFrame, edges: pd.DataFrame, loc_length_km: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E-R8: cut connected runs per river into locations of ≥ ``loc_length_km``.

    Parameters
    ----------
    reaches
        Retained COMIDs: ``comid, river, lengthkm, hydroseq, W, dbar``.
    edges
        PlusFlow ``from_comid, to_comid``. Two retained COMIDs are connected if an edge joins
        them and they carry the same river name.
    loc_length_km
        ``ehydro.loc_length_km``.

    Members of a run are walked from upstream (descending ``hydroseq``, then ``comid``). A
    location closes when its cumulative length reaches ``loc_length_km``; a final remainder
    shorter than half of it joins the preceding location of the same run. W and d̄ are
    length-weighted means and A = W·d̄.

    Returns
    -------
    tuple
        ``locations`` (``loc_id, river, n_comids, length_km, W, dbar, A``) and ``members``
        (``loc_id, comid``).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if reaches["comid"].duplicated().any():
        raise ValueError("reaches must have one row per COMID")
    r = reaches.sort_values("comid", kind="mergesort").reset_index(drop=True)
    pos = pd.Series(np.arange(len(r)), index=r["comid"].to_numpy())
    e = edges.loc[edges["from_comid"].isin(pos.index) & edges["to_comid"].isin(pos.index)]
    a = pos.loc[e["from_comid"].to_numpy()].to_numpy()
    b = pos.loc[e["to_comid"].to_numpy()].to_numpy()
    same = r["river"].to_numpy()[a] == r["river"].to_numpy()[b]
    a, b = a[same], b[same]
    graph = coo_matrix((np.ones(a.size), (a, b)), shape=(len(r), len(r)))
    _, labels = connected_components(graph, directed=False)
    r["_run"] = labels
    loc_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    order = r.groupby("_run")["comid"].min().sort_values(kind="mergesort").index
    for run in order:
        run_df = r.loc[r["_run"].eq(run)].sort_values(
            ["hydroseq", "comid"], ascending=[False, True], kind="mergesort"
        )
        chunks: list[list[int]] = []
        current: list[int] = []
        cum = 0.0
        for idx, length in zip(run_df.index, run_df["lengthkm"], strict=True):
            current.append(int(idx))
            cum += float(length)
            if cum >= loc_length_km:
                chunks.append(current)
                current, cum = [], 0.0
        if current:
            if chunks and cum < loc_length_km / 2:
                chunks[-1].extend(current)
            else:
                chunks.append(current)
        for chunk in chunks:
            part = r.loc[chunk]
            weights = part["lengthkm"].to_numpy(dtype=np.float64)
            w = float(np.average(part["W"].to_numpy(dtype=np.float64), weights=weights))
            dbar = float(np.average(part["dbar"].to_numpy(dtype=np.float64), weights=weights))
            loc_id = f"{part['river'].iloc[0]}:{int(part['comid'].iloc[0])}"
            loc_rows.append(
                {
                    "loc_id": loc_id,
                    "river": part["river"].iloc[0],
                    "n_comids": len(part),
                    "length_km": float(weights.sum()),
                    "W": w,
                    "dbar": dbar,
                    "A": w * dbar,
                }
            )
            member_rows += [{"loc_id": loc_id, "comid": int(c)} for c in part["comid"]]
    locations = pd.DataFrame(
        loc_rows, columns=["loc_id", "river", "n_comids", "length_km", "W", "dbar", "A"]
    )
    members = pd.DataFrame(member_rows, columns=["loc_id", "comid"])
    return locations, members


# ----------------------------------------------------------------------------- select CLI
def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _record(row: Mapping[Hashable, object]) -> dict[str, object]:
    """JSON-ready copy of one selected-survey row (missing values become ``None``)."""
    out: dict[str, object] = {}
    for name, value in row.items():
        key = str(name)
        if value is None or value is pd.NA or value is pd.NaT:
            out[key] = None
        elif isinstance(value, pd.Timestamp):
            out[key] = value.isoformat()
        elif isinstance(value, np.integer):
            out[key] = int(value)
        elif isinstance(value, float | np.floating):
            out[key] = None if np.isnan(value) else float(value)
        else:
            out[key] = value
    return out


def run_select(
    settings: Settings,
    get: Get = requests_get,
    head: Head = requests_head,
    flowlines: GeoDataFrame | None = None,
) -> dict[str, Any]:
    """Catalogue → eligibility → selection → HEAD pilot (E-R1 to E-R4); writes the report.

    Raises
    ------
    EhydroStop
        After the report is written, if the total size exceeds ``ehydro.size_cap_gb`` or a
        selected zip has no ``Content-Length`` (E-R4).
    """
    p: EhydroParams = settings.protocol.ehydro
    rt = settings.runtime
    if flowlines is None:
        flowlines = read_frame_flowlines(Path(rt.raw_dir) / "nhdplusv21", p.rivers)
    url = str(p.catalogue).rstrip("/")
    where = catalogue_where(p.date_start, p.date_end)
    queried = datetime.now(UTC).isoformat()
    features, pages = query_catalogue(url, where, get)
    snap = Path(rt.dvc_store).expanduser() / "ehydro_catalogue" / _utc_stamp()
    snap.mkdir(parents=True, exist_ok=False)
    page_hashes = []
    for i, body in enumerate(pages):
        (snap / f"page_{i:03d}.geojson").write_bytes(body)
        page_hashes.append(hashlib.sha256(body).hexdigest())
    cat, repaired = catalogue_frame(features)
    codes, type_counts = condition_codes(cat["surveytype"])
    eligible, counts = eligible_surveys(cat, flowlines, codes, p.date_start, p.date_end)
    selected = select_latest(eligible)
    sizes = head_sizes(selected["sourcedatalocation"].astype(str).tolist(), head)
    selected["content_length"] = pd.array(sizes, dtype="Int64")
    known = [s for s in sizes if s is not None]
    total = int(sum(known))
    no_length = int(sum(s is None for s in sizes))
    cap = int(p.size_cap_gb * 1e9)
    counts["selected"] = len(selected)
    counts["no_channel_area"] = int(
        selected["channelareaidfk"].astype("string").fillna("").str.len().eq(0).sum()
    )
    per_river = (
        selected.assign(river=selected["rivers"].str.split(";"))
        .explode("river")
        .groupby("river")
        .size()
        .reindex(list(p.rivers), fill_value=0)
    )
    report: dict[str, Any] = {
        "rules": "docs/3 section 11.3 E-R1 to E-R4 (pre-registered, commit 07e00d6)",
        "params_sha256": settings.params_sha256,
        "catalogue": url,
        "where": where,
        "queried_utc": queried,
        "snapshot_dir": str(snap),
        "page_sha256": page_hashes,
        "footprints_repaired": repaired,
        "surveytype_counts": type_counts,
        "condition_codes": list(codes),
        "counts": counts,
        "surveys_per_river": {k: int(v) for k, v in per_river.items()},
        "bytes_total_known": total,
        "n_without_content_length": no_length,
        "size_cap_bytes": cap,
        "stop": total > cap or no_length > 0,
        "selected": [
            _record(row)
            for row in selected[list(_SELECTED_COLUMNS)]
            .rename(columns=_SELECTED_COLUMNS)
            .to_dict("records")
        ],
    }
    out = Path(rt.reports_dir) / _SELECTION_REPORT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    LOG.info("selected %d surveys, %.2f GB known", len(selected), total / 1e9)
    if report["stop"]:
        raise EhydroStop(
            f"E-R4: {total / 1e9:.2f} GB known (cap {p.size_cap_gb} GB), "
            f"{no_length} zip(s) without Content-Length; see {out}"
        )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Run ``python -m src.data.ehydro select`` (catalogue, selection, HEAD pilot)."""
    from src.config import load_settings

    parser = argparse.ArgumentParser(description="eHydro survey selection (docs/3 11.3).")
    parser.add_argument("command", choices=["select"])
    parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_select(load_settings())


if __name__ == "__main__":
    main()

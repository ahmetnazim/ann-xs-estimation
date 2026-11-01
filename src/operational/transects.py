"""OPS-1 v2 terrain inputs: anchor WSE in NAVD88 and 3DEP cross-valley transects (docs/5 sec. 2-3).

Pure geometry and parsing functions plus a cached, resumable acquisition step:

* :func:`anchor_frame` — per primary site: COMID, coordinates, observed anchor width and the
  anchor water-surface elevation ``WSE_a = 0.3048 (alt_va + stage_a)`` in the gauge datum.
* :func:`transect_geometry` — centre on the NHDPlusV2.1 flowline ``offset`` upstream of the gauge
  projection (flowlines are digitized downstream), perpendicular line of length ``L``.
* :func:`parse_samples` / :func:`parse_ncat` — 3DEP ``getSamples`` and NOAA NCAT responses.
* :func:`channel_centre` — centre of the flat run containing the DEM minimum near the centre.
* :func:`acquire_site` — one site's requests, written to ``<cache>/<site_no>.json`` (skipped when
  present, so the step resumes after interruption).

Network calls are made only through an injectable ``get_json`` callable (tests need no network).
Owning phase: OPS-1 v2 (post-release extension).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import FT_TO_M
from src.operational.config_v2 import Operational2Params

__all__ = [
    "CRS_METRIC",
    "TRANSIENT",
    "GetJson",
    "TransectGeometry",
    "acquire_site",
    "anchor_frame",
    "channel_centre",
    "load_flowlines",
    "nwis_coordinates",
    "parse_ncat",
    "parse_nwis_rdb",
    "parse_samples",
    "requests_get_json",
    "requests_get_text",
    "transect_geometry",
    "transect_length",
]

FloatArray = NDArray[np.float64]
GetJson = Callable[[str, Mapping[str, str]], Any]
CRS_METRIC: Final[str] = "EPSG:5070"
CRS_GEO: Final[str] = "EPSG:4326"


def anchor_frame(
    site_table: pd.DataFrame, stage: pd.DataFrame, raw_alt: pd.DataFrame
) -> pd.DataFrame:
    """Per primary site: ``site_no, comid, lat, lon, w_a, stage_a_ft, alt_va_ft, datum, wse_a``.

    ``wse_a`` (m) is in the gauge datum ``datum``; NaN when ``alt_va`` or the anchor stage is
    missing. ``raw_alt`` holds ``site_no, alt_va, alt_datum_cd`` (one row per site).
    """
    prim = site_table.loc[site_table["variant"] == "primary"]
    out = prim[["site_no", "comid", "lat", "lon", "W", "row_id"]].rename(columns={"W": "w_a"})
    out = out.merge(stage[["row_id", "stage_ft"]], on="row_id", how="left", validate="1:1").rename(
        columns={"stage_ft": "stage_a_ft"}
    )
    alt = raw_alt.drop_duplicates("site_no")[["site_no", "alt_va", "alt_datum_cd"]]
    out = out.merge(alt, on="site_no", how="left", validate="1:1")
    out["alt_va_ft"] = pd.to_numeric(out["alt_va"], errors="coerce")
    out["datum"] = out["alt_datum_cd"].fillna("").astype(str).str.strip()
    out["wse_a"] = FT_TO_M * (out["alt_va_ft"] + out["stage_a_ft"])
    cols = ["site_no", "comid", "lat", "lon", "w_a", "stage_a_ft", "alt_va_ft", "datum", "wse_a"]
    return out[cols].reset_index(drop=True)


def transect_length(w_a: float, p: Operational2Params) -> float:
    """``L = clamp(length_factor_w * W_a, length_min_m, length_max_m)`` (docs/5 section 2)."""
    t = p.transect
    return float(min(max(t.length_factor_w * w_a, t.length_min_m), t.length_max_m))


@dataclass(frozen=True)
class TransectGeometry:
    """Transect in the metric CRS: centre, unit normal, end points, gauge-to-flowline distance."""

    cx: float
    cy: float
    nx: float
    ny: float
    length: float
    gauge_offset: float

    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """``(start, end)`` = centre -/+ normal * L/2."""
        h = self.length / 2.0
        return (
            (self.cx - h * self.nx, self.cy - h * self.ny),
            (self.cx + h * self.nx, self.cy + h * self.ny),
        )


def transect_geometry(
    line_xy: FloatArray,
    gauge_xy: tuple[float, float],
    *,
    offset: float,
    half_window: float,
    length: float,
) -> TransectGeometry:
    """Perpendicular transect ``offset`` upstream of the gauge projection on the flowline.

    ``line_xy`` is the ``(n, 2)`` flowline in the metric CRS, digitized downstream. The centre is
    at arc length ``max(s_g - offset, 0)``; the bearing uses the points at ``+-half_window``.
    """
    from shapely.geometry import LineString, Point

    if line_xy.ndim != 2 or line_xy.shape[0] < 2 or line_xy.shape[1] != 2:
        raise ValueError("line_xy must be an (n >= 2, 2) array")
    line = LineString(line_xy)
    if not line.length > 0:
        raise ValueError("degenerate flowline")
    g = Point(gauge_xy)
    s_g = float(line.project(g))
    s_c = max(s_g - offset, 0.0)
    a = line.interpolate(max(s_c - half_window, 0.0))
    b = line.interpolate(min(s_c + half_window, line.length))
    dx, dy = b.x - a.x, b.y - a.y
    norm = math.hypot(dx, dy)
    if not norm > 0:
        raise ValueError("cannot determine flowline bearing")
    c = line.interpolate(s_c)
    return TransectGeometry(
        cx=float(c.x),
        cy=float(c.y),
        nx=-dy / norm,
        ny=dx / norm,
        length=length,
        gauge_offset=float(line.distance(g)),
    )


def parse_samples(payload: Mapping[str, Any]) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(lon, lat, z)`` from a 3DEP ``getSamples`` response; ``NoData`` becomes NaN."""
    if "samples" not in payload:
        raise ValueError(f"3DEP response without samples: {str(payload)[:200]}")
    samples = sorted(payload["samples"], key=lambda s: int(s.get("locationId", 0)))
    lon = np.array([float(s["location"]["x"]) for s in samples])
    lat = np.array([float(s["location"]["y"]) for s in samples])
    z = np.array([_to_float(s.get("value")) for s in samples])
    return lon, lat, z


def _to_float(v: object) -> float:
    try:
        f = float(str(v))
    except ValueError:
        return float("nan")
    return f if math.isfinite(f) and f > -1.0e4 else float("nan")


def parse_ncat(payload: Mapping[str, Any]) -> float:
    """NGVD29 -> NAVD88 shift ``destOrthoht - srcOrthoht`` (m) from an NCAT response."""
    try:
        return float(payload["destOrthoht"]) - float(payload["srcOrthoht"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"NCAT response without orthometric heights: {exc}") from exc


def channel_centre(
    x: FloatArray, z: FloatArray, centre_x: float, search_half: float, flat_tol: float
) -> tuple[float, float]:
    """``(x_c, z_ch)``: centre of the flat run (``<= z_min + flat_tol``) containing the minimum.

    The minimum is searched within ``|x - centre_x| <= search_half`` on finite samples.
    """
    finite = np.isfinite(z)
    win = finite & (np.abs(x - centre_x) <= search_half)
    if not win.any():
        raise ValueError("no finite DEM sample in the channel search window")
    idx = np.flatnonzero(win)
    i0 = int(idx[np.argmin(z[idx])])
    zmin = float(z[i0])
    low = finite & (z <= zmin + flat_tol)
    lo = i0
    while lo > 0 and low[lo - 1]:
        lo -= 1
    hi = i0
    while hi < z.size - 1 and low[hi + 1]:
        hi += 1
    return 0.5 * (float(x[lo]) + float(x[hi])), zmin


def load_flowlines(paths: list[Path], comids: set[int]) -> dict[int, FloatArray]:
    """NHDPlusV2.1 ``NHDFlowline.shp`` geometries for ``comids`` as ``(n, 2)`` metric arrays.

    Multi-part lines keep their longest part; Z/M values are dropped.
    """
    import warnings

    import pyogrio
    from shapely import get_coordinates

    out: dict[int, FloatArray] = {}
    ids = ",".join(str(int(c)) for c in sorted(comids))
    for path in paths:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # measured geometries -> LineString Z
            gdf = pyogrio.read_dataframe(path, columns=["COMID"], where=f"COMID IN ({ids})")
        if gdf.empty:
            continue
        gdf = gdf.to_crs(CRS_METRIC)
        for comid, geom in zip(gdf["COMID"].to_numpy(), gdf.geometry, strict=True):
            parts = list(getattr(geom, "geoms", [geom]))
            best = max(parts, key=lambda g: g.length)
            out[int(comid)] = np.asarray(get_coordinates(best), dtype=np.float64)
    return out


def parse_nwis_rdb(text: str) -> pd.DataFrame:
    """Parse an NWIS site-service RDB table into ``site_no, lat, lon, coord_datum``.

    NAD27 coordinates are transformed to NAD83 (EPSG:4267 -> EPSG:4269), as in
    ``src.data.unit_map``; other datums are returned unchanged with their code.
    """
    from pyproj import Transformer

    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 2:
        return pd.DataFrame(columns=["site_no", "lat", "lon", "coord_datum"])
    header = lines[0].split("\t")
    rows = [dict(zip(header, ln.split("\t"), strict=False)) for ln in lines[2:]]
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "site_no": df["site_no"].astype(str),
            "lat": pd.to_numeric(df["dec_lat_va"], errors="coerce"),
            "lon": pd.to_numeric(df["dec_long_va"], errors="coerce"),
            "coord_datum": df["coord_datum_cd"].astype(str).str.strip(),
        }
    )
    nad27 = out["coord_datum"].eq("NAD27") & out["lat"].notna() & out["lon"].notna()
    if nad27.any():
        tr = Transformer.from_crs("EPSG:4267", "EPSG:4269", always_xy=True)
        x, y = tr.transform(out.loc[nad27, "lon"].to_numpy(), out.loc[nad27, "lat"].to_numpy())
        out.loc[nad27, "lon"] = x
        out.loc[nad27, "lat"] = y
        out.loc[nad27, "coord_datum"] = "NAD83(from NAD27)"
    return out.drop_duplicates("site_no").reset_index(drop=True)


def nwis_coordinates(
    site_nos: list[str], get_text: Callable[[str, Mapping[str, str]], str], chunk: int = 100
) -> pd.DataFrame:
    """Full-precision gauge coordinates from NWIS (HYDROSWOT rounds to 0.01 deg)."""
    url = "https://waterservices.usgs.gov/nwis/site/"
    parts = []
    for i in range(0, len(site_nos), chunk):
        q = {"format": "rdb", "siteOutput": "expanded", "sites": ",".join(site_nos[i : i + chunk])}
        parts.append(parse_nwis_rdb(get_text(url, q)))
    return pd.concat(parts, ignore_index=True) if parts else parse_nwis_rdb("")


def requests_get_text(url: str, params: Mapping[str, str], timeout: float = 60.0) -> str:
    """GET ``url`` and return the body text (lazy ``requests`` import)."""
    import requests

    resp = requests.get(url, params=dict(params), timeout=timeout)
    resp.raise_for_status()
    return str(resp.text)


def requests_get_json(url: str, params: Mapping[str, str], timeout: float = 60.0) -> Any:  # noqa: ANN401
    """GET ``url`` with query ``params`` and return the decoded JSON (lazy ``requests`` import)."""
    import requests

    resp = requests.get(url, params=dict(params), timeout=timeout)
    resp.raise_for_status()
    return resp.json()


#: Base of the exponential retry back-off in seconds (tests set it to 0).
BACKOFF_S: float = 2.0


def _retry(fn: Callable[[], Any], retries: int) -> Any:  # noqa: ANN401
    import time

    backoff_s = BACKOFF_S
    last: Exception | None = None
    for k in range(retries):
        try:
            return fn()
        except Exception as exc:  # network / decoding errors are retried, then recorded
            last = exc
            if k + 1 < retries and backoff_s > 0:
                time.sleep(backoff_s * 2**k)
    raise RuntimeError(f"failed after {retries} attempts: {last}") from last


#: Transient network failures are returned but not cached, so a re-run retries them.
TRANSIENT: Final[frozenset[str]] = frozenset({"excluded_3dep", "excluded_ncat"})


def acquire_site(
    site: Mapping[str, Any],
    line_xy: FloatArray | None,
    p: Operational2Params,
    cache_dir: Path,
    get_json: GetJson,
) -> dict[str, Any]:
    """Fetch (or reuse) one site's NCAT shift and 3DEP transect; return its cache record.

    The record has ``status`` = ``ok`` or an exclusion reason, the transect geometry and request,
    the raw samples and the datum shift. It is written to ``<cache_dir>/<site_no>.json``.
    """
    from pyproj import Transformer

    path = cache_dir / f"{site['site_no']}.json"
    if path.exists():
        cached: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cached
    rec: dict[str, Any] = {"site_no": str(site["site_no"]), "status": "ok"}
    datum = str(site["datum"])
    if not (math.isfinite(float(site["lat"])) and math.isfinite(float(site["lon"]))):
        rec["status"] = "excluded_coordinates"
    elif datum not in p.datum.accepted or not math.isfinite(float(site["wse_a"])):
        rec["status"] = "excluded_datum"
    elif line_xy is None:
        rec["status"] = "excluded_flowline_missing"
    if rec["status"] == "ok":
        to_m = Transformer.from_crs(CRS_GEO, CRS_METRIC, always_xy=True)
        to_g = Transformer.from_crs(CRS_METRIC, CRS_GEO, always_xy=True)
        gx, gy = to_m.transform(float(site["lon"]), float(site["lat"]))
        w_a = float(site["w_a"])
        assert line_xy is not None
        geom = transect_geometry(
            line_xy,
            (gx, gy),
            offset=max(p.transect.offset_factor_w * w_a, p.transect.offset_min_m),
            half_window=p.transect.bearing_half_window_m,
            length=transect_length(w_a, p),
        )
        rec["geometry"] = geom.__dict__
        if geom.gauge_offset > p.transect.max_flowline_offset_m:
            rec["status"] = "excluded_flowline_offset"
    if rec["status"] == "ok":
        rec["datum_shift_m"] = 0.0
        if datum == "NGVD29":
            q = {
                "lat": f"{float(site['lat']):.7f}",
                "lon": f"{float(site['lon']):.7f}",
                "inDatum": "NAD83(1986)",
                "outDatum": "NAD83(1986)",
                "orthoHt": f"{float(site['wse_a']):.3f}",
                "inVertDatum": "NGVD29",
                "outVertDatum": "NAVD88",
            }
            try:
                ncat = _retry(lambda: get_json(p.datum.ncat_url, q), p.acquisition.retries)
                rec["datum_shift_m"] = parse_ncat(ncat)
            except (RuntimeError, ValueError) as exc:
                rec["status"] = "excluded_ncat"
                rec["error"] = str(exc)
    if rec["status"] == "ok":
        (sx, sy), (ex, ey) = geom.endpoints()
        (slon, elon), (slat, elat) = to_g.transform([sx, ex], [sy, ey])  # (lons), (lats)
        g = {
            "paths": [[[slon, slat], [elon, elat]]],
            "spatialReference": {"wkid": 4326},
        }
        q3 = {
            "geometry": json.dumps(g),
            "geometryType": "esriGeometryPolyline",
            "sampleCount": str(p.transect.n_samples),
            "returnFirstValueOnly": "true",
            "f": "json",
        }
        rec["request"] = q3
        try:
            payload = _retry(lambda: get_json(p.transect.service_url, q3), p.acquisition.retries)
            lon, lat, z = parse_samples(payload)
            rec["samples"] = {"lon": lon.tolist(), "lat": lat.tolist(), "z": _nan_none(z)}
        except (RuntimeError, ValueError) as exc:
            rec["status"] = "excluded_3dep"
            rec["error"] = str(exc)
    if rec["status"] not in TRANSIENT:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, allow_nan=False), encoding="utf-8")
    return rec


def _nan_none(z: FloatArray) -> list[float | None]:
    return [float(v) if math.isfinite(v) else None for v in z]

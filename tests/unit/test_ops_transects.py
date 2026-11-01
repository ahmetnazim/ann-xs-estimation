"""Unit tests for OPS-1 v2 terrain inputs (src/operational/transects.py). No network."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import src.operational.transects as tr
from src.operational.config_v2 import load_operational_v2

pytestmark = pytest.mark.unit

P = load_operational_v2().params


def test_anchor_frame_wse_and_datum() -> None:
    site = pd.DataFrame(
        {
            "variant": ["primary", "primary", "sa2"],
            "site_no": ["1", "2", "1"],
            "comid": [10, 20, 10],
            "lat": [40.0, 41.0, 40.0],
            "lon": [-90.0, -91.0, -90.0],
            "W": [20.0, 30.0, 20.0],
            "row_id": [0, 1, 0],
        }
    )
    stage = pd.DataFrame({"row_id": [0, 1], "stage_ft": [5.0, np.nan]})
    alt = pd.DataFrame(
        {"site_no": ["1", "2"], "alt_va": ["100.0", ""], "alt_datum_cd": ["NGVD29", ""]}
    )
    out = tr.anchor_frame(site, stage, alt)
    assert out["site_no"].tolist() == ["1", "2"]
    assert out.loc[0, "wse_a"] == pytest.approx(0.3048 * 105.0)
    assert out.loc[0, "datum"] == "NGVD29"
    assert math.isnan(out.loc[1, "wse_a"])


def test_transect_length_clamps() -> None:
    assert tr.transect_length(1.0, P) == P.transect.length_min_m
    assert tr.transect_length(1.0e5, P) == P.transect.length_max_m
    assert tr.transect_length(50.0, P) == pytest.approx(40.0 * 50.0)


def test_transect_geometry_straight_east_flowing_line() -> None:
    """Flow eastward: upstream offset moves west; the normal is north; gauge offset measured."""
    line = np.array([[0.0, 0.0], [1000.0, 0.0]])
    g = tr.transect_geometry(line, (600.0, 25.0), offset=200.0, half_window=50.0, length=400.0)
    assert g.cx == pytest.approx(400.0)
    assert g.cy == pytest.approx(0.0)
    assert (g.nx, g.ny) == pytest.approx((0.0, 1.0))
    assert g.gauge_offset == pytest.approx(25.0)
    (sx, sy), (ex, ey) = g.endpoints()
    assert (sx, sy, ex, ey) == pytest.approx((400.0, -200.0, 400.0, 200.0))
    # offset beyond the upstream end clamps to the start of the line
    g2 = tr.transect_geometry(line, (100.0, 0.0), offset=500.0, half_window=50.0, length=10.0)
    assert g2.cx == pytest.approx(0.0)


def test_transect_geometry_guards() -> None:
    with pytest.raises(ValueError, match="array"):
        tr.transect_geometry(np.zeros((1, 2)), (0.0, 0.0), offset=1, half_window=1, length=1)
    with pytest.raises(ValueError, match="degenerate"):
        tr.transect_geometry(np.zeros((2, 2)), (0.0, 0.0), offset=1, half_window=1, length=1)


def test_parse_samples_and_ncat() -> None:
    payload = {
        "samples": [
            {"locationId": 1, "location": {"x": -90.1, "y": 40.0}, "value": "NoData"},
            {"locationId": 0, "location": {"x": -90.0, "y": 40.0}, "value": "12.5"},
            {"locationId": 2, "location": {"x": -90.2, "y": 40.0}, "value": "-3.4e38"},
        ]
    }
    lon, _lat, z = tr.parse_samples(payload)
    assert lon.tolist() == [-90.0, -90.1, -90.2]
    assert z[0] == 12.5
    assert np.isnan(z[1:]).all()
    with pytest.raises(ValueError, match="without samples"):
        tr.parse_samples({"error": {"code": 400}})
    assert tr.parse_ncat({"srcOrthoht": "40.000", "destOrthoht": "39.955"}) == pytest.approx(-0.045)
    with pytest.raises(ValueError, match="orthometric"):
        tr.parse_ncat({"x": 1})


def test_channel_centre_flat_run() -> None:
    x = np.arange(11, dtype=float)
    z = np.array([5, 4, 3, 1.0, 1.02, 1.01, 3, 4, np.nan, 0.0, 6])
    xc, zc = tr.channel_centre(x, z, centre_x=4.0, search_half=3.0, flat_tol=0.05)
    assert zc == 1.0
    assert xc == pytest.approx(4.0)  # run 3..5 (the deeper point at x=9 is outside the window)
    with pytest.raises(ValueError, match="no finite"):
        tr.channel_centre(x, np.full(11, np.nan), 4.0, 3.0, 0.05)


def test_parse_nwis_rdb_and_chunking() -> None:
    text = (
        "# comment\n"
        "agency_cd\tsite_no\tdec_lat_va\tdec_long_va\tcoord_datum_cd\n"
        "5s\t15s\t16s\t16s\t10s\n"
        "USGS\t01\t40.0\t-90.0\tNAD83\n"
        "USGS\t02\t35.0\t-100.0\tNAD27\n"
    )
    out = tr.parse_nwis_rdb(text)
    assert out["site_no"].tolist() == ["01", "02"]
    assert out.loc[0, "lat"] == 40.0
    assert out.loc[1, "coord_datum"] == "NAD83(from NAD27)"
    assert out.loc[1, "lon"] != -100.0
    assert tr.parse_nwis_rdb("# only comments\n").empty
    calls: list[str] = []

    def fake(url: str, q: Any) -> str:
        calls.append(q["sites"])
        return text

    res = tr.nwis_coordinates(["01", "02", "03"], fake, chunk=2)
    assert calls == ["01,02", "03"]
    assert len(res) == 4
    assert tr.nwis_coordinates([], fake).empty


def _site(**kw: Any) -> dict[str, Any]:
    base = {
        "site_no": "99",
        "lat": 40.0,
        "lon": -90.0,
        "w_a": 20.0,
        "datum": "NGVD29",
        "wse_a": 100.0,
    }
    return {**base, **kw}


def _line_through(lat: float, lon: float) -> np.ndarray:
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", tr.CRS_METRIC, always_xy=True)
    x, y = t.transform(lon, lat)
    return np.array([[x - 1000.0, y], [x + 1000.0, y]])


def _fake_get_json(fail: str | None = None) -> Any:
    def get(url: str, q: Any) -> Any:
        if fail and fail in url:
            raise OSError("boom")
        if "ncat" in url:
            return {"srcOrthoht": q["orthoHt"], "destOrthoht": str(float(q["orthoHt"]) - 0.2)}
        return {
            "samples": [
                {"locationId": i, "location": {"x": -90.0, "y": 40.0 + i}, "value": "1.0"}
                for i in range(3)
            ]
        }

    return get


def test_acquire_site_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ok (cached and reused), datum, coordinates, flowline and offset exclusions, transients."""
    monkeypatch.setattr(tr, "BACKOFF_S", 0.0)
    line = _line_through(40.0, -90.0)
    rec = tr.acquire_site(_site(), line, P, tmp_path, _fake_get_json())
    assert rec["status"] == "ok"
    assert rec["datum_shift_m"] == pytest.approx(-0.2)
    assert len(rec["samples"]["z"]) == 3
    assert json.loads((tmp_path / "99.json").read_text())["status"] == "ok"
    again = tr.acquire_site(_site(), line, P, tmp_path, _fake_get_json(fail="3DEP"))
    assert again == rec  # reused from cache, no request made
    st = lambda **kw: tr.acquire_site(  # noqa: E731
        _site(site_no=str(kw.pop("no")), **kw), kw.pop("line", line), P, tmp_path, _fake_get_json()
    )
    assert st(no=1, datum="COE1912")["status"] == "excluded_datum"
    assert st(no=2, lat=float("nan"))["status"] == "excluded_coordinates"
    assert (
        tr.acquire_site(_site(site_no="3"), None, P, tmp_path, _fake_get_json())["status"]
        == "excluded_flowline_missing"
    )
    far = line + np.array([0.0, 2000.0])
    assert (
        tr.acquire_site(_site(site_no="4"), far, P, tmp_path, _fake_get_json())["status"]
        == "excluded_flowline_offset"
    )
    nav = tr.acquire_site(_site(site_no="5", datum="NAVD88"), line, P, tmp_path, _fake_get_json())
    assert nav["datum_shift_m"] == 0.0
    bad = tr.acquire_site(_site(site_no="6"), line, P, tmp_path, _fake_get_json(fail="ncat"))
    assert bad["status"] == "excluded_ncat"
    assert not (tmp_path / "6.json").exists()  # transient: not cached
    bad3 = tr.acquire_site(
        _site(site_no="7"), line, P, tmp_path, _fake_get_json(fail="nationalmap")
    )
    assert bad3["status"] == "excluded_3dep"
    assert not (tmp_path / "7.json").exists()

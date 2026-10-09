"""Unit tests for src/data/ehydro.py (docs/3 11.3 E-R1 to E-R8; P6-D10). Synthetic only.

No network: the catalogue and HEAD calls are injected fakes. No eHydro data is read.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest
import shapely

from src.data import ehydro
from src.data.ehydro import (
    Cells,
    EhydroError,
    EhydroStop,
    build_locations,
    catalogue_frame,
    catalogue_where,
    condition_codes,
    coverage_fraction,
    depth_surface,
    eligible_surveys,
    pool_cells,
    query_catalogue,
    reach_values,
    select_latest,
)

pytestmark = pytest.mark.unit

_MS_2018 = 1_514_764_800_000  # 2018-01-01T00:00:00Z
_MS_2016 = 1_451_606_400_000  # 2016-01-01T00:00:00Z
_DAY_MS = 86_400_000


# ----------------------------------------------------------------------------- catalogue
def _feature(
    sid: str,
    lon: float,
    *,
    stype: str = "CS",
    area: str | None = "A1",
    start: int = _MS_2018,
    url: str | None = None,
    pa: float = 1.0,
) -> dict[str, Any]:
    ring = [[lon, 40.0], [lon + 0.01, 40.0], [lon + 0.01, 40.01], [lon, 40.01], [lon, 40.0]]
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {
            "surveyjobidpk": sid,
            "sdsfeaturename": "x",
            "surveytype": stype,
            "channelareaidfk": area,
            "usacedistrictcode": "LRL",
            "surveydatestart": start,
            "surveydateend": start,
            "sourcedatalocation": url or f"https://example.org/{sid}.ZIP",
            "sourceprojection": "x",
            "projectedarea": pa,
        },
    }


class FakeCatalogue:
    def __init__(self, features: list[dict[str, Any]], page: int, count: int | None = None):
        self.features = features
        self.page = page
        self.count = len(features) if count is None else count
        self.calls: list[dict[str, str]] = []

    def __call__(self, url: str, params: Any) -> bytes:
        self.calls.append(dict(params))
        if params.get("returnCountOnly") == "true":
            return json.dumps({"count": self.count}).encode()
        offset = int(params["resultOffset"])
        chunk = self.features[offset : offset + self.page]
        return json.dumps({"type": "FeatureCollection", "features": chunk}).encode()


def test_catalogue_where_is_inclusive_of_end_date() -> None:
    where = catalogue_where(date(2017, 1, 1), date(2023, 12, 31))
    assert where == ("surveydatestart >= DATE '2017-01-01' AND surveydatestart < DATE '2024-01-01'")


def test_query_catalogue_pages_until_empty_and_checks_count() -> None:
    feats = [_feature(f"s{i}", -85.0 + i * 0.1) for i in range(5)]
    fake = FakeCatalogue(feats, page=2)
    got, pages = query_catalogue("https://x/FeatureServer/0", "1=1", fake)
    assert [f["properties"]["surveyjobidpk"] for f in got] == [f"s{i}" for i in range(5)]
    assert len(pages) == 3
    assert [c.get("resultOffset") for c in fake.calls[1:]] == ["0", "2", "4", "5"]
    with pytest.raises(EhydroError, match="server count 6"):
        query_catalogue("https://x/FeatureServer/0", "1=1", FakeCatalogue(feats, 2, count=6))


def test_condition_codes_and_stop() -> None:
    codes, counts = condition_codes(pd.Series(["CS", "AD", "BD", None, "Condition Survey"]))
    assert codes == ("CS", "Condition Survey")
    assert counts["<null>"] == 1
    with pytest.raises(EhydroStop, match="E-R2"):
        condition_codes(pd.Series(["AD", "BD"]))


def test_catalogue_frame_repairs_invalid_and_parses_dates() -> None:
    bow = _feature("bow", -85.0)
    bow["geometry"]["coordinates"] = [
        [[-85.0, 40.0], [-84.99, 40.01], [-84.99, 40.0], [-85.0, 40.01], [-85.0, 40.0]]
    ]
    gdf, repaired = catalogue_frame([bow, _feature("ok", -84.0)])
    assert repaired == 1
    assert gdf.crs.to_epsg() == 5070
    assert bool(gdf.geometry.is_valid.all())
    assert gdf["surveydatestart"].iloc[0] == pd.Timestamp("2018-01-01", tz="UTC")


def _frame_flowlines(cat: Any) -> Any:
    import geopandas as gpd

    # One flowline crossing the footprint of every survey west of -84.5 deg.
    cx = cat.geometry.centroid
    west = cat.loc[cat.geometry.centroid.to_crs(4326).x < -84.5]
    lines = [shapely.linestrings([[p.x - 2000, p.y], [p.x + 2000, p.y]]) for p in cx[west.index]]
    return gpd.GeoDataFrame(
        {"comid": range(len(lines)), "river": "Ohio River", "vpu": "05", "lengthkm": 4.0},
        geometry=lines,
        crs=5070,
    )


def test_eligibility_steps_are_counted() -> None:
    feats = [
        _feature("ok", -85.0),
        _feature("ad", -85.1, stype="AD"),
        _feature("old", -85.2, start=_MS_2016),
        _feature("ftp", -85.3, url="ftp://example.org/x.zip"),
        _feature("east", -84.0),  # no frame flowline
    ]
    cat, _ = catalogue_frame(feats)
    flow = _frame_flowlines(cat)
    out, counts = eligible_surveys(cat, flow, ("CS",), date(2017, 1, 1), date(2023, 12, 31))
    assert out["surveyjobidpk"].tolist() == ["ok"]
    assert out["rivers"].tolist() == ["Ohio River"]
    assert counts == {
        "catalogue": 5,
        "condition_survey": 4,
        "in_window": 3,
        "https_zip": 2,
        "with_footprint": 2,
        "intersects_frame": 1,
    }


def test_select_latest_per_channel_area_with_ties() -> None:
    frame = pd.DataFrame(
        {
            "surveyjobidpk": ["a", "b", "c", "d", "e", "f"],
            "channelareaidfk": ["A1", "A1", "A1", "A2", None, None],
            "surveydatestart": pd.to_datetime(
                [_MS_2018, _MS_2018 + _DAY_MS, _MS_2018 + _DAY_MS, _MS_2018, _MS_2018, _MS_2018],
                unit="ms",
                utc=True,
            ),
            "projectedarea": [1.0, 2.0, 2.0, 1.0, 1.0, 1.0],
        }
    )
    out = select_latest(frame)
    # A1: b and c tie on date and area -> smaller ID b; e and f each form their own group.
    assert out["surveyjobidpk"].tolist() == ["b", "d", "e", "f"]


# ----------------------------------------------------------------------------- surface
def _grid_points(x0: float, x1: float, y0: float, y1: float, step: float, depth: Any) -> Any:
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    x, y = xs.ravel(), ys.ravel()
    return x, y, depth(x, y)


def test_depth_surface_reproduces_a_plane_inside_the_footprint() -> None:
    x, y, d = _grid_points(0.0, 1000.0, 0.0, 200.0, 2.5, lambda x, y: 2.0 + 0.001 * x)
    footprint = shapely.box(100.0, 20.0, 900.0, 180.0)
    cells = depth_surface(x, y, d, footprint, 5.0)
    assert len(cells) == 160 * 32  # every cell centre of the box, all wetted
    cx = (cells.ix + 0.5) * 5.0
    # binned value = plane at the mean point x (cell centre - 1.25 m), reproduced exactly
    np.testing.assert_allclose(cells.d, 2.0 + 0.001 * (cx - 1.25), rtol=0, atol=1e-12)


def test_depth_surface_hull_and_wetted_rules() -> None:
    # Soundings only on the left half; depth negative (a bar) for x > 303.
    x, y, d = _grid_points(0.0, 500.0, 0.0, 100.0, 5.0, lambda x, y: 1.0 - x / 303.0)
    footprint = shapely.box(0.0, 0.0, 1000.0, 100.0)
    cells = depth_surface(x, y, d, footprint, 5.0)
    cx = (cells.ix + 0.5) * 5.0
    # (iv) the binned point x = 305 is dry, so the last wetted centre is 302.5; (iii) no cell
    # beyond the hull (last binned centre 497.5) even though the footprint reaches 1000 m.
    assert cx.max() == 302.5
    assert bool((cells.d > 0).all())


def test_depth_surface_degenerate_inputs_give_no_cells() -> None:
    fp = shapely.box(0, 0, 100, 100)
    assert len(depth_surface(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.ones(2), fp, 5.0)) == 0
    line_x = np.arange(0.0, 100.0, 5.0)
    assert len(depth_surface(line_x, line_x, np.ones_like(line_x), fp, 5.0)) == 0


def test_depth_surface_tiles_match_untiled(monkeypatch: pytest.MonkeyPatch) -> None:
    x, y, d = _grid_points(0.0, 400.0, 0.0, 400.0, 7.0, lambda x, y: 3.0 + np.sin(x / 50.0))
    fp = shapely.Polygon([(10, 10), (390, 30), (370, 390), (20, 350)])
    whole = depth_surface(x, y, d, fp, 5.0)
    monkeypatch.setattr(ehydro, "_TILE", 7)
    tiled = depth_surface(x, y, d, fp, 5.0)
    a = pd.DataFrame({"ix": whole.ix, "iy": whole.iy, "d": whole.d}).sort_values(["ix", "iy"])
    b = pd.DataFrame({"ix": tiled.ix, "iy": tiled.iy, "d": tiled.d}).sort_values(["ix", "iy"])
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_pool_cells_later_survey_wins() -> None:
    old = Cells(np.array([0, 1]), np.array([0, 0]), np.array([1.0, 1.0]))
    new = Cells(np.array([1, 2]), np.array([0, 0]), np.array([5.0, 5.0]))
    pooled = pool_cells([old, new])
    got = dict(zip(pooled.ix.tolist(), pooled.d.tolist(), strict=True))
    assert got == {0: 1.0, 1: 5.0, 2: 5.0}
    assert len(pool_cells([])) == 0


def test_reach_values_follow_zarrabi_formulas() -> None:
    import geopandas as gpd

    # 20 x 4 cells of 5 m in catchment 1 (depth 2), 10 x 4 cells in catchment 2 (depth 4).
    ix, iy = np.meshgrid(np.arange(30), np.arange(4))
    ix, iy = ix.ravel(), iy.ravel()
    d = np.where(ix < 20, 2.0, 4.0)
    cells = Cells(ix.astype(np.int64), iy.astype(np.int64), d)
    catch = gpd.GeoDataFrame(
        {"comid": [1, 2]},
        geometry=[shapely.box(0, -10, 100, 30), shapely.box(100, -10, 150, 30)],
        crs=5070,
    )
    out = reach_values(cells, catch, {1: 0.1, 2: 0.05}, 5.0).set_index("comid")
    assert out.loc[1, "n_cells"] == 80
    assert out.loc[1, "dbar"] == pytest.approx(2.0)
    assert out.loc[1, "A"] == pytest.approx(80 * 2.0 * 25.0 / 100.0)  # Σ d c² / L
    assert out.loc[1, "W"] == pytest.approx(80 * 25.0 / 100.0)  # = wetted area / L = 20 m
    assert out.loc[2, "W"] == pytest.approx(40 * 25.0 / 50.0)
    assert out.loc[2, "A"] == pytest.approx(out.loc[2, "W"] * out.loc[2, "dbar"])


def test_coverage_fraction() -> None:
    import geopandas as gpd

    lines = gpd.GeoDataFrame(
        {"comid": [1, 2]},
        geometry=[shapely.linestrings([[0, 0], [1000, 0]]), shapely.linestrings([[0, 0], [0, 10]])],
        crs=5070,
    )
    cov = coverage_fraction(lines, [shapely.box(0, -50, 400, 50)], 100.0)
    assert cov.loc[1] == pytest.approx(0.5)  # 0..500 m inside the 100 m-buffered box
    assert cov.loc[2] == pytest.approx(1.0)
    assert (coverage_fraction(lines, [], 100.0) == 0.0).all()


# ----------------------------------------------------------------------------- locations
def _chain(n: int, river: str, start: int, length: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "comid": np.arange(start, start + n),
            "river": river,
            "lengthkm": length,
            "hydroseq": np.arange(start + n, start, -1),  # first COMID is the most upstream
            "W": 100.0,
            "dbar": 4.0,
        }
    )


def test_build_locations_chunks_and_remainder_rule() -> None:
    reaches = _chain(7, "Ohio River", 100, 3.0)  # 21 km: chunks 12 km, 9 km
    reaches.loc[reaches["comid"].eq(100), "W"] = 200.0
    edges = pd.DataFrame({"from_comid": np.arange(100, 106), "to_comid": np.arange(101, 107)})
    locs, members = build_locations(reaches, edges, 10.0)
    assert locs["n_comids"].tolist() == [4, 3]
    assert locs["length_km"].tolist() == [12.0, 9.0]
    assert locs["loc_id"].tolist() == ["Ohio River:100", "Ohio River:104"]
    assert locs.loc[0, "W"] == pytest.approx((200 + 3 * 100) / 4)
    assert locs.loc[0, "A"] == pytest.approx(locs.loc[0, "W"] * locs.loc[0, "dbar"])
    assert members.groupby("loc_id").size().tolist() == [4, 3]

    short = _chain(4, "Ohio River", 100, 3.0)
    locs2, _ = build_locations(short.iloc[:3].assign(lengthkm=[4.0, 4.0, 3.5]), edges, 10.0)
    assert locs2["n_comids"].tolist() == [3]
    tail = short.assign(lengthkm=[6.0, 5.0, 2.0, 2.0])  # 11 km closes; 4 km < 5 km joins it
    locs3, _ = build_locations(tail, edges, 10.0)
    assert locs3["n_comids"].tolist() == [4]


def test_build_locations_splits_runs_by_name_and_connectivity() -> None:
    a = _chain(2, "Ohio River", 1, 6.0)
    b = _chain(2, "Green River", 3, 6.0)
    c = _chain(1, "Ohio River", 10, 6.0)  # disconnected from a
    reaches = pd.concat([a, b, c], ignore_index=True)
    edges = pd.DataFrame({"from_comid": [1, 2, 3], "to_comid": [2, 3, 4]})  # 2 -> 3 changes name
    locs, members = build_locations(reaches, edges, 10.0)
    assert sorted(locs["loc_id"]) == ["Green River:3", "Ohio River:1", "Ohio River:10"]
    assert set(members.loc[members["loc_id"].eq("Ohio River:1"), "comid"]) == {1, 2}
    with pytest.raises(ValueError, match="one row per COMID"):
        build_locations(pd.concat([a, a]), edges, 10.0)


# ----------------------------------------------------------------------------- select CLI
def _tmp_settings(tmp_path: Any) -> Any:
    from src.config import load_settings

    settings = load_settings()
    runtime = settings.runtime.model_copy(
        update={"reports_dir": tmp_path / "reports", "dvc_store": tmp_path / "store"}
    )
    return settings.model_copy(update={"runtime": runtime})


def test_run_select_writes_report_and_stops_on_size(tmp_path: Any) -> None:
    feats = [
        _feature("s1", -85.0, area="A1"),
        _feature("s2", -85.0, area="A1", start=_MS_2018 + _DAY_MS),  # later: wins A1
        _feature("s3", -85.2, area="A2"),
    ]
    cat, _ = catalogue_frame(feats)
    flow = _frame_flowlines(cat)
    settings = _tmp_settings(tmp_path)
    report = ehydro.run_select(
        settings, get=FakeCatalogue(feats, page=2), head=lambda url: 1_000, flowlines=flow
    )
    assert [s["surveyjobidpk"] for s in report["selected"]] == ["s2", "s3"]
    assert report["bytes_total_known"] == 2_000
    assert report["stop"] is False
    assert report["condition_codes"] == ["CS"]
    assert report["surveys_per_river"]["Ohio River"] == 2
    assert report["params_sha256"] == settings.params_sha256
    on_disk = json.loads((tmp_path / "reports" / "ehydro_selection.json").read_text())
    assert on_disk["counts"]["selected"] == 2
    pages = sorted((tmp_path / "store" / "ehydro_catalogue").glob("*/page_*.geojson"))
    assert len(pages) == 2

    with pytest.raises(EhydroStop, match="E-R4"):
        ehydro.run_select(
            _tmp_settings(tmp_path / "b"),
            get=FakeCatalogue(feats, page=2),
            head=lambda url: None if url.endswith("s3.ZIP") else 1_000,
            flowlines=flow,
        )
    stopped = json.loads((tmp_path / "b" / "reports" / "ehydro_selection.json").read_text())
    assert stopped["stop"] is True
    assert stopped["n_without_content_length"] == 1

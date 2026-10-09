"""Unit tests for the eHydro survey reader and the ind_prepare stage (P6-D10). Synthetic only.

Sources: docs/3 11.3 (E-R5 to E-R12) and 11.4 (E-R5a to E-R5e). No eHydro survey is read: the
geodatabases below are written by the test itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pandera.errors
import pytest
import shapely

from src.config import FT_TO_M
from src.data.ehydro import (
    STATEMENT_RE,
    Cells,
    EhydroError,
    EhydroStop,
    classify_plane_text,
    classify_survey,
    decide_plane,
    plane_statement,
    read_survey,
)
from src.data.ind_prepare import (
    EXPECTED_PLANE_COUNTS,
    SurveyMeta,
    dedupe_l8,
    derive_locations,
    l8_status,
    plane_summary,
    selected_surveys,
)
from src.data.schemas import EhydroInd

pytestmark = pytest.mark.unit

_UTM15 = "EPSG:26915"


# ----------------------------------------------------------------------------- E-R5c text classes
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Soundings ... indicate depths below Low Water Reference Plane 2007 (LWRP07)", "water"),
        ("indicate depths below Columbia River Datum", "water"),
        ("NOS MLLW, Tidal Epoch: 1983-2001", "water"),
        ("depths below Low Water Datum of 569.2 Feet IGLD 1985", "water"),
        ("a datum plane which is 73.3 feet (Minimum Pool Elevation) above NAVD 88", "water"),
        ("indicated depths are referenced to water surface at time of collection", "water"),
        ("indicate depths on the date of survey", "water"),
        ("indicate depths below Mississippi River Datum (MRD)", "water"),
        ("indicate depths below National Geodetic Vertical Datum of 1929 (NGVD_29)", "geodetic"),
        ("Values are based on the North American Vertical Datum of 1988 (NAVD 88)", "geodetic"),
        ("MSL1912", "geodetic"),
        ("The Z-value represents the river bottom elevation", "geodetic"),
        ("Station datum TBD", None),
        ("", None),
        (None, None),
    ],
)
def test_classify_plane_text(text: str | None, expected: str | None) -> None:
    assert classify_plane_text(text) == expected


def test_statement_ends_before_a_capital_letter_only() -> None:
    text = "Vertical Datum: depths below LWRP at r. m. 5. The information depicted is old."
    match = STATEMENT_RE.search(text)
    assert match is not None
    assert match.group(1) == "depths below LWRP at r. m. 5"
    gen = "- Vertical Reference Datum: MLLW - Implied Vertical Accuracy: +/- 0.5 Feet"
    match = STATEMENT_RE.search(gen)
    assert match is not None
    assert match.group(1) == "MLLW"
    assert STATEMENT_RE.search("<vert>Vertical Datum Name: NAVD 88</vert>") is not None


# ----------------------------------------------------------------------------- E-R5c sources
def _xml(abstract: str = "", altdatum: str | None = None, edomv: str | None = None) -> str:
    alt = f"<vertdef><altsys><altdatum>{altdatum}</altdatum></altsys></vertdef>" if altdatum else ""
    attr = (
        f"<attr><attrlabl>elevationDatum</attrlabl><attrdomv><edom><edomv>{edomv}</edomv>"
        "<edomvd>NAVD_88 North American Vertical Datum of 1988</edomvd></edom></attrdomv></attr>"
        if edomv
        else ""
    )
    return f"<metadata><abstract>{abstract}</abstract>{alt}<eainfo>{attr}</eainfo></metadata>"


def test_plane_statement_sources_in_order(tmp_path: Path) -> None:
    a = tmp_path / "a"
    a.mkdir()
    (a / "a.XML").write_text(_xml("Survey. Vertical Datum: depths below CRD. Other text."))
    (a / "a.GEN").write_text("- Vertical Reference Datum: NAVD 88\n")
    assert plane_statement(a) == ("depths below CRD", ())

    b = tmp_path / "b"
    b.mkdir()
    (b / "b.XML").write_text(_xml("Channel survey.", altdatum="MLLW", edomv="NAVD_88"))
    (b / "b.GEN").write_text("- Vertical Reference Datum:          MLLW\r\n - Implied Vertical")
    assert plane_statement(b) == ("MLLW", ("NAVD_88",))  # .GEN statement before <altdatum>

    c = tmp_path / "c"
    c.mkdir()
    (c / "c.XML").write_text(_xml("Channel survey.", altdatum="MLLW"))
    assert plane_statement(c) == ("<altdatum> MLLW", ())

    d = tmp_path / "d"
    d.mkdir()
    assert plane_statement(d) == (None, ())


def test_decide_plane_precedence() -> None:
    s1 = decide_plane("depths below LWRP", ("NAVD_88",), ("NGVD1929",))
    assert (s1.cls, s1.source) == ("water", "S1")
    geo = decide_plane("Values are based on NAVD 88", (), ("MLLW",))
    assert (geo.cls, geo.source) == ("geodetic", "S1")
    fallback = decide_plane("Station datum", (), ("mllw",))
    assert (fallback.cls, fallback.source) == ("water", "S3")
    both = decide_plane(None, ("MLLW",), ("mllw",))
    assert (both.cls, both.source) == ("water", "S2+S3")
    conflict = decide_plane(
        None, ("SD (NAVD_88 North American Vertical Datum of 1988)",), ("mllw",)
    )
    assert (conflict.cls, conflict.source) == ("unresolved", "conflict")
    none = decide_plane(None, (), ("<null>",))
    assert (none.cls, none.source) == ("unresolved", "none")


# ----------------------------------------------------------------------------- E-R5a/b/d reader
def _write_gdb(
    survey: Path,
    z_ft: list[float | None],
    *,
    uom: str = "usSurveyFoot",
    datum: str = "",
    polygon: shapely.Polygon | None = None,
) -> None:
    survey.mkdir(parents=True, exist_ok=True)
    gdb = survey / f"{survey.name}.gdb"
    n = len(z_ft)
    pts = gpd.GeoDataFrame(
        {
            "Z_depth": pd.array(z_ft, dtype="Float64").astype("float64"),
            "elevationUOM": [uom] * n,
            "elevationDatum": [datum] * n,
        },
        geometry=shapely.points(500_000.0 + 10.0 * np.arange(n), np.full(n, 4_400_000.0)),
        crs=_UTM15,
    )
    pts.to_file(gdb, layer="SurveyPoint", driver="OpenFileGDB")
    poly = polygon or shapely.box(499_990.0, 4_399_990.0, 500_000.0 + 10.0 * n, 4_400_010.0)
    job = gpd.GeoDataFrame({"SurveyId": ["x"]}, geometry=[poly], crs=_UTM15)
    job.to_file(gdb, layer="SurveyJob", driver="OpenFileGDB")


def test_read_survey_converts_feet_drops_nulls_and_reprojects(tmp_path: Path) -> None:
    survey = tmp_path / "S1"
    _write_gdb(survey, [10.0, None, 20.0, 30.0])
    out = read_survey(survey)
    assert out.n_read == 4
    assert out.n_null_depth == 1
    assert out.n_empty_geometry == 0
    assert out.units == ("usSurveyFoot",)
    np.testing.assert_allclose(out.depth_m, np.array([10.0, 20.0, 30.0]) * FT_TO_M, rtol=0, atol=0)
    expected = gpd.GeoSeries(
        shapely.points([500_000.0, 500_020.0, 500_030.0], [4_400_000.0] * 3), crs=_UTM15
    ).to_crs("EPSG:5070")
    np.testing.assert_allclose(out.x, expected.x.to_numpy(), atol=1e-6)
    np.testing.assert_allclose(out.y, expected.y.to_numpy(), atol=1e-6)
    assert out.x.dtype == np.float64
    assert out.footprint.contains(shapely.Point(out.x[0], out.y[0]))
    assert out.footprint_repaired is False


def test_read_survey_repairs_footprint_and_stops_on_units(tmp_path: Path) -> None:
    bowtie = shapely.Polygon(
        [(499_990, 4_399_990), (500_050, 4_400_010), (500_050, 4_399_990), (499_990, 4_400_010)]
    )
    _write_gdb(tmp_path / "S2", [1.0, 2.0], polygon=bowtie)
    assert read_survey(tmp_path / "S2").footprint_repaired is True
    _write_gdb(tmp_path / "S3", [1.0, 2.0], uom="meter")
    with pytest.raises(EhydroStop, match="E-R5b"):
        read_survey(tmp_path / "S3")
    (tmp_path / "S4").mkdir()
    with pytest.raises(EhydroError, match="0 geodatabases"):
        read_survey(tmp_path / "S4")


def test_classify_survey_uses_field_values_last(tmp_path: Path) -> None:
    _write_gdb(tmp_path / "S5", [1.0, 2.0], datum="MLLW")
    (tmp_path / "S5" / "S5.XML").write_text(_xml("Channel survey."))
    decision = classify_survey(tmp_path / "S5")
    assert (decision.cls, decision.source, decision.field_values) == ("water", "S3", ("MLLW",))


# ----------------------------------------------------------------------------- selection, planes
def _record(name: str, start: str = "2021-05-20T00:00:00+00:00") -> dict[str, Any]:
    return {
        "surveyjobidpk": name,
        "surveydatestart": start,
        "district": "CELRP",
        "rivers": "Ohio River;Green River",
        "vpus": "05",
        "url": f"https://example.org/eh/CELRP/{name}.ZIP",
    }


def test_selected_surveys_checks_url_order() -> None:
    sel = {"selected": [_record("A_1"), _record("B%202")]}
    urls = [r["url"] for r in sel["selected"]]
    out = selected_surveys(sel, urls)
    assert [s.name for s in out] == ["A_1", "B 2"]
    assert out[0].rivers == ("Ohio River", "Green River")
    assert out[0].surveydatestart == pd.Timestamp("2021-05-20", tz="UTC")
    with pytest.raises(EhydroError, match="differ"):
        selected_surveys(sel, urls[::-1])


def test_plane_summary_counts_and_stop_rule() -> None:
    from src.data.ehydro import PlaneDecision

    surveys = [
        SurveyMeta(n, n, pd.Timestamp("2021-01-01", tz="UTC"), "D", ("Ohio River",), ("05",))
        for n in ("a", "b", "c", "d")
    ]

    def dec(cls: str) -> PlaneDecision:
        return PlaneDecision(cls, "S1", "x", (), ())

    half = {"a": dec("water"), "b": dec("water"), "c": dec("geodetic"), "d": dec("unresolved")}
    out = plane_summary(surveys, half, ["Ohio River", "Green River"])
    assert out["classes"] == {"water": 2, "geodetic": 1, "unresolved": 1}
    assert out["stop"] is False  # exactly 50 % excluded: not more than half
    assert out["surveys_per_river"] == {
        "Ohio River": {"selected": 4, "water": 2},
        "Green River": {"selected": 0, "water": 0},
    }
    assert out["rivers_with_water_survey"] == 1
    assert out["matches_expected"] is False
    most = dict(half, b=dec("geodetic"))
    assert plane_summary(surveys, most, ["Ohio River"])["stop"] is True
    assert sum(EXPECTED_PLANE_COUNTS.values()) == 483


# ----------------------------------------------------------------------------- L8
def _l8_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame({"site_no": ["00000001"], "comid": [5], "x5070": [0.0], "y5070": [0.0]})
    line = shapely.linestrings
    locs = pd.DataFrame(
        {
            "loc_id": ["a", "a", "b", "b", "c"],
            "comid": [7, 5, 8, 10, 9],
            "geometry": [
                line([[50_000.0, 0.0], [51_000.0, 0.0]]),
                line([[51_000.0, 0.0], [52_000.0, 0.0]]),
                line([[5_000.0, 0.0], [3_000.0, 0.0]]),
                line([[3_000.0, 0.0], [900.0, 0.0]]),
                line([[5_000.0, 0.0], [6_000.0, 0.0]]),
            ],
        }
    )
    return locs, train


def test_l8_status_matches_dedupe_counts() -> None:
    locs, train = _l8_inputs()
    status = l8_status(locs, train, 1000.0)
    assert status.to_dict() == {"a": "dropped_comid", "b": "dropped_distance", "c": "kept"}
    kept, counts = dedupe_l8(locs, train, 1000.0)
    assert kept["loc_id"].tolist() == ["c"]
    assert counts == {"locations": 3, "dropped_comid": 1, "dropped_distance_only": 1, "kept": 1}


# ----------------------------------------------------------------------------- E-R6 to E-R9
def _params() -> Any:
    from src.config import load_settings

    return load_settings().protocol.ehydro  # cell 5 m, coverage 0.8, buffer 100 m, 10 km


def _river_fixture() -> dict[str, Any]:
    """Four 6 km COMIDs along y = 0; a 100 m wide, 2 m deep channel surveyed over x < 18 km."""
    c = 5.0
    ix, iy = np.meshgrid(np.arange(0, int(18_000 / c)), np.arange(int(-50 / c), int(50 / c)))
    cells = Cells(ix.ravel().astype(np.int64), iy.ravel().astype(np.int64), np.full(ix.size, 2.0))
    comids = [1, 2, 3, 4]
    flowlines = gpd.GeoDataFrame(
        {
            "comid": comids,
            "river": ["Test River"] * 4,
            "vpu": ["05", "05", "05", "06"],
            "lengthkm": [6.0] * 4,
        },
        geometry=[
            shapely.linestrings([[6000.0 * i, 0.0], [6000.0 * (i + 1), 0.0]]) for i in range(4)
        ],
        crs="EPSG:5070",
    )
    catchments = gpd.GeoDataFrame(
        {"comid": [*comids, 99]},
        geometry=[shapely.box(6000.0 * i, -1000.0, 6000.0 * (i + 1), 1000.0) for i in range(4)]
        + [shapely.box(0.0, 1000.0, 1.0, 1001.0)],
        crs="EPSG:5070",
    )
    vaa = pd.DataFrame(
        {"comid": comids, "hydroseq": [40, 30, 20, 10], "DA_km2": [100.0, 110.0, 120.0, 130.0]}
    )
    edges = pd.DataFrame({"from_comid": [1, 2, 3], "to_comid": [2, 3, 4]})
    footprints = [shapely.box(0.0, -60.0, 18_000.0, 60.0)]
    return {
        "cells": cells,
        "footprints": footprints,
        "flowlines": flowlines,
        "catchments": catchments,
        "vaa": vaa,
        "edges": edges,
    }


def test_derive_locations_end_to_end() -> None:
    fx = _river_fixture()
    sites = pd.DataFrame(
        {"site_no": ["00000001"], "comid": [77], "x5070": [15_000.0], "y5070": [500.0]}
    )
    table, counts = derive_locations(**fx, train_sites=sites, params=_params(), radius_m=1000.0)
    assert counts["comids_with_cells"] == 3
    assert counts["comids_dropped_by_coverage"] == 0
    assert counts["comids_retained"] == 3  # COMID 4: 100 m of 6 km covered, no cells
    assert table["loc_id"].tolist() == ["Test River:1", "Test River:3"]
    assert table["comids"].tolist() == ["1;2", "3"]
    assert table["comid_outlet"].tolist() == [2, 3]
    np.testing.assert_allclose(table["DA_km2"], [110.0, 120.0])
    np.testing.assert_allclose(table["dbar"], [2.0, 2.0])
    np.testing.assert_allclose(table["W"], [100.0, 100.0])  # A = 2 m x 100 m
    np.testing.assert_allclose(table["A"], [200.0, 200.0])
    assert table["dedupe_status"].tolist() == ["kept", "dropped_distance"]
    assert counts["l8"] == {"locations": 2, "dropped_comid": 0, "dropped_distance": 1, "kept": 1}
    EhydroInd.validate(table)


def test_derive_locations_coverage_and_duplicates() -> None:
    fx = _river_fixture()
    fx["footprints"] = [shapely.box(0.0, -60.0, 9_000.0, 60.0)]  # COMID 2 only half covered
    sites = pd.DataFrame({"site_no": ["00000001"], "comid": [1], "x5070": [1e7], "y5070": [1e7]})
    table, counts = derive_locations(**fx, train_sites=sites, params=_params(), radius_m=1000.0)
    assert counts["comids_dropped_by_coverage"] == 2  # COMIDs 2 and 3 have cells, not coverage
    assert table["comids"].tolist() == ["1"]
    assert table["dedupe_status"].tolist() == ["dropped_comid"]
    dup = dict(fx, flowlines=pd.concat([fx["flowlines"], fx["flowlines"].iloc[[0]]]))
    with pytest.raises(EhydroError, match="E-R1"):
        derive_locations(**dup, train_sites=sites, params=_params(), radius_m=1000.0)


def test_ehydro_ind_schema_rejects_inconsistent_rows() -> None:
    fx = _river_fixture()
    sites = pd.DataFrame({"site_no": ["00000001"], "comid": [77], "x5070": [1e7], "y5070": [1e7]})
    table, _ = derive_locations(**fx, train_sites=sites, params=_params(), radius_m=1000.0)
    bad_area = table.assign(A=table["A"] * 1.01)
    with pytest.raises(pandera.errors.SchemaError):
        EhydroInd.validate(bad_area)
    bad_status = table.assign(dedupe_status="maybe")
    with pytest.raises(pandera.errors.SchemaError):
        EhydroInd.validate(bad_status)
    bad_count = table.assign(n_comids=np.int64(5))
    with pytest.raises(pandera.errors.SchemaError):
        EhydroInd.validate(bad_count)


# ----------------------------------------------------------------------------- stage STOP
def test_run_stops_before_reading_depths_when_most_surveys_are_excluded(tmp_path: Path) -> None:
    from src.config import AcquireUnit, load_settings
    from src.data import ind_prepare

    raw = tmp_path / "raw"
    for name in ("S_A", "S_B"):
        _write_gdb(raw / "ehydro" / name, [1.0, 2.0])  # no metadata, blank datum: unresolved
    (raw / "ehydro" / "MANIFEST.sha256").write_text("x\n")
    (raw / "nhdplusv21").mkdir(parents=True)
    (raw / "nhdplusv21" / "MANIFEST.sha256").write_text("y\n")
    (tmp_path / "proc").mkdir()
    (tmp_path / "proc" / "site_table.parquet").write_bytes(b"z")
    records = [_record("S_A"), _record("S_B")]
    reports = tmp_path / "reports"
    reports.mkdir()
    selection = {"selected": records, "counts": {}, "bytes_total_known": 0, "condition_codes": []}
    (reports / "ehydro_selection.json").write_text(json.dumps(selection))

    settings = load_settings()
    runtime = settings.runtime.model_copy(
        update={"raw_dir": raw, "reports_dir": reports, "processed_dir": tmp_path / "proc"}
    )
    acquire = settings.protocol.acquire.model_copy(
        update={"ehydro": AcquireUnit(urls=tuple(r["url"] for r in records))}
    )
    protocol = settings.protocol.model_copy(update={"acquire": acquire})
    stopped = settings.model_copy(update={"runtime": runtime, "protocol": protocol})
    with pytest.raises(EhydroStop, match="E-R5"):
        ind_prepare.run(stopped)
    report = json.loads((reports / "ind_prepare.json").read_text())
    assert report["reference_plane"]["classes"] == {"water": 0, "geodetic": 0, "unresolved": 2}
    assert "surveys_read" not in report
    assert not (tmp_path / "proc" / "ind").exists()

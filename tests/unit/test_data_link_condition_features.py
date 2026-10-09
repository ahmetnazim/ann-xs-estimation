"""Unit tests for src/data/link.py, condition_extract.py and features.py (Phase 5, Batch C).

Synthetic only: flowlines are shapely lines in EPSG:5070; the aridity raster is a 4 x 4 GeoTIFF
written to ``tmp_path``.

Sources: docs/3 2.3-2.4 (R12-R14), 2.5 (gage-reach link),
2.6 (condition extraction), 3.5 L4 (banned predictors), 10.7 (aridity zonal mean).
"""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from shapely.geometry import LineString, box

from src.data import condition_extract as ce
from src.data import features as ft
from src.data.nhdplus import CRS_ALBERS
from src.data.schemas import assert_no_banned

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------------- link
def hline(x0: float, y: float) -> LineString:
    """A 2 km horizontal flowline centred on ``x0`` at height ``y``."""
    return LineString([(x0 - 1000.0, y), (x0 + 1000.0, y)])


@pytest.fixture
def network() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    sites = pd.DataFrame(
        {
            "site_no": ["0000000A", "0000000B", "0000000C", "0000000D", "0000000E", "0000000F"],
            "x5070": [0.0, 10_000.0, 0.0, 20_000.0, 30_000.0, 40_000.0],
            "y5070": [0.0, 0.0, -20.0, 0.0, 0.0, 0.0],
            "DA_site_km2": [100.0, 100.0, 500.0, 100.0, 100.0, 100.0],
        }
    )
    flow = gpd.GeoDataFrame(
        {"comid": [1, 2, 3, 7, 5, 4, 6, 8]},
        geometry=[
            hline(0, 10),  # 1: GageLoc for A and C
            hline(10_000, 100),  # 2: B snaps here (DA passes)
            hline(10_000, -50),  # 3: nearer to B but DA 1000 fails
            hline(20_000, 600),  # 7: beyond 500 m of D
            hline(30_000, 200),  # 5: tie with 4 for E
            hline(30_000, -200),  # 4: tie with 5 for E -> smaller COMID
            hline(40_000, 10),  # 6: non-network GageLoc COMID of F
            hline(40_000, 300),  # 8: F snaps here
        ],
        crs=CRS_ALBERS,
    )
    vaa = pd.DataFrame(
        {
            "comid": [1, 2, 3, 7, 5, 4, 6, 8],
            "DA_km2": [100.0, 100.0, 1000.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "non_network": [False, False, False, False, False, False, True, False],
        }
    )
    gageloc = pd.DataFrame({"site_no": ["0000000A", "0000000C", "0000000F"], "comid": [1, 1, 6]})
    return sites, gageloc, vaa, flow


def test_link_rules(network: tuple[pd.DataFrame, ...]) -> None:
    from src.data.link import link_sites

    sites, gageloc, vaa, flow = network
    out, counts = link_sites(sites, gageloc, vaa, flow, snap_max_m=500.0, da_ratio_max=1.25)
    by = out.set_index("site_no")
    assert "0000000D" not in by.index  # nothing within 500 m
    assert by.loc["0000000A", ["comid", "link_method", "F_DA"]].tolist() == [1, "gageloc", False]
    assert by.loc["0000000C", ["comid", "link_method", "F_DA"]].tolist() == [1, "gageloc", True]
    assert by.loc["0000000B", ["comid", "link_method"]].tolist() == [2, "snap"]
    assert by.loc["0000000B", "snap_dist_m"] == pytest.approx(100.0)
    assert by.loc["0000000B", "knn_comid"] == 3  # diagnostic ignores the DA check
    assert by.loc["0000000E", "comid"] == 4  # distance tie -> smaller COMID
    assert by.loc["0000000F", ["comid", "link_method"]].tolist() == [8, "snap"]
    assert by["F_snap"].tolist() == [False, True, False, True, True]
    assert counts["unlinked"] == 1
    assert counts["gageloc_comid_not_in_network"] == 1
    assert counts["F_DA_gageloc"] == 1


def test_da_log_ratio_rejects_non_positive_areas() -> None:
    from src.data.link import da_log_ratio

    got = da_log_ratio(np.array([125.0, 0.0, 10.0]), np.array([100.0, 10.0, -1.0]))
    assert got[0] == pytest.approx(math.log(1.25))
    assert np.isinf(got[1:]).all()


# ----------------------------------------------------------------------------- condition
def init_rows() -> pd.DataFrame:
    rows = [
        # site_no, row_id, date, Q_obs, Q_meas, F_eps, F_t
        ("00000001", 0, "2009-06-01", 10.0, 10.0, True, False),  # best, but R8 -> not primary
        ("00000001", 1, "2010-06-01", 20.0, 11.0, False, False),
        ("00000001", 2, "2012-06-01", 20.0, 5.0, False, False),  # ties row 1; later date
        ("00000001", 3, "2012-06-01", 20.0, 5.0, False, False),  # ties row 2; higher row_id
        ("00000002", 4, "2010-06-01", 10.0, 10.0, False, False),  # Q_E = 0 -> R13
        ("00000003", 5, "2010-06-01", 30.0, 30.0, False, True),  # R9 only -> sa3 only
    ]
    frame = pd.DataFrame(
        rows, columns=["site_no", "row_id", "q_meas_dt", "Q_obs", "Q_meas", "F_eps", "F_t"]
    )
    frame["q_meas_dt"] = pd.to_datetime(frame["q_meas_dt"])
    for col in ("lat", "lon", "W", "dbar", "dmax", "A", "V", "t_obs", "eps_A", "DA_site_km2"):
        frame[col] = 1.0
    frame["division"] = "INTERIOR PLAINS"
    frame["F_qual"] = False
    frame["F_tout"] = False
    return frame


LINKS = pd.DataFrame(
    {
        "site_no": ["00000001", "00000002", "00000003"],
        "comid": [1, 2, 3],
        "link_method": ["gageloc"] * 3,
        "F_snap": [False] * 3,
        "F_DA": [False] * 3,
        "x5070": [0.0] * 3,
        "y5070": [0.0] * 3,
    }
)
EROM = pd.DataFrame(
    {"comid": [1, 2, 3], "Q_E": [10.0, 0.0, 10.0], "Q_C": [10.0, 0.0, 10.0], "V_E": [1.0] * 3}
)


def test_condition_variants_ties_and_r13() -> None:
    out, counts = ce.extract_conditions(init_rows(), LINKS, EROM, qsel_ratio_flag=2.0)
    got = out.set_index(["site_no", "variant"])["row_id"].to_dict()
    assert got == {
        ("00000001", "primary"): 2,
        ("00000001", "sa2"): 1,
        ("00000001", "sa3"): 0,
        ("00000003", "sa3"): 5,
    }
    assert counts["R13_QE_le_0_or_missing"] == 1
    assert counts["sites_union"] == 2
    f_qsel = out.set_index(["site_no", "variant"])["F_Qsel"]
    assert not f_qsel[("00000001", "primary")]  # |ln 2| is not > ln 2
    assert f_qsel[("00000003", "sa3")]  # ln 3 > ln 2
    assert out.loc[0, "n_rows_site"] == 4


@settings(max_examples=25, deadline=None)
@given(st.permutations(list(range(6))))
def test_condition_is_row_order_invariant(order: list[int]) -> None:
    rows = init_rows()
    expected, _ = ce.extract_conditions(rows, LINKS, EROM, 2.0)
    shuffled, _ = ce.extract_conditions(rows.iloc[order].reset_index(drop=True), LINKS, EROM, 2.0)
    pd.testing.assert_frame_equal(expected, shuffled)


# ----------------------------------------------------------------------------- features
def test_landcover_sum_is_missing_if_any_class_is_missing() -> None:
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
    got = ft.landcover_sums(frame, {"g": ["a", "b"]})
    assert got["g"].iloc[0] == 3.0
    assert math.isnan(got["g"].iloc[1])


@pytest.mark.parametrize("name", ["Q_obs", "V_E", "A_J", "DA_site_km2", "W"])
def test_banned_predictors_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="banned"):
        assert_no_banned(["DA_km2", name])
    with pytest.raises(ValueError, match="banned"):
        ft.build_site_table(pd.DataFrame(), pd.DataFrame(), ["DA_km2", name], [])


def test_build_site_table_r12_r14_and_m1_flag() -> None:
    core = pd.DataFrame(
        {
            "site_no": ["s1", "s1", "s2", "s3", "s4"],
            "variant": ["primary", "sa3", "sa3", "primary", "primary"],
            "comid": [1, 1, 2, 3, 4],
            "division": ["X"] * 5,
            "Q_E": [10.0, 10.0, 5.0, 8.0, 4.0],
            "V_E": [0.5, 0.5, 0.5, 0.5, np.nan],
        }
    )
    per_site = pd.DataFrame(
        {
            "comid": [1, 2, 3, 4],
            "S": [1e-3, np.nan, 1e-3, 1e-3],  # s2: R12
            "AI": [0.5, 0.5, np.nan, 0.5],  # s3: R14
            "m1": [1.0, 1.0, 1.0, np.nan],  # s4: F_M1_missing only
        }
    )
    table, counts = ft.build_site_table(core, per_site, ["S", "AI"], ["m1"])
    assert table["site_no"].tolist() == ["s1", "s1", "s4"]
    assert table["F_M1_missing"].tolist() == [False, False, True]
    assert table["A_J"].iloc[0] == pytest.approx(20.0)
    assert math.isnan(table["A_J"].iloc[2])
    assert counts["R12_slope_missing"] == 1
    assert counts["R14_core_missing"] == 1
    assert counts["sites_out"] == 2
    assert counts["A_J_missing_rows"] == 1


def test_aridity_zonal_mean_fill_and_touched_fallback(tmp_path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin

    band = np.array(
        [
            [1000, 2000, 0, 0],
            [3000, 0, 0, 0],
            [65535, 65535, 0, 0],
            [65535, 65535, 0, 5000],
        ],
        dtype="uint16",
    )
    path = tmp_path / "ai.tif"
    profile = {
        "driver": "GTiff",
        "height": 4,
        "width": 4,
        "count": 1,
        "dtype": "uint16",
        "crs": CRS_ALBERS,
        "transform": from_origin(0, 400, 100, 100),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(band, 1)
    cats = gpd.GeoDataFrame(
        {"comid": [1, 2, 3]},
        geometry=[
            box(0, 200, 200, 400),  # four pixel centres, one fill
            box(310, 10, 340, 40),  # inside one pixel, no centre -> touched fallback
            box(0, 0, 200, 200),  # all 65535
        ],
        crs=CRS_ALBERS,
    )
    got = ft.aridity_zonal_mean(cats, path, 1e-4, (0, 65535)).set_index("comid")
    assert got.loc[1, "AI"] == pytest.approx(0.2)
    assert got.loc[1, "ai_pixels"] == 3
    assert not got.loc[1, "F_AI_touched"]
    assert got.loc[2, "AI"] == pytest.approx(0.5)
    assert got.loc[2, "F_AI_touched"]
    assert math.isnan(got.loc[3, "AI"])
    assert got.loc[3, "ai_pixels"] == 0

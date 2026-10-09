"""Unit tests for src/data/unit_map.py and src/data/qc.py (Phase 5, Batch C). Synthetic only.

Sources: docs/2 1.2 (lambda_L = 0.3048 m/ft); docs/3 2.1 (unit map),
2.2 (bounds), 2.3 (datum, physiography), 2.4 (QC rules R1-R11), 10.5 (QC interpretations).
"""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from shapely.geometry import box

from src.config import FT_TO_M, load_settings
from src.data import qc
from src.data import unit_map as um
from src.data.schemas import HYDROSWOT_REQUIRED, HydroswotInit

pytestmark = pytest.mark.unit

SETTINGS = load_settings()


def raw_row(**over: str) -> dict[str, str]:
    """One delivered HYDROSWOT row as text (a clean single-channel ST measurement)."""
    row = dict.fromkeys(HYDROSWOT_REQUIRED, "")
    row.update(
        site_no="01234567",
        q_meas_no="1",
        q_meas_dt="06/01/2010 12:00:00",
        dec_lat_va="40.0",
        dec_long_va="-90.0",
        coord_datum_cd="NAD83",
        site_tp_cd="ST",
        drain_area_va="100",
        q_meas_chan_nu="1.00",
        q_va="100",
        meas_q_va="100",
        stream_wdth_va="50",
        xsec_area_va="100",
        mean_depth_va="2",
        max_depth_va="3",
        mean_vel_va="1",
        max_vel_va="1.5",
        q_meas_qual_cd="GOOD",
        q_meas_used_fg="Y",
        q_meas_type_cd="QSCMM",
        data_aging_cd="A",
        q_meas_td="CST",
        chan_mat_cd="SAND",
        chan_stability_cd="",
        base_flw_cd="",
    )
    for flag in ("q1_Outside5Percent", "q2_q_va_OutsideBounds", "q2_meas_q_va_OutsideBounds",
                 "q4_OutsideBounds", "q5_Mean_GT_Max", "q6_mean_vel_GT_max_vel"):  # fmt: skip
        row[flag] = "0"
    row.update(over)
    return row


def raw_frame(*rows: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), dtype=str)


# ----------------------------------------------------------------------------- unit map
def test_exact_factors_derive_from_the_international_foot() -> None:
    ft = Decimal("0.3048")
    assert Decimal(str(um.FT2_TO_M2)) == ft**2
    assert Decimal(str(um.CFS_TO_CMS)) == ft**3
    assert Decimal(str(um.MI2_TO_KM2)) == Decimal("1.609344") ** 2
    assert Fraction(FT_TO_M).limit_denominator(10**6) == Fraction(3048, 10000)


def test_sentinels_are_masked_before_scaling() -> None:
    frame, counts = um.map_hydroswot(raw_frame(raw_row(stream_wdth_va="-9999", q_va="-9998")))
    assert math.isnan(frame.loc[0, "W"])
    assert math.isnan(frame.loc[0, "Q_obs"])
    assert counts["sentinel_stream_wdth_va"] == 1
    assert counts["sentinel_q_va"] == 1


def test_unit_map_values_and_schema() -> None:
    frame, _ = um.map_hydroswot(raw_frame(raw_row()))
    assert frame.loc[0, "W"] == pytest.approx(50 * 0.3048, rel=1e-15)
    assert frame.loc[0, "A"] == pytest.approx(100 * 0.09290304, rel=1e-15)
    assert frame.loc[0, "Q_obs"] == pytest.approx(100 * 0.028316846592, rel=1e-15)
    assert frame.loc[0, "DA_site_km2"] == pytest.approx(258.9988110336, rel=1e-15)
    assert frame.loc[0, "q_meas_chan_nu"] == 1
    assert frame["row_id"].tolist() == [0]


def test_nad27_is_transformed_and_unknown_datum_missing() -> None:
    frame, counts = um.map_hydroswot(
        raw_frame(
            raw_row(coord_datum_cd="NAD27"),
            raw_row(q_meas_no="2", coord_datum_cd="OLD"),
            raw_row(q_meas_no="3"),
        )
    )
    assert counts["datum_nad27_transformed"] == 1
    assert counts["datum_unknown_set_missing"] == 1
    assert frame.loc[0, "lat"] != 40.0 or frame.loc[0, "lon"] != -90.0
    assert abs(frame.loc[0, "lat"] - 40.0) < 1e-3
    assert math.isnan(frame.loc[1, "lat"])
    assert frame.loc[2, "lat"] == 40.0


# ----------------------------------------------------------------------------- QC
def mapped(*rows: dict[str, str]) -> pd.DataFrame:
    return um.map_hydroswot(raw_frame(*rows))[0]


def run_qc(frame: pd.DataFrame, division: str | None = "INTERIOR PLAINS"):
    div = pd.Series([division] * len(frame), index=frame.index, dtype="string")
    return qc.apply_qc(frame, div, SETTINGS)


def test_clean_row_survives_and_schema_holds() -> None:
    out, log = run_qc(mapped(raw_row()))
    assert len(out) == 1
    HydroswotInit.validate(out)
    assert out.loc[0, "t_obs"] == pytest.approx(math.log(1.5))
    assert out.loc[0, "eps_A"] == pytest.approx(0.0, abs=1e-12)
    assert list(log.rules) == [
        "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "F_tout",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("over", "rule"),
    [
        ({"site_tp_cd": "ST-CA"}, "R1"),
        ({"drain_area_va": ""}, "R4"),
        ({"mean_vel_va": "0"}, "R4"),
        ({"stream_wdth_va": "20000"}, "R5"),
        ({"mean_depth_va": "4"}, "R6"),
        ({"mean_vel_va": "2", "max_vel_va": "1.5", "q_va": "200", "meas_q_va": "200"}, "R6"),
        ({"meas_q_va": "110"}, "R7"),
    ],
)
def test_dropping_rules(over: dict[str, str], rule: str) -> None:
    out, log = run_qc(mapped(raw_row(**over)))
    assert out.empty
    assert log.rules[rule]["rows_removed"] == 1


def test_r2_drops_rows_without_division() -> None:
    out, log = run_qc(mapped(raw_row()), division=None)
    assert out.empty
    assert log.rules["R2"]["rows_removed"] == 1


def test_r3_drops_whole_multichannel_measurement() -> None:
    out, log = run_qc(
        mapped(
            raw_row(q_meas_chan_nu="1.00"),
            raw_row(q_meas_chan_nu="2.00"),
            raw_row(q_meas_no="2", q_meas_chan_nu="0.00"),
        )
    )
    assert out["q_meas_no"].tolist() == ["2"]
    assert log.rules["R3"]["rows_removed"] == 2
    assert log.rules["R3"]["rows_chan_0_kept"] == 1


def test_flags_r8_r9_r10_are_kept() -> None:
    out, _ = run_qc(
        mapped(
            # eps_A = ln 1.2 > ln 1.05; Q = A V keeps R7 satisfied
            raw_row(xsec_area_va="120", q_va="120", meas_q_va="120"),
            raw_row(q_meas_no="2", max_depth_va="2"),  # dmax == dbar
            raw_row(q_meas_no="3", data_aging_cd="W"),
        )
    )
    assert out["F_eps"].tolist() == [True, False, False]
    assert out["F_t"].tolist() == [False, True, False]
    assert out["F_qual"].tolist() == [False, False, True]
    assert out.loc[1, "F_tout"]  # t_obs = 0 lies outside [ln 1.1, ln 3]


def test_r11_keeps_lowest_row_id() -> None:
    out, log = run_qc(mapped(raw_row(), raw_row(q_meas_chan_nu="0.00")))
    assert out["row_id"].tolist() == [0]
    assert log.rules["R11"]["rows_removed"] == 1


@settings(max_examples=25, deadline=None)
@given(st.permutations(list(range(6))))
def test_qc_is_row_order_invariant(order: list[int]) -> None:
    rows = [
        raw_row(q_meas_no=str(i), q_va=str(100 + i), meas_q_va=str(100 + i)) for i in range(4)
    ] + [raw_row(q_meas_no="0"), raw_row(q_meas_no="9", site_tp_cd="LK")]
    frame = mapped(*rows)
    expected, _ = run_qc(frame)
    shuffled, _ = run_qc(frame.iloc[order].reset_index(drop=True))
    pd.testing.assert_frame_equal(expected, shuffled)


def test_assign_division_single_boundary_and_outside() -> None:
    polys = gpd.GeoDataFrame(
        {"DIVISION": ["A", "B", pd.NA]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(5, 5, 6, 6)],
        crs="EPSG:4269",
    )
    lat = pd.Series([0.5, 0.5, 0.5, 5.5, np.nan, 0.5])
    lon = pd.Series([0.5, 1.0, 9.0, 5.5, 0.5, 0.5])
    div = qc.assign_division(lat, lon, polys)
    assert div.tolist()[0] == "A"
    assert div.isna().tolist() == [False, True, True, True, True, False]
    assert div.tolist()[5] == "A"  # duplicate coordinates resolved once

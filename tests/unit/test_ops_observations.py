"""Unit tests for the OPS-1 observed responses (src/operational/observations.py). Synthetic."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.operational.config import ObservationParams
from src.operational.observations import OBS_COLUMNS, build_observations, read_stage_fields

pytestmark = pytest.mark.unit

CFG = ObservationParams(
    exclude_ad_site_cd=("VELQ", "SLPQ"), channel_number=1, window_years=5.0, max_abs_dh_m=70.0
)


def _unit() -> pd.DataFrame:
    t0 = pd.Timestamp("2010-06-01")
    rows = [
        # site A: anchor row 0; 1 normal; 2 duplicate q_meas_no of 1; 3 channel 2;
        # 4 outside window; 5 stage missing; 6 implausible stage; 7 zero flow; 8 no width
        (0, "A", "1", t0, 1, 10.0, 20.0),
        (1, "A", "2", t0 + pd.Timedelta(days=30), 1, 40.0, 25.0),
        (2, "A", "2", t0 + pd.Timedelta(days=30), 1, 41.0, 25.0),
        (3, "A", "3", t0, 2, 5.0, 10.0),
        (4, "A", "4", t0 + pd.Timedelta(days=int(6 * 365.25)), 1, 20.0, 22.0),
        (5, "A", "5", t0, 1, 12.0, 21.0),
        (6, "A", "6", t0, 1, 12.0, 21.0),
        (7, "A", "7", t0, 1, 0.0, 21.0),
        (8, "A", "8", t0, 1, 5.0, np.nan),
        # site B: index-velocity gauge (excluded)
        (9, "B", "1", t0, 1, 3.0, 5.0),
        (10, "B", "2", t0, 1, 6.0, 6.0),
        # site C: anchor stage missing (excluded)
        (11, "C", "1", t0, 1, 3.0, 5.0),
        (12, "C", "2", t0, 1, 6.0, 6.0),
    ]
    df = pd.DataFrame(
        rows, columns=["row_id", "site_no", "q_meas_no", "q_meas_dt", "chan", "Q", "W"]
    )
    return df.rename(columns={"chan": "q_meas_chan_nu", "Q": "Q_obs"}).astype(
        {"q_meas_chan_nu": "Int16"}
    )


def _stage() -> pd.DataFrame:
    stage = [5.0, 8.0, 8.1, 4.0, 6.0, np.nan, 400.0, 5.0, 4.0, 1.0, 2.0, np.nan, 2.0]
    code = ["OTHR"] * 9 + ["VELQ"] * 2 + ["OTHR"] * 2
    return pd.DataFrame({"row_id": np.arange(13), "stage_ft": stage, "ad_site_cd": code})


def _anchors() -> pd.DataFrame:
    return pd.DataFrame({"site_no": ["A", "B", "C"], "row_id": [0, 9, 11], "division": ["d"] * 3})


def test_exclusions_and_responses() -> None:
    """Every pre-registered exclusion fires once and the responses are datum-free."""
    obs, counts = build_observations(_unit(), _stage(), _anchors(), CFG, window_years=5.0)
    assert tuple(obs.columns) == OBS_COLUMNS
    assert counts["sites_excluded_ad_site_cd"] == 1
    assert counts["sites_excluded_anchor_stage_missing"] == 1
    assert counts["sites_eligible"] == 1
    assert counts["rows_excluded_duplicate_measurement"] == 1
    assert counts["rows_excluded_channel"] == 1
    assert counts["rows_anchor"] == 1
    assert counts["rows_excluded_flow_or_stage_missing"] == 2
    assert counts["rows_excluded_time_window"] == 1
    assert counts["rows_excluded_implausible_dh"] == 1
    assert obs["row_id"].tolist() == [1, 8]
    first = obs.iloc[0]
    assert first["q_ratio"] == pytest.approx(4.0)
    assert first["ln_q_ratio"] == pytest.approx(math.log(4.0))
    assert first["dh_obs"] == pytest.approx(0.3048 * 3.0)
    assert first["dlnw_obs"] == pytest.approx(math.log(25.0 / 20.0))
    assert math.isnan(obs.iloc[1]["dlnw_obs"])
    assert counts["rows_with_width"] == 1


def test_window_disabled_keeps_old_rows() -> None:
    """window_years=None (sensitivity run) keeps the row outside the window."""
    obs, counts = build_observations(_unit(), _stage(), _anchors(), CFG, window_years=None)
    assert counts["rows_excluded_time_window"] == 0
    assert 4 in obs["row_id"].tolist()


def test_input_guards() -> None:
    """Missing columns, duplicated anchors and foreign anchor rows are refused."""
    with pytest.raises(KeyError, match="missing columns"):
        build_observations(_unit().drop(columns="W"), _stage(), _anchors(), CFG, window_years=5.0)
    dup = pd.concat([_anchors(), _anchors().iloc[:1]])
    with pytest.raises(ValueError, match="one row per site"):
        build_observations(_unit(), _stage(), dup, CFG, window_years=5.0)
    foreign = _anchors().assign(row_id=[9, 0, 11])
    with pytest.raises(ValueError, match="does not belong"):
        build_observations(_unit(), _stage(), foreign, CFG, window_years=5.0)
    with pytest.raises(ValueError, match="one-to-one"):
        build_observations(
            _unit(), pd.concat([_stage(), _stage().iloc[:1]]), _anchors(), CFG, window_years=5.0
        )


def test_read_stage_fields(tmp_path: Path) -> None:
    """Raw reader: sentinels masked, empty codes missing, row_id = 0-based file row."""
    p = tmp_path / "SWOT_ADCP_Dataset.txt"
    p.write_text(
        '"site_no"\t"stage_va"\t"ad_site_cd"\n"01"\t"3.5"\t"OTHR"\n"02"\t"-9999"\t""\n',
        encoding="latin-1",
    )
    out = read_stage_fields(p)
    assert out["row_id"].tolist() == [0, 1]
    assert out["stage_ft"].iloc[0] == 3.5
    assert math.isnan(out["stage_ft"].iloc[1])
    assert out["ad_site_cd"].iloc[0] == "OTHR"
    assert pd.isna(out["ad_site_cd"].iloc[1])

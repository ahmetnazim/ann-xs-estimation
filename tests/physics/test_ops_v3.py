"""Physical-invariant tests for OPS-1 v3 dynamic resistance & CCHE1D geometry (evaluate_v3.py).

ADL-160 (OPS-1 v3 pre-registration); docs/2 Prop. 2.1 (power-law section), Prop. 2.2 (perimeter
factor), and docs/6 OPS3-E1..E7.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.operational.evaluate import HIGH_BIN
from src.operational.evaluate_v3 import (
    ALL_FLOWS_BIN,
    _compute_ln_r0,
    add_all_flows_bin,
    oracle_anchor_geometry,
)
from src.operational.rating import solve_lambda_np, stage_change

pytestmark = pytest.mark.physics


def test_all_flows_bin_observation_weighted_identity() -> None:
    """V3-1 (ADL-160, docs/2): all_flows MSE equals observation-weighted average across bins."""
    df = pd.DataFrame(
        {
            "variant": ["floodplain"] * 3,
            "family": ["S-ID"] * 3,
            "model": ["M0"] * 3,
            "repeat": [0] * 3,
            "site_no": ["s1"] * 3,
            "bin": ["[0,0.5)", "[2,10)", HIGH_BIN],
            "n": [2, 8, 8],
            "mse_h": [1.0, 6.0, 6.0],
            "me_h": [-0.5, 1.5, 1.5],
            "n_w": [0, 0, 0],
            "mse_w": [np.nan, np.nan, np.nan],
            "me_w": [np.nan, np.nan, np.nan],
        }
    )
    out = add_all_flows_bin(df)
    af = out.loc[out["bin"] == ALL_FLOWS_BIN].iloc[0]
    assert int(af["n"]) == 10
    assert float(af["mse_h"]) == pytest.approx((2 * 1.0 + 8 * 6.0) / 10.0)
    assert float(af["me_h"]) == pytest.approx((2 * (-0.5) + 8 * 1.5) / 10.0)


def test_dynamic_roughness_attenuates_high_flow_stage_rise() -> None:
    """V3-2 (ADL-160, docs/2 Prop. 6.2): negative gamma (-0.32) reduces high-flow stage rise."""
    ln_q = np.log(np.array([2.0, 5.0, 15.0]))
    r = np.full(3, 1.8)
    ln_a0 = np.full(3, math.log(1.5 / 40.0))
    dmax0 = np.full(3, 2.25)
    lam_const, _ = solve_lambda_np(
        ln_q, r, ln_a0, level=6, trunc=4.0, tol=1e-11, max_iter=40, chunk_rows=100
    )
    lam_dyn, _ = solve_lambda_np(
        (1.0 - 0.32) * ln_q, r, ln_a0, level=6, trunc=4.0, tol=1e-11, max_iter=40, chunk_rows=100
    )
    dh_const = stage_change(dmax0, lam_const)
    dh_dyn = stage_change(dmax0, lam_dyn)
    assert bool(np.all(dh_dyn > 0.0))
    assert bool(np.all(dh_dyn < dh_const))
    assert bool(np.all(np.diff(dh_dyn) > 0.0))


def test_cche1d_dingman_hydraulic_radius_bounds_and_wide_limit() -> None:
    """V3-3 (ADL-160, docs/2 Prop. 2.2): 0 < R_0 <= dbar_0 and R_0 -> dbar_0 as W/dbar -> inf."""
    ln_w = np.log(np.array([10.0, 50.0, 5000.0]))
    ln_dbar = np.log(np.array([2.0, 2.0, 2.0]))
    ln_dmax = np.log(np.array([1.9, 3.0, 3.0]))  # first row tests shape_ratio_floor (1.9 < 2.0)
    ln_m, ln_r0 = _compute_ln_r0(
        ln_w,
        ln_dbar,
        ln_dmax,
        shape_floor=1.01,
        r_bounds=(0.1, 100.0),
        quad_level=6,
        quad_trunc=4.0,
    )
    r0 = np.exp(ln_r0)
    dbar = np.exp(ln_dbar)
    assert math.exp(ln_m[0]) == pytest.approx(2.0 * 1.01)
    assert bool(np.all((r0 > 0.0) & (r0 <= dbar)))
    # In the wide-channel limit (W = 5000 m, dbar = 2 m), R_0 -> dbar_0 within 0.1%
    assert r0[2] == pytest.approx(dbar[2], rel=1e-3)


def test_oracle_anchor_geometry_exact_replication() -> None:
    """V3-4 (ADL-160, docs/2 Prop. 2.1): Oracle geometry recovers exact W, dbar, dmax and r."""
    prim = pd.DataFrame(
        {
            "site_no": ["s1", "s2"],
            "W": [30.0, 60.0],
            "dbar": [1.5, 2.0],
            "dmax": [2.25, 1.8],  # s2 has dmax < dbar -> filtered out
        }
    )
    g = oracle_anchor_geometry(prim, repeats=[0, 1], r_bounds=(0.1, 100.0))
    assert set(g["site_no"]) == {"s1"}
    assert len(g) == 2
    assert bool(g["valid"].all())
    assert float(g["r"].iloc[0]) == pytest.approx(1.5 / (2.25 - 1.5))

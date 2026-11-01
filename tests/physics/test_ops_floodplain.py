"""Physical-invariant tests for the OPS-1 v2 composite section (src/operational/floodplain.py).

ADL-157 (OPS-1 v2 pre-registration); docs/2 Prop. 2.1 (power-law section) and the v2 strip
conveyance OPS2-E1. Invariants: F1 exact segment integrals; F2 control strip == analytic power
law; F3 high terrain walls reproduce the control; F4 a low floodplain lowers high-flow stage;
F5 water behind a levee is not counted (contiguity); F6 truncation flag; F7 terrain preparation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.operational.floodplain import (
    composite_grid,
    control_stage_change,
    conveyance,
    floodplain_stage_change,
    prepare_terrain,
    segment_integrals,
)

pytestmark = pytest.mark.physics

W0, D0, R = 40.0, 1.5, 1.8
ZB = 10.0
WSE = ZB + D0
L = np.log(np.array([0.1, 0.5, 1.0, 2.0, 5.0, 20.0]))


def test_segment_integrals_exact() -> None:
    """F1 (ADL-157): exact per-segment integral of max(d,0)^{5/3} on linear d (docs/2)."""
    x = np.array([0.0, 2.0, 5.0, 6.0, 9.0])
    d = np.array([-1.0, 1.0, 2.0, 2.0, -0.5])
    from scipy.integrate import quad

    got = segment_integrals(x, d)
    for j in range(4):
        ref, _ = quad(
            lambda t: max(float(np.interp(t, x, d)), 0.0) ** (5 / 3),
            x[j],
            x[j + 1],
            points=[x[j] + (x[j + 1] - x[j]) * 0.5],
            epsabs=1e-13,
            epsrel=1e-12,
        )
        assert got[j] == pytest.approx(ref, rel=1e-8)
    assert segment_integrals(x, np.full(5, -1.0)).sum() == 0.0


def test_control_strip_equals_power_law() -> None:
    """F2 (ADL-157): numeric strip conveyance on the power-law bed == analytic (docs/2 2.1)."""
    x, b, ic = composite_grid(
        None, w0=W0, dmax0=D0, r=R, z_b=ZB, z_top=ZB + 20, channel_points=20001
    )
    z = np.union1d(ZB + D0 * np.geomspace(1e-3, 20 / D0, 3000), [WSE])
    k, _trunc = conveyance(x, b, ic, z)
    assert bool(np.all(np.diff(k) > 0))
    lr = np.log(k) - np.log(k[np.searchsorted(z, WSE)])
    num = np.interp(L, lr, z) - WSE
    assert num == pytest.approx(control_stage_change(D0, R, L), abs=1e-4)
    assert control_stage_change(D0, R, [0.0])[0] == 0.0


def _terrain(z_fn: object, half: float = 1000.0) -> object:
    xt = np.linspace(-half, half, 1001)
    zt = z_fn(xt)  # type: ignore[operator]
    return prepare_terrain(xt, zt, 0.0, 100.0, 0.05)


def test_high_walls_reproduce_control() -> None:
    """F3 (ADL-157): terrain far above every stage leaves the model section unchanged."""
    t = _terrain(lambda x: np.full_like(x, WSE + 500.0))
    dh, trunc = floodplain_stage_change(
        t, wse_a=WSE, w0=W0, dmax0=D0, r=R, ln_q_ratio=L, channel_points=4001, z_levels=600
    )
    assert dh == pytest.approx(control_stage_change(D0, R, L), abs=5e-3)
    assert not trunc.any()


def test_low_floodplain_lowers_high_flow_stage() -> None:
    """F4 (ADL-157): a wide floodplain 1 m above the anchor caps the high-flow stage rise."""
    t = _terrain(lambda x: np.where(np.abs(x) > 900, WSE + 30.0, WSE + 1.0))
    dh, trunc = floodplain_stage_change(
        t, wse_a=WSE, w0=W0, dmax0=D0, r=R, ln_q_ratio=L, channel_points=401, z_levels=300
    )
    ctrl = control_stage_change(D0, R, L)
    assert dh[2] == pytest.approx(0.0, abs=1e-9)
    assert dh[-1] < ctrl[-1] - 1.0
    assert dh[-1] < 2.0
    assert bool(np.all(np.diff(dh) > 0))
    assert not trunc.any()


def test_levee_blocks_disconnected_storage() -> None:
    """F5 (ADL-157): low ground behind a levee does not add conveyance below the levee crest."""
    x = np.linspace(-100, 100, 2001)
    bed = np.where(np.abs(x) < 10, 0.0, 5.0)
    bed = np.where((x > 30) & (x < 40), 8.0, bed)  # levee crest 8 m
    bed = np.where(x >= 40, 1.0, bed)  # low land behind the levee
    ic = int(np.argmin(np.abs(x)))
    k_lo, _ = conveyance(x, bed, ic, np.array([7.0]))
    bed_no = np.where(x >= 40, 8.0, bed)  # same levee, no low land behind it
    k_ref, _ = conveyance(x, bed_no, ic, np.array([7.0]))
    assert k_lo[0] == pytest.approx(k_ref[0])
    _k_hi, trunc = conveyance(x, bed, ic, np.array([9.0]))
    assert trunc[0]  # above the crest the water reaches the transect end


def test_truncation_flag_when_valley_overtopped() -> None:
    """F6 (ADL-157): stages above the lower transect end are flagged as truncated."""
    t = _terrain(lambda x: WSE + 0.5 + 0.001 * np.abs(x), half=300.0)
    assert t.z_end_low == pytest.approx(WSE + 0.8)
    _dh, trunc = floodplain_stage_change(
        t,
        wse_a=WSE,
        w0=W0,
        dmax0=D0,
        r=R,
        ln_q_ratio=np.log(np.array([1.0, 1000.0])),
        channel_points=401,
        z_levels=300,
    )
    assert not trunc[0]
    assert trunc[1]


def test_prepare_terrain_fill_and_errors() -> None:
    """F7 (ADL-157): unsorted input is sorted, interior gaps filled, no-data share reported."""
    x = np.array([4.0, 0.0, 1.0, 2.0, 3.0])
    z = np.array([9.0, 9.0, np.nan, 1.0, 5.0])
    t = prepare_terrain(x, z, 2.0, 2.0, 0.05)
    assert t.x.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert t.z[1] == pytest.approx(5.0)
    assert t.nodata_share == pytest.approx(0.2)
    assert t.z_ch == 1.0
    assert t.x_c == 2.0
    assert t.z_end_low == 9.0
    with pytest.raises(ValueError, match="fewer than 3"):
        prepare_terrain(x, np.array([1.0, np.nan, np.nan, np.nan, 2.0]), 2.0, 2.0, 0.05)


def test_floodplain_guards() -> None:
    """ADL-157: unreachable flows and zero anchor conveyance raise."""
    t = _terrain(lambda x: np.full_like(x, WSE + 500.0))
    with pytest.raises(RuntimeError, match="did not reach"):
        floodplain_stage_change(
            t,
            wse_a=WSE,
            w0=W0,
            dmax0=D0,
            r=R,
            ln_q_ratio=np.array([math.log(1e300)]),
            channel_points=101,
            z_levels=50,
            max_doublings=2,
        )
    low = _terrain(lambda x: np.full_like(x, ZB - 50.0))  # terrain below the thalweg everywhere
    dh, trunc = floodplain_stage_change(
        low, wse_a=WSE, w0=W0, dmax0=D0, r=R, ln_q_ratio=L, channel_points=101, z_levels=50
    )
    assert trunc.all()
    assert np.isfinite(dh).all()

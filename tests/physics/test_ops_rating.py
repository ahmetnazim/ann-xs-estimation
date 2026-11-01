"""Physical-invariant tests for the OPS-1 synthetic rating curve (src/operational/rating.py).

Invariants (docs/4_operational_stage_response.md section 2):
I1 the anchor is reproduced (L = 0 -> lam = 0, Delta h = 0);
I2 round trip: the forward map evaluated at the root returns ln(Q/Q_a) to machine precision;
I3 Proposition OPS-1: the root lies inside the derivative bracket and f is strictly increasing;
I4 wide-channel limit a -> 0: lam = L / (1/r + 5/3) (at-a-station exponent 3r/(3 + 5r));
I5 rectangular limit (large r): stage grows as Q^(3/5) in the wide limit;
I6 roughness and slope cancel: a Manning curve with arbitrary n, S calibrated at the anchor gives
   the same stage response as the solver;
I7 the width response is lam / r and is monotone in Q.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.operational.rating import (
    derivative_bounds,
    rating_log_q_ratio,
    solve_lambda,
    solve_lambda_np,
    stage_change,
    width_log_change,
)
from src.physics.shape import section_terms

pytestmark = pytest.mark.physics

LEVEL, TRUNC = 6, 4.0
KW = {"level": LEVEL, "trunc": TRUNC, "tol": 1e-12, "max_iter": 80}


def _t(*xs: float) -> torch.Tensor:
    return torch.tensor(xs, dtype=torch.float64)


def test_anchor_is_reproduced() -> None:
    """I1: L = 0 gives lam = 0 for any shape and aspect ratio."""
    sol = solve_lambda(_t(0.0, 0.0, 0.0), _t(0.5, 1.0, 10.0), _t(-6.0, -2.0, -0.5), **KW)
    assert torch.equal(sol.lam, torch.zeros(3, dtype=torch.float64))
    assert np.all(stage_change([1.0, 2.0, 3.0], sol.lam.numpy()) == 0.0)


@settings(max_examples=60, deadline=None)
@given(
    big_l=st.floats(-4.0, 6.0),
    r=st.floats(0.5, 10.0),
    ln_a0=st.floats(math.log(1e-4), math.log(0.5)),
)
def test_round_trip_and_bracket(big_l: float, r: float, ln_a0: float) -> None:
    """I2 and I3: forward(root) == L and the root lies inside the Prop. OPS-1 bracket."""
    lq, rr, la = _t(big_l), _t(r), _t(ln_a0)
    sol = solve_lambda(lq, rr, la, **KW)
    back = rating_log_q_ratio(sol.lam, rr, la, level=LEVEL, trunc=TRUNC)
    assert abs(float(back[0]) - big_l) <= 1e-11
    d_lo, d_hi = derivative_bounds(rr)
    lo, hi = sorted((big_l / float(d_hi[0]), big_l / float(d_lo[0])))
    assert lo - 1e-12 <= float(sol.lam[0]) <= hi + 1e-12


@pytest.mark.parametrize("r", [0.5, 0.8, 1.0, 2.0, 5.0, 10.0])
def test_forward_map_strictly_increasing(r: float) -> None:
    """I3: f is strictly increasing in lam (unique root)."""
    lam = torch.linspace(-3.0, 4.0, 401, dtype=torch.float64)
    f = rating_log_q_ratio(
        lam, torch.full_like(lam, r), torch.full_like(lam, math.log(0.05)), level=LEVEL, trunc=TRUNC
    )
    assert bool((f[1:] > f[:-1]).all())


@pytest.mark.parametrize("r", [0.5, 1.0, 3.0, 10.0])
def test_wide_channel_limit(r: float) -> None:
    """I4: as a -> 0, k -> 0 and lam = L / (1/r + 5/3)."""
    big_l = _t(-2.0, 0.7, 3.0)
    sol = solve_lambda(big_l, torch.full_like(big_l, r), torch.full_like(big_l, -25.0), **KW)
    expected = big_l / (1.0 / r + 5.0 / 3.0)
    assert torch.allclose(sol.lam, expected, rtol=0, atol=1e-12)


def test_rectangular_limit_exponent() -> None:
    """I5: wide rectangle-like section (r large): y ~ Q^(3/5) to O(1/r)."""
    r = 1.0e4
    lam = solve_lambda(_t(math.log(10.0)), _t(r), _t(-25.0), **KW).lam
    assert abs(float(lam[0]) - 0.6 * math.log(10.0)) < 1e-3


@pytest.mark.parametrize(("n", "s"), [(0.02, 1e-4), (0.05, 3e-3), (0.2, 0.05)])
def test_roughness_and_slope_cancel(n: float, s: float) -> None:
    """I6: an explicit Manning curve calibrated at the anchor gives the solver's stage."""
    w0, dbar0, r = 40.0, 1.2, 1.7
    dmax0 = dbar0 * (r + 1.0) / r

    def manning_q(y: float, n_: float) -> float:
        w = w0 * (y / dmax0) ** (1.0 / r)
        dbar = r / (r + 1.0) * y
        terms = section_terms(_t(math.log(dbar)), _t(math.log(w)), _t(r), level=LEVEL, trunc=TRUNC)
        kappa = float(terms.kappa[0])
        return w * dbar ** (5.0 / 3.0) * kappa ** (2.0 / 3.0) * math.sqrt(s) / n_

    q_a = manning_q(dmax0, n)
    # recalibrate n to a different anchor flow; S and n must cancel in the response
    q_anchor = 0.37 * q_a
    n_cal = n * q_a / q_anchor
    y_target = 2.3 * dmax0
    q_i = manning_q(y_target, n_cal)
    lam = solve_lambda(
        _t(math.log(q_i / q_anchor)), _t(r), _t(math.log(dbar0 / w0)), **KW
    ).lam.numpy()
    assert stage_change([dmax0], lam)[0] == pytest.approx(y_target - dmax0, rel=1e-10)


def test_width_response_monotone_and_np_wrapper() -> None:
    """I7 and the chunked NumPy wrapper (chunking does not change results)."""
    big_l = np.linspace(-2.0, 3.0, 23)
    r = np.full(23, 2.5)
    la = np.full(23, math.log(0.03))
    lam_a, it_a = solve_lambda_np(big_l, r, la, chunk_rows=5, **KW)
    lam_b, _ = solve_lambda_np(big_l, r, la, chunk_rows=1000, **KW)
    assert np.array_equal(lam_a, lam_b)
    assert 1 <= it_a <= KW["max_iter"]
    dlw = width_log_change(lam_a, r)
    assert np.all(np.diff(dlw) > 0)
    assert np.allclose(dlw, lam_a / 2.5)


def test_np_wrapper_empty_and_bad_inputs() -> None:
    """Empty input returns empty output; malformed inputs raise."""
    lam, it = solve_lambda_np([], [], [], chunk_rows=10, **KW)
    assert lam.size == 0
    assert it == 0
    with pytest.raises(ValueError, match="equal length"):
        solve_lambda_np([0.0, 1.0], [1.0], [0.0], chunk_rows=10, **KW)
    with pytest.raises(ValueError, match="chunk_rows"):
        solve_lambda_np([0.0], [1.0], [0.0], chunk_rows=0, **KW)


def test_solver_input_validation_and_nonconvergence() -> None:
    """Invalid r / tol raise ValueError; an impossible iteration cap raises RuntimeError."""
    with pytest.raises(ValueError, match="r must be > 0"):
        solve_lambda(_t(1.0), _t(-1.0), _t(-3.0), **KW)
    with pytest.raises(ValueError, match="tol"):
        solve_lambda(_t(1.0), _t(1.0), _t(-3.0), level=LEVEL, trunc=TRUNC, tol=0.0, max_iter=5)
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_lambda(_t(4.0), _t(0.6), _t(-0.7), level=LEVEL, trunc=TRUNC, tol=1e-300, max_iter=1)

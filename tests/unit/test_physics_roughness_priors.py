"""Smoke tests for src/physics/roughness_priors.py (Phase 6, Batch A). Synthetic; no data.

Values: protocol 4.1 (P3-07 Strickler, P3-08 Jarrett) from params ``priors``; proofs 10.2 E5.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from src.config import load_settings
from src.physics import roughness_priors as rp

pytestmark = pytest.mark.unit

PRIORS = load_settings().protocol.priors
JARRETT, STRICKLER = PRIORS.jarrett, PRIORS.strickler


def t64(*values: float) -> Tensor:
    return torch.tensor(values, dtype=torch.float64)


def test_e5_unit_constants() -> None:
    """Tier 3 (proofs 10.2 E5): lambda_L^3 exact; k_US and k_US^(3/5) to the printed digit."""
    assert rp.LAMBDA_L == 0.3048
    assert abs(rp.LAMBDA_L**3 - 0.028316846592) <= 1e-17
    k_us = rp.LAMBDA_L ** (-1.0 / 3.0)
    assert abs(k_us - 1.48592) <= 1e-5
    assert abs(k_us**0.6 - 1.26822) <= 1e-5


def test_strickler_one_millimetre() -> None:
    nu = rp.strickler_nu(t64(math.log(1e-3)), STRICKLER)
    assert abs(math.exp(float(nu[0])) - 0.047 * 1e-3 ** (1.0 / 6.0)) <= 1e-15
    assert abs(math.exp(float(nu[0])) - 0.014863) <= 1e-6


def test_jarrett_si_constant_matches_us_form() -> None:
    assert abs(JARRETT.c_j_us * rp.LAMBDA_L**0.16 - JARRETT.c_j_si) < 5e-5


def test_jarrett_worked_example_si_equals_us() -> None:
    s, r_m = 0.02, 0.6096  # R = 2 ft
    nu = rp.jarrett_nu(t64(math.log(s)), t64(math.log(r_m)), JARRETT)
    n_si = math.exp(float(nu[0]))
    n_us = JARRETT.c_j_us * s**JARRETT.a_j * (r_m / rp.LAMBDA_L) ** JARRETT.b_j
    assert abs(n_si - 0.0789) <= 5e-4
    assert abs(n_si / n_us - 1.0) <= 5e-5 / JARRETT.c_j_si


def test_jarrett_branch_and_validity_flags() -> None:
    s = torch.log(t64(0.001, 0.002, 0.03, 0.05))
    assert rp.jarrett_branch(s, JARRETT).tolist() == [False, True, True, True]
    ln_r = torch.log(t64(0.6096, 0.6096, 3.0, 0.6096))  # 2 ft, 2 ft, 9.84 ft, 2 ft
    assert rp.jarrett_valid(s, ln_r, JARRETT).tolist() == [False, True, False, False]
    edge = torch.log(t64(0.501 * rp.LAMBDA_L, 6.99 * rp.LAMBDA_L))  # just inside the R bounds
    s_ok = torch.log(t64(0.01, 0.01))
    assert rp.jarrett_valid(s_ok, edge, JARRETT).tolist() == [True, True]


def test_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        rp.strickler_nu(t64(float("nan")), STRICKLER)
    with pytest.raises(TypeError):
        rp.jarrett_nu(torch.zeros(2), torch.zeros(2), JARRETT)
    with pytest.raises(ValueError, match="length"):
        rp.jarrett_valid(t64(1.0), t64(1.0, 2.0), JARRETT)

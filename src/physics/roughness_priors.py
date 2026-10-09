"""Empirical roughness: Strickler n_S = c_S D50^(1/6) and Jarrett n_J = c_J' S^a R^b (SI).

Sources: proofs section 7.5 (``nu_S = ln c_S + (1/6) ln D50``; ablations A3-SP and A3-SR),
section 7.9 and Proposition 7.5 (M3: Strickler for ``S < S_J``, Jarrett for ``S >= S_J``;
``b_J < 5/3``); protocol section 4.1 (P3-07: ``c_S = 0.047``, D50 in m, Strickler 1923 ⚠️ no
DOI; P3-08: Jarrett 1985 eq. 6, SI form ``n = 0.3225 S^0.38 R_m^-0.16``, ``S_J = 0.002``, validity
``0.002 <= S <= 0.04`` and ``0.5 <= R <= 7 ft``; rows outside are **flagged, never clipped**).
Values come from params ``priors``.

Inputs are log coordinates: ``ln_d50 = ln(D50 / 1 m)``, ``s = ln S``, ``ln_r_hyd = ln(R / 1 m)``.
Torch-only, float64, CPU, row-wise. Owning phase: 6.
"""

from __future__ import annotations

import math
from typing import Final

from torch import Tensor

from src.config import JarrettParams, StricklerParams
from src.physics.shape import check_rows

__all__ = [
    "LAMBDA_L",
    "jarrett_branch",
    "jarrett_nu",
    "jarrett_valid",
    "strickler_nu",
]

#: International foot in metres, exact (proofs 1.2).
LAMBDA_L: Final[float] = 0.3048


def strickler_nu(ln_d50: Tensor, strickler: StricklerParams) -> Tensor:
    """``nu_S = ln c_S + (1/6) ln D50`` with D50 in m (proofs 7.5; protocol 4.1, P3-07)."""
    check_rows(ln_d50=ln_d50)
    return math.log(strickler.c_s) + ln_d50 / 6.0


def jarrett_nu(s: Tensor, ln_r_hyd: Tensor, jarrett: JarrettParams) -> Tensor:
    """``nu_J = ln c_J' + a_J s + b_J ln R`` (protocol 4.1 SI form; proofs 7.9)."""
    check_rows(s=s, ln_r_hyd=ln_r_hyd)
    return math.log(jarrett.c_j_si) + jarrett.a_j * s + jarrett.b_j * ln_r_hyd


def jarrett_branch(s: Tensor, jarrett: JarrettParams) -> Tensor:
    """Boolean ``S >= S_J`` (proofs 7.9: Jarrett branch; Strickler otherwise)."""
    check_rows(s=s)
    return s >= math.log(jarrett.s_j)


def jarrett_valid(s: Tensor, ln_r_hyd: Tensor, jarrett: JarrettParams) -> Tensor:
    """Boolean validity of Jarrett's eq. 6 (``S`` and ``R`` ranges; R bounds given in ft).

    A flag for reporting only (protocol 4.1); predictions are never clipped by it.
    """
    check_rows(s=s, ln_r_hyd=ln_r_hyd)
    s_lo, s_hi = jarrett.validity_s
    r_lo, r_hi = jarrett.validity_r_ft
    ln_r_ft = ln_r_hyd - math.log(LAMBDA_L)
    in_s = (s >= math.log(s_lo)) & (s <= math.log(s_hi))
    in_r = (ln_r_ft >= math.log(r_lo)) & (ln_r_ft <= math.log(r_hi))
    out: Tensor = in_s & in_r
    return out

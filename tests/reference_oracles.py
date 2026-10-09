"""Independent mpmath oracles for the Phase 7 harness (docs/2 sections 2, 3 and 10.2).

Nothing here imports ``src``: every value is computed from the docs/2 definitions at
``MP_DPS = 40`` digits (P6-D11, P7-D11; docs/2 section 10.1 "Limits of P2-D1 (a)").

- ``F`` by its definition (Proposition 2.2, the u-integral), written cancellation-free as
  ``F - 1 = int_0^1 s / (sqrt(1 + s) + 1) du`` with ``s = beta^2 u^(2r-2)`` and split at the
  ``s = 1`` transition; ``F_beta`` and ``F_r`` by differentiating under the integral.
- Proposition 2.4 closed forms at ``r = 1/2, 1, 2``.
- The Manning residual ``rho`` (eq. 3.1) and its root (Proposition 3.1) by bracketed root finding
  in the Proposition 3.2 bracket; Picard iterates (Proposition 3.2 (2)).
- The tier-3 rule of ADL-080: one unit in the last printed digit of docs/2 section 10.2.
"""

from __future__ import annotations

from typing import Any

import mpmath as mp

MP_DPS = 40
CLOSED_FORM_R = (0.5, 1.0, 2.0)

Mpf = Any  # mpmath has no type stubs


def f_closed(beta: Mpf, r: float) -> Mpf:
    """Proposition 2.4: (a) ``r = 1``, (b) ``r = 2``, (d) ``r = 1/2``."""
    b = mp.mpf(beta)
    if r == 1.0:
        return mp.sqrt(1 + b**2)
    if r == 2.0:
        return (mp.sqrt(1 + b**2) + mp.asinh(b) / b) / 2
    if r == 0.5:
        return mp.sqrt(1 + b**2) + b**2 * mp.asinh(1 / b)
    raise ValueError(f"no closed form at r = {r}")


def _split(b: Mpf, rr: Mpf) -> list[Any]:
    pts: list[Any] = [0, 1]
    if rr != 1:
        us = b ** (-1 / (rr - 1))
        if 0 < us < 1:
            pts = [0, us, 1]
    return pts


def f_integrals(beta: float, r: float) -> tuple[Mpf, Mpf, Mpf]:
    """``(F - 1, F_beta, F_r)`` from the Proposition 2.2 u-integral at ``MP_DPS`` digits."""
    with mp.workdps(MP_DPS):
        b, rr = mp.mpf(beta), mp.mpf(r)

        def s(u: Mpf) -> Mpf:
            return b**2 * u ** (2 * rr - 2)

        pts = _split(b, rr)
        fm1 = mp.quad(lambda u: s(u) / (mp.sqrt(1 + s(u)) + 1), pts, maxdegree=10)
        f_beta = mp.quad(lambda u: s(u) / mp.sqrt(1 + s(u)), pts, maxdegree=10) / b
        f_r = mp.quad(lambda u: mp.log(u) * s(u) / mp.sqrt(1 + s(u)), pts, maxdegree=10)
        return fm1, f_beta, f_r


def f_mp(beta: Mpf, r: float) -> Mpf:
    """``F(beta, r)``: the closed form where Proposition 2.4 gives one, else the integral."""
    if r in CLOSED_FORM_R:
        return f_closed(beta, r)
    return 1 + f_integrals(float(beta), r)[0]


def rho(ell: Mpf, w: Mpf, r: float, nu: Mpf, q: Mpf, s: Mpf, *, kappa: bool = True) -> Mpf:
    """Eq. 3.1: ``rho = (5/3) ell - (2/3) ln F + w + s/2 - nu - q``, ``beta = 2(r+1)e^(ell-w)``."""
    with mp.workdps(MP_DPS):
        ell, w, nu, q, s = (mp.mpf(v) for v in (ell, w, nu, q, s))
        k = -mp.log(f_mp(2 * (mp.mpf(r) + 1) * mp.exp(ell - w), r)) if kappa else mp.mpf(0)
        return mp.mpf(5) / 3 * ell + mp.mpf(2) / 3 * k + w + s / 2 - nu - q


def ell0(w: Mpf, nu: Mpf, q: Mpf, s: Mpf) -> Mpf:
    """Wide-channel depth ``(3/5)(q + nu - w - s/2)`` (docs/2 section 3.2, ablation A1)."""
    with mp.workdps(MP_DPS):
        return mp.mpf(3) / 5 * (mp.mpf(q) + mp.mpf(nu) - mp.mpf(w) - mp.mpf(s) / 2)


def root(w: Mpf, r: float, nu: Mpf, q: Mpf, s: Mpf) -> Mpf:
    """Unique root ``ell*`` of ``rho`` (Prop 3.1), bracketed by Prop 3.2 (1)."""
    with mp.workdps(MP_DPS):
        l0 = ell0(w, nu, q, s)
        k0 = -mp.log(f_mp(2 * (mp.mpf(r) + 1) * mp.exp(l0 - mp.mpf(w)), r))
        lo, hi = l0 + mp.mpf(2) / 5 * abs(k0), l0 + mp.mpf(2) / 3 * abs(k0)

        def fn(x: Mpf) -> Mpf:
            return rho(x, w, r, nu, q, s)

        if hi == lo:
            return lo
        return mp.findroot(fn, (lo, hi), solver="anderson", tol=mp.mpf(10) ** (-(MP_DPS - 5)))


def picard(w: Mpf, r: float, nu: Mpf, q: Mpf, s: Mpf, m: int) -> list[Mpf]:
    """``[ell_0, ..., ell_m]`` of ``G(ell) = ell_0 + (2/5) ln F`` (Proposition 3.2 (2))."""
    with mp.workdps(MP_DPS):
        l0 = ell0(w, nu, q, s)
        out = [l0]
        for _ in range(m):
            beta = 2 * (mp.mpf(r) + 1) * mp.exp(out[-1] - mp.mpf(w))
            out.append(l0 + mp.mpf(2) / 5 * mp.log(f_mp(beta, r)))
        return out


def eta(beta: Mpf, r: float) -> Mpf:
    """``eta = beta F_beta / F`` (Proposition 2.7); ``F_beta`` by ``mp.diff`` of ``F``."""
    with mp.workdps(MP_DPS):
        b = mp.mpf(beta)
        if r in CLOSED_FORM_R:
            return b * mp.diff(lambda x: f_closed(x, r), b) / f_closed(b, r)
        fm1, f_beta, _ = f_integrals(float(beta), r)
        return b * f_beta / (1 + fm1)


def one_minus_eta(beta: float, r: float) -> Mpf:
    """``1 - eta = (F - beta F_beta) / F`` from the u-integrals (Prop 2.7), at 40 digits.

    Used by the closed-end rule O5 (docs/3 section 12; P7-A1/A2/A3 (a)): rows with float
    ``eta >= 1 - 4 eps`` must satisfy ``|eta_float - (1 - this value)| <= 4 eps``.
    """
    with mp.workdps(MP_DPS):
        fm1, f_beta, _ = f_integrals(beta, r)
        f = 1 + fm1
        return (f - mp.mpf(beta) * f_beta) / f


def ift_partials(ell: Mpf, w: Mpf, r: float) -> dict[str, Mpf]:
    """Proposition 3.3 at depth ``ell``: ``d ell*/d(q, nu, s, w, r)`` from the u-integrals.

    ``eta = beta F_beta / F``, ``k_r = -(2 a F_beta + F_r) / F`` (Props 2.7, 2.8),
    ``D = 5/3 - (2/3) eta``; then ``1/D``, ``-1/(2D)``, ``-(1 + 2 eta/3)/D``, ``-(2/3) k_r/D``.
    """
    with mp.workdps(MP_DPS):
        beta_f = float(2 * (mp.mpf(r) + 1) * mp.exp(mp.mpf(ell) - mp.mpf(w)))
        beta = mp.mpf(beta_f)
        a = beta / (2 * (mp.mpf(r) + 1))
        fm1, f_beta, f_r = f_integrals(beta_f, r)
        f = 1 + fm1
        eta_v = beta * f_beta / f
        k_r = -(2 * a * f_beta + f_r) / f
        d = mp.mpf(5) / 3 - mp.mpf(2) / 3 * eta_v
        return {
            "q": 1 / d,
            "nu": 1 / d,
            "s": -1 / (2 * d),
            "w": -(1 + mp.mpf(2) / 3 * eta_v) / d,
            "r": -(mp.mpf(2) / 3) * k_r / d,
        }


def rel(got: float, ref: Mpf) -> float:
    """``|got - ref| / |ref|`` evaluated in mpmath."""
    with mp.workdps(MP_DPS):
        return float(abs(mp.mpf(got) - ref) / abs(ref))


def printed_unit(printed: str) -> float:
    """One unit in the last printed digit (tier 3, ADL-080)."""
    return 10.0 ** -len(printed.split(".")[1])


def tier3(got: float, printed: str) -> tuple[bool, float, float]:
    """``(ok, |got - printed|, unit)`` under the ADL-080 one-unit rule."""
    unit = printed_unit(printed)
    err = abs(got - float(printed))
    return err <= unit, err, unit

"""OPS-1 v2 composite section and strip-conveyance rating curve (docs/5_operational_floodplain_v2).

Composite bed for a model section burned into a 3DEP transect::

    b_m(x) = z_b + dmax0 (2 |x - x_c| / W0)^r ,   z_b = WSE_a - dmax0       (Dingman power law)
    b(x)   = min(b_m(x), b_dem(x))                (floodplain variant)

Strip (local-depth) conveyance with one calibrated roughness (OPS2-E1)::

    K(z) = integral over the contiguous wetted run around x_c of (z - b(x))^{5/3} dx,
    Q(z) = Q_a K(z) / K(WSE_a).

On a piecewise-linear bed the integral is exact per segment (:func:`segment_integrals`).
``K`` is continuous and strictly increasing in ``z``; stages are obtained by monotone
interpolation of ``ln K`` on a geometric depth grid. The control variant (no terrain) on a pure
power-law bed has ``Q ∝ (z - z_b)^{1/r + 5/3}`` exactly (:func:`control_stage_change`).

NumPy only. Owning phase: OPS-1 v2 (post-release extension).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Terrain",
    "composite_grid",
    "control_stage_change",
    "conveyance",
    "floodplain_stage_change",
    "prepare_terrain",
    "segment_integrals",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolLike = NDArray[np.bool_]
P53: Final[float] = 5.0 / 3.0
P83: Final[float] = 8.0 / 3.0


@dataclass(frozen=True)
class Terrain:
    """A transect in local coordinates: ``x`` (m, increasing), ``z`` (m NAVD88, gaps filled)."""

    x: FloatArray
    z: FloatArray
    x_c: float
    z_ch: float
    z_end_low: float
    nodata_share: float


def prepare_terrain(
    x: ArrayLike, z: ArrayLike, centre_x: float, search_half: float, flat_tol: float
) -> Terrain:
    """Sort, fill interior no-data linearly, locate the channel centre and the lower end height.

    ``z_end_low`` is the lower of the first and last finite elevations: above it the transect is
    truncated (water would leave the sampled valley).
    """
    from src.operational.transects import channel_centre

    xa = np.asarray(x, dtype=np.float64)
    za = np.asarray(z, dtype=np.float64)
    order = np.argsort(xa, kind="mergesort")
    xa, za = xa[order], za[order]
    finite = np.isfinite(za)
    if finite.sum() < 3:
        raise ValueError("fewer than 3 finite DEM samples")
    share = float(1.0 - finite.mean())
    x_c, z_ch = channel_centre(xa, za, centre_x, search_half, flat_tol)
    zf = np.interp(xa, xa[finite], za[finite])
    fi = np.flatnonzero(finite)
    z_end_low = float(min(za[fi[0]], za[fi[-1]]))
    return Terrain(x=xa, z=zf, x_c=x_c, z_ch=z_ch, z_end_low=z_end_low, nodata_share=share)


def segment_integrals(x: FloatArray, d: FloatArray) -> FloatArray:
    """Exact ``∫ max(d, 0)^{5/3} dx`` per segment of a piecewise-linear depth ``d`` on ``x``.

    ``d`` may be ``(N,)`` or ``(Z, N)``; returns ``(N-1,)`` or ``(Z, N-1)``.
    """
    dx = np.diff(x)
    d0, d1 = d[..., :-1], d[..., 1:]
    p0, p1 = np.maximum(d0, 0.0), np.maximum(d1, 0.0)
    both = (d0 > 0) & (d1 > 0)
    one = (d0 > 0) ^ (d1 > 0)
    out = np.zeros(np.broadcast(d0, dx).shape)
    # both wet: dx (d1^{8/3} - d0^{8/3}) / ((8/3)(d1 - d0)), or dx d^{5/3} when equal
    diff = d1 - d0
    with np.errstate(divide="ignore", invalid="ignore"):
        exact = dx * (p1**P83 - p0**P83) / (P83 * diff)
    flat = dx * p0**P53
    out = np.where(both & (np.abs(diff) > 1e-12 * np.maximum(p0, p1)), exact, out)
    out = np.where(both & ~(np.abs(diff) > 1e-12 * np.maximum(p0, p1)), flat, out)
    # one wet: wetted fraction f = d+/(d+ - d-), integral = (3/8) f dx d+^{5/3}
    dp = np.maximum(p0, p1)
    dn = np.minimum(d0, d1)
    with np.errstate(divide="ignore", invalid="ignore"):
        part = 0.375 * dx * dp / (dp - dn) * dp**P53
    out = np.where(one, part, out)
    return np.asarray(out, dtype=np.float64)


def conveyance(
    x: FloatArray, b: FloatArray, i_c: int, z: FloatArray
) -> tuple[FloatArray, BoolLike]:
    """``K(z)`` over the contiguous wetted run containing node ``i_c``; and a truncation flag.

    Returns ``(K, truncated)`` for every level in ``z`` (``(Z,)``); ``truncated`` is True where the
    wetted run reaches either end of the transect.
    """
    d = z[:, None] - b[None, :]
    dry = d <= 0
    n = x.size
    left = dry[:, : i_c + 1][:, ::-1]
    has_l = left.any(axis=1)
    lb = np.where(has_l, i_c - np.argmax(left, axis=1), -1)
    right = dry[:, i_c:]
    has_r = right.any(axis=1)
    rb = np.where(has_r, i_c + np.argmax(right, axis=1), n)
    seg = segment_integrals(x, d)
    j = np.arange(n - 1)[None, :]
    lo = np.maximum(lb, 0)[:, None]
    hi = np.minimum(rb, n - 1)[:, None]
    keep = (j >= lo) & (j < hi)
    k = np.where(keep, seg, 0.0).sum(axis=1)
    truncated = (~has_l) | (~has_r)
    return k, truncated


def composite_grid(
    terrain: Terrain | None,
    *,
    w0: float,
    dmax0: float,
    r: float,
    z_b: float,
    z_top: float,
    channel_points: int,
) -> tuple[FloatArray, FloatArray, int]:
    """Nodes ``x``, composite bed ``b`` and the centre node index for one model section.

    With ``terrain=None`` (control) the grid spans the power-law channel up to ``z_top``.
    """
    half = 0.5 * w0 * (max(z_top - z_b, 0.0) / dmax0) ** (1.0 / r)
    if terrain is None:
        xc = 0.0
        xs = np.linspace(-half, half, channel_points)
        xs = np.union1d(xs, [xc])
        bm = z_b + dmax0 * (2.0 * np.abs(xs - xc) / w0) ** r
        return xs, bm, int(np.searchsorted(xs, xc))
    xc = terrain.x_c
    lo, hi = terrain.x[0], terrain.x[-1]
    dense = np.linspace(max(xc - half, lo), min(xc + half, hi), channel_points)
    xs = np.union1d(np.union1d(terrain.x, dense), [xc])
    bm = z_b + dmax0 * (2.0 * np.abs(xs - xc) / w0) ** r
    bd = np.interp(xs, terrain.x, terrain.z)
    return xs, np.minimum(bm, bd), int(np.searchsorted(xs, xc))


def control_stage_change(dmax0: ArrayLike, r: ArrayLike, ln_q_ratio: ArrayLike) -> FloatArray:
    """Control variant: ``Δh = dmax0 ((Q/Q_a)^{1/(1/r + 5/3)} - 1)`` (exact strip result)."""
    d = np.asarray(dmax0, dtype=np.float64)
    e = 1.0 / (1.0 / np.asarray(r, dtype=np.float64) + P53)
    out: FloatArray = d * np.expm1(e * np.asarray(ln_q_ratio, dtype=np.float64))
    return out


def floodplain_stage_change(
    terrain: Terrain,
    *,
    wse_a: float,
    w0: float,
    dmax0: float,
    r: float,
    ln_q_ratio: FloatArray,
    channel_points: int,
    z_levels: int,
    max_doublings: int = 12,
) -> tuple[FloatArray, BoolLike]:
    """Floodplain variant: predicted ``Δh`` at each ``ln(Q_i/Q_a)`` and a truncation flag.

    Levels are geometric in depth above the thalweg from ``1e-3 dmax0`` to ``H``; ``H`` starts at
    ``3 dmax0`` and doubles until ``K(z_b + H) / K(WSE_a)`` exceeds the largest requested ratio.
    """
    z_b = wse_a - dmax0
    need = float(np.max(ln_q_ratio)) if ln_q_ratio.size else 0.0
    h = 3.0 * dmax0
    for _ in range(max_doublings):
        x, b, i_c = composite_grid(
            terrain, w0=w0, dmax0=dmax0, r=r, z_b=z_b, z_top=z_b + h, channel_points=channel_points
        )
        depths = dmax0 * np.geomspace(1e-3, h / dmax0, z_levels)
        z = np.union1d(z_b + depths, [wse_a])
        k, trunc = conveyance(x, b, i_c, z)
        k_a = float(k[np.searchsorted(z, wse_a)])
        if not k_a > 0:
            raise ValueError("zero conveyance at the anchor stage")
        ln_rel = np.log(np.maximum(k, 1e-300)) - np.log(k_a)
        if ln_rel[-1] >= need:
            break
        h *= 2.0
    else:
        raise RuntimeError("rating curve did not reach the largest requested flow")
    # strictly increasing ln K (guard against flat steps from the 1e-300 floor)
    ok = np.concatenate([[True], np.diff(ln_rel) > 0])
    zz, ll, tt = z[ok], ln_rel[ok], trunc[ok]
    z_i = np.interp(ln_q_ratio, ll, zz)
    trunc_i = tt[np.clip(np.searchsorted(ll, ln_q_ratio), 0, ll.size - 1)]
    return z_i - wse_a, trunc_i

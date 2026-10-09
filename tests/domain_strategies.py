"""Hypothesis strategies on the admissible domain D and shared assertion helpers (Phase 7).

Sources: docs/2 section 1.3 (domain D: ``W > 0``, ``r in [r_min, r_max]``, ``n in [n_min, n_max]``,
``Q_E > 0``, ``S >= S_floor``), section 7.6 (bounded logistic transforms), Theorem 4.1 note on
invariants 1-2 in floating point; docs/3 section 2.2 (P3-01..P3-03 bounds in params ``domain``);
P7-D6 (a) (profiles and boundary atoms; ADL-101).

Boundary atoms (P7-D6 (a)): ``r = 1/2`` and ``r = 10`` exactly; ``S = S_floor`` with
``s = np.log(S_floor)`` exactly (the loader's NumPy log, ``src.data.dataset``); logits
``z = -40`` and ``z = +40`` (``sigma(z)`` rounds to 0 or 1). Beta atoms ``1e-8`` and ``1e3``
are used by the quadrature reference tests.

Ranges not fixed by docs/2 (test-design choices, recorded in docs/3 section 12): ``W`` in
[0.1, 5000] m and ``Q_E`` in [1e-4, 1e5] m3/s (log-uniform), ``S`` up to 0.1. Docs/2 section 1.3
bounds only ``r``, ``n`` and ``S`` from below; ``W`` and ``Q_E`` may be any positive reals.

This module is importable because ``pyproject.toml`` sets ``pythonpath = ["tests"]``:
``@given`` needs its strategies at import time (P7-D4 (a) places them beside ``conftest.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from hypothesis import strategies as st
from torch import Tensor

from src.config import load_settings
from src.data.normstats import NormStats
from src.models.heads import BoundedTransforms, LogFeature

PROTOCOL = load_settings().protocol
DOMAIN = PROTOCOL.domain
SOLVER = PROTOCOL.solver
TR = BoundedTransforms.from_domain(DOMAIN)
TAU_L: float = SOLVER.tau_l
TAU_Q: float = SOLVER.tau_q
S_FLOOR: float = DOMAIN.s_floor
LN_S_FLOOR: float = float(np.log(S_FLOOR))
N_MIN, N_MAX = DOMAIN.n_bounds
R_MIN, R_MAX = DOMAIN.r_bounds

W_RANGE = (0.1, 5000.0)
Q_RANGE = (1e-4, 1e5)
S_MAX = 0.1
LOGIT_ATOM = 40.0
MAX_ROWS = 32

#: Synthetic feature layout of the ladder tests (batch contract ADL-077): F = 9, I_blind routes
#: out columns 3, 4, 8 (they play Q_E, S, F_S); column 6 is D50 (log feature, for A3-SP/SR).
N_FEATURES = 9
D50_COL = 6
BLIND_IDX = torch.tensor([0, 1, 2, 5, 6, 7])
STATS = NormStats(
    names=tuple(f"f{j}" if j != D50_COL else "D50_m" for j in range(N_FEATURES)),
    kinds=tuple("log" if j == D50_COL else "identity" for j in range(N_FEATURES)),
    offsets=(0.0,) * N_FEATURES,
    mean=tuple(math.log(0.02) if j == D50_COL else 0.0 for j in range(N_FEATURES)),
    std=tuple(1.5 if j == D50_COL else 1.0 for j in range(N_FEATURES)),
    n_fit=100,
)
LN_D50 = LogFeature.from_stats(STATS, "D50_m")


def t64(*values: float) -> Tensor:
    return torch.tensor(values, dtype=torch.float64)


def _nu_of(z: float) -> float:
    return float(TR.nu(t64(z))[0])


def _r_of(z: float) -> float:
    return float(TR.r(t64(z))[0])


logits = st.one_of(st.sampled_from([-LOGIT_ATOM, LOGIT_ATOM]), st.floats(-LOGIT_ATOM, LOGIT_ATOM))
w_values = st.floats(math.log(W_RANGE[0]), math.log(W_RANGE[1]))
q_values = st.floats(math.log(Q_RANGE[0]), math.log(Q_RANGE[1]))
s_values = st.one_of(st.just(LN_S_FLOOR), st.floats(LN_S_FLOOR, math.log(S_MAX)))
nu_values = logits.map(_nu_of)
r_values = st.one_of(st.sampled_from([R_MIN, R_MAX]), logits.map(_r_of))


@dataclass(frozen=True)
class Rows:
    """A batch of domain rows ``(w, r, nu, q, s)`` in log coordinates (docs/2 section 1.1.1)."""

    w: Tensor
    r: Tensor
    nu: Tensor
    q: Tensor
    s: Tensor

    @property
    def n(self) -> int:
        return int(self.w.shape[0])

    def args(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Positional ``(w, r, nu, q, s)`` for :func:`src.physics.manning_layer.phi`."""
        return self.w, self.r, self.nu, self.q, self.s

    def phys(self) -> Tensor:
        """``(B, 2)`` ``[ln Q_E, ln S]`` (batch contract, ADL-077)."""
        return torch.stack([self.q, self.s], dim=1)

    def describe(self, i: int) -> str:
        """Offending-row report: inputs ``(w, nu, ln r, q, s)`` at full precision."""
        vals = (
            float(self.w[i]),
            float(self.nu[i]),
            math.log(float(self.r[i])),
            float(self.q[i]),
            float(self.s[i]),
        )
        return "row {}: (w, nu, ln r, q, s) = ({:.17g}, {:.17g}, {:.17g}, {:.17g}, {:.17g})".format(
            i, *vals
        )

    @classmethod
    def from_si(cls, table: list[tuple[float, float, float, float, float]]) -> Rows:
        """Rows from SI tuples ``(W, r, n, Q_E, S)``."""
        arr = np.asarray(table, dtype=np.float64)
        return cls(
            w=torch.from_numpy(np.log(arr[:, 0])),
            r=torch.from_numpy(arr[:, 1].copy()),
            nu=torch.from_numpy(np.log(arr[:, 2])),
            q=torch.from_numpy(np.log(arr[:, 3])),
            s=torch.from_numpy(np.log(arr[:, 4])),
        )


@st.composite
def phi_rows(draw: st.DrawFn, min_size: int = 1, max_size: int = MAX_ROWS) -> Rows:
    """Batches of 1..``max_size`` rows of D with the P7-D6 boundary atoms."""
    table = draw(
        st.lists(
            st.tuples(w_values, r_values, nu_values, q_values, s_values),
            min_size=min_size,
            max_size=max_size,
        )
    )
    cols = [torch.tensor(c, dtype=torch.float64) for c in zip(*table, strict=True)]
    return Rows(w=cols[0], r=cols[1], nu=cols[2], q=cols[3], s=cols[4])


def model_batch(rows: Rows, seed: int) -> dict[str, Tensor]:
    """Synthetic ADL-077 batch on ``rows`` (``q, s`` used; ``x`` seeded normal; ``y = 0``)."""
    g = torch.Generator().manual_seed(seed)
    b = rows.n
    return {
        "x": torch.randn(b, N_FEATURES, dtype=torch.float64, generator=g),
        "x_blind_idx": BLIND_IDX,
        "phys": rows.phys(),
        "y": torch.zeros(b, 4, dtype=torch.float64),
        "omega": torch.full((b,), 1.0 / b, dtype=torch.float64),
        "row": torch.arange(b),
    }


# ----------------------------------------------------------------------------- assertions
def require(ok: Tensor, magnitude: Tensor, bound: str, what: str, ref: str, rows: Rows) -> None:
    """Assert ``ok`` on every row; report the worst violation, its bound and the row.

    ``magnitude`` is the per-row violation size (positive = worse); NaN counts as a violation.
    """
    bad = ~ok
    if bool(bad.any()):
        mag = torch.where(bad, magnitude.detach().nan_to_num(math.inf), -math.inf)
        i = int(mag.argmax())
        raise AssertionError(
            f"{what} violated ({ref}): magnitude {float(magnitude[i]):.6e}, required {bound}; "
            f"{int(bad.sum())} of {rows.n} rows; worst {rows.describe(i)}"
        )


def require_le(value: Tensor, tol: float, what: str, ref: str, rows: Rows) -> None:
    """Assert ``value <= tol`` row-wise (``value`` is a non-negative error measure)."""
    v = value.detach()
    require(v <= tol, v, f"<= {tol:.3e}", what, ref, rows)


def require_equal(got: Tensor, want: Tensor, what: str, ref: str, rows: Rows) -> None:
    """Assert bit-equality row-wise; the magnitude is ``|got - want|``."""
    g, w = got.detach(), want.detach()
    require(g == w, (g - w).abs(), "bit-equal", what, ref, rows)


def up2(x: float) -> float:
    """``x`` moved up by 2 ulp (``np.nextafter`` twice; docs/3 section 12, P7-Q3 (a))."""
    return float(np.nextafter(np.nextafter(x, math.inf), math.inf))


def down2(x: float) -> float:
    """``x`` moved down by 2 ulp (``np.nextafter`` twice; docs/3 section 12, P7-Q3 (a))."""
    return float(np.nextafter(np.nextafter(x, -math.inf), -math.inf))


def ulp_excess(x: Tensor, lo: float, hi: float) -> Tensor:
    """Distance outside ``[lo, hi]`` in units of the bound's ulp (0 inside)."""
    xv = x.detach()
    below = (lo - xv).clamp(min=0.0) / float(np.spacing(abs(lo)))
    above = (xv - hi).clamp(min=0.0) / float(np.spacing(abs(hi)))
    return torch.maximum(below, above)


def require_in(
    x: Tensor, lo: float, hi: float, what: str, ref: str, rows: Rows, *, ulps: int = 0
) -> None:
    """Assert ``lo' <= x <= hi'`` where the bounds are widened by ``ulps`` (0 or 2) ulp."""
    if ulps not in (0, 2):
        raise ValueError("only exact (0) or the P7-Q3 (a) 2-ulp rule are allowed")
    lo_w, hi_w = (down2(lo), up2(hi)) if ulps == 2 else (lo, hi)
    xv = x.detach()
    ok = (xv >= lo_w) & (xv <= hi_w)
    require(ok, ulp_excess(xv, lo, hi), f"in [{lo!r}, {hi!r}] +/- {ulps} ulp", what, ref, rows)

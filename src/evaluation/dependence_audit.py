"""Spatial and network dependence audit (diagnostic only; it cannot change the splits).

Sources: protocol section 3.6 (P3-D4): 3.6.2 global Moran's I (weights W1 = kNN-8 binary
row-standardized; W2 = inverse distance within 100 km, row-standardized, isolates get a zero row;
distance-class correlogram with binary weights; 999 permutations; one-sided pseudo-p
``(1 + #{I* >= I}) / 1000``; permutation z-score); 3.6.3 LISA ``I_i = (z_i - z_bar)/m_2
sum_j w_ij (z_j - z_bar)`` with ``m_2 = sum (z_i - z_bar)^2 / N``, 999 conditional permutations,
pseudo-p < 0.01; 3.6.4 semivariogram (20 equal-pair-count lag classes up to 500 km; spherical,
exponential and pure-nugget fits weighted by ``N_k``; pure nugget within 5 % of the best weighted
SSE -> ``rho_hat = 0``; practical range at 95 % of the sill; ``rho* = min(rho_hat, 250 km)``);
3.6.5 nested-basin check; 3.6.6 area of applicability; 3.6.7 reporting rule (conditions D and N).
Values from ``params.yaml`` -> ``audit`` and ``audit_rules``.

Implementation choices (ADL-106; PI decisions 2026-10-09):
- Global Moran's I: own sparse formula and an own permutation null drawn with
  ``numpy.random.default_rng(seed)``, because esda's ``p_sim`` is folded (two-directional) while
  the protocol fixes a one-sided pseudo-p. esda's ``Moran`` is the test oracle for ``I``.
- LISA: ``esda.Moran_Local`` (pinned 2.10.0) with ``alternative='two-sided'`` and a seed. esda
  scales ``I_i`` by ``(n-1)/n`` relative to the protocol formula; the value is rescaled by
  ``n/(n-1)``, which leaves pseudo-p and quadrants unchanged. esda seeds NumPy's legacy global RNG
  inside its conditional randomization.
- Practical range: 95 % of the total sill ``c0 + c``; 0 if the nugget alone reaches it.
- Equal-count lag classes: pairs sorted by ``(distance, i, j)`` and split with
  ``numpy.array_split`` (class sizes differ by at most one).
- Coordinates are in km (EPSG:5070 metres / 1000).
- Nested basins: generic ``FromComID -> ToComID`` edges (``0`` = no node, dropped); reachability
  by breadth-first search with ``scipy.sparse.csgraph``; pairs must share the terminal path.
- AOA: predictors standardized with the training mean and sample SD (``ddof = 1``), weighted by
  the importances; ``DI`` = nearest-training distance / mean pairwise training distance; the
  training ``DI`` uses the nearest training site in another CV fold; threshold
  ``Q3 + 1.5 IQR`` (``numpy.percentile``, linear).

Torch-free and xgboost-free (import-isolation test). Owning phase: 7 (P7-D1 (a)).

Phase 8 additions (docs/3 13.9, ADL-117; known-answer tests before Event 1):
- W2 with co-located sites: pairs at ``d_ij = 0`` get weight 0 in W2 and are counted
  (:func:`colocated_pairs`); W1 is unchanged. This replaces the Phase 7 refusal on ``1/0``.
- AOA importance (:func:`permutation_importance`): for each M2-XGB member, on its ``es_val``
  rows, ``AOA_PERMUTATIONS`` permutations per feature drawn from
  ``default_rng(SeedSequence([member_seed, feature_index]))``; importance = the mean increase in
  the d-bar log-RMSE, averaged over members, negative values set to 0. The booster enters as a
  ``predict`` callable, so this module stays xgboost-free.
- Nested-basin BFS: :func:`nested_status` is unchanged; its inputs from the raw ``nhdplusv21``
  unit are built by :func:`src.data.network.network_inputs`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final, Literal

import numpy as np
from esda.moran import Moran_Local
from libpysal.weights import KNN, W
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.optimize import brentq, least_squares
from scipy.sparse.csgraph import breadth_first_order
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist

from src.config import AuditParams, AuditRulesParams

__all__ = [
    "AOA_PERMUTATIONS",
    "AuditLabel",
    "ImportanceMember",
    "LisaResult",
    "MoranResult",
    "NestedResult",
    "Variogram",
    "VgmFit",
    "VgmSelection",
    "aoa_threshold",
    "audit_label",
    "band_weights",
    "colocated_pairs",
    "correlogram",
    "dependence_radius",
    "dissimilarity_index",
    "empirical_variogram",
    "far_mask",
    "fit_variogram",
    "idw_weights",
    "knn_w",
    "lisa",
    "median_nn_distance",
    "moran_global",
    "nearest_distance",
    "nested_status",
    "permutation_importance",
    "practical_range",
    "select_variogram",
    "training_cv_di",
    "vgm_model",
]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
VgmName = Literal["spherical", "exponential", "nugget"]
QUADRANTS = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}  # esda Moran_Local.q, geoda_quads=False
VGM_TOL = 1e-12  # least-squares termination tolerances (scipy default 1e-8 is too loose)
#: docs/3 13.9: permutations per feature and member for the AOA importance.
AOA_PERMUTATIONS: Final[int] = 10


# ----------------------------------------------------------------------------- helpers
def _coords(xy: ArrayLike, name: str = "coords") -> FloatArray:
    a = np.asarray(xy, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] < 1 or not bool(np.isfinite(a).all()):
        raise ValueError(f"{name} must be a finite (N, 2) array in km, got {a.shape}")
    return a


def _values(z: ArrayLike, n: int, name: str = "z") -> FloatArray:
    a = np.asarray(z, dtype=np.float64)
    if a.shape != (n,) or not bool(np.isfinite(a).all()):
        raise ValueError(f"{name} must be a finite ({n},) array, got {a.shape}")
    return a


# ----------------------------------------------------------------------------- weights
def knn_w(coords_km: ArrayLike, k: int) -> W:
    """W1: ``k`` nearest neighbours, binary, row-standardized, ``w_ii = 0`` (protocol 3.6.2)."""
    xy = _coords(coords_km)
    if not 1 <= k < xy.shape[0]:
        raise ValueError(f"need 1 <= k < N, got k={k}, N={xy.shape[0]}")
    w = KNN.from_array(xy, k=k)
    w.transform = "r"
    return w


def idw_weights(coords_km: ArrayLike, band_km: float) -> tuple[sparse.csr_array, NDArray[np.int64]]:
    """W2: ``1/d_ij`` for ``0 < d_ij <= band``, row-standardized; returns ``(W, isolates)``.

    Isolates (no neighbour within the band) keep a zero row (protocol 3.6.2). Co-located pairs
    (``d_ij = 0``) get weight 0 (docs/3 13.9); count them with :func:`colocated_pairs`. A site
    whose only neighbours within the band are co-located is an isolate.
    """
    xy = _coords(coords_km)
    n = xy.shape[0]
    pairs = cKDTree(xy).query_pairs(r=band_km, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    d = np.linalg.norm(xy[i] - xy[j], axis=1)
    pos = d > 0
    i, j, d = i[pos], j[pos], d[pos]
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    vals = np.concatenate([1.0 / d, 1.0 / d])
    mat = sparse.csr_array((vals, (rows, cols)), shape=(n, n))
    rs = np.asarray(mat.sum(axis=1)).ravel()
    isolates = np.flatnonzero(rs == 0).astype(np.int64)
    scale = np.divide(1.0, rs, out=np.zeros_like(rs), where=rs > 0)
    return sparse.csr_array(sparse.diags_array(scale) @ mat), isolates


def colocated_pairs(coords_km: ArrayLike) -> int:
    """Count unordered site pairs at distance exactly 0 (weight 0 in W2; docs/3 13.9)."""
    xy = _coords(coords_km)
    return int(cKDTree(xy).query_pairs(r=0.0, output_type="ndarray").shape[0])


def band_weights(coords_km: ArrayLike, lower_km: float, upper_km: float) -> sparse.csr_array:
    """Binary weights for the distance class ``(lower, upper]`` (protocol 3.6.2 correlogram)."""
    xy = _coords(coords_km)
    if not 0.0 <= lower_km < upper_km:
        raise ValueError("need 0 <= lower < upper")
    n = xy.shape[0]
    pairs = cKDTree(xy).query_pairs(r=upper_km, output_type="ndarray")
    d = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
    keep = pairs[d > lower_km]
    rows = np.concatenate([keep[:, 0], keep[:, 1]])
    cols = np.concatenate([keep[:, 1], keep[:, 0]])
    return sparse.csr_array((np.ones(rows.size), (rows, cols)), shape=(n, n))


# ----------------------------------------------------------------------------- Moran
@dataclass(frozen=True)
class MoranResult:
    """Global Moran's I with its permutation inference (protocol 3.6.2)."""

    i: float
    expected: float
    z_perm: float
    pseudo_p: float
    s0: float


def moran_global(
    z: ArrayLike, w: sparse.sparray | sparse.spmatrix, permutations: int, seed: int
) -> MoranResult:
    """``I = (N/S0) sum_ij w_ij (z_i - z_bar)(z_j - z_bar) / sum_i (z_i - z_bar)^2`` (3.6.2).

    One-sided pseudo-p ``(1 + #{I* >= I}) / (permutations + 1)`` over random permutations of
    ``z``; z-score ``(I - mean I*) / sd I*`` (population SD).
    """
    mat = sparse.csr_array(w)
    n = mat.shape[0]
    zv = _values(z, n)
    if mat.shape != (n, n):
        raise ValueError("weights must be square (N, N)")
    s0 = float(mat.sum())
    if not s0 > 0.0:
        raise ValueError("S0 = 0: no neighbour pairs")
    zc = zv - zv.mean()
    den = float(zc @ zc)
    if not den > 0.0:
        raise ValueError("z has zero variance")
    stat = n / s0 * float(zc @ (mat @ zc)) / den
    if permutations < 1:
        raise ValueError("permutations must be >= 1")
    rng = np.random.default_rng(seed)
    perm = rng.permuted(np.tile(zc, (permutations, 1)), axis=1)
    lag = (mat @ perm.T).T
    sims = n / s0 * np.einsum("ij,ij->i", perm, lag) / den
    p = (1.0 + float(np.sum(sims >= stat))) / (permutations + 1.0)
    sd = float(sims.std())
    zs = (stat - float(sims.mean())) / sd if sd > 0 else math.nan
    return MoranResult(i=stat, expected=-1.0 / (n - 1), z_perm=zs, pseudo_p=p, s0=s0)


def correlogram(
    z: ArrayLike, coords_km: ArrayLike, cfg: AuditParams, seed: int
) -> list[tuple[float, float, MoranResult | None]]:
    """Moran's I per distance class ``(0, 10], (10, 25], ..., (250, 500]`` km (3.6.2).

    Returns ``(lower, upper, result)``; ``result`` is ``None`` for a class with no pairs.
    """
    edges = (0.0, *cfg.dist_classes_km)
    out: list[tuple[float, float, MoranResult | None]] = []
    for lo, hi in pairwise(edges):
        wb = band_weights(coords_km, lo, hi)
        res = moran_global(z, wb, cfg.permutations, seed) if wb.nnz else None
        out.append((lo, hi, res))
    return out


@dataclass(frozen=True)
class LisaResult:
    """Local Moran's I (protocol 3.6.3); descriptive only."""

    local_i: FloatArray
    pseudo_p: FloatArray
    quadrant: NDArray[np.str_]
    significant: BoolArray


def lisa(z: ArrayLike, w: W, cfg: AuditParams, seed: int) -> LisaResult:
    """LISA with W1 via esda ``Moran_Local`` (two-sided conditional permutations, seeded).

    ``local_i`` is on the protocol 3.6.3 scale (esda value times ``n/(n-1)``).
    """
    n = int(w.n)
    zv = _values(z, n)
    if not float(zv.std()) > 0.0:
        raise ValueError("z has zero variance")
    # A site with z_i == z_bar has a degenerate conditional null (all I_i* = 0); esda then
    # computes its unused z_sim as 0/0. Only that warning is silenced; the outputs used here are
    # checked for finiteness below.
    with np.errstate(invalid="ignore"):
        lm = Moran_Local(
            zv,
            w,
            transformation="r",
            permutations=cfg.permutations,
            seed=seed,
            alternative="two-sided",
            keep_simulations=True,
        )
    local_i = np.asarray(lm.Is, dtype=np.float64) * n / (n - 1)
    p = np.asarray(lm.p_sim, dtype=np.float64)
    if not (bool(np.isfinite(local_i).all()) and bool(np.isfinite(p).all())):
        raise ValueError("non-finite LISA statistic or pseudo-p")
    quad = np.array([QUADRANTS[int(q)] for q in np.asarray(lm.q)], dtype=np.str_)
    return LisaResult(local_i=local_i, pseudo_p=p, quadrant=quad, significant=p < cfg.lisa_alpha)


# ----------------------------------------------------------------------------- variogram
@dataclass(frozen=True)
class Variogram:
    """Empirical semivariogram: class mean lag ``h``, ``gamma`` and pair count ``n_pairs``."""

    h: FloatArray
    gamma: FloatArray
    n_pairs: NDArray[np.int64]


def empirical_variogram(e: ArrayLike, coords_km: ArrayLike, cfg: AuditParams) -> Variogram:
    """``gamma(h_k) = (1/(2 N_k)) sum_{(i,j) in P_k} (e_i - e_j)^2``, equal-count classes.

    Protocol 3.6.4: ``vgm_bins`` classes of equal pair count over pairs with ``d <= vgm_max_km``.
    """
    xy = _coords(coords_km)
    ev = _values(e, xy.shape[0], "e")
    pairs = cKDTree(xy).query_pairs(r=cfg.vgm_max_km, output_type="ndarray")
    if pairs.shape[0] < cfg.vgm_bins:
        raise ValueError(f"{pairs.shape[0]} pairs within {cfg.vgm_max_km} km < {cfg.vgm_bins}")
    i, j = pairs[:, 0], pairs[:, 1]
    d = np.linalg.norm(xy[i] - xy[j], axis=1)
    order = np.lexsort((j, i, d))
    sq = 0.5 * (ev[i] - ev[j]) ** 2
    h, g, cnt = [], [], []
    for part in np.array_split(order, cfg.vgm_bins):
        h.append(float(d[part].mean()))
        g.append(float(sq[part].mean()))
        cnt.append(part.size)
    return Variogram(h=np.array(h), gamma=np.array(g), n_pairs=np.array(cnt, dtype=np.int64))


def vgm_model(name: VgmName, h: ArrayLike, c0: float, c: float, a: float) -> FloatArray:
    """Spherical, exponential or pure-nugget model at lags ``h > 0`` (protocol 3.6.4)."""
    hv = np.asarray(h, dtype=np.float64)
    if name == "nugget":
        return np.full_like(hv, c0)
    if not a > 0.0:
        raise ValueError("range parameter a must be > 0")
    if name == "spherical":
        u = np.minimum(hv / a, 1.0)
        out: FloatArray = c0 + c * (1.5 * u - 0.5 * u**3)
        return out
    if name == "exponential":
        out = c0 + c * (1.0 - np.exp(-hv / a))
        return out
    raise ValueError(f"unknown variogram model {name!r}")


@dataclass(frozen=True)
class VgmFit:
    """A fitted variogram model and its pair-count-weighted SSE."""

    name: VgmName
    c0: float
    c: float
    a: float
    wsse: float


def fit_variogram(vg: Variogram, rules: AuditRulesParams) -> dict[str, VgmFit]:
    """Least-squares fits weighted by ``N_k`` for every model in ``rules.vgm_models`` (3.6.4)."""
    h, g = vg.h, vg.gamma
    wts = vg.n_pairs.astype(np.float64)
    fits: dict[str, VgmFit] = {}
    for name in rules.vgm_models:
        if name == "nugget":
            c0 = float(np.sum(wts * g) / np.sum(wts))
            fits[name] = VgmFit(name, c0, 0.0, math.nan, float(np.sum(wts * (g - c0) ** 2)))
            continue
        gmax = float(g.max())
        x0 = np.array([0.1 * gmax, 0.9 * gmax, float(h.max()) / 3.0])

        def resid(x: FloatArray, name: VgmName = name) -> FloatArray:
            out: FloatArray = np.sqrt(wts) * (vgm_model(name, h, x[0], x[1], x[2]) - g)
            return out

        sol = least_squares(
            resid,
            x0,
            bounds=([0.0, 0.0, 1e-9], [np.inf, np.inf, np.inf]),
            x_scale="jac",
            ftol=VGM_TOL,
            xtol=VGM_TOL,
            gtol=VGM_TOL,
        )
        c0, c, a = (float(v) for v in sol.x)
        wsse = float(np.sum(wts * (vgm_model(name, h, c0, c, a) - g) ** 2))
        fits[name] = VgmFit(name, c0, c, a, wsse)
    return fits


def practical_range(fit: VgmFit, frac: float) -> float:
    """Distance where the model reaches ``frac`` (0.95) of its total sill ``c0 + c`` (3.6.4)."""
    if fit.name == "nugget" or fit.c <= 0.0:
        return 0.0
    target = frac * (fit.c0 + fit.c)
    if fit.c0 >= target:
        return 0.0
    q = (target - fit.c0) / fit.c  # in (0, 1)
    if fit.name == "exponential":
        return -fit.a * math.log1p(-q)
    return fit.a * float(brentq(lambda u: 1.5 * u - 0.5 * u**3 - q, 0.0, 1.0, xtol=1e-14))


@dataclass(frozen=True)
class VgmSelection:
    """Selected model, practical range ``rho_hat`` and nugget/sill ratio (protocol 3.6.4)."""

    best: VgmFit
    rho_hat: float
    unstructured: bool
    nugget_sill: float


def select_variogram(fits: Mapping[str, VgmFit], rules: AuditRulesParams) -> VgmSelection:
    """Lowest weighted SSE; pure nugget within 5 % of the best -> unstructured, ``rho_hat = 0``."""
    if not fits:
        raise ValueError("no fitted models")
    best = min(fits.values(), key=lambda f: f.wsse)
    nug = fits.get("nugget")
    unstructured = nug is not None and nug.wsse <= (1.0 + rules.nugget_sse_margin) * best.wsse
    rho = 0.0 if unstructured else practical_range(best, rules.range_sill_frac)
    sill = best.c0 + best.c
    ratio = 1.0 if best.name == "nugget" else (best.c0 / sill if sill > 0 else math.nan)
    return VgmSelection(best=best, rho_hat=rho, unstructured=unstructured, nugget_sill=ratio)


def dependence_radius(rho_hat: float, cfg: AuditParams) -> float:
    """``rho* = min(rho_hat, 250 km)`` (protocol 3.6.4)."""
    if not (math.isfinite(rho_hat) and rho_hat >= 0.0):
        raise ValueError("rho_hat must be finite and >= 0")
    return min(rho_hat, cfg.rho_cap_km)


# ----------------------------------------------------------------------------- distances, label
def nearest_distance(test_km: ArrayLike, train_km: ArrayLike) -> FloatArray:
    """Euclidean distance from each test site to its nearest training site (km)."""
    d, _ = cKDTree(_coords(train_km, "train")).query(_coords(test_km, "test"), k=1)
    out: FloatArray = np.asarray(d, dtype=np.float64)
    return out


def median_nn_distance(test_km: ArrayLike, train_km: ArrayLike) -> float:
    """``d~_NN``: median test-to-nearest-training distance (protocol 3.6.7 condition D)."""
    return float(np.median(nearest_distance(test_km, train_km)))


def far_mask(test_km: ArrayLike, train_km: ArrayLike, rho_star: float) -> BoolArray:
    """Far stratum: no training site within ``rho*`` (nearest distance ``> rho*``; 3.6.4)."""
    out: BoolArray = nearest_distance(test_km, train_km) > rho_star
    return out


@dataclass(frozen=True)
class AuditLabel:
    """Conditions D and N of protocol 3.6.7 and the resulting reporting label."""

    d: bool
    n: bool

    @property
    def optimistic(self) -> bool:
        """S-ID headline numbers are labelled 'optimistic under spatial/network dependence'."""
        return self.d or self.n


def audit_label(
    moran: MoranResult,
    rho_hat: float,
    d_nn_median: float,
    nested_share: float,
    cfg: AuditParams,
    rules: AuditRulesParams,
) -> AuditLabel:
    """D: ``(p < 0.01 and I >= 0.10) or rho_hat > d~_NN``; N: nested share ``>= 0.20`` (3.6.7)."""
    d = (moran.pseudo_p < rules.moran_pseudo_p and moran.i >= cfg.moran_material) or (
        rho_hat > d_nn_median
    )
    return AuditLabel(d=d, n=nested_share >= cfg.nested_share_material)


# ----------------------------------------------------------------------------- nested basins
@dataclass(frozen=True)
class NestedResult:
    """Per test site: maximum ``psi`` over nested training sites (0 if none) and the strata."""

    psi_max: FloatArray
    strong: BoolArray
    weak: BoolArray

    @property
    def share_strong(self) -> float:
        """Share of test sites with at least one strongly nested training site (3.6.5)."""
        return float(np.mean(self.strong))


def nested_status(
    train_comids: Sequence[int],
    test_comids: Sequence[int],
    edges: ArrayLike,
    terminal_path: Mapping[int, int],
    drainage_km2: Mapping[int, float],
    cfg: AuditParams,
) -> NestedResult:
    """Nested-basin check on a ``FromComID -> ToComID`` graph (protocol 3.6.5).

    A training and a test site are nested if one COMID is reachable from the other along the
    flow graph (divergences included) and both share the terminal path. ``psi = DA_small /
    DA_large``; strongly nested if ``psi >= 0.5``.
    """
    e = np.asarray(edges, dtype=np.int64)
    if e.ndim != 2 or e.shape[1] != 2:
        raise ValueError("edges must be an (E, 2) integer array of (FromComID, ToComID)")
    e = e[(e[:, 0] != 0) & (e[:, 1] != 0)]
    sites = [int(c) for c in (*train_comids, *test_comids)]
    for c in sites:
        if c not in terminal_path or c not in drainage_km2 or not drainage_km2[c] > 0:
            raise ValueError(f"COMID {c} lacks a terminal path or a positive drainage area")
    nodes = np.unique(np.concatenate([e.ravel(), np.asarray(sites, dtype=np.int64)]))
    idx = {int(c): k for k, c in enumerate(nodes)}
    rows = np.searchsorted(nodes, e[:, 0])
    cols = np.searchsorted(nodes, e[:, 1])
    graph = sparse.csr_array((np.ones(rows.size), (rows, cols)), shape=(nodes.size, nodes.size))
    graph_t = sparse.csr_array(graph.T)
    psi_max = np.zeros(len(test_comids))
    for t, tc in enumerate(int(c) for c in test_comids):
        down = breadth_first_order(graph, idx[tc], directed=True, return_predecessors=False)
        up = breadth_first_order(graph_t, idx[tc], directed=True, return_predecessors=False)
        reach = set(down.tolist()) | set(up.tolist())
        for rc in (int(c) for c in train_comids):
            if idx[rc] in reach and terminal_path[rc] == terminal_path[tc]:
                a, b = drainage_km2[rc], drainage_km2[tc]
                psi_max[t] = max(psi_max[t], min(a, b) / max(a, b))
    strong = psi_max >= cfg.nested_psi
    weak = (psi_max > 0.0) & ~strong
    return NestedResult(psi_max=psi_max, strong=strong, weak=weak)


# ----------------------------------------------------------------------------- AOA
def _standardize(
    train_x: ArrayLike, importance: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    x = np.asarray(train_x, dtype=np.float64)
    wv = np.asarray(importance, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or wv.shape != (x.shape[1],):
        raise ValueError("train_x must be (N >= 2, F) and importance (F,)")
    if not (bool(np.isfinite(x).all()) and bool(np.isfinite(wv).all())):
        raise ValueError("non-finite predictors or importances")
    if bool((wv < 0).any()) or not float(wv.sum()) > 0:
        raise ValueError("importances must be non-negative with a positive sum")
    mean, sd = x.mean(axis=0), x.std(axis=0, ddof=1)
    if not bool((sd > 0).all()):
        raise ValueError("a predictor is constant on the training set")
    return mean, sd, wv


def _scaled(x: ArrayLike, mean: FloatArray, sd: FloatArray, wv: FloatArray) -> FloatArray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != mean.size or not bool(np.isfinite(a).all()):
        raise ValueError("predictor matrix has the wrong shape or non-finite values")
    out: FloatArray = (a - mean) / sd * wv
    return out


def dissimilarity_index(train_x: ArrayLike, test_x: ArrayLike, importance: ArrayLike) -> FloatArray:
    """DI of each test site (protocol 3.6.6; Meyer and Pebesma 2021)."""
    mean, sd, wv = _standardize(train_x, importance)
    tr = _scaled(train_x, mean, sd, wv)
    te = _scaled(test_x, mean, sd, wv)
    dbar = float(pdist(tr).mean())
    out: FloatArray = cdist(te, tr).min(axis=1) / dbar
    return out


def training_cv_di(train_x: ArrayLike, importance: ArrayLike, folds: ArrayLike) -> FloatArray:
    """Cross-validated DI of the training sites: nearest training site in another fold."""
    mean, sd, wv = _standardize(train_x, importance)
    tr = _scaled(train_x, mean, sd, wv)
    f = np.asarray(folds)
    if f.shape != (tr.shape[0],) or np.unique(f).size < 2:
        raise ValueError("folds must label every training site with at least two folds")
    dist = cdist(tr, tr)
    dist[f[:, None] == f[None, :]] = np.inf
    out: FloatArray = dist.min(axis=1) / float(pdist(tr).mean())
    return out


def aoa_threshold(train_di: ArrayLike, rules: AuditRulesParams) -> float:
    """AOA threshold ``Q3 + 1.5 IQR`` of the training cross-validated DI (protocol 3.6.6)."""
    d = np.asarray(train_di, dtype=np.float64)
    if d.ndim != 1 or d.size == 0 or not bool(np.isfinite(d).all()):
        raise ValueError("train_di must be a non-empty finite 1-D array")
    q1, q3 = np.percentile(d, [25.0, 75.0])
    return float(q3 + rules.aoa_iqr_mult * (q3 - q1))


@dataclass(frozen=True)
class ImportanceMember:
    """One M2-XGB member for the AOA importance (docs/3 13.9).

    ``predict`` maps a predictor matrix to log d-bar predictions (the member's d-bar booster);
    ``x`` and ``y_log`` are the member's ``es_val`` rows; ``seed`` is its ADL-088 member seed.
    """

    predict: Callable[[FloatArray], FloatArray]
    x: FloatArray
    y_log: FloatArray
    seed: int


def _log_rmse(pred: FloatArray, y: FloatArray) -> float:
    p = np.asarray(pred, dtype=np.float64)
    if p.shape != y.shape or not bool(np.isfinite(p).all()):
        raise ValueError("predict must return finite values, one per row")
    return math.sqrt(float(np.mean((p - y) ** 2)))


def permutation_importance(
    members: Sequence[ImportanceMember], permutations: int = AOA_PERMUTATIONS
) -> FloatArray:
    """AOA importance per feature: mean d-bar log-RMSE increase under permutation (3.6.6).

    docs/3 13.9: for member ``m`` and feature ``f`` the generator is
    ``default_rng(SeedSequence([seed_m, f]))`` and draws ``permutations`` row permutations in
    turn; feature ``f`` of the ``es_val`` rows is replaced by its permuted copy. The increase over
    the unpermuted log-RMSE is averaged over permutations, then over members; negative averages
    are set to 0.
    """
    if not members:
        raise ValueError("need at least one member")
    if permutations < 1:
        raise ValueError("permutations must be >= 1")
    n_feat = -1
    per_member: list[FloatArray] = []
    for mem in members:
        x = np.asarray(mem.x, dtype=np.float64)
        y = np.asarray(mem.y_log, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] < 2 or y.shape != (x.shape[0],):
            raise ValueError("member x must be (N >= 2, F) and y_log (N,)")
        if not bool(np.isfinite(y).all()):
            raise ValueError("y_log contains non-finite values")
        if n_feat not in (-1, x.shape[1]):
            raise ValueError("members disagree on the number of features")
        n_feat = x.shape[1]
        base = _log_rmse(mem.predict(x), y)
        inc = np.zeros(n_feat)
        for f in range(n_feat):
            rng = np.random.default_rng(np.random.SeedSequence([int(mem.seed), f]))
            for _ in range(permutations):
                xp = x.copy()
                xp[:, f] = x[rng.permutation(x.shape[0]), f]
                inc[f] += _log_rmse(mem.predict(xp), y) - base
        per_member.append(inc / permutations)
    out: FloatArray = np.maximum(np.mean(np.stack(per_member), axis=0), 0.0)
    return out

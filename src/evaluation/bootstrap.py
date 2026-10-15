"""Paired cluster bootstrap stratified by division; percentile and Bonferroni intervals.

Sources: proofs section 9.5 (paired cluster bootstrap: draw ``|J|`` sites with replacement, keep
all of each drawn site's observations, compute the statistic for both models on the same draw;
percentile interval; H3 two-sample bootstrap); protocol section 4.4 (``B = 10,000``,
``alpha_b = 0.05``, stratified by division, seed 20261008; S-ID statistic averaged over the
three repeats in each draw; S-IND clusters are rivers) and section 4.5 (S2 Bonferroni level
``1 - 0.05/4``, O-7). ``B``, ``alpha_b`` and the seed come from ``params.yaml`` ->
``evaluation.bootstrap`` (:class:`src.config.BootstrapParams`).

Design (P7-D1 (a)):
- Clusters are generic labels (sites, or rivers for S-IND). With ``strata`` given, each stratum
  resamples its own clusters with replacement, ``n_s`` draws for ``n_s`` clusters; every cluster
  must lie in exactly one stratum.
- A statistic receives ``(idx, groups)``: the observation indices of the draw and, per index, the
  position of the drawn cluster. A cluster drawn twice therefore counts as two clusters in the
  site-weighted statistics of proofs 9.5 (:func:`site_weighted_rmse`, :func:`site_weighted_bias`).
- The generator is ``numpy.random.default_rng(seed)`` (PCG64): the same inputs and seed give
  bit-identical replicates. The percentile interval uses ``numpy.quantile`` (linear method).

Torch-free and xgboost-free. Owning phase: 7 (P7-D1 (a)).

Phase 8 addition (Batch C3c, docs/3 13.13): :func:`bootstrap_matrix` and
:func:`two_sample_bootstrap_matrix` evaluate a vector of statistics on each draw. They consume the
generator exactly as :func:`paired_cluster_bootstrap` and :func:`two_sample_cluster_bootstrap`, so
column ``k`` equals the scalar bootstrap of statistic ``k`` (verified by test).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.config import BootstrapParams, S2Params

__all__ = [
    "ClusterDesign",
    "average_over_repeats",
    "bonferroni_level",
    "bootstrap_matrix",
    "cluster_design",
    "paired_cluster_bootstrap",
    "percentile_ci",
    "site_weighted_bias",
    "site_weighted_rmse",
    "two_sample_bootstrap_matrix",
    "two_sample_cluster_bootstrap",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Stat = Callable[[IntArray, IntArray], float]
Stat2 = Callable[[IntArray, IntArray, IntArray, IntArray], float]
StatVec = Callable[[IntArray, IntArray], ArrayLike]
StatVec2 = Callable[[IntArray, IntArray, IntArray, IntArray], ArrayLike]


@dataclass(frozen=True)
class ClusterDesign:
    """Observation indices per cluster and cluster indices per stratum."""

    members: tuple[IntArray, ...]
    strata: tuple[IntArray, ...]

    @property
    def n_clusters(self) -> int:
        """Number of clusters."""
        return len(self.members)


def cluster_design(clusters: ArrayLike, strata: ArrayLike | None = None) -> ClusterDesign:
    """Group observations by cluster label and clusters by stratum label.

    Raises if the inputs are empty or misaligned, or if a cluster spans two strata.
    """
    c = np.asarray(clusters)
    if c.ndim != 1 or c.size == 0:
        raise ValueError("clusters must be a non-empty 1-D array")
    labels, inverse = np.unique(c, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(labels.size + 1))
    members = tuple(order[bounds[k] : bounds[k + 1]].astype(np.int64) for k in range(labels.size))
    if strata is None:
        return ClusterDesign(members=members, strata=(np.arange(labels.size, dtype=np.int64),))
    s = np.asarray(strata)
    if s.shape != c.shape:
        raise ValueError(f"strata shape {s.shape} must equal clusters shape {c.shape}")
    s_labels, s_inv = np.unique(s, return_inverse=True)
    cluster_stratum = np.empty(labels.size, dtype=np.int64)
    for k, idx in enumerate(members):
        values = np.unique(s_inv[idx])
        if values.size != 1:
            raise ValueError(f"cluster {labels[k]!r} spans {values.size} strata")
        cluster_stratum[k] = values[0]
    by_stratum = tuple(
        np.flatnonzero(cluster_stratum == g).astype(np.int64) for g in range(s_labels.size)
    )
    return ClusterDesign(members=members, strata=by_stratum)


def _draw(design: ClusterDesign, rng: np.random.Generator) -> tuple[IntArray, IntArray]:
    chosen = np.concatenate(
        [pool[rng.integers(0, pool.size, size=pool.size)] for pool in design.strata]
    )
    parts = [design.members[k] for k in chosen]
    idx = np.concatenate(parts)
    groups = np.repeat(np.arange(len(parts), dtype=np.int64), [p.size for p in parts])
    return idx, groups


def _finite(value: float, b: int) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"bootstrap statistic is non-finite at replicate {b}")
    return v


def paired_cluster_bootstrap(
    stat: Stat,
    clusters: ArrayLike,
    strata: ArrayLike | None,
    cfg: BootstrapParams,
    *,
    n_boot: int | None = None,
) -> FloatArray:
    """Return ``B`` replicates of ``stat`` under the paired (stratified) cluster bootstrap.

    ``stat(idx, groups)`` must evaluate **both** models on the same draw (e.g. a Delta RMSE), so
    the pairing is the caller's statistic. ``B`` and the seed come from ``cfg`` (protocol 4.4);
    ``n_boot`` overrides ``B`` (synthetic tests only).
    """
    design = cluster_design(clusters, strata)
    b_total = cfg.b if n_boot is None else n_boot
    if b_total < 1:
        raise ValueError("n_boot must be >= 1")
    rng = np.random.default_rng(cfg.seed)
    out = np.empty(b_total, dtype=np.float64)
    for b in range(b_total):
        idx, groups = _draw(design, rng)
        out[b] = _finite(stat(idx, groups), b)
    return out


def two_sample_cluster_bootstrap(
    stat: Stat2,
    clusters_a: ArrayLike,
    strata_a: ArrayLike | None,
    clusters_b: ArrayLike,
    strata_b: ArrayLike | None,
    cfg: BootstrapParams,
    *,
    n_boot: int | None = None,
) -> FloatArray:
    """Two-sample bootstrap for H3 (proofs 9.5): resample sets A and B independently.

    ``stat(idx_a, groups_a, idx_b, groups_b)`` is paired across models within each set.
    """
    design_a = cluster_design(clusters_a, strata_a)
    design_b = cluster_design(clusters_b, strata_b)
    b_total = cfg.b if n_boot is None else n_boot
    if b_total < 1:
        raise ValueError("n_boot must be >= 1")
    rng = np.random.default_rng(cfg.seed)
    out = np.empty(b_total, dtype=np.float64)
    for b in range(b_total):
        idx_a, g_a = _draw(design_a, rng)
        idx_b, g_b = _draw(design_b, rng)
        out[b] = _finite(stat(idx_a, g_a, idx_b, g_b), b)
    return out


def _finite_row(value: ArrayLike, k: int | None, b: int) -> FloatArray:
    v = np.asarray(value, dtype=np.float64)
    if v.ndim != 1 or v.size == 0 or (k is not None and v.size != k):
        raise ValueError(f"vector statistic has shape {v.shape} at replicate {b}")
    if not bool(np.isfinite(v).all()):
        raise ValueError(f"bootstrap statistic is non-finite at replicate {b}")
    return v


def bootstrap_matrix(
    stat: StatVec,
    clusters: ArrayLike,
    strata: ArrayLike | None,
    cfg: BootstrapParams,
    *,
    n_boot: int | None = None,
) -> FloatArray:
    """``(B, K)`` replicates of a vector statistic under the paired (stratified) bootstrap.

    The draws are those of :func:`paired_cluster_bootstrap` with the same inputs and seed, so
    column ``k`` equals the scalar bootstrap of component ``k``.
    """
    design = cluster_design(clusters, strata)
    b_total = cfg.b if n_boot is None else n_boot
    if b_total < 1:
        raise ValueError("n_boot must be >= 1")
    rng = np.random.default_rng(cfg.seed)
    rows: list[FloatArray] = []
    for b in range(b_total):
        idx, groups = _draw(design, rng)
        rows.append(_finite_row(stat(idx, groups), rows[0].size if rows else None, b))
    return np.vstack(rows)


def two_sample_bootstrap_matrix(
    stat: StatVec2,
    clusters_a: ArrayLike,
    strata_a: ArrayLike | None,
    clusters_b: ArrayLike,
    strata_b: ArrayLike | None,
    cfg: BootstrapParams,
    *,
    n_boot: int | None = None,
) -> FloatArray:
    """``(B, K)`` replicates of a vector statistic under the H3 two-sample bootstrap.

    The draws are those of :func:`two_sample_cluster_bootstrap` (set A first in each replicate).
    """
    design_a = cluster_design(clusters_a, strata_a)
    design_b = cluster_design(clusters_b, strata_b)
    b_total = cfg.b if n_boot is None else n_boot
    if b_total < 1:
        raise ValueError("n_boot must be >= 1")
    rng = np.random.default_rng(cfg.seed)
    rows: list[FloatArray] = []
    for b in range(b_total):
        idx_a, g_a = _draw(design_a, rng)
        idx_b, g_b = _draw(design_b, rng)
        value = stat(idx_a, g_a, idx_b, g_b)
        rows.append(_finite_row(value, rows[0].size if rows else None, b))
    return np.vstack(rows)


def percentile_ci(stats: ArrayLike, level: float) -> tuple[float, float]:
    """Percentile interval ``[q_{(1-level)/2}, q_{1-(1-level)/2}]`` (proofs 9.5)."""
    s = np.asarray(stats, dtype=np.float64)
    if s.ndim != 1 or s.size == 0 or not bool(np.isfinite(s).all()):
        raise ValueError("stats must be a non-empty finite 1-D array")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    half = (1.0 - level) / 2.0
    lo, hi = np.quantile(s, [half, 1.0 - half])
    return float(lo), float(hi)


def bonferroni_level(cfg: S2Params) -> float:
    """Simultaneous level ``1 - alpha / n_contrasts`` (protocol 4.5, O-7; 0.9875)."""
    return 1.0 - cfg.alpha / cfg.n_contrasts


def _site_means(values: FloatArray, groups: ArrayLike) -> FloatArray:
    g = np.asarray(groups)
    if values.ndim != 1 or g.shape != values.shape or values.size == 0:
        raise ValueError("values and groups must be equal non-empty 1-D arrays")
    if not bool(np.isfinite(values).all()):
        raise ValueError("non-finite errors")
    _, inv, counts = np.unique(g, return_inverse=True, return_counts=True)
    out: FloatArray = np.bincount(inv, weights=values) / counts
    return out


def site_weighted_rmse(err: ArrayLike, groups: ArrayLike) -> float:
    """``[(1/|J|) sum_j (1/|I_j|) sum_{i in I_j} e_i^2]^(1/2)`` (proofs 9.5)."""
    e = np.asarray(err, dtype=np.float64)
    return math.sqrt(float(np.mean(_site_means(e * e, groups))))


def site_weighted_bias(err: ArrayLike, groups: ArrayLike) -> float:
    """``(1/|J|) sum_j (1/|I_j|) sum_{i in I_j} e_i`` (proofs 9.5)."""
    e = np.asarray(err, dtype=np.float64)
    return float(np.mean(_site_means(e, groups)))


def average_over_repeats(stats: Sequence[Stat]) -> Stat:
    """Average a per-repeat statistic over the S-ID repeats in each draw (protocol 4.4)."""
    if not stats:
        raise ValueError("need at least one repeat")
    fns = tuple(stats)

    def averaged(idx: IntArray, groups: IntArray) -> float:
        return float(np.mean([f(idx, groups) for f in fns]))

    return averaged

"""Stage split: site-COMID groups and the frozen fold-assignment table (protocol 3.1-3.3).

Sources: protocol section 3.1 (groups = connected components of the site-COMID graph;
``group_id`` = first 16 hex characters of SHA-256 over the sorted, comma-joined member
``site_no``), section 3.2 (split families), section 3.3 (fold table), section 3.4 (grouped inner
CV, member-specific grouped early-stopping split), section 3.5 (L1, L2, L5), P5-D6 (folds are
built **once over the union** of the site-table variants), P5-D7 (S-RND in a separate table).

This is the **only** module that creates folds. Every other module reads the tables written here.

Outputs
-------
``data/processed/splits/fold_assignment.parquet``
    Long table, one row per (``family``, ``repeat``, ``outer_fold``, ``site_no``). Every block
    (``family``, ``repeat``, ``outer_fold``) lists the whole site set U (common support); the
    block's outer test set is ``is_test``. ``inner_fold`` (0 .. inner_k-1) and ``es_member_m``
    (m = 1 .. ensemble_m) describe the outer **training** set of that block; for test rows they
    are -1 and ``False``. ``lro_division`` is the (merged) division of the S-LRO block.
``data/processed/splits/srnd_assignment.parquet``
    S-RND (secondary, never used for a hypothesis): row-level random 75:25 split of the
    ``hydroswot_init`` measurements of the sites in U, keyed by (``site_no``, ``q_meas_no``).
``reports/split_summary.json``
    Counts per block, strata, division merges, tail cutpoints and the table SHA-256 values.

Determinism: inputs are sorted before every random draw; random draws use
``numpy.random.SeedSequence`` entropy built from the protocol seeds and the block coordinates;
outputs are written by :func:`src.data.io.write_parquet` (stable sort, no wall-clock metadata),
so a re-run gives byte-identical files.

Owning phase: 5.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src.config import Settings, load_settings
from src.data.io import read_parquet, sha256_file, unit_manifest_sha256, write_json, write_parquet
from src.data.schemas import FoldAssignment, SiteTable, SrndAssignment

__all__ = [
    "FAMILIES",
    "SRND_TEST_FRAC",
    "BlockSpec",
    "assert_fold_table",
    "build_fold_table",
    "build_srnd",
    "group_ids",
    "group_table",
    "main",
    "merge_divisions",
]

LOG = logging.getLogger(__name__)

#: Families stored in ``fold_assignment`` (S-RND has its own table, P5-D7).
FAMILIES: Final[tuple[str, ...]] = ("S-ID", "S-LRO", "S-TAIL-DA", "S-TAIL-SO", "S-IND")
#: Protocol 3.2: S-RND is a row-level random 75:25 split.
SRND_TEST_FRAC: Final[float] = 0.25
#: Fill value of ``inner_fold`` for outer-test rows.
NO_FOLD: Final[int] = -1
_SITE_COLUMNS: Final[tuple[str, ...]] = ("site_no", "comid", "division", "DA_km2", "stream_order")


# ----------------------------------------------------------------------------- groups (3.1)
def group_ids(site_no: pd.Series, comid: pd.Series) -> pd.Series:
    """Return ``group_id`` per site: connected components of the bipartite site-COMID graph.

    Parameters
    ----------
    site_no, comid
        One entry per final link (a site may appear more than once if it had several links).

    Returns
    -------
    pandas.Series
        Indexed by unique ``site_no`` (sorted), value = 16-hex ``group_id``.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    sites = np.sort(site_no.astype(str).unique())
    comids = np.sort(comid.astype("int64").unique())
    i = np.searchsorted(sites, site_no.astype(str).to_numpy())
    j = len(sites) + np.searchsorted(comids, comid.astype("int64").to_numpy())
    n = len(sites) + len(comids)
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    members = pd.Series(sites, index=labels[: len(sites)]).groupby(level=0).agg(list)
    gid = {
        label: hashlib.sha256(",".join(sorted(m)).encode("utf-8")).hexdigest()[:16]
        for label, m in members.items()
    }
    return pd.Series([gid[k] for k in labels[: len(sites)]], index=sites, name="group_id")


def group_table(sites: pd.DataFrame) -> pd.DataFrame:
    """One row per group: first-listed site, its division, max ``DA_km2``, stream order.

    The division of a group is that of its first-listed (smallest) ``site_no`` (protocol 3.2,
    S-LRO). ``DA_km2`` is the group maximum (S-ID strata). Drainage-area decile and stream order
    must be the same for every member (they are reach attributes); otherwise a ``ValueError`` is
    raised, because a tail split would then cut a group.
    """
    ordered = sites.sort_values("site_no", kind="mergesort")
    grouped = ordered.groupby("group_id", sort=True)
    table = pd.DataFrame(
        {
            "first_site": grouped["site_no"].first(),
            "n_sites": grouped.size(),
            "division": grouped["division"].first(),
            "DA_max": grouped["DA_km2"].max(),
            "stream_order": grouped["stream_order"].max(),
            "tail_da": grouped["tail_da"].any(),
            "tail_so": grouped["tail_so"].any(),
        }
    )
    mixed = (grouped["tail_da"].nunique() > 1) | (grouped["tail_so"].nunique() > 1)
    if mixed.any():
        raise ValueError(f"tail membership differs inside groups: {list(mixed[mixed].index)}")
    return table


# ----------------------------------------------------------------------------- S-LRO merges
def merge_divisions(
    counts: Mapping[str, int],
    min_groups: int,
    boundary_km: Callable[[], Mapping[tuple[str, str], float]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Merge divisions with fewer than ``min_groups`` groups (protocol 3.2, S-LRO).

    The smallest division (ties: alphabetical) is merged into the neighbouring (merged) division
    with which it shares the longest boundary; this repeats until every division has at least
    ``min_groups`` groups. ``boundary_km`` is called only if a merge is needed and returns the
    shared boundary length of every pair of original divisions (keys sorted alphabetically).

    Returns
    -------
    tuple
        Mapping original division -> merged division name (the absorbing division keeps its
        name), and the list of merges in order.
    """
    owner = {d: d for d in counts}
    size = dict(counts)
    merges: list[dict[str, Any]] = []
    lengths: Mapping[tuple[str, str], float] | None = None
    while len(size) > 1 and min(size.values()) < min_groups:
        small = min(size, key=lambda d: (size[d], d))
        if lengths is None:
            lengths = boundary_km()
        shared: dict[str, float] = {}
        for (a, b), km in lengths.items():
            oa, ob = owner.get(a), owner.get(b)
            if oa is None or ob is None or oa == ob or small not in (oa, ob):
                continue
            other = ob if oa == small else oa
            shared[other] = shared.get(other, 0.0) + km
        if not shared or max(shared.values()) <= 0:
            raise ValueError(f"division {small!r} has no neighbour to merge into")
        into = max(sorted(shared), key=shared.__getitem__)  # ties: alphabetical first
        merges.append(
            {"from": small, "into": into, "groups": size[small], "shared_km": shared[into]}
        )
        size[into] += size.pop(small)
        for d, o in owner.items():
            if o == small:
                owner[d] = into
    return owner, merges


def _boundary_lengths(physio_dir: Path) -> dict[tuple[str, str], float]:
    """Shared boundary length (km, EPSG:5070) between every pair of F&J divisions."""
    from src.data.nhdplus import CRS_ALBERS
    from src.data.qc import load_divisions

    polys = load_divisions(physio_dir).dropna(subset=["DIVISION"])
    dissolved = polys.dissolve(by="DIVISION").to_crs(CRS_ALBERS)
    names = sorted(dissolved.index)
    out: dict[tuple[str, str], float] = {}
    for k, a in enumerate(names):
        for b in names[k + 1 :]:
            line = dissolved.geometry[a].boundary.intersection(dissolved.geometry[b].boundary)
            out[(a, b)] = float(line.length) / 1000.0
    return out


# ----------------------------------------------------------------------------- blocks
@dataclass(frozen=True)
class BlockSpec:
    """One outer split: the test groups of (``family``, ``repeat``, ``outer_fold``)."""

    family: str
    repeat: int
    outer_fold: int
    seed: int
    test_groups: frozenset[str]
    lro_division: str | None = None


def _quantile_labels(values: pd.Series, q: int) -> np.ndarray:
    """Equal-count quantile labels 0 .. q-1 (covariate only; L5)."""
    return np.asarray(pd.qcut(values, q, labels=False, duplicates="raise"), dtype="int64")


def _sgkf(groups: pd.DataFrame, k: int, seed: int) -> np.ndarray:
    """Fold label per group from ``StratifiedGroupKFold`` with DA-quintile strata."""
    from sklearn.model_selection import StratifiedGroupKFold

    strata = _quantile_labels(groups["DA_max"], 5)
    folds = np.full(len(groups), NO_FOLD, dtype="int64")
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    dummy = np.zeros((len(groups), 1))
    for f, (_, test) in enumerate(cv.split(dummy, strata, groups.index.to_numpy())):
        folds[test] = f
    return folds


def _family_index(family: str) -> int:
    return FAMILIES.index(family)


def build_blocks(groups: pd.DataFrame, settings: Settings) -> tuple[list[BlockSpec], dict[str, Any]]:
    """Return every outer block of the five site-level families and their metadata."""
    sp, seeds = settings.protocol.splits, settings.protocol.seeds
    blocks: list[BlockSpec] = []
    meta: dict[str, Any] = {}
    gids = groups.index.to_numpy()

    # S-ID: StratifiedGroupKFold, K x R, strata = DA quintile of the group maximum.
    for r, seed in enumerate(seeds.sid_repeats[: sp.sid_r]):
        folds = _sgkf(groups, sp.sid_k, seed)
        for f in range(sp.sid_k):
            blocks.append(BlockSpec("S-ID", r, f, seed, frozenset(gids[folds == f])))
    meta["S-ID"] = {
        "strata_quintile_groups": np.bincount(_quantile_labels(groups["DA_max"], 5)).tolist()
    }

    # S-LRO: leave one (merged) division out.
    raw_counts = {
        str(k): int(v) for k, v in groups["division"].value_counts().sort_index().items()
    }
    owner, merges = merge_divisions(raw_counts, sp.lro_min_groups, settings_boundary(settings))
    merged = groups["division"].map(owner)
    for f, div in enumerate(sorted(merged.unique())):
        test = frozenset(gids[(merged == div).to_numpy()])
        blocks.append(BlockSpec("S-LRO", 0, f, seeds.global_seed, test, lro_division=div))
    meta["S-LRO"] = {"groups_per_division": raw_counts, "merges": merges}

    # S-TAIL-DA, S-TAIL-SO (deterministic; tail flags computed on sites, see run()).
    blocks.append(
        BlockSpec(
            "S-TAIL-DA", 0, 0, seeds.global_seed, frozenset(gids[groups["tail_da"].to_numpy()])
        )
    )
    blocks.append(
        BlockSpec(
            "S-TAIL-SO", 0, 0, seeds.global_seed, frozenset(gids[groups["tail_so"].to_numpy()])
        )
    )
    # S-IND: train on all sites (the test set is the eHydro set, ind_prepare).
    blocks.append(BlockSpec("S-IND", 0, 0, seeds.global_seed, frozenset()))
    return blocks, meta


def settings_boundary(settings: Settings) -> Callable[[], Mapping[tuple[str, str], float]]:
    """Lazy boundary-length provider for :func:`merge_divisions`."""
    return lambda: _boundary_lengths(settings.runtime.raw_dir / "physiography")


def _block_rows(
    block: BlockSpec, groups: pd.DataFrame, sites: pd.DataFrame, settings: Settings
) -> pd.DataFrame:
    """Rows of one block: outer test flag, inner folds and ES membership of the training set."""
    sp = settings.protocol.splits
    n_members = settings.protocol.uq.ensemble_m
    is_test_g = groups.index.isin(list(block.test_groups))
    train = groups.loc[~is_test_g]
    inner = pd.Series(NO_FOLD, index=groups.index, dtype="int64")
    inner.loc[train.index] = _sgkf(train, sp.inner_k, block.seed)
    es = pd.DataFrame(False, index=groups.index, columns=[f"es_member_{m}" for m in range(1, n_members + 1)])
    n_es = math.ceil(sp.es_frac * len(train))
    for m in range(1, n_members + 1):
        entropy = [block.seed, _family_index(block.family), block.repeat, block.outer_fold, m]
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        chosen = rng.permutation(train.index.to_numpy())[:n_es]
        es.loc[chosen, f"es_member_{m}"] = True
    per_group = pd.concat(
        [pd.Series(is_test_g, index=groups.index, name="is_test"), inner.rename("inner_fold"), es],
        axis=1,
    )
    rows = sites[["group_id", "site_no"]].merge(per_group, left_on="group_id", right_index=True)
    rows.insert(0, "family", block.family)
    rows.insert(1, "repeat", block.repeat)
    rows.insert(2, "outer_fold", block.outer_fold)
    rows["lro_division"] = block.lro_division
    rows["seed"] = block.seed
    return rows


def build_fold_table(
    sites: pd.DataFrame, settings: Settings, site_table_sha256: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the long fold-assignment table for the five site-level families.

    Parameters
    ----------
    sites
        One row per site in U: ``site_no``, ``comid``, ``division``, ``DA_km2``,
        ``stream_order``, ``group_id``, ``tail_da``, ``tail_so``.
    settings
        Validated settings (``seeds``, ``splits``, ``model.training.ensemble_m``).
    site_table_sha256
        SHA-256 of the input ``site_table.parquet`` (copied into every row).
    """
    groups = group_table(sites)
    blocks, meta = build_blocks(groups, settings)
    table = pd.concat([_block_rows(b, groups, sites, settings) for b in blocks], ignore_index=True)
    table["site_table_sha256"] = site_table_sha256
    table = table.astype(
        {
            "family": "str",
            "group_id": "str",
            "site_no": "str",
            "repeat": "int8",
            "outer_fold": "int8",
            "inner_fold": "int8",
            "seed": "int64",
            "lro_division": "str",
            "site_table_sha256": "str",
        }
    )
    table = table.sort_values(
        ["family", "repeat", "outer_fold", "site_no"], kind="mergesort"
    ).reset_index(drop=True)
    table = FoldAssignment.validate(table)
    assert_fold_table(table, sites)
    meta["blocks"] = _block_summary(table)
    meta["groups"] = {
        "n_groups": len(groups),
        "n_sites": len(sites),
        "group_size_counts": groups["n_sites"].value_counts().sort_index().to_dict(),
    }
    return table, meta


def assert_fold_table(table: pd.DataFrame, sites: pd.DataFrame) -> None:
    """L1, L2, common support and S-ID partition checks (raise ``AssertionError``)."""
    universe = set(sites["site_no"])
    comid = sites.set_index("site_no")["comid"]
    for key, block in table.groupby(["family", "repeat", "outer_fold"], sort=True):
        if set(block["site_no"]) != universe or block["site_no"].duplicated().any():
            raise AssertionError(f"{key}: block does not list the site set U exactly once")
        test, train = block.loc[block["is_test"]], block.loc[~block["is_test"]]
        if set(test["group_id"]) & set(train["group_id"]):
            raise AssertionError(f"{key}: L1 violated (group in train and test)")
        if set(comid.reindex(test["site_no"])) & set(comid.reindex(train["site_no"])):
            raise AssertionError(f"{key}: L2 violated (COMID in train and test)")
        if (test["inner_fold"] != NO_FOLD).any() or (train["inner_fold"] == NO_FOLD).any():
            raise AssertionError(f"{key}: inner folds must cover exactly the training rows")
        es_cols = [c for c in block.columns if c.startswith("es_member_")]
        if test[es_cols].to_numpy().any():
            raise AssertionError(f"{key}: early-stopping rows must be training rows")
    sid = table.loc[table["family"].eq("S-ID") & table["is_test"]]
    if (sid.groupby(["repeat", "site_no"]).size() != 1).any() or sid.groupby(
        "repeat"
    )["site_no"].nunique().ne(len(universe)).any():
        raise AssertionError("S-ID: every site must be tested exactly once per repeat")


def _block_summary(table: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for _, block in table.groupby(["family", "repeat", "outer_fold"]):
        test = block["is_test"]
        out.append(
            {
                "family": str(block["family"].iloc[0]),
                "repeat": int(block["repeat"].iloc[0]),
                "outer_fold": int(block["outer_fold"].iloc[0]),
                "lro_division": block["lro_division"].iloc[0]
                if block["lro_division"].notna().any()
                else None,
                "test_sites": int(test.sum()),
                "test_groups": int(block.loc[test, "group_id"].nunique()),
                "train_sites": int((~test).sum()),
                "train_groups": int(block.loc[~test, "group_id"].nunique()),
                "inner_fold_sites": block.loc[~test, "inner_fold"].value_counts().sort_index().to_dict(),
                "es_member_1_sites": int(block["es_member_1"].sum()),
            }
        )
    return out


# ----------------------------------------------------------------------------- S-RND
def build_srnd(init: pd.DataFrame, universe: set[str], seed: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Row-level random 75:25 split of the ``hydroswot_init`` rows of the sites in U (P5-D7)."""
    rows = init.loc[init["site_no"].isin(universe), ["site_no", "q_meas_no", "row_id"]]
    rows = rows.sort_values(["site_no", "q_meas_no", "row_id"], kind="mergesort").reset_index(drop=True)
    n_test = round(SRND_TEST_FRAC * len(rows))
    rng = np.random.default_rng(np.random.SeedSequence([seed, len(FAMILIES)]))
    is_test = np.zeros(len(rows), dtype=bool)
    is_test[rng.permutation(len(rows))[:n_test]] = True
    out = rows.assign(is_test=is_test, seed=np.int64(seed)).astype(
        {"site_no": "str", "q_meas_no": "str", "row_id": "int64", "seed": "int64"}
    )
    out = SrndAssignment.validate(out)
    counts = {
        "rows_init": len(init),
        "rows_outside_U": int((~init["site_no"].isin(universe)).sum()),
        "rows": len(out),
        "test_rows": int(is_test.sum()),
        "sites_in_both": int(
            len(set(out.loc[out["is_test"], "site_no"]) & set(out.loc[~out["is_test"], "site_no"]))
        ),
    }
    return out, counts


# ----------------------------------------------------------------------------- stage
def site_universe(site_table: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One row per site of U with ``group_id`` and the tail flags (cutpoints from covariates)."""
    sp = settings.protocol.splits
    site_table = SiteTable.validate(site_table)
    cols = list(_SITE_COLUMNS)
    per_site = site_table[cols].drop_duplicates()
    if per_site["site_no"].duplicated().any():
        raise ValueError("site attributes differ between variants of one site")
    sites = per_site.sort_values("site_no", kind="mergesort").reset_index(drop=True)
    sites["group_id"] = group_ids(sites["site_no"], sites["comid"]).reindex(sites["site_no"]).to_numpy()
    decile = _quantile_labels(sites["DA_km2"], 10) + 1
    cuts = np.quantile(sites["DA_km2"].to_numpy(), np.linspace(0.1, 0.9, 9))
    sites["tail_da"] = np.isin(decile, list(sp.tail_deciles))
    sites["tail_so"] = sites["stream_order"].to_numpy() >= sp.tail_order_min
    meta = {
        "S-TAIL-DA": {"decile_cutpoints_km2": cuts.tolist(), "test_deciles": list(sp.tail_deciles)},
        "S-TAIL-SO": {"test_order_min": sp.tail_order_min},
    }
    return sites, meta


def run(settings: Settings) -> None:
    """Write ``fold_assignment.parquet``, ``srnd_assignment.parquet`` and the summary."""
    rt = settings.runtime
    p_site = rt.processed_dir / "site_table.parquet"
    p_init = rt.processed_dir / "qc" / "hydroswot_init.parquet"
    out_dir = rt.processed_dir / "splits"
    site_sha = sha256_file(p_site)
    sites, tail_meta = site_universe(read_parquet(p_site), settings)
    table, meta = build_fold_table(sites, settings, site_sha)
    srnd, srnd_counts = build_srnd(
        read_parquet(p_init), set(sites["site_no"]), settings.protocol.seeds.global_seed
    )
    inputs = {
        "site_table.parquet": site_sha,
        "qc/hydroswot_init.parquet": sha256_file(p_init),
        "physiography/MANIFEST.sha256": unit_manifest_sha256(rt.raw_dir / "physiography"),
    }
    p_fold, p_srnd = out_dir / "fold_assignment.parquet", out_dir / "srnd_assignment.parquet"
    write_parquet(
        table,
        p_fold,
        stage="split",
        params_sha256=settings.params_sha256,
        inputs=inputs,
        sort_by=["family", "repeat", "outer_fold", "site_no"],
    )
    write_parquet(
        srnd,
        p_srnd,
        stage="split",
        params_sha256=settings.params_sha256,
        inputs=inputs,
        sort_by=["site_no", "q_meas_no", "row_id"],
    )
    LOG.info("fold_assignment: %d rows; %d sites; %d groups", len(table), len(sites), meta["groups"]["n_groups"])
    write_json(
        rt.reports_dir / "split_summary.json",
        {
            "params_sha256": settings.params_sha256,
            "site_table_sha256": site_sha,
            "fold_assignment_sha256": sha256_file(p_fold),
            "srnd_assignment_sha256": sha256_file(p_srnd),
            "rows": len(table),
            **meta,
            **tail_meta,
            "S-RND": srnd_counts,
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run ``python -m src.data.split`` (the dvc.yaml ``split`` stage)."""
    argparse.ArgumentParser(description="Groups and frozen folds (protocol 3.1-3.3).").parse_args(
        argv
    )
    settings = load_settings()
    logging.basicConfig(
        level=settings.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    LOG.info("params_sha256=%s", settings.params_sha256)
    run(settings)


if __name__ == "__main__":
    main()

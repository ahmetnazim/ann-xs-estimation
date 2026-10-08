"""Unit tests for src/data/split.py and ind_prepare.py (Phase 5, Batch D). Synthetic only."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_settings
from src.data import split
from src.data.ind_prepare import dedupe_l8
from src.data.ind_prepare import main as ind_main
from src.data.io import sha256_file, write_parquet

pytestmark = pytest.mark.unit

SETTINGS = load_settings()
SHA = "0" * 64


def synthetic_site_table(n_sites: int = 360) -> pd.DataFrame:
    """Sites on ``n_sites - 30`` COMIDs (30 COMIDs carry two sites), three divisions."""
    rng = np.random.default_rng(7)
    site_no = [f"{10_000_000 + i:08d}" for i in range(n_sites)]
    comid = np.arange(n_sites) + 1000
    comid[1:60:2] = comid[0:60:2]  # 30 shared COMIDs -> 30 two-site groups
    da = np.exp(rng.normal(5.0, 2.0, n_sites))
    order = rng.integers(1, 10, n_sites).astype("int8")
    first = pd.Series(np.arange(n_sites)).groupby(comid).transform("min").to_numpy()
    da, order = da[first], order[first]  # reach attributes are per COMID
    division = np.array(["ALPHA", "BETA", "GAMMA"])[first % 3]
    base = pd.DataFrame(
        {
            "site_no": site_no,
            "comid": comid.astype("int64"),
            "division": division,
            "DA_km2": da,
            "stream_order": order,
            "Q_E": 1.0,
            "S": 1e-3,
        }
    )
    sa3 = base.assign(variant="sa3")
    primary = base.iloc[::2].assign(variant="primary")
    return pd.concat([primary, sa3], ignore_index=True)


@pytest.fixture(scope="module")
def built() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sites, _ = split.site_universe(synthetic_site_table(), SETTINGS)
    table, meta = split.build_fold_table(sites, SETTINGS, SHA)
    return sites, table, meta


# ----------------------------------------------------------------------------- groups
def test_group_ids_are_components_with_hashed_members() -> None:
    site_no = pd.Series(["00000003", "00000001", "00000002", "00000002", "00000004"])
    comid = pd.Series([10, 10, 11, 10, 99])  # site 2 links to 11 and 10 -> joins the group
    gid = split.group_ids(site_no, comid)
    expect = hashlib.sha256(b"00000001,00000002,00000003").hexdigest()[:16]
    assert gid.loc[["00000001", "00000002", "00000003"]].tolist() == [expect] * 3
    assert gid.loc["00000004"] == hashlib.sha256(b"00000004").hexdigest()[:16]


def test_tail_membership_must_not_cut_a_group() -> None:
    sites = pd.DataFrame(
        {
            "site_no": ["00000001", "00000002"],
            "group_id": ["g", "g"],
            "division": ["A", "A"],
            "DA_km2": [1.0, 2.0],
            "stream_order": [3, 3],
            "tail_da": [True, False],
            "tail_so": [False, False],
        }
    )
    with pytest.raises(ValueError, match="tail membership"):
        split.group_table(sites)


# ----------------------------------------------------------------------------- S-LRO merges
def test_merge_divisions_longest_boundary_and_order() -> None:
    lengths = {("A", "B"): 5.0, ("A", "D"): 1.0, ("B", "D"): 3.0, ("B", "C"): 9.0, ("C", "D"): 0}
    calls = []

    def boundary() -> dict[tuple[str, str], float]:
        calls.append(1)
        return lengths

    owner, merges = split.merge_divisions({"A": 100, "B": 20, "C": 60, "D": 10}, 50, boundary)
    assert [(m["from"], m["into"]) for m in merges] == [("D", "B"), ("B", "C")]
    assert owner == {"A": "A", "B": "C", "C": "C", "D": "C"}
    assert merges[1]["groups"] == 30
    assert len(calls) == 1


def test_merge_divisions_skips_boundaries_when_not_needed() -> None:
    def boundary() -> dict[tuple[str, str], float]:
        raise AssertionError("must not be called")

    owner, merges = split.merge_divisions({"A": 60, "B": 50}, 50, boundary)
    assert merges == []
    assert owner == {"A": "A", "B": "B"}


# ----------------------------------------------------------------------------- fold table
def test_fold_table_blocks_and_partitions(built: tuple) -> None:
    sites, table, meta = built
    blocks = table.groupby(["family", "repeat", "outer_fold"]).size()
    assert (blocks == len(sites)).all()
    per_family = blocks.groupby(level="family").size().to_dict()
    assert per_family == {"S-ID": 15, "S-IND": 1, "S-LRO": 3, "S-TAIL-DA": 1, "S-TAIL-SO": 1}
    assert not table.loc[table["family"].eq("S-IND"), "is_test"].any()
    lro = table.loc[table["family"].eq("S-LRO") & table["is_test"]]
    assert lro.groupby("outer_fold")["lro_division"].nunique().eq(1).all()
    assert meta["groups"]["group_size_counts"] == {1: 300, 2: 30}


def test_inner_and_early_stopping_cover_training_groups_only(built: tuple) -> None:
    _, table, _ = built
    block = table.loc[
        table["family"].eq("S-ID") & table["repeat"].eq(1) & table["outer_fold"].eq(2)
    ]
    train = block.loc[~block["is_test"]]
    assert set(train["inner_fold"]) == {0, 1, 2}
    n_groups = train["group_id"].nunique()
    for m in range(1, 11):
        es_groups = train.loc[train[f"es_member_{m}"], "group_id"]
        assert es_groups.nunique() == math.ceil(0.10 * n_groups)
        # grouped: a group is wholly in or out of the ES split
        assert not set(es_groups) & set(train.loc[~train[f"es_member_{m}"], "group_id"])
    assert not train["es_member_1"].equals(train["es_member_2"])


def test_tail_families_follow_covariate_cutpoints(built: tuple) -> None:
    sites, table, _ = built
    da = table.loc[table["family"].eq("S-TAIL-DA")].merge(sites, on="site_no")
    lo, hi = np.quantile(sites["DA_km2"], [0.1, 0.9])
    assert da.loc[da["is_test"], "DA_km2"].pipe(lambda s: ((s <= lo) | (s > hi)).all())
    so = table.loc[table["family"].eq("S-TAIL-SO")].merge(sites, on="site_no")
    assert (so["is_test"] == (so["stream_order"] >= 7)).all()


def test_assertions_catch_l1(built: tuple) -> None:
    sites, table, _ = built
    bad = table.copy()
    pair = sites.loc[sites["group_id"].duplicated(keep=False), "site_no"].iloc[0]
    row = bad.index[bad["family"].eq("S-TAIL-DA") & bad["site_no"].eq(pair)][0]
    bad.loc[row, "is_test"] = not bad.loc[row, "is_test"]
    bad.loc[row, "inner_fold"] = -1 if bad.loc[row, "is_test"] else 0
    for c in [c for c in bad.columns if c.startswith("es_member_")]:
        bad.loc[row, c] = False
    with pytest.raises(AssertionError, match="L1"):
        split.assert_fold_table(bad, sites)


def test_fold_table_is_byte_identical_on_rerun(built: tuple, tmp_path: Path) -> None:
    sites, table, _ = built
    again, _ = split.build_fold_table(sites, SETTINGS, SHA)
    pd.testing.assert_frame_equal(table, again)
    paths = [tmp_path / "a.parquet", tmp_path / "b.parquet"]
    for frame, path in zip((table, again), paths, strict=True):
        write_parquet(frame, path, stage="split", params_sha256=SHA, inputs={}, sort_by=None)
    assert sha256_file(paths[0]) == sha256_file(paths[1])


# ----------------------------------------------------------------------------- S-RND
def test_srnd_is_a_seeded_75_25_row_split_on_u() -> None:
    init = pd.DataFrame(
        {
            "site_no": [f"{i % 50:08d}" for i in range(400)],
            "q_meas_no": [str(i) for i in range(400)],
            "row_id": np.arange(400, dtype="int64"),
        }
    )
    universe = {f"{i:08d}" for i in range(40)}
    out, counts = split.build_srnd(init, universe, 20261008)
    assert len(out) == 320
    assert counts["rows_outside_U"] == 80
    assert out["is_test"].sum() == 80
    again, _ = split.build_srnd(init.sample(frac=1.0, random_state=1), universe, 20261008)
    pd.testing.assert_frame_equal(out, again)


# ----------------------------------------------------------------------------- ind_prepare (L8)
def test_dedupe_l8_drops_comid_overlap_and_near_locations() -> None:
    train = pd.DataFrame({"site_no": ["00000001"], "comid": [5], "x5070": [0.0], "y5070": [0.0]})
    locs = pd.DataFrame(
        {
            "loc_id": ["a", "a", "b", "c"],
            "comid": [7, 5, 8, 9],  # a: member COMID 5 is a training COMID
            "x5070": [50_000.0, 50_000.0, 900.0, 5_000.0],  # b within 1 km; c beyond
            "y5070": [0.0, 0.0, 0.0, 0.0],
        }
    )
    kept, counts = dedupe_l8(locs, train, 1000.0)
    assert kept["loc_id"].tolist() == ["c"]
    assert counts == {"locations": 3, "dropped_comid": 1, "dropped_distance_only": 1, "kept": 1}


def test_ind_prepare_stage_is_pending() -> None:
    with pytest.raises(NotImplementedError, match="P5-D8"):
        ind_main([])

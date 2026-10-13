"""Tests for the Phase 8 fit plan (src.models.plan) and run tracking (src.models.tracking).

Synthetic inputs only. The docs/3 13.6 count table is checked verbatim with the frozen
``params.yaml`` and the realised block structure (S-ID 3 x 5, S-LRO 8, S-TAIL-DA, S-TAIL-SO,
S-IND = 26 blocks); no site or fold table is read. Merkle known answers: the RFC 6962 test
vectors of transparency-dev/merkle ``testonly/constants.go`` (leaf inputs, node and root
hashes; https://github.com/transparency-dev/merkle).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from src.config import load_settings
from src.models import plan as pl
from src.models import tracking as tr
from src.models.registry import ModelSpec, fit_dir, write_json
from src.models.runner import baseline_dir

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()
PROTOCOL = SETTINGS.protocol
BLOCKS_26 = (
    [("S-ID", r, f) for r in range(3) for f in range(5)]
    + [("S-LRO", 0, f) for f in range(8)]
    + [("S-TAIL-DA", 0, 0), ("S-TAIL-SO", 0, 0), ("S-IND", 0, 0)]
)
HASHES = {k: f"{i:x}" * 64 for i, k in enumerate(pl.TABLE_KEYS, start=1)}

# transparency-dev/merkle testonly/constants.go (RFC 6962 hashing)
KAT_INPUTS = [
    "",
    "00",
    "10",
    "2021",
    "3031",
    "40414243",
    "5051525354555657",
    "606162636465666768696a6b6c6d6e6f",
]
KAT_LEAVES = [
    "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
    "96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7",
    "0298d122906dcfc10892cb53a73992fc5b9f493ea4c9badb27b791b4127a7fe7",
    "07506a85fd9dd2f120eb694f86011e5bb4662e5c415a62917033d4a9624487e7",
    "bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b",
    "4271a26be0d8a84f0bd54c8c302e7cb3a3b5d1fa6780a40bcce2873477dab658",
    "b08693ec2e721597130641e8211e7eedccb4c26413963eee6c1e2ed16ffb1a5f",
    "46f6ffadd3d06a09ff3c5860d2755c8b9819db7df44251788c7d8e3180de8eb1",
]
KAT_ROOTS = [
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
    "fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125",
    "aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77",
    "d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7",
    "4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4",
    "76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef",
    "ddb89be403809e325750d3d263cd78929c2942b7942a34b77e122c9594a74c8c",
    "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328",
]


@pytest.fixture(scope="module")
def full_plan() -> tuple[pl.PlannedFit, ...]:
    return pl.build_plan(PROTOCOL, BLOCKS_26)


# ----------------------------------------------------------------------------- counts
def test_plan_counts_equal_the_13_6_table(full_plan: tuple[pl.PlannedFit, ...]) -> None:
    counts = pl.count_groups(full_plan)
    assert counts == dict(pl.EXPECTED_COUNTS_13_6)
    assert pl.expected_counts(PROTOCOL, len(BLOCKS_26)) == dict(pl.EXPECTED_COUNTS_13_6)
    assert len(full_plan) == 10348 + 6728 + 208 + 78


def test_scope_rules_13_8(full_plan: tuple[pl.PlannedFit, ...]) -> None:
    sa1 = {f.model for f in full_plan if f.analysis == "SA1"}
    assert sa1 == {"M5"}  # docs/3 13.8: SA1 needs a Phi
    a4 = {f.model for f in full_plan if f.ablation == "A4"}
    assert a4 == {"M5", "M1", "M2-MLP", "M2-XGB"}  # Q5
    sa = {f.model for f in full_plan if f.analysis not in ("primary",)}
    assert sa == {"M5", "M2-MLP", "M1", "M2-XGB"}  # Q6
    srnd = [f for f in full_plan if f.family == "S-RND"]
    assert {f.model for f in srnd} == {"M1", "M2-XGB"}
    assert {f.stage for f in srnd} == {"train@xgb"}
    assert all(f.kind == "member" for f in srnd)
    # ablations and SAs reuse a selection: no inner fit or selection of their own
    assert not [
        f
        for f in full_plan
        if (f.ablation or f.analysis != "primary") and f.kind in ("inner", "select")
    ]


def test_stage_order_xgb_before_nn(full_plan: tuple[pl.PlannedFit, ...]) -> None:
    stages = [pl.STAGES.index(f.stage) for f in full_plan]
    assert stages == sorted(stages)
    nn = {f.stage for f in full_plan if f.model in ("M2-MLP", "M3", "M4", "M5", "M6-NN")}
    assert nn <= pl.NN_STAGES
    assert not {f.stage for f in full_plan if f.model in ("M1", "M2-XGB", "M6-XGB")} & pl.NN_STAGES


def test_plan_is_sorted_and_topological(full_plan: tuple[pl.PlannedFit, ...]) -> None:
    assert list(full_plan) == sorted(full_plan, key=pl.PlannedFit.sort_key)
    pos = {f.fit_id: i for i, f in enumerate(full_plan)}
    deps = pl.dependency_map(full_plan)
    assert all(pos[d] < pos[fid] for fid, ds in deps.items() for d in ds)
    srnd = next(f for f in full_plan if f.family == "S-RND" and f.model == "M1")
    assert len(deps[srnd.fit_id]) == 15 + 150  # every S-ID primary selection and member
    sa = next(f for f in full_plan if f.analysis == "SA4" and f.model == "M5")
    assert deps[sa.fit_id] == ("M5/primary/S-ID/r0/f0/select",)
    abl = next(f for f in full_plan if f.ablation == "A1")
    assert deps[abl.fit_id] == ("M5/primary/S-ID/r0/f0/select",)


def test_fit_ids_match_the_runner_layout(
    full_plan: tuple[pl.PlannedFit, ...], tmp_path: Path
) -> None:
    for f in full_plan:
        if f.kind == "outer":
            path = baseline_dir(tmp_path, f.model, f.family, f.repeat, f.fold)
        elif f.kind == "select":
            assert f.completion_path(tmp_path).parent == tmp_path / f.block_id
            continue
        else:
            spec = ModelSpec(f.model, f.ablation)
            path = fit_dir(
                tmp_path, spec, f.analysis, f.family, f.repeat, f.fold, f.kind, f.index, f.grid
            )
        assert path == tmp_path / f.fit_id


def test_plan_hash_and_run_id_are_deterministic(full_plan: tuple[pl.PlannedFit, ...]) -> None:
    doc = pl.plan_document(full_plan, SETTINGS.params_sha256, HASHES)
    sha = pl.plan_sha256(doc)
    assert sha == pl.plan_sha256(
        pl.plan_document(pl.build_plan(PROTOCOL, BLOCKS_26), SETTINGS.params_sha256, HASHES)
    )
    assert pl.run_id_for(sha) == "p8-" + sha[:12]
    other = dict(HASHES, fold_table="f" * 64)
    assert pl.plan_sha256(pl.plan_document(full_plan, SETTINGS.params_sha256, other)) != sha
    roundtrip = [
        pl.PlannedFit.from_json(json.loads(json.dumps(f.to_json()))) for f in full_plan[:50]
    ]
    assert tuple(roundtrip) == full_plan[:50]
    code = (
        "import sys; from src.config import load_settings; from src.models import plan as pl; "
        f"s = load_settings(); b = {BLOCKS_26!r}; h = {HASHES!r}; "
        "d = pl.plan_document(pl.build_plan(s.protocol, b), s.params_sha256, h); "
        "print(pl.plan_sha256(d)); "
        "print('torch' in sys.modules, 'xgboost' in sys.modules)"
    )
    for seed in ("1", "999"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
            env=env,
        ).stdout.split("\n")
        assert out[0] == sha
        assert out[1] == "False False"


def test_canonical_json_format() -> None:
    assert pl.canonical_json({"b": 1, "a": [None, "é"]}) == '{"a":[null,"é"],"b":1}'.encode()
    with pytest.raises(ValueError, match="not JSON compliant"):
        pl.canonical_json({"x": float("nan")})


def test_planned_fit_validation() -> None:
    with pytest.raises(ValueError, match="stage"):
        pl.PlannedFit("train@x", "M5", None, "primary", "S-ID", 0, 0, "member", 1, None)
    with pytest.raises(ValueError, match="grid"):
        pl.PlannedFit("train@nn_primary", "M5", None, "primary", "S-ID", 0, 0, "inner", 0, None)
    with pytest.raises(ValueError, match="grid"):
        pl.PlannedFit("train@nn_primary", "M5", None, "primary", "S-ID", 0, 0, "member", 1, 3)
    with pytest.raises(ValueError, match="hex"):
        pl.run_id_for("XYZ")
    with pytest.raises(ValueError, match="tables"):
        pl.plan_document([], "0" * 64, {"site_table": "0" * 64})


def test_blocks_from_folds_order_and_refusal() -> None:
    folds = pd.DataFrame(
        {
            "family": ["S-IND", "S-ID", "S-ID", "S-LRO", "S-ID"],
            "repeat": [0, 1, 0, 0, 0],
            "outer_fold": [0, 0, 1, 2, 1],
        }
    )
    assert pl.blocks_from_folds(folds) == [
        ("S-ID", 0, 1),
        ("S-ID", 1, 0),
        ("S-LRO", 0, 2),
        ("S-IND", 0, 0),
    ]
    with pytest.raises(ValueError, match="unknown fold families"):
        pl.blocks_from_folds(pd.DataFrame({"family": ["S-RND"], "repeat": [0], "outer_fold": [0]}))


def test_table_hashes(tmp_path: Path) -> None:
    paths = pl.table_paths(tmp_path)
    with pytest.raises(FileNotFoundError, match="0a/0b"):
        pl.table_hashes(paths)
    for i, p in enumerate(paths.values()):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(bytes([i]))
    hashes = pl.table_hashes(paths)
    assert hashes["site_table"] == hashlib.sha256(b"\x00").hexdigest()
    with pytest.raises(ValueError, match="exactly"):
        pl.table_hashes({"site_table": paths["site_table"]})


def test_dependency_map_refuses_incomplete_plans() -> None:
    member = pl.PlannedFit(
        "train@nn_primary", "M5", None, "primary", "S-ID", 0, 0, "member", 1, None
    )
    with pytest.raises(RuntimeError, match="dependencies"):
        pl.dependency_map([member])
    select = pl.PlannedFit(
        "train@nn_primary", "M5", None, "primary", "S-ID", 0, 0, "select", 0, None
    )
    with pytest.raises(RuntimeError, match="no inner fits"):
        pl.dependency_map([select])
    srnd = pl.PlannedFit("train@xgb", "M1", None, "primary", "S-RND", 0, 0, "member", 1, None)
    with pytest.raises(RuntimeError, match="dependencies"):
        pl.dependency_map([srnd])


# ----------------------------------------------------------------------------- Merkle
def test_rfc6962_known_answers() -> None:
    leaves = [tr.rfc6962_leaf_hash(bytes.fromhex(x)) for x in KAT_INPUTS]
    assert [h.hex() for h in leaves] == KAT_LEAVES
    for n, root in enumerate(KAT_ROOTS):
        assert tr.merkle_root(leaves[:n]) == root


def test_merkle_leaf_definition() -> None:
    sha = "ab" * 32
    want = hashlib.sha256(
        b"\x00" + b"M5/x/member_1" + b"\x00" + b"run.json" + b"\x00" + sha.encode()
    ).digest()
    assert tr.merkle_leaf("M5/x/member_1", "run.json", sha) == want
    with pytest.raises(ValueError, match="NUL"):
        tr.merkle_leaf("a\x00b", "run.json", sha)


# ----------------------------------------------------------------------------- registry, summaries
def _fake_run(run_dir: Path, fit: pl.PlannedFit, body: dict, extras: dict | None = None) -> None:
    doc = {"wall_s": 1.5, "rows": {"test_blanked": 7}, "fit": body, "extras": extras or {}}
    write_json(run_dir / fit.fit_id / "run.json", doc)


def _history(n: int, xi: bool = False) -> list[dict]:
    return [
        {"epoch": e, "train": 10.0 - e, "monitor": 12.0 - e, "xi": 0.5 if xi else None}
        for e in range(n)
    ]


def test_registry_merkle_and_summaries(tmp_path: Path) -> None:
    fits = [
        f
        for f in pl.build_plan(PROTOCOL, [("S-ID", 0, 0)])
        if (f.model in ("M4", "M2-XGB") and f.analysis == "primary" and f.ablation is None)
    ]
    fits = [f for f in fits if f.family != "S-RND"]
    m4_members = [f for f in fits if f.model == "M4" and f.kind == "member"]
    xgb_member = next(f for f in fits if f.model == "M2-XGB" and f.kind == "member")
    for f in m4_members[:2]:
        _fake_run(
            tmp_path,
            f,
            {"epochs_run": 3, "history": _history(3, xi=True)},
            {"mean_picard_iterations": 4.5, "max_picard_iterations": 7},
        )
        (tmp_path / f.fit_id / "state.npz").write_bytes(b"weights")
    _fake_run(tmp_path, xgb_member, {"W": {"best_iteration": 3}})
    sel = next(f for f in fits if f.model == "M4" and f.kind == "select")
    write_json(sel.completion_path(tmp_path), {"kind": "nn"})
    (tmp_path / m4_members[2].fit_id).mkdir(parents=True)  # incomplete: no run.json
    reg = tr.build_registry(fits, tmp_path)
    assert list(reg.columns) == list(tr.REGISTRY_COLUMNS)
    assert len(reg) == len(fits)
    assert int(reg["status"].eq("done").sum()) == 4
    row = reg.set_index("fit_id").loc[m4_members[0].fit_id]
    assert (row["epochs"], row["max_picard"], row["test_blanked"], row["n_files"]) == (3, 7, 7, 2)
    assert row["xi_final"] == 0.5
    assert (
        json.loads(reg.set_index("fit_id").loc[sel.fit_id, "files_json"])[0][0] == "selection.json"
    )
    root = tr.registry_merkle_root(reg)
    assert root == tr.registry_merkle_root(tr.build_registry(fits, tmp_path))
    pairs = sorted(
        (fid, p, s)
        for fid, fj in reg.loc[reg["status"].eq("done"), ["fit_id", "files_json"]].itertuples(
            index=False
        )
        for p, s in json.loads(fj)
    )
    assert root == tr.merkle_root([tr.merkle_leaf(*t) for t in pairs])
    (tmp_path / m4_members[0].fit_id / "state.npz").write_bytes(b"tampered")
    assert tr.registry_merkle_root(tr.build_registry(fits, tmp_path)) != root
    reg_path = tmp_path / "out" / "run_registry.parquet"
    sha = tr.write_registry(reg, reg_path, params_sha256="0" * 64, plan_sha256="1" * 64)
    assert len(sha) == 64
    assert len(pd.read_parquet(reg_path)) == len(fits)
    summ = tr.training_summaries(fits, tmp_path)
    assert summ["summary"]["stages"]["train@nn_primary"]["done"] == 3
    assert summ["summary"]["models"]["M4"]["picard_max"] == 7
    assert [r["n"] for r in summ["curves"]] == [2, 2, 2]
    assert summ["curves"][0] == {"model": "M4", "epoch": 0, "train": 10.0, "es_val": 12.0, "n": 2}
    assert [r["epoch"] for r in summ["xi"]] == [0, 1, 2]
    out = tr.write_training_reports(tmp_path / "training", summ)
    assert sorted(out) == ["curves", "summary", "xi"]
    assert isinstance(json.loads((tmp_path / "training" / "curves.json").read_text()), list)
    stage_docs = {
        "train@nn_primary": {
            "invocations": [
                {
                    "utc_start": "2026-01-01T00:00:00Z",
                    "git_start": {"commit": "a", "dirty": False},
                    "git_end": {"commit": "b", "dirty": False},
                    "failure": None,
                    "reruns": [{"fit_id": "x", "reason": "incomplete directory", "utc": "t"}],
                    "cleanup": [],
                    "wall_s": 2.0,
                }
            ]
        }
    }
    manifest = tr.build_manifest(
        run_id="p8-000000000000",
        plan_sha256="1" * 64,
        plan_header={"params_sha256": "0" * 64, "tables": HASHES},
        uv_lock_sha256="2" * 64,
        registry=reg,
        registry_sha256=sha,
        stage_docs=stage_docs,
        n_workers=10,
        torch_threads_per_worker=1,
        training_reports=out,
        pilot=None,
        archive=None,
    )
    assert manifest["merkle_root"] == tr.registry_merkle_root(reg)
    assert manifest["git"] == {
        "start": {"commit": "a", "dirty": False},
        "end": {"commit": "b", "dirty": False},
    }
    assert manifest["stages"]["train@nn_primary"]["reruns"][0]["fit_id"] == "x"
    assert manifest["stages"]["train@xgb"]["planned"] == sum(f.stage == "train@xgb" for f in fits)
    assert "torch" in manifest["host"]["packages"]


# ----------------------------------------------------------------------------- pilot, archive
def _tree(root: Path) -> None:
    (root / "M5" / "member_1").mkdir(parents=True)
    (root / "M5" / "member_1" / "run.json").write_text("{}")
    (root / "M5" / "member_1" / "state.npz").write_bytes(b"\x00\x01")
    (root / "a.json").write_text("[]")


def test_move_pilot(tmp_path: Path) -> None:
    src, dest = tmp_path / "checkpoints", tmp_path / "runs" / "pilot_adl090"
    _tree(src)
    before = tr.tree_digests(src)
    assert tr.move_pilot(src, dest) == before
    assert not src.exists()
    assert tr.tree_digests(dest) == before
    with pytest.raises(FileExistsError):
        tr.move_pilot(dest, dest)
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        tr.move_pilot(tmp_path / "empty", tmp_path / "x")


def test_archive_is_deterministic_ustar(tmp_path: Path) -> None:
    run = tmp_path / "p8-abc"
    _tree(run)
    a = tr.archive_run(run, tmp_path / "a" / "p8-abc.tar")
    os.utime(run / "a.json", (1, 1))  # mtimes do not enter the archive
    b = tr.archive_run(run, tmp_path / "b" / "p8-abc.tar")
    assert a["sha256"] == b["sha256"]
    assert a["n_files"] == 3
    with tarfile.open(tmp_path / "a" / "p8-abc.tar") as tar:
        members = tar.getmembers()
        assert [m.name for m in members] == sorted(m.name for m in members)
        assert {(m.mtime, m.uid, m.gid, m.uname, m.gname) for m in members} == {(0, 0, 0, "", "")}
    with pytest.raises(FileExistsError):
        tr.archive_run(run, tmp_path / "a" / "p8-abc.tar")

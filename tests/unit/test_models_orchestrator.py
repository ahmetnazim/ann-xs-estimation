"""Tests for the Phase 8 orchestrator and ``scripts/run_experiment.py`` (synthetic data only).

Reduced ``params.yaml`` in ``tmp_path`` (tiny backbone, 6 epochs, 2-point grids, M = 2), one
synthetic S-ID block. Pool tests use the production path: spawn workers, the torch-free pool for
baselines / M1 / M2-XGB and the torch pool for NN models. Bit-identical reseeding is checked on
every non-``run.json`` artefact of two independent runs (``run.json`` carries wall times).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import Settings, load_settings
from src.data.io import sha256_file, write_parquet
from src.evaluation.ledger import Ledger
from src.models import orchestrator as orch
from src.models import plan as pl
from src.models import runner
from src.models.registry import read_json, write_json
from src.models.runner import Tables
from src.models.tracking import build_registry, registry_merkle_root
from src.utils.seeding import fit_seed
from synthetic_sites import fold_table, site_frame

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CLEAN = {"commit": "abc123", "dirty": False}
RUN_ID = "p8-0123456789ab"


@pytest.fixture(scope="module")
def small(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    raw = yaml.safe_load((REPO_ROOT / "params.yaml").read_text())
    raw["model"]["backbone"].update(hidden_layers=1, hidden_width=8)
    raw["model"]["optimizer"].update(batch_size=16, max_epochs=6, patience=2)
    raw["model"]["inner_grid"].update(weight_decay=[0.0, 1.0e-4], lr=[1.0e-3], m4_lambda_max=[10.0])
    raw["model"]["xgb"].update(
        max_depth=[2, 3], eta=[0.3], min_child_weight=[1], max_rounds=20, early_stopping_rounds=5
    )
    raw["model"]["baselines"]["m1_max_depth"] = [2]
    raw["uq"].update(warm_start_epochs=2, ensemble_m=2)
    path = tmp_path_factory.mktemp("params") / "params.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_settings(path)


@pytest.fixture(scope="module")
def tables_on_disk(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Tables]:
    root = tmp_path_factory.mktemp("tables")
    frame = site_frame()
    site, folds = root / "site_table.parquet", root / "fold_assignment.parquet"
    frame.to_parquet(site)
    fold_table(frame["site_no"]).to_parquet(folds)
    return site, folds, Tables.load(site, folds)


def ctx_for(
    run_dir: Path, settings: Settings, tables: tuple[Path, Path, Tables]
) -> orch.RunContext:
    site, folds, t = tables
    return orch.RunContext(
        params_path=str(settings.params_path),
        site_path=str(site),
        fold_path=str(folds),
        run_dir=str(run_dir),
        params_sha256=settings.params_sha256,
        site_sha256=t.site_sha256,
        fold_sha256=t.fold_sha256,
    )


def subset(settings: Settings, models: tuple[str, ...]) -> list[pl.PlannedFit]:
    fits = pl.build_plan(settings.protocol, [("S-ID", 0, 0)])
    return [
        f
        for f in fits
        if f.model in models
        and f.ablation is None
        and f.analysis == "primary"
        and f.family != "S-RND"
    ]


def ledger_with_checkpoint(path: Path, settings: Settings, run_id: str = RUN_ID) -> Ledger:
    led = Ledger.create(
        path,
        run_id=run_id,
        plan_sha256="1" * 64,
        params_sha256=settings.params_sha256,
        uv_lock_sha256="2" * 64,
    )
    common: dict[str, Any] = {"git": CLEAN, "params_sha256": settings.params_sha256}
    led.append(
        "seal", inputs=[{"path": "p.parquet", "sha256": "a" * 64}], merkle_root="c" * 64, **common
    )
    led.append("event_1", inputs=[{"path": "p.parquet", "sha256": "a" * 64}], **common)
    led.append("checkpoint_1", note="PI approval text", **common)
    return led


def run(
    stage: str,
    fits: list[pl.PlannedFit],
    run_dir: Path,
    settings: Settings,
    tables: tuple[Path, Path, Tables],
    *,
    ledger: Path | None = None,
    inline: bool = True,
    n_workers: int = 2,
) -> orch.StageResult:
    return orch.run_stage(
        stage,
        fits,
        ctx=ctx_for(run_dir, settings, tables),
        settings=settings,
        run_id=RUN_ID,
        ledger_path=ledger or run_dir.parent / "no_ledger.json",
        n_workers=n_workers,
        inline=inline,
    )


def artefacts(run_dir: Path) -> dict[str, bytes]:
    """Every artefact except ``run.json`` (wall times) and logs/stage records."""
    skip = {"run.json"}
    return {
        p.relative_to(run_dir).as_posix(): p.read_bytes()
        for p in sorted(run_dir.rglob("*"))
        if p.is_file()
        and p.name not in skip
        and p.parts[len(run_dir.parts)] not in ("logs", "stages")
    }


# ----------------------------------------------------------------------------- inline
def test_baselines_inline_resume_and_rerun(
    small: Settings, tables_on_disk: Any, tmp_path: Path
) -> None:
    fits = subset(small, pl.BASELINES)
    run_dir = tmp_path / RUN_ID
    res = run("train@baselines", fits, run_dir, small, tables_on_disk)
    assert (res.planned, res.already_done, res.executed, res.reruns) == (3, 0, 3, [])
    for f in fits:
        doc = read_json(f.completion_path(run_dir))
        assert doc["rows"]["test_blanked"] == 12
        assert doc["header"]["params_sha256"] == small.params_sha256
        assert (run_dir / f.fit_id / "params.json").is_file()
    m0 = read_json(run_dir / fits[2].fit_id / "params.json")
    assert m0["model"] == "M0"
    assert m0["coef"] == {}  # 24 sites per division < 30 -> pooled
    stamps = {f.fit_id: f.completion_path(run_dir).stat().st_mtime_ns for f in fits}
    again = run("train@baselines", fits, run_dir, small, tables_on_disk)
    assert (again.already_done, again.executed) == (3, 0)
    assert stamps == {f.fit_id: f.completion_path(run_dir).stat().st_mtime_ns for f in fits}
    # a crash mid-fit leaves a directory without run.json: removed and re-run, logged
    victim = fits[0]
    victim.completion_path(run_dir).unlink()
    third = run("train@baselines", fits, run_dir, small, tables_on_disk)
    assert third.executed == 1
    assert [r["fit_id"] for r in third.reruns] == [victim.fit_id]
    (run_dir / "m6xgb_stale").mkdir()
    fourth = run("train@baselines", fits, run_dir, small, tables_on_disk)
    assert [c["path"] for c in fourth.cleanup] == ["m6xgb_stale"]
    stage_doc = read_json(run_dir / "stages" / "train@baselines.json")
    assert [inv["executed"] for inv in stage_doc["invocations"]] == [3, 0, 1, 0]
    reg = build_registry(fits, run_dir)
    assert reg["status"].eq("done").all()
    assert registry_merkle_root(reg) == registry_merkle_root(build_registry(fits, run_dir))


def test_complete_fit_that_disagrees_stops_the_run(
    small: Settings, tables_on_disk: Any, tmp_path: Path
) -> None:
    fits = subset(small, ("M-null",))
    run_dir = tmp_path / RUN_ID
    run("train@baselines", fits, run_dir, small, tables_on_disk)
    path = fits[0].completion_path(run_dir)
    for key, value in (("site_table_sha256", "f" * 64), ("header", {"params_sha256": "0" * 64})):
        doc = read_json(path)
        original = dict(doc)
        doc[key] = value if key != "header" else {**doc["header"], **value}
        write_json(path, doc)
        with pytest.raises(orch.ResumeMismatchError, match=fits[0].fit_id):
            run("train@baselines", fits, run_dir, small, tables_on_disk)
        write_json(path, original)
    doc = read_json(path)
    doc["header"]["outer_fold"] = 3
    write_json(path, doc)
    with pytest.raises(orch.ResumeMismatchError, match="coordinates"):
        run("train@baselines", fits, run_dir, small, tables_on_disk)


def test_nn_stages_need_checkpoint_1(small: Settings, tables_on_disk: Any, tmp_path: Path) -> None:
    fits = subset(small, ("M2-MLP",))
    run_dir = tmp_path / RUN_ID
    ledger = tmp_path / "results" / "scoring_ledger.json"
    with pytest.raises(orch.StageRefusedError, match="no scoring ledger"):
        run("train@nn_primary", fits, run_dir, small, tables_on_disk, ledger=ledger)
    led = Ledger.create(
        ledger,
        run_id=RUN_ID,
        plan_sha256="1" * 64,
        params_sha256=small.params_sha256,
        uv_lock_sha256="2" * 64,
    )
    led.append(
        "seal",
        inputs=[{"path": "p", "sha256": "a" * 64}],
        merkle_root="c" * 64,
        git=CLEAN,
        params_sha256=small.params_sha256,
    )
    with pytest.raises(orch.StageRefusedError, match="until checkpoint_1"):
        run("train@nn_primary", fits, run_dir, small, tables_on_disk, ledger=ledger)
    other = tmp_path / "other" / "scoring_ledger.json"
    ledger_with_checkpoint(other, small, run_id="p8-ffffffffffff")
    with pytest.raises(orch.StageRefusedError, match="belongs to"):
        run("train@nn_primary", fits, run_dir, small, tables_on_disk, ledger=other)
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"header": {}, "events": []}))
    with pytest.raises(orch.StageRefusedError, match="invalid"):
        run("train@nn_primary", fits, run_dir, small, tables_on_disk, ledger=broken)
    assert not run_dir.exists()  # refusals write nothing


def test_cross_stage_dependencies_must_be_complete(
    small: Settings, tables_on_disk: Any, tmp_path: Path
) -> None:
    fits = [
        f
        for f in pl.build_plan(small.protocol, [("S-ID", 0, 0)])
        if f.model == "M5" and f.family != "S-RND"
    ]
    ledger = tmp_path / "results" / "scoring_ledger.json"
    ledger_with_checkpoint(ledger, small)
    with pytest.raises(orch.StageRefusedError, match="dependencies of train@ablations incomplete"):
        run("train@ablations", fits, tmp_path / RUN_ID, small, tables_on_disk, ledger=ledger)


def test_nn_stage_inline_and_failure_drains(
    small: Settings, tables_on_disk: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fits = subset(small, ("M2-MLP",))
    ledger = tmp_path / "results" / "scoring_ledger.json"
    ledger_with_checkpoint(ledger, small)
    run_dir = tmp_path / RUN_ID
    res = run("train@nn_primary", fits, run_dir, small, tables_on_disk, ledger=ledger)
    assert res.executed == len(fits) == 2 * 3 + 1 + 2
    sel = read_json(run_dir / "M2-MLP/primary/S-ID/r0/f0/selection.json")
    assert sel["kind"] == "nn"
    member = run_dir / "M2-MLP/primary/S-ID/r0/f0/member_2"
    assert {p.name for p in member.iterdir()} == {
        "state.npz",
        "model.json",
        "normstats.json",
        "run.json",
    }
    with np.load(member / "state.npz", allow_pickle=False) as npz:
        assert npz.files
    # failure: the first member raises; scheduling stops, the report names fit, seed, coordinates
    real = orch.execute_fit

    def flaky(fit: pl.PlannedFit, *args: Any, **kwargs: Any) -> None:
        if fit.kind == "member" and fit.index == 1:
            raise FloatingPointError("non-finite loss")
        real(fit, *args, **kwargs)

    monkeypatch.setattr(orch, "execute_fit", flaky)
    run_dir2 = tmp_path / "second"
    with pytest.raises(orch.RunFailureError) as info:
        run("train@nn_primary", fits, run_dir2, small, tables_on_disk, ledger=ledger, n_workers=1)
    rep = info.value.report
    assert rep["fit_id"] == "M2-MLP/primary/S-ID/r0/f0/member_1"
    assert rep["error"] == "FloatingPointError"
    assert rep["seed"] == orch.fit_seed(small.protocol.seeds, "S-ID", 0, 0, "member", 1)
    assert rep["coordinates"]["kind"] == "member"
    assert not (run_dir2 / "M2-MLP/primary/S-ID/r0/f0/member_2").exists()  # never scheduled
    inv = read_json(run_dir2 / "stages" / "train@nn_primary.json")["invocations"][-1]
    assert inv["failure"]["fit_id"] == rep["fit_id"]


def test_inline_worker_refuses_torch_free_with_torch(
    small: Settings, tables_on_disk: Any, tmp_path: Path
) -> None:
    assert "torch" in sys.modules  # loaded by the session fixture
    with pytest.raises(RuntimeError, match="torch-free worker"):
        orch._init_worker(ctx_for(tmp_path, small, tables_on_disk).to_json(), True, False)
    bad = orch.RunContext(
        **{
            **json.loads(ctx_for(tmp_path, small, tables_on_disk).to_json()),
            "site_sha256": "0" * 64,
        }
    )
    with pytest.raises(RuntimeError, match="table hashes"):
        orch._init_worker(bad.to_json(), False, False)
    bad = orch.RunContext(**{**json.loads(bad.to_json()), "params_sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="params_sha256"):
        orch._init_worker(bad.to_json(), False, False)


def test_execute_fit_srnd_refuses_in_a_torch_process(
    small: Settings, tables_on_disk: Any, tmp_path: Path
) -> None:
    """S-RND now executes (Batch C2); without a frame, and in a torch process, it refuses."""
    fit = pl.PlannedFit("train@xgb", "M1", None, "primary", "S-RND", 0, 0, "member", 1, None)
    with pytest.raises(ValueError, match="srnd_frame table"):
        orch.execute_fit(fit, tables_on_disk[2], small, tmp_path, logging.getLogger("t"))
    assert orch._seed_of(fit, small) == fit_seed(small.protocol.seeds, "S-RND", 0, 0, "member", 1)


# ----------------------------------------------------------------------------- S-RND (13.6)
def test_round_half_up_and_mode_and_median_known_answers() -> None:
    assert [runner.round_half_up(x) for x in (2.5, 2.4999, 3.5, 7.0)] == [3, 2, 4, 7]
    with pytest.raises(ValueError, match="non-finite"):
        runner.round_half_up(float("nan"))
    sel = [{"targets": {"W": {"selected": g}, "dbar": {"selected": 1}}} for g in (3, 0, 5, 3, 0)]
    mode = runner.srnd_hyper_mode(sel, ("W", "dbar"))
    assert mode["W"] == {"grid": 0, "votes": {"0": 2, "3": 2, "5": 1}}  # tie -> lower index
    assert mode["dbar"]["grid"] == 1
    members = [{"fit": {"W": {"best_iteration": b}}} for b in (9, 10, 11, 12)]
    rounds = runner.srnd_rounds(members, ("W",))
    assert rounds["W"] == {"rounds": 12, "median": 11.5, "n_members": 4}  # 11.5 -> 12
    assert runner.srnd_rounds(members[:3], ("W",))["W"]["rounds"] == 11
    with pytest.raises(ValueError, match="no S-ID selections"):
        runner.srnd_hyper_mode([], ("W",))
    with pytest.raises(ValueError, match="no S-ID members"):
        runner.srnd_rounds([], ("W",))


def _srnd_frame_file(path: Path, site_path: Path, site_sha: str) -> pd.DataFrame:
    """Three measurements per synthetic site; every 4th row (by row_id) is an S-RND test row."""
    sites = site_frame()
    rng = np.random.default_rng(5)
    rows = sites.loc[sites.index.repeat(3)].reset_index(drop=True)
    rows["row_id"] = np.arange(len(rows), dtype=np.int64)
    rows["q_meas_no"] = [f" {k}" for k in range(len(rows))]
    rows["is_test"] = rows["row_id"] % 4 == 0
    rows["seed"] = np.int64(7)
    for col in ("W", "dbar", "dmax", "A"):
        rows[col] = rows[col] * np.exp(rng.normal(0.0, 0.05, len(rows)))
    rows["comid"] = 1000 + rows.index // 3
    for flag in ("F_eps", "F_t", "F_qual", "F_tout"):
        rows[flag] = False
    write_parquet(
        rows,
        path,
        stage="srnd_frame",
        params_sha256="0" * 64,
        inputs={"site_table.parquet": site_sha},
        sort_by=["row_id"],
    )
    assert site_path.is_file()
    return rows


@pytest.fixture(scope="module")
def srnd_on_disk(tables_on_disk: Any, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    site, _, t = tables_on_disk
    path = tmp_path_factory.mktemp("srnd") / "srnd_frame.parquet"
    _srnd_frame_file(path, site, t.site_sha256)
    return path, sha256_file(path)


def test_tables_load_checks_srnd_provenance(
    tables_on_disk: Any, srnd_on_disk: tuple[Path, str], tmp_path: Path
) -> None:
    site, folds, _ = tables_on_disk
    t = Tables.load(site, folds, srnd_on_disk[0])
    assert t.srnd_frame is not None
    assert t.srnd_sha256 == srnd_on_disk[1]
    other = tmp_path / "other.parquet"
    _srnd_frame_file(other, site, "f" * 64)
    with pytest.raises(ValueError, match="not built from this site_table"):
        Tables.load(site, folds, other)
    ctx = orch.RunContext(
        **{
            **json.loads(ctx_for(tmp_path, small_settings_for_ctx(), tables_on_disk).to_json()),
            "srnd_path": str(srnd_on_disk[0]),
            "srnd_sha256": "0" * 64,
        }
    )
    with pytest.raises(RuntimeError, match="srnd_frame hash"):
        orch._init_worker(ctx.to_json(), False, False)


def small_settings_for_ctx() -> Settings:
    """The checkout's settings (only the params path and hash enter the context)."""
    return load_settings()


def _write_sid_sources(root: Path, settings: Settings, t: Tables, model: str) -> None:
    """One S-ID selection and M members of ``model`` with known best iterations."""
    targets = ("W", "dbar") if model == "M1" else ("W", "dbar", "dmax", "A")
    base = root / model / "primary" / "S-ID" / "r0" / "f0"
    write_json(
        base / "selection.json",
        {
            "kind": "xgb",
            "params_sha256": settings.params_sha256,
            "site_table_sha256": t.site_sha256,
            "fold_table_sha256": t.fold_sha256,
            "targets": {k: {"selected": 0} for k in targets},
        },
    )
    for m in range(1, settings.protocol.uq.ensemble_m + 1):
        write_json(
            base / f"member_{m}" / "run.json",
            {
                "header": {
                    "params_sha256": settings.params_sha256,
                    "spec": model,
                    "analysis": "primary",
                    "family": "S-ID",
                    "kind": "member",
                    "index": m,
                },
                "site_table_sha256": t.site_sha256,
                "fold_table_sha256": t.fold_sha256,
                "fit": {k: {"best_iteration": 3 + m} for k in targets},
            },
        )


def test_run_srnd_member_refusals_and_source_checks(
    small: Settings, tables_on_disk: Any, srnd_on_disk: tuple[Path, str], tmp_path: Path
) -> None:
    site, folds, plain = tables_on_disk
    t = Tables.load(site, folds, srnd_on_disk[0])
    with pytest.raises(ValueError, match="S-RND trains"):
        runner.run_srnd_member("M6-XGB", 1, t, small, tmp_path)
    with pytest.raises(ValueError, match="numbered"):
        runner.run_srnd_member("M1", 0, t, small, tmp_path)
    with pytest.raises(ValueError, match="srnd_frame table"):
        runner.run_srnd_member("M1", 1, plain, small, tmp_path)
    with pytest.raises(FileNotFoundError, match=r"selection\.json"):
        runner.run_srnd_member("M1", 1, t, small, tmp_path)
    _write_sid_sources(tmp_path, small, t, "M1")
    with pytest.raises(RuntimeError, match="without torch"):  # sources pass; torch is loaded
        runner.run_srnd_member("M1", 1, t, small, tmp_path)
    member2 = tmp_path / "M1" / "primary" / "S-ID" / "r0" / "f0" / "member_2" / "run.json"
    doc = read_json(member2)
    doc["header"]["index"] = 1
    write_json(member2, doc)
    with pytest.raises(ValueError, match=r"member_2/run\.json does not match"):
        runner.run_srnd_member("M1", 1, t, small, tmp_path)
    member2.unlink()
    with pytest.raises(FileNotFoundError, match="member_2"):
        runner.run_srnd_member("M1", 1, t, small, tmp_path)
    sel = tmp_path / "M1" / "primary" / "S-ID" / "r0" / "f0" / "selection.json"
    write_json(sel, {**read_json(sel), "params_sha256": "0" * 64})
    with pytest.raises(ValueError, match=r"selection\.json does not match"):
        runner.run_srnd_member("M1", 1, t, small, tmp_path)
    no_sid = Tables(t.site_table, t.folds.assign(family="S-LRO"), "a", "b", t.srnd_frame, "c")
    with pytest.raises(ValueError, match="no S-ID block"):
        runner.run_srnd_member("M1", 1, no_sid, small, tmp_path / "x")


def test_srnd_members_spawn_pool_known_answers_and_bit_identical(
    small: Settings, tables_on_disk: Any, srnd_on_disk: tuple[Path, str], tmp_path: Path
) -> None:
    """S-ID M1/M2-XGB blocks, then S-RND members, in the torch-free spawn pool (twice).

    Known answers recomputed from the artefacts: the grid point (one S-ID block, so the mode is
    its selection), the rounds (median of best_iteration + 1 over the S-ID members, half up), the
    number of trees in each booster JSON, the seed and the row counts.
    """
    fits = [
        f
        for f in pl.build_plan(small.protocol, [("S-ID", 0, 0)])
        if f.model in ("M1", "M2-XGB") and f.ablation is None and f.analysis == "primary"
    ]
    srnd = [f for f in fits if f.family == "S-RND"]
    assert len(srnd) == 2 * small.protocol.uq.ensemble_m
    frame = pd.read_parquet(srnd_on_disk[0])
    results = {}
    for name in ("a", "b"):
        run_dir = tmp_path / name / RUN_ID
        ctx = orch.RunContext(
            **{
                **json.loads(ctx_for(run_dir, small, tables_on_disk).to_json()),
                "srnd_path": str(srnd_on_disk[0]),
                "srnd_sha256": srnd_on_disk[1],
            }
        )
        res = orch.run_stage(
            "train@xgb",
            fits,
            ctx=ctx,
            settings=small,
            run_id=RUN_ID,
            ledger_path=tmp_path / "none.json",
            n_workers=2,
        )
        assert res.executed == len(fits)
        results[name] = artefacts(run_dir)
        for model, targets in (("M1", ("W", "dbar")), ("M2-XGB", ("W", "dbar", "dmax", "A"))):
            block = run_dir / model / "primary" / "S-ID" / "r0" / "f0"
            sel = read_json(block / "selection.json")
            sid = [
                read_json(block / f"member_{m}" / "run.json")
                for m in range(1, small.protocol.uq.ensemble_m + 1)
            ]
            for m in range(1, small.protocol.uq.ensemble_m + 1):
                member = run_dir / model / "primary" / "S-RND" / "r0" / "f0" / f"member_{m}"
                doc = read_json(member / "run.json")
                assert doc["seed"] == fit_seed(small.protocol.seeds, "S-RND", 0, 0, "member", m)
                assert doc["srnd_frame_sha256"] == srnd_on_disk[1]
                assert doc["rows"]["test_blanked"] == int(frame["is_test"].sum())
                train = ~frame["is_test"]
                if model == "M1":
                    train &= ~frame["F_M1_missing"]
                assert doc["rows"]["train"] == int(train.sum())
                for t in targets:
                    n_trees = [s["fit"][t]["best_iteration"] + 1 for s in sid]
                    want = runner.round_half_up(float(np.median(n_trees)))
                    assert doc["fit"][t]["rounds"] == want
                    assert doc["fit"][t]["grid"] == sel["targets"][t]["selected"]
                    booster = json.loads((member / f"booster_{t}.json").read_bytes())
                    model_param = booster["learner"]["gradient_booster"]["model"]
                    assert int(model_param["gbtree_model_param"]["num_trees"]) == want
    assert results["a"] == results["b"]
    assert any("/S-RND/" in k and k.endswith("booster_dmax.json") for k in results["a"])


def test_run_stage_argument_checks(small: Settings, tables_on_disk: Any, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        run("train@x", [], tmp_path, small, tables_on_disk)
    with pytest.raises(ValueError, match="n_workers"):
        run("train@baselines", [], tmp_path, small, tables_on_disk, n_workers=0)


def test_thread_env_and_keep_awake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("VECLIB_MAXIMUM_THREADS", raising=False)
    with orch.thread_env():
        assert all(os.environ[k] == "1" for k in orch.THREAD_ENV)
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert "VECLIB_MAXIMUM_THREADS" not in os.environ
    with orch.keep_awake(enabled=False) as proc:
        assert proc is None
    with orch.keep_awake() as proc:
        if sys.platform == "darwin":
            assert proc is not None
            assert proc.poll() is None
    if proc is not None:
        assert proc.poll() is not None


# ----------------------------------------------------------------------------- spawn pools
def test_pools_bit_identical_reseeding_and_isolation(
    small: Settings, tables_on_disk: Any, tmp_path: Path
) -> None:
    xgb = subset(small, ("M1", "M2-XGB"))
    nn = subset(small, ("M2-MLP",))
    ledger = tmp_path / "results" / "scoring_ledger.json"
    ledger_with_checkpoint(ledger, small)
    results = {}
    for name in ("a", "b"):
        run_dir = tmp_path / name / RUN_ID
        rx = run("train@xgb", xgb + nn, run_dir, small, tables_on_disk, inline=False, ledger=ledger)
        rn = run(
            "train@nn_primary",
            xgb + nn,
            run_dir,
            small,
            tables_on_disk,
            inline=False,
            ledger=ledger,
        )
        assert (rx.executed, rn.executed) == (len(xgb), len(nn))
        assert rx.workers
        assert rn.workers
        assert not set(rx.workers) & set(rn.workers)  # torch-free and torch pools differ
        assert (run_dir / "logs").is_dir()
        results[name] = artefacts(run_dir)
    assert results["a"] == results["b"]
    assert any(k.endswith("booster_dbar.json") for k in results["a"])
    assert any(k.endswith("state.npz") for k in results["a"])
    assert not [k for k in results["a"] if k.endswith((".pkl", ".pt", ".pickle"))]

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
import pytest
import yaml

from src.config import Settings, load_settings
from src.evaluation.ledger import Ledger
from src.models import orchestrator as orch
from src.models import plan as pl
from src.models.registry import read_json, write_json
from src.models.runner import Tables
from src.models.tracking import build_registry, registry_merkle_root
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


def test_execute_fit_refuses_srnd_until_batch_c(
    small: Settings, tables_on_disk: Any, tmp_path: Path
) -> None:
    fit = pl.PlannedFit("train@xgb", "M1", None, "primary", "S-RND", 0, 0, "member", 1, None)
    with pytest.raises(NotImplementedError, match="srnd_frame"):
        orch.execute_fit(fit, tables_on_disk[2], small, tmp_path, logging.getLogger("t"))


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

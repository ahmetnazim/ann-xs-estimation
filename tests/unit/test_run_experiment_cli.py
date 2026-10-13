"""Tests for ``scripts/run_experiment.py`` in a temporary repository (synthetic tables only).

``ANNXS_REPO_ROOT`` points at ``tmp_path/repo`` (not a git checkout, so the tree counts as
dirty and ``--allow-dirty`` is needed), ``ANNXS_RUN_ROOT`` and ``ANNXS_DVC_STORE`` at
``tmp_path/store``. The plan uses a reduced ``params.yaml`` (``--reduced``).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from src.config import load_settings
from src.data.io import sha256_file
from src.evaluation.ledger import Ledger, LedgerError
from src.models.orchestrator import RunContext, StageRefusedError, run_stage
from synthetic_sites import fold_table, site_frame

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def _cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_experiment", REPO_ROOT / "scripts" / "run_experiment.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_experiment"] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    proc = repo / "data" / "processed"
    (proc / "splits").mkdir(parents=True)
    frame = site_frame()
    frame.to_parquet(proc / "site_table.parquet")
    fold_table(frame["site_no"]).to_parquet(proc / "splits" / "fold_assignment.parquet")
    for rel, data in (
        ("splits/srnd_assignment.parquet", b"srnd"),
        ("ind/ehydro_ind.parquet", b"ind"),
        ("ind/ehydro_ind_features.parquet", b"indf"),
        ("srnd/srnd_frame.parquet", b"frame"),
    ):
        (proc / rel).parent.mkdir(parents=True, exist_ok=True)
        (proc / rel).write_bytes(data)
    (repo / "uv.lock").write_text("lock")
    raw = yaml.safe_load((REPO_ROOT / "params.yaml").read_text())
    raw["model"]["inner_grid"].update(weight_decay=[0.0], lr=[1.0e-3], m4_lambda_max=[10.0])
    raw["model"]["xgb"].update(max_depth=[2], eta=[0.3], min_child_weight=[1])
    raw["model"]["baselines"]["m1_max_depth"] = [2]
    raw["uq"]["ensemble_m"] = 2
    params = tmp_path / "params.yaml"
    params.write_text(yaml.safe_dump(raw, sort_keys=False))
    store = tmp_path / "store"
    monkeypatch.setenv("ANNXS_REPO_ROOT", str(repo))
    monkeypatch.setenv("ANNXS_DVC_STORE", str(store))
    monkeypatch.setenv("ANNXS_RUN_ROOT", str(store / "runs"))
    monkeypatch.setenv("ANNXS_N_WORKERS", "2")
    return repo, store, params


def test_cli_plan_run_status_track(
    repo: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    root, store, params = repo
    cli = _cli()
    base = ["--params", str(params), "--reduced"]
    with pytest.raises(RuntimeError, match=r"13\.6 table"):
        cli.main(["plan", "--params", str(params)])
    with pytest.raises(FileNotFoundError, match="plan` first"):
        cli.main(["track", *base])
    assert cli.main(["plan", *base]) == 0
    out = capsys.readouterr().out
    run_id = out.split()[1]
    assert run_id.startswith("p8-")
    plan_file = store / "runs" / run_id / "plan.json"
    assert plan_file.is_file()
    assert cli.main(["plan", *base]) == 0  # idempotent
    with pytest.raises(RuntimeError, match="dirty"):
        cli.main(["run", "--stage", "train@baselines", *base])
    assert cli.main(["run", "--stage", "train@baselines", *base, "--allow-dirty"]) == 0
    assert "merkle_root" in capsys.readouterr().out
    manifest = json.loads((root / "reports" / "run_manifest.json").read_text())
    assert manifest["run_id"] == run_id
    assert manifest["stages"]["train@baselines"]["done"] == 3
    assert manifest["pilot"] is None
    assert manifest["archive"] is None
    assert (root / "data" / "processed" / "runs" / "run_registry.parquet").is_file()
    assert (root / "reports" / "training" / "curves.json").is_file()
    with pytest.raises(StageRefusedError, match="checkpoint_1"):
        cli.main(["run", "--stage", "train@nn_primary", *base, "--allow-dirty"])
    assert cli.main(["status", *base]) == 0
    assert "train@baselines      planned      3  done      3" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="archive refused"):
        cli.main(["archive", *base])
    (root / "data" / "checkpoints" / "M5").mkdir(parents=True)
    (root / "data" / "checkpoints" / "M5" / "run.json").write_text("{}")
    assert cli.main(["move-pilot", *base]) == 0
    assert cli.main(["track", *base]) == 0
    manifest = json.loads((root / "reports" / "run_manifest.json").read_text())
    assert list(manifest["pilot"]) == ["M5/run.json"]
    plan_file.write_bytes(plan_file.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="differs from the recomputed plan"):
        cli.main(["run", "--stage", "train@baselines", *base, "--allow-dirty"])
    with pytest.raises(RuntimeError, match="different content"):
        cli.main(["plan", *base])


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )


def test_cli_predict_and_seal(
    repo: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    root, _, params = repo
    cli = _cli()
    base = ["--params", str(params), "--reduced"]
    audit = ["--stage", "predict@audit", *base]
    assert cli.main(["plan", *base]) == 0
    run_id = capsys.readouterr().out.split()[1]
    with pytest.raises(FileNotFoundError, match=r"predictions_audit\.json"):
        cli.main(["seal", *audit])
    with pytest.raises(RuntimeError, match="fits incomplete"):
        cli.main(["predict", *audit])
    assert cli.main(["run", "--stage", "train@baselines", *base, "--allow-dirty"]) == 0
    settings = load_settings(params)
    built = cli._build(settings, reduced=True)
    fits = [
        f
        for f in built.fits
        if f.model == "M2-XGB"
        and f.ablation is None
        and f.analysis == "primary"
        and f.family == "S-ID"
    ]
    ctx = RunContext(
        params_path=str(settings.params_path),
        site_path=str(built.paths["site_table"]),
        fold_path=str(built.paths["fold_table"]),
        run_dir=str(built.run_dir(settings)),
        params_sha256=settings.params_sha256,
        site_sha256=built.hashes["site_table"],
        fold_sha256=built.hashes["fold_table"],
    )
    ledger_path = root / "reports" / "results" / "scoring_ledger.json"
    run_stage(
        "train@xgb",
        fits,
        ctx=ctx,
        settings=settings,
        run_id=run_id,
        ledger_path=ledger_path,
        n_workers=2,
    )
    capsys.readouterr()
    assert cli.main(["predict", *audit]) == 0
    rel = "data/processed/predictions/audit_inputs.parquet"
    out = capsys.readouterr().out
    assert rel in out
    assert "predict@audit: 2 blocks, 1 files" in out
    report = json.loads((root / "reports" / "predictions_audit.json").read_text())
    assert (report["run_id"], report["plan_sha256"]) == (run_id, built.sha256)
    assert report["files"][rel]["sha256"] == sha256_file(root / rel)
    with pytest.raises(StageRefusedError, match="checkpoint_1"):
        cli.main(["predict", "--stage", "predict@all", *base])
    with pytest.raises(RuntimeError, match="clean, committed tree"):
        cli.main(["seal", *audit])
    assert not ledger_path.exists()
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "synthetic")
    assert cli.main(["seal", *audit]) == 0
    assert "seal seq 1: 1 files" in capsys.readouterr().out
    ledger = Ledger.load(ledger_path)
    assert ledger.sealed() == {rel: report["files"][rel]["sha256"]}
    assert ledger.header["run_id"] == run_id
    manifest = json.loads((root / "reports" / "run_manifest.json").read_text())
    assert ledger.events[0]["merkle_root"] == manifest["merkle_root"]
    with pytest.raises(LedgerError, match="out-of-order"):
        cli.main(["seal", *audit])
    sealed = (root / rel).read_bytes()
    (root / rel).write_bytes(sealed + b"x")
    with pytest.raises(RuntimeError, match="changed since it was written"):
        cli.main(["seal", *audit])
    assert cli.main(["predict", *audit]) == 0  # same inputs: the sealed bytes come back
    assert (root / rel).read_bytes() == sealed

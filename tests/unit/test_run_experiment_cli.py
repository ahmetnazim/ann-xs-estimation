"""Tests for ``scripts/run_experiment.py`` in a temporary repository (synthetic tables only).

``ANNXS_REPO_ROOT`` points at ``tmp_path/repo`` (not a git checkout, so the tree counts as
dirty and ``--allow-dirty`` is needed), ``ANNXS_RUN_ROOT`` and ``ANNXS_DVC_STORE`` at
``tmp_path/store``. The plan uses a reduced ``params.yaml`` (``--reduced``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from src.models.orchestrator import StageRefusedError
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

"""Smoke tests for scripts/verify_models.py (Phase 6, Batch E). Synthetic only; no data.

The full script run is the Phase 6 sanity record (PHASE_06 Step 5); these tests exercise the
report logic, the synthetic tables and a quick subset of sections in-process.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from src.config import load_settings

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()


def _load() -> ModuleType:
    path = REPO_ROOT / "scripts" / "verify_models.py"
    spec = importlib.util.spec_from_file_location("verify_models", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_models"] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


VM = _load()


def test_report_records_limits_and_non_finite_values() -> None:
    rep = VM.Report()
    assert rep.le("s", "small", 1e-9, 1e-8, "src")
    assert not rep.le("s", "large", 1e-7, 1e-8, "src")
    assert not rep.le("s", "nan", math.nan, 1.0, "src")
    assert not rep.add("s", "inf", True, math.inf, "finite", "src")
    assert rep.flag("s", "flag", True, "src")
    assert [c.name for c in rep.failures] == ["large", "nan", "inf"]
    doc = rep.to_json({"params_sha256": "0" * 64})
    assert doc["summary"] == {"n_checks": 5, "n_failed": 3}
    json.dumps(doc, allow_nan=False)  # non-finite values are stored as text


def test_a_raising_section_is_a_failed_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(rep: object, ctx: object) -> None:
        raise RuntimeError("synthetic failure")

    monkeypatch.setitem(VM.SECTIONS, "boom", boom)
    rep = VM.run(["boom", "ensemble"], SETTINGS, quick=True, work=tmp_path)
    failed = rep.failures
    assert len(failed) == 1
    assert failed[0].section == "boom"
    assert "synthetic failure" in failed[0].value
    assert any(c.section == "ensemble" for c in rep.checks)  # later sections still run


def test_synthetic_tables_follow_the_contracts() -> None:
    frame = VM.synthetic_site_table(SETTINGS.protocol, 60, seed=3)
    folds = VM.synthetic_fold_table(frame["site_no"])
    assert set(SETTINGS.protocol.transforms.features) <= set(frame.columns)
    assert (frame["S"] >= SETTINGS.protocol.domain.s_floor).all()
    assert np.allclose(frame["A"], frame["W"] * frame["dbar"])
    test = folds["is_test"].to_numpy()
    es = folds[[f"es_member_{m}" for m in range(1, 11)]].to_numpy()
    assert not (es & test[:, None]).any()
    assert (folds.loc[test, "inner_fold"] == -1).all()


def test_quick_physics_and_ensemble_sections_pass(tmp_path: Path) -> None:
    rep = VM.run(["hand", "grad", "ensemble"], SETTINGS, quick=True, work=tmp_path)
    assert not rep.failures, rep.failures
    assert len(rep.checks) > 30


def test_main_writes_a_passing_report(tmp_path: Path) -> None:
    out = tmp_path / "verify.json"
    assert VM.main(["--quick", "--only", "quad,isolation", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["header"]["params_sha256"] == SETTINGS.params_sha256
    assert doc["header"]["sections"] == ["quad", "isolation"]
    assert doc["summary"]["n_failed"] == 0
    assert doc["info"]["quad_grid_points"] == len(VM.R_QUICK) * len(VM.B_QUICK)
    with pytest.raises(SystemExit, match="unknown sections"):
        VM.main(["--only", "nope", "--out", str(out)])

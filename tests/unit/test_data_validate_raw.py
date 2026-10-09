"""Unit tests for src/data/validate_raw.py (Phase 5, Batch C). Synthetic only.

Sources: docs/3 1.1 (manifest, publisher checksums), 2.3 (raw schemas),
2.7; ADL-058.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.config import load_settings
from src.data import validate_raw as vr

pytestmark = pytest.mark.unit


def make_unit(tmp_path: Path, payload: bytes = b"hello\n") -> Path:
    """A one-file raw unit with matching MANIFEST.json and MANIFEST.sha256."""
    unit = tmp_path / "unit"
    unit.mkdir()
    (unit / "a.txt").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    record = {"name": "a.txt", "size": len(payload), "sha256": digest}
    (unit / "MANIFEST.json").write_text(json.dumps({"files": [record]}), encoding="utf-8")
    (unit / "MANIFEST.sha256").write_text(f"{digest}  a.txt\n", encoding="utf-8")
    return unit


def test_manifest_passes_when_intact(tmp_path: Path) -> None:
    report = vr.UnitReport("unit")
    vr.check_manifest(make_unit(tmp_path), report)
    assert report.passed
    assert report.info["v1"] == {"files": 1, "bytes": 6, "extracted": 0}


def test_manifest_detects_size_change(tmp_path: Path) -> None:
    unit = make_unit(tmp_path)
    (unit / "a.txt").write_bytes(b"hello world\n")
    report = vr.UnitReport("unit")
    vr.check_manifest(unit, report)
    assert not report.passed
    assert "size" in report.failures[0]


def test_manifest_detects_same_size_content_change(tmp_path: Path) -> None:
    unit = make_unit(tmp_path)
    (unit / "a.txt").write_bytes(b"HELLO\n")
    report = vr.UnitReport("unit")
    vr.check_manifest(unit, report)
    assert report.failures == ["V1: a.txt: SHA-256 differs from acquisition"]


def test_manifest_missing_files(tmp_path: Path) -> None:
    unit = make_unit(tmp_path)
    (unit / "MANIFEST.sha256").unlink()
    report = vr.UnitReport("unit")
    vr.check_manifest(unit, report)
    assert not report.passed


def test_only_v1_to_v3_fail_a_unit() -> None:
    report = vr.UnitReport("unit")
    with pytest.raises(ValueError, match="only V1-V3"):
        report.fail("V4", "a reported-only check")
    report.fail("V2", "schema")
    assert report.as_dict()["status"] == "fail"


def test_streamcat_fields_come_from_the_acquire_urls() -> None:
    fields = vr.streamcat_fields(load_settings())
    assert fields
    assert all(name.endswith("ws") for name in fields)
    assert len(set(fields)) == len(fields)


def test_every_raw_unit_has_a_checker() -> None:
    assert set(vr.VALIDATED_UNITS) == set(vr.CHECKERS)

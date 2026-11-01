"""Unit tests for the OPS-1 parameter schema (src/operational/config.py)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.operational.config import OPS_PARAMS_PATH, load_operational

pytestmark = pytest.mark.unit


def _write(tmp_path: Path, data: dict[str, object]) -> Path:
    p = tmp_path / "ops.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def _base() -> dict[str, object]:
    raw = yaml.safe_load(OPS_PARAMS_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def test_committed_file_loads_with_provenance() -> None:
    """The committed pre-registration validates and carries its SHA-256."""
    s = load_operational()
    p = s.params
    assert p.version == 1
    assert p.predictions.primary_family == "S-ID"
    assert p.predictions.reference_model == "M0"
    assert p.metrics.flow_bin_edges[0] == 0.0
    assert p.metrics.flow_bin_edges[-1] == float("inf")
    assert len(s.sha256) == 64


def test_unknown_key_rejected(tmp_path: Path) -> None:
    """extra='forbid' at every level."""
    data = _base()
    data["surprise"] = 1
    with pytest.raises(ValueError, match="surprise"):
        load_operational(_write(tmp_path, data))


def test_duplicate_key_rejected(tmp_path: Path) -> None:
    """Duplicate YAML keys are refused before validation."""
    p = tmp_path / "dup.yaml"
    p.write_text(OPS_PARAMS_PATH.read_text(encoding="utf-8") + "\nversion: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        load_operational(p)


def test_non_mapping_rejected(tmp_path: Path) -> None:
    """A YAML list is not a parameter file."""
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_operational(p)


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        ("metrics", "flow_bin_edges", [0.5, 2.0, float("inf")], "start at 0"),
        ("metrics", "flow_bin_edges", [0.0, 2.0, 1.0, float("inf")], "strictly increasing"),
        ("metrics", "contrasts", [["M5", "M5"]], "itself"),
        ("predictions", "primary_family", "S-LRO", "primary_family"),
        ("predictions", "reference_model", "M5", None),
        ("predictions", "models", ["M5", "M5"], "unique"),
    ],
)
def test_validators(
    tmp_path: Path, section: str, key: str, value: object, match: str | None
) -> None:
    """Cross-field validators reject inconsistent pre-registrations."""
    data = _base()
    block = data[section]
    assert isinstance(block, dict)
    block[key] = value
    if section == "predictions" and key == "primary_family":
        block["families"] = ["S-ID"]
    if section == "predictions" and key == "reference_model":
        block["models"] = ["M0", "M2-XGB"]
        match = "reference_model"
    with pytest.raises(ValueError, match=match):
        load_operational(_write(tmp_path, data))


def test_contrast_and_fallback_must_be_scored_models(tmp_path: Path) -> None:
    """Contrasts and the fallback-shape model must be among the scored models."""
    data = _base()
    preds = data["predictions"]
    assert isinstance(preds, dict)
    preds["models"] = ["M0", "M2-XGB", "M3"]
    metrics = data["metrics"]
    assert isinstance(metrics, dict)
    metrics["contrasts"] = [["M2-XGB", "M0"], ["M5", "M0"]]
    with pytest.raises(ValueError, match=r"outside predictions\.models"):
        load_operational(_write(tmp_path, data))
    metrics["contrasts"] = [["M2-XGB", "M0"]]
    preds["models"] = ["M0", "M2-XGB"]
    with pytest.raises(ValueError, match="fallback_shape_model"):
        load_operational(_write(tmp_path, data))

"""Unit tests for the OPS-1 v3 parameter schema (src/operational/config_v3.py)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.operational.config import OPS_PARAMS_PATH
from src.operational.config_v2 import OPS2_PARAMS_PATH
from src.operational.config_v3 import OPS3_PARAMS_PATH, load_operational_v3

pytestmark = pytest.mark.unit


def _base() -> dict[str, object]:
    raw = yaml.safe_load(OPS3_PARAMS_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _write(tmp_path: Path, data: dict[str, object]) -> Path:
    (tmp_path / "params_operational.yaml").write_bytes(OPS_PARAMS_PATH.read_bytes())
    (tmp_path / "params_operational_v2.yaml").write_bytes(OPS2_PARAMS_PATH.read_bytes())
    p = tmp_path / "params_operational_v3.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_committed_v3_loads_and_pins_sha256() -> None:
    """params_operational_v3.yaml is byte-identical to pre-registration commit 578b342."""
    s = load_operational_v3()
    assert s.params.version == 3
    assert s.sha256 == "3c2cdd15a8cc80e3255fc234d36c6d4d234eb93b2e3f20c592e67594de5e1fda"
    assert s.v1.sha256 == s.params.inherits.v1_sha256
    assert s.v2.sha256 == s.params.inherits.v2_sha256
    assert s.params.cche1d.in_domain_model == "M2-XGB"
    assert s.params.cche1d.tail_model == "M1"


def test_inherited_digest_mismatch_rejected(tmp_path: Path) -> None:
    """Any mismatch in v1 or v2 SHA-256 is rejected."""
    data = _base()
    inh = data["inherits"]
    assert isinstance(inh, dict)
    inh["v2_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        load_operational_v3(_write(tmp_path, data))

    data2 = _base()
    inh2 = data2["inherits"]
    assert isinstance(inh2, dict)
    inh2["v1_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        load_operational_v3(_write(tmp_path, data2))


def test_duplicate_and_non_mapping_rejected(tmp_path: Path) -> None:
    """Duplicate YAML keys and non-mapping YAML documents are rejected."""
    p_dup = tmp_path / "dup.yaml"
    p_dup.write_text(
        OPS3_PARAMS_PATH.read_text(encoding="utf-8") + "\nversion: 3\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        load_operational_v3(p_dup)

    p_list = tmp_path / "list.yaml"
    p_list.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_operational_v3(p_list)


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        ("dynamic_roughness", "oracle_dyn_abs_bias_max_m", 0.60, "oracle_dyn_abs_bias_max_m"),
        ("cche1d", "splits", ["S-ID", "S-ID"], "unique"),
        ("cche1d", "tail_model", "M2-XGB", "distinct"),
        ("ablation_models", None, ["M5_A1", "M5_A1"], "unique"),
    ],
)
def test_v3_validators(
    tmp_path: Path, section: str, key: str | None, value: object, match: str
) -> None:
    """Cross-field validators reject invalid v3 configurations."""
    data = _base()
    if key is None:
        data[section] = value
    else:
        block = data[section]
        assert isinstance(block, dict)
        block[key] = value
    with pytest.raises(ValueError, match=match):
        load_operational_v3(_write(tmp_path, data))

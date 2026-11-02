"""Schema and loader for ``params_operational_v3.yaml`` (OPS-1 v3 diagnostics & CCHE1D benchmark).

Sources: docs/6_operational_diagnostics_v3.md sections 2-3. v3 inherits every rule from
``params_operational.yaml`` (v1) and ``params_operational_v2.yaml`` (v2); the loader verifies both
inherited files' SHA-256 digests so a v3 result can never be computed against edited v1/v2
pre-registrations.

Owning phase: OPS-1 v3 (post-release extension).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Final, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config import REPO_ROOT
from src.operational.config import ModelId, OperationalSettings, _reject_duplicate_keys
from src.operational.config_v2 import Operational2Settings, load_operational_v2

__all__ = [
    "ABLATION_IDS",
    "CCHE1D_SPLITS",
    "OPS3_PARAMS_PATH",
    "AblationId",
    "Cche1dParams",
    "Cche1dSplit",
    "DynamicRoughnessParams",
    "Inherit3Params",
    "Operational3Params",
    "Operational3Settings",
    "Scoring3Params",
    "load_operational_v3",
]

OPS3_PARAMS_PATH: Final[Path] = REPO_ROOT / "params_operational_v3.yaml"

AblationId = Literal["M5_A1", "M5_wA1", "M5_A2", "M5_A3-SP", "M5_A3-SR", "M5_A4"]
ABLATION_IDS: Final[tuple[str, ...]] = (
    "M5_A1",
    "M5_wA1",
    "M5_A2",
    "M5_A3-SP",
    "M5_A3-SR",
    "M5_A4",
)
Cche1dSplit = Literal["S-ID", "S-LRO", "S-TAIL-DA", "S-TAIL-SO"]
CCHE1D_SPLITS: Final[tuple[str, ...]] = ("S-ID", "S-LRO", "S-TAIL-DA", "S-TAIL-SO")

PosFloat = Annotated[float, Field(gt=0)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class Inherit3Params(_Frozen):
    """The v1 and v2 pre-registrations this file extends."""

    v1_file: str
    v1_sha256: Sha256
    v2_file: str
    v2_sha256: Sha256


class DynamicRoughnessParams(_Frozen):
    """Leopold-Maddock at-a-station dynamic resistance and Oracle thresholds (section 2.2, 3)."""

    min_abs_ln_q_ratio: PosFloat
    oracle_const_bias_min_m: PosFloat
    oracle_dyn_abs_bias_max_m: PosFloat

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if self.oracle_dyn_abs_bias_max_m >= self.oracle_const_bias_min_m:
            raise ValueError("oracle_dyn_abs_bias_max_m must be < oracle_const_bias_min_m")
        return self


class Cche1dParams(_Frozen):
    """CCHE1D cross-section continuity and domain-gated hybrid configuration (section 2.4)."""

    shape_ratio_floor: Annotated[float, Field(gt=1.0, le=1.5)]
    continuity_rel_tol: Annotated[float, Field(gt=0.0, lt=1.0)]
    splits: tuple[Cche1dSplit, ...] = Field(min_length=2)
    in_domain_model: ModelId
    tail_model: ModelId

    @model_validator(mode="after")
    def _splits_and_models(self) -> Self:
        if len(set(self.splits)) != len(self.splits):
            raise ValueError("cche1d.splits must be unique")
        if self.in_domain_model == self.tail_model:
            raise ValueError("in_domain_model and tail_model must be distinct")
        return self


class Scoring3Params(_Frozen):
    """Families and H-V3-1 thresholds (section 3)."""

    families: tuple[Literal["S-ID", "S-LRO"], ...] = Field(min_length=1)
    noninferiority_margin_m: PosFloat
    fs_penalty_min_m: PosFloat


class Operational3Params(_Frozen):
    """Every OPS-1 v3 pre-registered value; mirrors ``params_operational_v3.yaml``."""

    version: Literal[3]
    inherits: Inherit3Params
    dynamic_roughness: DynamicRoughnessParams
    cche1d: Cche1dParams
    ablation_models: tuple[AblationId, ...] = Field(min_length=1)
    scoring: Scoring3Params

    @model_validator(mode="after")
    def _unique_ablations(self) -> Self:
        if len(set(self.ablation_models)) != len(self.ablation_models):
            raise ValueError("ablation_models must be unique")
        return self


class Operational3Settings(BaseModel):
    """Validated v3 parameters, inherited v1 and v2 settings, and provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    params: Operational3Params
    v1: OperationalSettings
    v2: Operational2Settings
    path: Path
    sha256: Sha256


def load_operational_v3(path: Path | str = OPS3_PARAMS_PATH) -> Operational3Settings:
    """Load v3, then the inherited v2 and v1 files, and verify both SHA-256 digests.

    Raises
    ------
    ValueError
        Duplicate key, non-mapping document, failed validation, or an inherited file whose
        SHA-256 differs from the pre-registered value.
    """
    p = Path(path).expanduser().resolve()
    raw = p.read_bytes()
    _reject_duplicate_keys(yaml.compose(raw, Loader=yaml.SafeLoader), p.name)
    data: object = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a YAML mapping")
    params = Operational3Params.model_validate(data)
    v2 = load_operational_v2(p.parent / params.inherits.v2_file)
    if v2.sha256 != params.inherits.v2_sha256:
        raise ValueError(
            f"inherited {params.inherits.v2_file} SHA-256 {v2.sha256} != pre-registered "
            f"{params.inherits.v2_sha256}"
        )
    if v2.v1.sha256 != params.inherits.v1_sha256:
        raise ValueError(
            f"inherited {params.inherits.v1_file} SHA-256 {v2.v1.sha256} != pre-registered "
            f"{params.inherits.v1_sha256}"
        )
    return Operational3Settings(
        params=params,
        v1=v2.v1,
        v2=v2,
        path=p,
        sha256=hashlib.sha256(raw).hexdigest(),
    )

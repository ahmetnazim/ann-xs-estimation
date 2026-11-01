"""Schema and loader for ``params_operational_v2.yaml`` (OPS-1 v2 floodplain pre-registration).

Sources: docs/5_operational_floodplain_v2.md sections 2-3. v2 inherits every v1 rule from
``params_operational.yaml``; the loader verifies the inherited file's SHA-256 so a v2 result can
never be computed against an edited v1 pre-registration.

Owning phase: OPS-1 v2 (post-release extension).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Final, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config import REPO_ROOT
from src.operational.config import OperationalSettings, load_operational

__all__ = [
    "OPS2_PARAMS_PATH",
    "AcquisitionParams",
    "DatumParams",
    "InheritParams",
    "Operational2Params",
    "Operational2Settings",
    "Scoring2Params",
    "SectionParams",
    "TransectParams",
    "load_operational_v2",
]

OPS2_PARAMS_PATH: Final[Path] = REPO_ROOT / "params_operational_v2.yaml"

PosFloat = Annotated[float, Field(gt=0)]
PosInt = Annotated[int, Field(ge=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class InheritParams(_Frozen):
    """The v1 pre-registration this file extends."""

    file: str
    sha256: Sha256


class DatumParams(_Frozen):
    """Vertical datum handling (section 2)."""

    accepted: tuple[Literal["NAVD88", "NGVD29"], ...] = Field(min_length=1)
    ncat_url: str
    anchor_dem_max_offset_m: PosFloat


class TransectParams(_Frozen):
    """3DEP cross-valley transect geometry (section 2)."""

    service_url: str
    n_samples: PosInt
    length_factor_w: PosFloat
    length_min_m: PosFloat
    length_max_m: PosFloat
    offset_factor_w: PosFloat
    offset_min_m: PosFloat
    bearing_half_window_m: PosFloat
    channel_search_factor_w: PosFloat
    channel_search_min_m: PosFloat
    flat_tol_m: PosFloat
    max_flowline_offset_m: PosFloat
    max_nodata_share: Annotated[float, Field(ge=0, lt=1)]

    @model_validator(mode="after")
    def _lengths(self) -> Self:
        if self.length_min_m >= self.length_max_m:
            raise ValueError("length_min_m must be < length_max_m")
        return self


class AcquisitionParams(_Frozen):
    """Network politeness and robustness."""

    workers: PosInt
    timeout_s: PosFloat
    retries: PosInt


class SectionParams(_Frozen):
    """Composite-section discretization (section 2)."""

    channel_points: PosInt
    z_levels: Annotated[int, Field(ge=10)]


class Scoring2Params(_Frozen):
    """Variants, families and the H-FP2 margin (section 3)."""

    variants: tuple[Literal["control", "floodplain"], ...] = Field(min_length=2, max_length=2)
    families: tuple[Literal["S-ID", "S-LRO"], ...] = Field(min_length=1)
    noninferiority_margin_m: PosFloat

    @model_validator(mode="after")
    def _variants(self) -> Self:
        if set(self.variants) != {"control", "floodplain"}:
            raise ValueError("variants must be control and floodplain")
        return self


class Operational2Params(_Frozen):
    """Every OPS-1 v2 pre-registered value; mirrors ``params_operational_v2.yaml``."""

    version: Literal[2]
    inherits: InheritParams
    datum: DatumParams
    transect: TransectParams
    acquisition: AcquisitionParams
    section: SectionParams
    scoring: Scoring2Params


class Operational2Settings(BaseModel):
    """Validated v2 parameters, the inherited v1 settings, and provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    params: Operational2Params
    v1: OperationalSettings
    path: Path
    sha256: Sha256


def load_operational_v2(path: Path | str = OPS2_PARAMS_PATH) -> Operational2Settings:
    """Load v2, then the inherited v1 file, and verify the v1 SHA-256.

    Raises
    ------
    ValueError
        Non-mapping document, failed validation, or an inherited file whose SHA-256 differs from
        the pre-registered value.
    """
    p = Path(path).expanduser().resolve()
    raw = p.read_bytes()
    data: object = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a YAML mapping")
    params = Operational2Params.model_validate(data)
    v1 = load_operational(p.parent / params.inherits.file)
    if v1.sha256 != params.inherits.sha256:
        raise ValueError(
            f"inherited {params.inherits.file} SHA-256 {v1.sha256} != pre-registered "
            f"{params.inherits.sha256}"
        )
    return Operational2Settings(
        params=params, v1=v1, path=p, sha256=hashlib.sha256(raw).hexdigest()
    )

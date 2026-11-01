"""Schema and loader for ``params_operational.yaml`` (OPS-1 pre-registration).

Sources: docs/4_operational_stage_response.md sections 3.1-3.6. Frozen, closed to unknown keys,
duplicate YAML keys rejected (same rules as :func:`src.config.load_settings`). Every OPS-1 output
records :attr:`OperationalSettings.sha256` of the exact file bytes.

Owning phase: OPS-1 (post-release extension).
"""

from __future__ import annotations

import hashlib
import math
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Final, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config import REPO_ROOT

__all__ = [
    "MODEL_IDS",
    "OPS_PARAMS_PATH",
    "HypothesesParams",
    "MetricsParams",
    "ObservationParams",
    "OperationalParams",
    "OperationalSettings",
    "PredictionParams",
    "RatingParams",
    "SensitivityParams",
    "load_operational",
]

OPS_PARAMS_PATH: Final[Path] = REPO_ROOT / "params_operational.yaml"

ModelId = Literal["M-null", "M0", "M1", "M2-MLP", "M2-XGB", "M3", "M4", "M5", "M6-NN", "M6-XGB"]
MODEL_IDS: Final[tuple[str, ...]] = (
    "M-null",
    "M0",
    "M1",
    "M2-MLP",
    "M2-XGB",
    "M3",
    "M4",
    "M5",
    "M6-NN",
    "M6-XGB",
)
Family = Literal["S-ID", "S-LRO"]
PosFloat = Annotated[float, Field(gt=0)]
PosInt = Annotated[int, Field(ge=1)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class PredictionParams(_Frozen):
    """Which frozen predictions are scored (section 3.1)."""

    analysis: Literal["primary"]
    primary_family: Family
    families: tuple[Family, ...] = Field(min_length=1)
    models: tuple[ModelId, ...] = Field(min_length=2)
    fallback_shape_model: ModelId
    reference_model: ModelId

    @model_validator(mode="after")
    def _members(self) -> Self:
        if self.primary_family not in self.families:
            raise ValueError("primary_family must be listed in families")
        if len(set(self.models)) != len(self.models):
            raise ValueError("models must be unique")
        if self.reference_model not in self.models:
            raise ValueError("reference_model must be listed in models")
        return self


class ObservationParams(_Frozen):
    """Observation exclusions (section 3.2)."""

    exclude_ad_site_cd: tuple[str, ...]
    channel_number: PosInt
    window_years: PosFloat
    max_abs_dh_m: PosFloat


class RatingParams(_Frozen):
    """Root-finding controls for OPS-E1 (section 2)."""

    newton_tol: PosFloat
    max_iter: PosInt
    chunk_rows: PosInt


class MetricsParams(_Frozen):
    """Flow bins and contrasts (section 3.4)."""

    flow_bin_edges: tuple[float, ...] = Field(min_length=2)
    high_flow_ratio: PosFloat
    contrasts: tuple[tuple[ModelId, ModelId], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _edges(self) -> Self:
        e = self.flow_bin_edges
        if e[0] != 0.0 or not math.isinf(e[-1]):
            raise ValueError("flow_bin_edges must start at 0 and end at +inf")
        if any(b <= a for a, b in pairwise(e)):
            raise ValueError("flow_bin_edges must be strictly increasing")
        for a, b in self.contrasts:
            if a == b:
                raise ValueError(f"contrast {a} - {b} compares a model with itself")
        return self


class HypothesesParams(_Frozen):
    """Decision thresholds (section 3.5)."""

    noninferiority_margin_m: PosFloat


class SensitivityParams(_Frozen):
    """Pre-registered sensitivity runs (section 3.2)."""

    window_years_none: bool


class OperationalParams(_Frozen):
    """Every OPS-1 pre-registered value; mirrors ``params_operational.yaml`` exactly."""

    version: Literal[1]
    predictions: PredictionParams
    observations: ObservationParams
    rating: RatingParams
    metrics: MetricsParams
    hypotheses: HypothesesParams
    sensitivity: SensitivityParams

    @model_validator(mode="after")
    def _cross(self) -> Self:
        models = set(self.predictions.models)
        for a, b in self.metrics.contrasts:
            if a not in models or b not in models:
                raise ValueError(f"contrast {a} - {b} uses a model outside predictions.models")
        if self.predictions.fallback_shape_model not in models:
            raise ValueError("fallback_shape_model must be listed in models")
        return self


class OperationalSettings(BaseModel):
    """Validated OPS-1 parameters plus the provenance of the file they came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    params: OperationalParams
    path: Path
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def _reject_duplicate_keys(node: yaml.Node | None, where: str) -> None:
    if isinstance(node, yaml.MappingNode):
        seen: set[str] = set()
        for key_node, value_node in node.value:
            key = str(key_node.value)
            if key in seen:
                raise ValueError(f"duplicate key {key!r} in {where}")
            seen.add(key)
            _reject_duplicate_keys(value_node, f"{where}:{key}")
    elif isinstance(node, yaml.SequenceNode):
        for index, item in enumerate(node.value):
            _reject_duplicate_keys(item, f"{where}[{index}]")


def load_operational(path: Path | str = OPS_PARAMS_PATH) -> OperationalSettings:
    """Load and validate ``params_operational.yaml``.

    Raises
    ------
    ValueError
        On a duplicate key, a non-mapping document or any failed validator.
    """
    p = Path(path).expanduser().resolve()
    raw = p.read_bytes()
    _reject_duplicate_keys(yaml.compose(raw, Loader=yaml.SafeLoader), p.name)
    data: object = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a YAML mapping")
    return OperationalSettings(
        params=OperationalParams.model_validate(data),
        path=p,
        sha256=hashlib.sha256(raw).hexdigest(),
    )

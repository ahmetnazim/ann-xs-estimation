"""Type-safe configuration: pre-registered protocol values and runtime settings.

Two layers (P4-D4 (a); ADL-052):

``ProtocolParams``
    Pre-registered scientific values from docs/3_experimental_protocol.md (``a98feff``).
    Loaded only from the root ``params.yaml``; frozen; unknown keys rejected; never read
    from environment variables. Changing any value requires a new ADL row.
``RuntimeSettings``
    Operational settings only (paths, workers, threads, device, dtype, log level, DVC
    store), read from ``ANNXS_*`` environment variables and an optional ``.env`` file.

Design rules: importing this module has no side effects (directories are created only by
:meth:`Settings.ensure_dirs`); torch is never imported (device and dtype are validated as
strings and fixed to ``cpu``/``float64`` for every NN model, O-8, because PyTorch MPS has no
float64); every run logs :attr:`Settings.params_sha256`, the SHA-256 of the exact
``params.yaml`` bytes.

Owning phase: 4 (scaffolding). Consumers: Phases 5-8.
"""

from __future__ import annotations

import hashlib
import math
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Final, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "ACQUIRE_UNITS",
    "PARAMS_PATH",
    "REPO_ROOT",
    "ProtocolParams",
    "RuntimeSettings",
    "Settings",
    "load_settings",
]

# ----------------------------------------------------------------------------- constants
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PARAMS_PATH: Final[Path] = REPO_ROOT / "params.yaml"

FT_TO_M: Final[float] = 0.3048  # exact international foot (protocol 2.1; proofs 1.2)
R_MIN_BOUNDARY: Final[float] = 0.5  # boundary case of Proposition 2.6; O-9
S_FLOOR_DATA: Final[float] = 1.0e-5  # NHDPlusV2 slope floor, a data property (A1.1; 2.2)
B_J_MAX: Final[float] = 5.0 / 3.0  # Proposition 7.5: b_J < 5/3 gives a unique Jarrett root

#: The dvc.yaml ``acquire`` foreach items, in order (protocol 1.1, 1.5; ADL-043).
ACQUIRE_UNITS: Final[tuple[str, ...]] = (
    "hydroswot",
    "nhdplusv21",
    "streamcat",
    "usgs_attrs",
    "d50",
    "aridity",
    "physiography",
    "ehydro",
    "ehydro_author",
    "zarrabi_product",
    "hyg",
)

_DRIVE_MARKERS: Final[tuple[str, ...]] = ("CloudStorage/GoogleDrive", "Google Drive")

# ----------------------------------------------------------------------------- types
PosFloat = Annotated[float, Field(gt=0)]
NonNegFloat = Annotated[float, Field(ge=0)]
OpenUnit = Annotated[float, Field(gt=0, lt=1)]
UnitInterval = Annotated[float, Field(gt=0, le=1)]
PosInt = Annotated[int, Field(ge=1)]
Seed = Annotated[int, Field(ge=0, lt=2**32)]
Md5Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def _increasing(values: tuple[float, ...], name: str) -> None:
    """Raise unless ``values`` is strictly increasing."""
    if any(b <= a for a, b in pairwise(values)):
        raise ValueError(f"{name} must be strictly increasing, got {values}")


class _Frozen(BaseModel):
    """Base of every protocol group: immutable and closed to unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


# ----------------------------------------------------------------------------- protocol 1.5
class SeedsParams(_Frozen):
    """Random seeds (protocol 1.5, 3.2, 4.4)."""

    global_seed: Seed = Field(alias="global")
    sid_repeats: tuple[Seed, ...] = Field(min_length=1)

    @field_validator("sid_repeats")
    @classmethod
    def _unique(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(value)) != len(value):
            raise ValueError("sid_repeats must be unique")
        return value


class AcquireUnit(_Frozen):
    """One DVC acquisition unit (protocol 1.1, 1.5); filled in Phase 5."""

    urls: tuple[HttpUrl, ...] = ()
    md5: tuple[Md5Hex, ...] | None = None
    sha256_pinned: Sha256Hex | None = None

    @model_validator(mode="after")
    def _one_md5_per_url(self) -> Self:
        if self.md5 is not None and len(self.md5) != len(self.urls):
            raise ValueError("md5 must list exactly one checksum per URL")
        return self


class AcquireParams(_Frozen):
    """All eleven acquisition units; field names equal the dvc.yaml foreach list."""

    hydroswot: AcquireUnit
    nhdplusv21: AcquireUnit
    streamcat: AcquireUnit
    usgs_attrs: AcquireUnit
    d50: AcquireUnit
    aridity: AcquireUnit
    physiography: AcquireUnit
    ehydro: AcquireUnit
    ehydro_author: AcquireUnit
    zarrabi_product: AcquireUnit
    hyg: AcquireUnit


class QcParams(_Frozen):
    """QC tolerances and gross-error bounds (protocol 2.2, 2.4)."""

    q_tol: OpenUnit
    eps_a_tol: OpenUnit = Field(alias="epsA_tol")
    w_max_m: PosFloat = Field(alias="W_max_m")
    dbar_max_m: PosFloat
    dmax_max_m: PosFloat
    v_max_ms: PosFloat = Field(alias="V_max_ms")

    @model_validator(mode="after")
    def _dmax_bound_covers_dbar(self) -> Self:
        if self.dmax_max_m < self.dbar_max_m:
            raise ValueError("dmax_max_m must be >= dbar_max_m (invariant 4)")
        return self

    @property
    def eps_a_log_tol(self) -> float:
        """Return the R8 closure tolerance ln(1 + epsA_tol) (= ln 1.05)."""
        return math.log1p(self.eps_a_tol)


class LinkParams(_Frozen):
    """Gage-reach link settings (protocol 2.5)."""

    snap_max_m: PosFloat
    da_ratio_max: float = Field(gt=1)

    @property
    def da_log_tol(self) -> float:
        """Return the drainage-area tolerance ln(da_ratio_max) (= ln 1.25)."""
        return math.log(self.da_ratio_max)


class ConditionParams(_Frozen):
    """Mean-flow condition extraction (protocol 2.6)."""

    qsel_ratio_flag: float = Field(gt=1)
    tie_break: Literal["latest_date"]


class DomainParams(_Frozen):
    """Admissible domain and transform bounds (protocol 2.2; proofs 1.3, 7.6)."""

    n_bounds: tuple[PosFloat, PosFloat]
    n_lit: tuple[PosFloat, PosFloat]
    r_bounds: tuple[PosFloat, PosFloat]
    s_floor: PosFloat

    @model_validator(mode="after")
    def _check(self) -> Self:
        n_min, n_max = self.n_bounds
        lit_min, lit_max = self.n_lit
        r_min, r_max = self.r_bounds
        if not 0 < n_min < lit_min < lit_max < n_max:
            raise ValueError("require 0 < n_min < n_lit_min < n_lit_max < n_max (protocol 2.2)")
        if r_min < R_MIN_BOUNDARY:
            raise ValueError("r_min must be >= 0.5 (Proposition 2.6 boundary case; O-9)")
        if not r_min < r_max:
            raise ValueError("r_bounds must be strictly increasing")
        if self.s_floor != S_FLOOR_DATA:
            raise ValueError("s_floor is a data property and must equal 1e-5 (A1.1)")
        return self

    @property
    def delta_n(self) -> float:
        """Return Delta_n = ln(n_max / n_min) (= ln 120)."""
        return math.log(self.n_bounds[1] / self.n_bounds[0])

    @property
    def delta_r(self) -> float:
        """Return Delta_r = ln(r_max / r_min)."""
        return math.log(self.r_bounds[1] / self.r_bounds[0])

    @property
    def nu_bounds(self) -> tuple[float, float]:
        """Return (nu_min, nu_max) = (ln n_min, ln n_max)."""
        return (math.log(self.n_bounds[0]), math.log(self.n_bounds[1]))

    @property
    def t_band(self) -> tuple[float, float]:
        """Return the shape-target band [ln(1 + 1/r_max), ln(1 + 1/r_min)]."""
        return (math.log1p(1 / self.r_bounds[1]), math.log1p(1 / self.r_bounds[0]))


class SplitsParams(_Frozen):
    """Split families and the fold table (protocol 3.2-3.4)."""

    sid_k: int = Field(ge=2)
    sid_r: PosInt
    inner_k: int = Field(ge=2)
    es_frac: OpenUnit
    lro_min_groups: PosInt
    tail_deciles: tuple[Annotated[int, Field(ge=1, le=10)], ...] = Field(min_length=1)
    tail_order_min: PosInt
    ind_dedupe_m: PosFloat

    @field_validator("tail_deciles")
    @classmethod
    def _unique_deciles(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(value)) != len(value):
            raise ValueError("tail_deciles must be unique")
        return value


class AuditParams(_Frozen):
    """Spatial and network dependence audit; diagnostic only (protocol 3.6)."""

    knn_k: PosInt
    idw_band_km: PosFloat
    permutations: PosInt
    lisa_alpha: OpenUnit
    dist_classes_km: tuple[PosFloat, ...] = Field(min_length=1)
    vgm_bins: int = Field(ge=2)
    vgm_max_km: PosFloat
    rho_cap_km: PosFloat
    moran_material: PosFloat
    nested_psi: UnitInterval
    nested_share_material: OpenUnit

    @model_validator(mode="after")
    def _check(self) -> Self:
        _increasing(self.dist_classes_km, "dist_classes_km")
        if self.vgm_max_km < self.rho_cap_km:
            raise ValueError("vgm_max_km must be >= rho_cap_km")
        return self


class AuditRulesParams(_Frozen):
    """Decision constants of the dependence audit (protocol 3.6.4, 3.6.6, 3.6.7)."""

    vgm_models: tuple[Literal["spherical", "exponential", "nugget"], ...] = Field(min_length=1)
    nugget_sse_margin: OpenUnit
    range_sill_frac: OpenUnit
    moran_pseudo_p: OpenUnit
    aoa_iqr_mult: PosFloat

    @field_validator("vgm_models")
    @classmethod
    def _unique_models(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("vgm_models must be unique")
        return value


# ----------------------------------------------------------------------------- protocol 4.x
class BackboneParams(_Frozen):
    """Shared MLP backbone (protocol 4.2, P3-17)."""

    hidden_layers: PosInt
    hidden_width: PosInt
    activation: Literal["silu"]
    #: Row-wise normalization only: Proposition 3.4 forbids batch coupling upstream of Phi.
    norm: Literal["layernorm"]


class OptimizerParams(_Frozen):
    """Optimizer and early stopping (protocol 4.2)."""

    name: Literal["adamw"]
    lr: PosFloat
    schedule: Literal["cosine"]
    batch_size: PosInt
    max_epochs: PosInt
    patience: PosInt

    @model_validator(mode="after")
    def _patience_within_budget(self) -> Self:
        if self.patience > self.max_epochs:
            raise ValueError("patience must not exceed max_epochs")
        return self


class InnerGridParams(_Frozen):
    """Nested inner-CV grid (protocol 4.1, 4.2)."""

    weight_decay: tuple[NonNegFloat, ...] = Field(min_length=1)
    lr: tuple[PosFloat, ...] = Field(min_length=1)
    m4_lambda_max: tuple[PosFloat, ...] = Field(min_length=1)


class LossWeightParams(_Frozen):
    """Data-loss weights (protocol 4.2, P3-06)."""

    w_w: NonNegFloat = Field(alias="w_W")
    w_dbar: NonNegFloat
    w_t: NonNegFloat
    w_a: NonNegFloat = Field(alias="w_A")


class M4ScheduleParams(_Frozen):
    """Capped M4 penalty schedule lambda_M(e) = min(lambda0 * gamma**e, lambda_max)."""

    lambda0: PosFloat
    gamma: float = Field(gt=1)


class XgbParams(_Frozen):
    """XGBoost grid for M1, M2-XGB and M6-XGB (protocol 4.2)."""

    max_depth: tuple[PosInt, ...] = Field(min_length=1)
    eta: tuple[OpenUnit, ...] = Field(min_length=1)
    min_child_weight: tuple[PosFloat, ...] = Field(min_length=1)
    subsample: UnitInterval
    colsample: UnitInterval
    max_rounds: PosInt
    early_stopping_rounds: PosInt


class BaselineParams(_Frozen):
    """Baseline fallbacks (protocol 4.1)."""

    m0_min_division_sites: PosInt
    m1_floor_quantile: OpenUnit


class ModelParams(_Frozen):
    """Model configuration shared across the ladder (protocol 4.1, 4.2)."""

    backbone: BackboneParams
    optimizer: OptimizerParams
    inner_grid: InnerGridParams
    loss_weights: LossWeightParams
    m4_schedule: M4ScheduleParams
    xgb: XgbParams
    baselines: BaselineParams


class SolverParams(_Frozen):
    """Phi solver and quadrature (protocol 4.2, P3-12; proofs 3.3-3.6)."""

    tau_l: PosFloat
    k_max: PosInt
    tau_q: PosFloat
    #: Per-row freezing keeps the computed iterate row-wise (Proposition 3.4 (c)).
    stopping: Literal["per_row"]
    quadrature: Literal["tanh_sinh"]


class UqParams(_Frozen):
    """Deep-ensemble UQ (protocol 4.2, 4.6; P3-13, P3-18)."""

    ensemble_m: int = Field(ge=2)
    sigma_min: PosFloat
    beta_nll: float = Field(ge=0, le=1)
    warm_start_epochs: int = Field(ge=0)
    alpha: OpenUnit
    coverage_band: tuple[OpenUnit, OpenUnit]
    correlation_head: Literal[False]
    conformal: Literal[False]

    @model_validator(mode="after")
    def _band_contains_nominal(self) -> Self:
        lo, hi = self.coverage_band
        if not lo <= 1 - self.alpha <= hi:
            raise ValueError("coverage_band must contain the nominal level 1 - alpha")
        return self


class StricklerParams(_Frozen):
    """Strickler coefficient (protocol 4.1, P3-07, O-3)."""

    c_s: PosFloat


class JarrettParams(_Frozen):
    """Jarrett (1985) roughness for the M3 high-gradient branch (protocol 4.1, P3-08)."""

    c_j_us: PosFloat
    c_j_si: PosFloat
    a_j: PosFloat
    b_j: float
    s_j: PosFloat
    validity_s: tuple[PosFloat, PosFloat]
    validity_r_ft: tuple[PosFloat, PosFloat]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.b_j < B_J_MAX:
            raise ValueError("b_j must be < 5/3 (Proposition 7.5)")
        # R_ft = R_m / 0.3048, so c_SI = c_US * 0.3048**(-b_j), rounded as in protocol 4.1.
        if round(self.c_j_us * FT_TO_M ** (-self.b_j), 4) != self.c_j_si:
            raise ValueError("c_j_si must equal round(c_j_us * 0.3048**(-b_j), 4)")
        _increasing(self.validity_s, "validity_s")
        _increasing(self.validity_r_ft, "validity_r_ft")
        return self


class StricklerPriorParams(_Frozen):
    """Ablation A3-SP: Gaussian prior on nu around nu_S (proofs 7.5 (a))."""

    tau_p: PosFloat
    lambda_p: PosFloat

    @model_validator(mode="after")
    def _lambda_from_tau(self) -> Self:
        if not math.isclose(self.lambda_p, 1 / (2 * self.tau_p**2)):
            raise ValueError("lambda_p must equal 1 / (2 tau_p**2)")
        return self


class StricklerResidualParams(_Frozen):
    """Ablation A3-SR: logit-domain residual around Strickler (proofs 7.5 (b))."""

    varpi: float = Field(gt=0, lt=0.5)
    lambda_delta: PosFloat


class PriorsParams(_Frozen):
    """Empirical roughness priors (protocol 4.1)."""

    strickler: StricklerParams
    jarrett: JarrettParams
    strickler_prior: StricklerPriorParams
    strickler_residual: StricklerResidualParams


class BootstrapParams(_Frozen):
    """Paired cluster bootstrap (protocol 4.4)."""

    b: PosInt
    alpha_b: OpenUnit
    seed: Seed
    stratify: Literal["division"]


class S2Params(_Frozen):
    """Secondary contrasts with Bonferroni intervals (protocol 4.5, O-7)."""

    n_contrasts: PosInt
    alpha: OpenUnit

    @property
    def level(self) -> float:
        """Return the simultaneous confidence level 1 - alpha / n_contrasts."""
        return 1 - self.alpha / self.n_contrasts


class H4Params(_Frozen):
    """H4 falsification thresholds (protocol 4.4, P3-16)."""

    vartheta: OpenUnit
    pi_boundary_max: OpenUnit
    pi_lit_min: OpenUnit


class EvaluationParams(_Frozen):
    """Pre-registered statistics (protocol 4.4-4.7)."""

    mstar_candidates: tuple[Literal["M1", "M2-MLP", "M2-XGB"], ...] = Field(min_length=1)
    bootstrap: BootstrapParams
    s2: S2Params
    h4: H4Params
    no_loss_margin_log: PosFloat


class ComputeParams(_Frozen):
    """Compute-budget trigger (protocol 4.8; ADL-048)."""

    pilot_max_fit_s: PosFloat


class ProtocolParams(_Frozen):
    """Every pre-registered value; mirrors ``params.yaml`` exactly (top-level keys included)."""

    seeds: SeedsParams
    acquire: AcquireParams
    qc: QcParams
    link: LinkParams
    condition: ConditionParams
    domain: DomainParams
    splits: SplitsParams
    audit: AuditParams
    audit_rules: AuditRulesParams
    model: ModelParams
    solver: SolverParams
    uq: UqParams
    priors: PriorsParams
    evaluation: EvaluationParams
    compute: ComputeParams

    @model_validator(mode="after")
    def _cross_group(self) -> Self:
        if len(self.seeds.sid_repeats) != self.splits.sid_r:
            raise ValueError("len(seeds.sid_repeats) must equal splits.sid_r")
        # Proposition 7.2 match at pi_S = 1/2: lambda_delta = lambda_p * Delta_n**2 / 16,
        # recorded to three decimals in protocol 4.1.
        expected = self.priors.strickler_prior.lambda_p * self.domain.delta_n**2 / 16
        if not math.isclose(self.priors.strickler_residual.lambda_delta, expected, abs_tol=5e-4):
            raise ValueError("lambda_delta must equal lambda_p * Delta_n**2 / 16 (3 d.p.)")
        return self


# ----------------------------------------------------------------------------- runtime
class RuntimeSettings(BaseSettings):
    """Operational settings, overridable through ``ANNXS_*`` variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="ANNXS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    repo_root: Path = REPO_ROOT
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    checkpoint_dir: Path = Path("data/checkpoints")
    reports_dir: Path = Path("reports")
    dvc_store: Path = Path("~/dvc_store/ANN_XS_Estimation")
    #: O-8: every NN model runs on the CPU in float64 (PyTorch MPS has no float64).
    device: Literal["cpu"] = "cpu"
    dtype: Literal["float64"] = "float64"
    n_workers: PosInt = 10
    torch_threads_per_worker: PosInt = 1
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("repo_root", mode="after")
    @classmethod
    def _resolve_root(cls, value: Path) -> Path:
        root = value.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"repo_root is not a directory: {root}")
        return root

    @field_validator("raw_dir", "processed_dir", "checkpoint_dir", "reports_dir", mode="after")
    @classmethod
    def _inside_repo(cls, value: Path, info: ValidationInfo) -> Path:
        root = info.data.get("repo_root")
        if not isinstance(root, Path):
            raise ValueError("repo_root failed validation")
        path = value.expanduser()
        path = (path if path.is_absolute() else root / path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"{info.field_name} must lie inside the repository: {path}")
        return path

    @field_validator("dvc_store", mode="after")
    @classmethod
    def _outside_drive(cls, value: Path, info: ValidationInfo) -> Path:
        store = value.expanduser().resolve()
        root = info.data.get("repo_root")
        if isinstance(root, Path) and store.is_relative_to(root):
            raise ValueError("dvc_store must lie outside the repository (P3-D2)")
        if any(marker in store.as_posix() for marker in _DRIVE_MARKERS):
            raise ValueError("dvc_store must lie outside Google Drive (P3-D2; O-13)")
        return store


# ----------------------------------------------------------------------------- composite
class Settings(BaseModel):
    """Validated configuration of one run: protocol values, runtime settings, provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol: ProtocolParams
    runtime: RuntimeSettings
    params_path: Path
    params_sha256: Sha256Hex

    def ensure_dirs(self, *, include_dvc_store: bool = False) -> tuple[Path, ...]:
        """Create the runtime directories explicitly and return those that were created.

        ``data/raw`` itself may be created, never ``data/raw/<unit>``: those are DVC stage
        outputs (ADL-043). With ``include_dvc_store`` the external ``cache``, ``remote`` and
        ``inbox`` directories are created as well (Phase 5).
        """
        rt = self.runtime
        targets = [rt.raw_dir, rt.processed_dir, rt.checkpoint_dir, rt.reports_dir]
        if include_dvc_store:
            targets += [rt.dvc_store / name for name in ("cache", "remote", "inbox")]
        created: list[Path] = []
        for path in targets:
            if not path.exists():
                path.mkdir(parents=True)
                created.append(path)
        return tuple(created)


def _reject_duplicate_keys(node: yaml.Node | None, where: str = "params.yaml") -> None:
    """Raise if a YAML mapping repeats a key (PyYAML would silently keep the last one)."""
    if isinstance(node, yaml.MappingNode):
        seen: set[str] = set()
        for key_node, value_node in node.value:
            key = str(key_node.value)
            if key in seen:
                raise ValueError(f"duplicate key {key!r} at {where}")
            seen.add(key)
            _reject_duplicate_keys(value_node, f"{where}:{key}")
    elif isinstance(node, yaml.SequenceNode):
        for index, item in enumerate(node.value):
            _reject_duplicate_keys(item, f"{where}[{index}]")


def load_settings(
    params_path: Path | str = PARAMS_PATH,
    *,
    runtime: RuntimeSettings | None = None,
) -> Settings:
    """Load ``params.yaml`` and the runtime settings into one validated, frozen object.

    Parameters
    ----------
    params_path
        Pre-registration file. Defaults to the ``params.yaml`` of this checkout and never
        depends on an environment variable.
    runtime
        Pre-built runtime settings (used by tests); by default read from ``ANNXS_*``.

    Returns
    -------
    Settings
        Frozen settings carrying ``params_sha256`` of the exact file bytes.

    Raises
    ------
    ValueError
        On a duplicate YAML key, a non-mapping document or any failed validator.
    """
    path = Path(params_path).expanduser().resolve()
    raw = path.read_bytes()
    _reject_duplicate_keys(yaml.compose(raw, Loader=yaml.SafeLoader))
    data: object = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return Settings(
        protocol=ProtocolParams.model_validate(data),
        runtime=runtime if runtime is not None else RuntimeSettings(),
        params_path=path,
        params_sha256=hashlib.sha256(raw).hexdigest(),
    )

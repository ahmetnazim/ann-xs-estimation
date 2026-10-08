"""Unit tests for src/config.py (Phase 4 gate item 2).

Every pre-registered value is compared with docs/3_experimental_protocol.md (a98feff); the
validators must reject bad inputs; environment variables must never reach ProtocolParams.
"""

from __future__ import annotations

import copy
import hashlib
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from src.config import (
    ACQUIRE_UNITS,
    PARAMS_PATH,
    REPO_ROOT,
    ProtocolParams,
    RuntimeSettings,
    load_settings,
)

pytestmark = pytest.mark.unit

_UNIT = {"urls": [], "md5": None, "sha256_pinned": None}

# Transcribed from docs/3_experimental_protocol.md (a98feff); section numbers in comments.
EXPECTED: dict[str, dict[str, Any]] = {
    "seeds": {"global": 20261008, "sid_repeats": [20261008, 20261009, 20261010]},  # 1.5, 3.2
    "acquire": dict.fromkeys(ACQUIRE_UNITS, _UNIT),  # 1.5 (filled in Phase 5)
    "qc": {  # 1.5, 2.2, 2.4
        "q_tol": 0.05,
        "epsA_tol": 0.05,
        "W_max_m": 5000,
        "dbar_max_m": 50,
        "dmax_max_m": 70,
        "V_max_ms": 6,
    },
    "link": {"snap_max_m": 500, "da_ratio_max": 1.25},  # 1.5, 2.5
    "condition": {"qsel_ratio_flag": 2.0, "tie_break": "latest_date"},  # 1.5, 2.6
    "domain": {  # 1.5, 2.2
        "n_bounds": [0.005, 0.6],
        "n_lit": [0.012, 0.33],
        "r_bounds": [0.5, 10],
        "s_floor": 1e-5,
    },
    "splits": {  # 1.5, 3.2-3.5
        "sid_k": 5,
        "sid_r": 3,
        "inner_k": 3,
        "es_frac": 0.10,
        "lro_min_groups": 50,
        "tail_deciles": [1, 10],
        "tail_order_min": 7,
        "ind_dedupe_m": 1000,
    },
    "audit": {  # 1.5, 3.6
        "knn_k": 8,
        "idw_band_km": 100,
        "permutations": 999,
        "lisa_alpha": 0.01,
        "dist_classes_km": [10, 25, 50, 100, 250, 500],
        "vgm_bins": 20,
        "vgm_max_km": 500,
        "rho_cap_km": 250,
        "moran_material": 0.10,
        "nested_psi": 0.5,
        "nested_share_material": 0.20,
    },
    "audit_rules": {  # 3.6.4, 3.6.6, 3.6.7
        "vgm_models": ["spherical", "exponential", "nugget"],
        "nugget_sse_margin": 0.05,
        "range_sill_frac": 0.95,
        "moran_pseudo_p": 0.01,
        "aoa_iqr_mult": 1.5,
    },
    "model": {  # 4.1, 4.2
        "backbone": {
            "hidden_layers": 3,
            "hidden_width": 128,
            "activation": "silu",
            "norm": "layernorm",
        },
        "optimizer": {
            "name": "adamw",
            "lr": 1e-3,
            "schedule": "cosine",
            "batch_size": 256,
            "max_epochs": 500,
            "patience": 50,
        },
        "inner_grid": {
            "weight_decay": [0, 1e-5, 1e-4, 1e-3],
            "lr": [1e-3, 3e-4],
            "m4_lambda_max": [10, 1e2, 1e3],
        },
        "loss_weights": {"w_W": 1, "w_dbar": 1, "w_t": 1, "w_A": 0},
        "m4_schedule": {"lambda0": 1, "gamma": 1.1},
        "xgb": {
            "max_depth": [4, 6, 8],
            "eta": [0.03, 0.1],
            "min_child_weight": [1, 5],
            "subsample": 0.8,
            "colsample": 0.8,
            "max_rounds": 2000,
            "early_stopping_rounds": 100,
        },
        "baselines": {"m0_min_division_sites": 30, "m1_floor_quantile": 0.001},
    },
    "solver": {  # 4.2 (P3-12)
        "tau_l": 1e-10,
        "k_max": 40,
        "tau_q": 1e-8,
        "stopping": "per_row",
        "quadrature": "tanh_sinh",
    },
    "uq": {  # 4.2 (P3-13), 4.6
        "ensemble_m": 10,
        "sigma_min": 0.01,
        "beta_nll": 0.5,
        "warm_start_epochs": 20,
        "alpha": 0.10,
        "coverage_band": [0.85, 0.95],
        "correlation_head": False,
        "conformal": False,
    },
    "priors": {  # 4.1 (P3-07, P3-08)
        "strickler": {"c_s": 0.047},
        "jarrett": {
            "c_j_us": 0.39,
            "c_j_si": 0.3225,
            "a_j": 0.38,
            "b_j": -0.16,
            "s_j": 0.002,
            "validity_s": [0.002, 0.04],
            "validity_r_ft": [0.5, 7],
        },
        "strickler_prior": {"tau_p": 0.5, "lambda_p": 2},
        "strickler_residual": {"varpi": 0.01, "lambda_delta": 2.865},
    },
    "evaluation": {  # 4.4-4.7
        "mstar_candidates": ["M1", "M2-MLP", "M2-XGB"],
        "bootstrap": {"b": 10000, "alpha_b": 0.05, "seed": 20261008, "stratify": "division"},
        "s2": {"n_contrasts": 4, "alpha": 0.05},
        "h4": {"vartheta": 0.05, "pi_boundary_max": 0.10, "pi_lit_min": 0.80},
        "no_loss_margin_log": 0.05,
    },
    "compute": {"pilot_max_fit_s": 120},  # 4.8
}


# ----------------------------------------------------------------------------- helpers
def _get(data: dict[str, Any], dotted: str) -> Any:
    node: Any = data
    for part in dotted.split("."):
        node = node[part]
    return node


def _set(data: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node: Any = data
    for part in parents:
        node = node[part]
    node[leaf] = value


def _clean_runtime(**overrides: Any) -> RuntimeSettings:
    return RuntimeSettings(_env_file=None, **overrides)


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def raw_params() -> dict[str, Any]:
    return yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dvc_stages() -> dict[str, Any]:
    return yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]


# ----------------------------------------------------------------------------- structure
def test_params_file_is_repo_root_params_yaml():
    assert PARAMS_PATH == REPO_ROOT / "params.yaml"
    assert PARAMS_PATH.is_file()


def test_top_level_keys_match_expected_order(raw_params):
    assert list(raw_params) == list(EXPECTED)


def test_dvc_yaml_params_resolve_in_params_yaml(raw_params, dvc_stages):
    referenced = [
        key for stage in dvc_stages.values() for key in stage.get("do", stage).get("params", [])
    ]
    assert referenced, "dvc.yaml declares no params"
    for dotted in referenced:
        _get(raw_params, dotted)  # KeyError if a referenced key is missing


def test_acquire_units_match_dvc_foreach(raw_params, dvc_stages, settings):
    assert tuple(dvc_stages["acquire"]["foreach"]) == ACQUIRE_UNITS
    assert tuple(raw_params["acquire"]) == ACQUIRE_UNITS
    assert tuple(settings.protocol.acquire.model_dump()) == ACQUIRE_UNITS


# ----------------------------------------------------------------------------- protocol equality
@pytest.mark.parametrize("group", list(EXPECTED))
def test_group_equals_protocol(settings, group):
    dumped = settings.protocol.model_dump(mode="json", by_alias=True)
    assert dumped[group] == EXPECTED[group]


@pytest.mark.parametrize(
    "dotted",
    ["domain.s_floor", "solver.tau_l", "solver.tau_q", "model.optimizer.lr", "uq.alpha"],
)
def test_float_literals_parse_as_floats(raw_params, dotted):
    assert isinstance(_get(raw_params, dotted), float)


def test_derived_quantities(settings):
    p = settings.protocol
    assert p.domain.delta_n == pytest.approx(math.log(120))
    assert round(p.domain.delta_n, 4) == 4.7875  # 2.2
    t_lo, t_hi = p.domain.t_band
    assert (round(t_lo, 4), round(t_hi, 4)) == (0.0953, 1.0986)  # 2.2
    assert p.qc.eps_a_log_tol == pytest.approx(math.log(1.05))  # 2.2 R8
    assert p.link.da_log_tol == pytest.approx(math.log(1.25))  # 2.5
    assert p.evaluation.s2.level == pytest.approx(0.9875)  # 4.5
    n_min, n_max = p.domain.n_bounds
    band = p.evaluation.h4.vartheta * p.domain.delta_n
    assert round(n_min * math.exp(band), 4) == 0.0064  # 4.4 H4 (a)
    assert round(n_max * math.exp(-band), 2) == 0.47
    assert round(0.39 * 0.3048**0.16, 4) == p.priors.jarrett.c_j_si  # 4.1
    expected_lambda_delta = p.priors.strickler_prior.lambda_p * p.domain.delta_n**2 / 16
    assert p.priors.strickler_residual.lambda_delta == pytest.approx(
        expected_lambda_delta, abs=5e-4
    )


def test_params_sha256_is_exact_file_hash(settings):
    assert settings.params_sha256 == hashlib.sha256(PARAMS_PATH.read_bytes()).hexdigest()
    assert settings.params_path == PARAMS_PATH.resolve()


# ----------------------------------------------------------------------------- immutability
def test_protocol_is_frozen(settings):
    with pytest.raises(ValidationError):
        settings.protocol.domain.s_floor = 1.0


@pytest.mark.parametrize("dotted", ["unknown_group", "qc.unknown_key", "model.backbone.dropout"])
def test_unknown_keys_rejected(raw_params, dotted):
    data = copy.deepcopy(raw_params)
    _set(data, dotted, 1)
    with pytest.raises(ValidationError):
        ProtocolParams.model_validate(data)


def test_duplicate_yaml_key_rejected(tmp_path):
    bad = tmp_path / "params.yaml"
    bad.write_text(PARAMS_PATH.read_text(encoding="utf-8") + "\nqc:\n  q_tol: 0.5\n")
    with pytest.raises(ValueError, match="duplicate key"):
        load_settings(bad, runtime=_clean_runtime())


# ----------------------------------------------------------------------------- validators
BAD_VALUES: list[tuple[str, Any]] = [
    ("domain.n_bounds", [0.6, 0.005]),
    ("domain.n_lit", [0.004, 0.33]),
    ("domain.r_bounds", [0.4, 10.0]),
    ("domain.s_floor", 1.0e-4),
    ("priors.jarrett.b_j", 1.7),
    ("priors.jarrett.c_j_si", 0.39),
    ("priors.strickler_prior.lambda_p", 1.0),
    ("priors.strickler_residual.lambda_delta", 3.0),
    ("uq.alpha", 0.2),
    ("uq.ensemble_m", 1),
    ("uq.sigma_min", 0.0),
    ("uq.beta_nll", 1.5),
    ("uq.conformal", True),
    ("solver.tau_l", 0.0),
    ("solver.k_max", 0),
    ("solver.stopping", "batch_global"),
    ("model.backbone.norm", "batchnorm"),
    ("model.optimizer.patience", 501),
    ("audit.dist_classes_km", [10, 25, 25, 100]),
    ("audit.vgm_max_km", 200),
    ("seeds.sid_repeats", [20261008, 20261008, 20261010]),
    ("seeds.sid_repeats", [20261008, 20261009]),
    ("seeds.global", -1),
    ("seeds.global", 2**32),
    ("qc.dmax_max_m", 40),
    ("acquire.hydroswot.md5", ["0" * 32]),
    ("acquire.hydroswot.sha256_pinned", "not-a-hash"),
    ("condition.tie_break", "first"),
    ("evaluation.mstar_candidates", ["M5"]),
    ("audit_rules.range_sill_frac", 1.5),
    ("audit_rules.vgm_models", ["gaussian"]),
    ("compute.pilot_max_fit_s", 0),
]


@pytest.mark.parametrize(("dotted", "value"), BAD_VALUES)
def test_validators_reject(raw_params, dotted, value):
    data = copy.deepcopy(raw_params)
    _set(data, dotted, value)
    with pytest.raises(ValidationError):
        ProtocolParams.model_validate(data)


# ----------------------------------------------------------------------------- environment
def test_environment_cannot_override_protocol(monkeypatch, settings):
    hostile = {
        "ANNXS_SEEDS__GLOBAL": "42",
        "ANNXS_SEEDS": '{"global": 42}',
        "ANNXS_DOMAIN__S_FLOOR": "0.1",
        "ANNXS_PROTOCOL": "{}",
        "ANNXS_PARAMS_PATH": "/nonexistent/params.yaml",
        "SEEDS__GLOBAL": "42",
    }
    for name, value in hostile.items():
        monkeypatch.setenv(name, value)
    reloaded = load_settings()
    assert reloaded.protocol == settings.protocol
    assert reloaded.params_sha256 == settings.params_sha256
    assert reloaded.protocol.seeds.global_seed == 20261008


def test_runtime_settings_read_environment(monkeypatch):
    monkeypatch.setenv("ANNXS_N_WORKERS", "4")
    monkeypatch.setenv("ANNXS_LOG_LEVEL", "DEBUG")
    runtime = _clean_runtime()
    assert (runtime.n_workers, runtime.log_level) == (4, "DEBUG")


def test_runtime_defaults(monkeypatch):
    for field in RuntimeSettings.model_fields:
        monkeypatch.delenv(f"ANNXS_{field.upper()}", raising=False)
    runtime = _clean_runtime()
    assert (runtime.device, runtime.dtype) == ("cpu", "float64")
    assert (runtime.n_workers, runtime.torch_threads_per_worker) == (10, 1)
    assert runtime.repo_root == REPO_ROOT
    assert runtime.raw_dir == (REPO_ROOT / "data" / "raw").resolve()
    assert runtime.processed_dir == (REPO_ROOT / "data" / "processed").resolve()
    assert runtime.checkpoint_dir == (REPO_ROOT / "data" / "checkpoints").resolve()
    assert runtime.reports_dir == (REPO_ROOT / "reports").resolve()
    assert runtime.dvc_store == (Path.home() / "dvc_store" / "ANN_XS_Estimation").resolve()


@pytest.mark.parametrize(
    ("device", "dtype"), [("mps", "float64"), ("mps", "float32"), ("cpu", "float32")]
)
def test_only_cpu_float64_allowed(device, dtype):
    with pytest.raises(ValidationError):
        _clean_runtime(device=device, dtype=dtype)


@pytest.mark.parametrize("field", ["raw_dir", "processed_dir", "checkpoint_dir", "reports_dir"])
def test_data_dirs_must_be_inside_repo(tmp_path, field):
    with pytest.raises(ValidationError, match="inside the repository"):
        _clean_runtime(**{field: tmp_path / "elsewhere"})


@pytest.mark.parametrize(
    "store",
    [
        REPO_ROOT / "dvc_store",
        Path.home() / "Library/CloudStorage/GoogleDrive-x@example.com/My Drive/dvc_store",
    ],
)
def test_dvc_store_must_be_outside_repo_and_drive(store):
    with pytest.raises(ValidationError, match="outside"):
        _clean_runtime(dvc_store=store)


# ----------------------------------------------------------------------------- side effects
def test_ensure_dirs_is_explicit_and_idempotent(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    store = tmp_path / "store"
    runtime = _clean_runtime(repo_root=repo, dvc_store=store)
    settings = load_settings(runtime=runtime)
    assert list(repo.iterdir()) == []  # constructing settings creates nothing
    created = settings.ensure_dirs()
    assert set(created) == {
        runtime.raw_dir,
        runtime.processed_dir,
        runtime.checkpoint_dir,
        runtime.reports_dir,
    }
    assert list(runtime.raw_dir.iterdir()) == []  # never data/raw/<unit> (DVC outputs)
    assert not store.exists()
    settings.ensure_dirs(include_dvc_store=True)
    assert sorted(p.name for p in store.iterdir()) == ["cache", "inbox", "remote"]
    assert settings.ensure_dirs(include_dvc_store=True) == ()


def test_config_does_not_import_torch():
    code = "import sys, src.config; print('torch' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    assert result.stdout.strip() == "False"

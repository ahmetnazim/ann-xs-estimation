"""No-pickle model artefacts: ``state.npz`` + ``model.json``; model construction from a spec.

Sources: P6-D6 (a) (``state_dict`` as ``.npz`` loaded with ``allow_pickle=False``, plus
``model.json``; one ``normstats.json`` per fit carrying ``folds.fit_provenance``), P5-D9 and
ADL-076 (no pickle or joblib anywhere), protocol section 4.1 (ladder rows and ablation switches),
P6-D12 (a), P6-Q2 (a) (``ln D50`` recovered from the fit's ``NormStats``).

``model.json`` records everything needed to rebuild the module (spec, features, ``in_features``,
``x_blind_idx``, hyperparameters, M3 ``r_bar``), the ``params_sha256`` it was trained under and
the SHA-256 of ``state.npz``; :func:`load_model` refuses a mismatch of either.

Torch-only, float64, CPU. Owning phase: 6.
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import Tensor

from src.config import ProtocolParams
from src.data.normstats import NormStats
from src.models.backbone import assert_row_wise
from src.models.heads import LadderNet, LogFeature
from src.models.m2_mlp import M2MLP, M2Config
from src.models.m3_physics import M3Config, M3Physics
from src.models.m4_soft import M4Config, M4Soft
from src.models.m5_hard import M5Config, M5Hard, Prior
from src.models.m6_two_stage import M6NN, M6Config
from src.models.registry import ModelSpec, NnHyper, read_json, write_json

__all__ = [
    "MODEL_FILE",
    "NORMSTATS_FILE",
    "RUN_FILE",
    "STATE_FILE",
    "build_model",
    "load_model",
    "load_state",
    "save_model",
    "save_state",
]

STATE_FILE: Final[str] = "state.npz"
MODEL_FILE: Final[str] = "model.json"
NORMSTATS_FILE: Final[str] = "normstats.json"
RUN_FILE: Final[str] = "run.json"
_FORMAT: Final[str] = "annxs-model-v1"
_PRIORS: Final[dict[str | None, Prior]] = {"A3-SP": "SP", "A3-SR": "SR"}


def _construct(
    spec: ModelSpec,
    protocol: ProtocolParams,
    *,
    stats: NormStats,
    blind_idx: Tensor,
    hyper: NnHyper,
    r_bar: float | None = None,
) -> LadderNet:
    if not spec.is_nn:
        raise ValueError(f"{spec.name} is not a torch rung")
    f = len(stats.names)
    backbone = protocol.model.backbone
    ab = spec.ablation
    if spec.model == "M2-MLP":
        return M2MLP(f, M2Config.from_protocol(protocol), backbone)
    if spec.model == "M3":
        if r_bar is None:
            raise ValueError("M3 needs r_bar (protocol 4.1)")
        cfg3 = M3Config.from_protocol(protocol, r_bar=r_bar)
        return M3Physics(f, cfg3, backbone, LogFeature.from_stats(stats, "D50_m"))
    if spec.model == "M4":
        if hyper.lambda_max is None:
            raise ValueError("M4 needs lambda_max (P3-09)")
        return M4Soft(
            f, blind_idx, M4Config.from_protocol(protocol, lambda_max=hyper.lambda_max), backbone
        )
    if spec.model == "M5":
        prior = _PRIORS.get(ab, "none")
        cfg5 = M5Config.from_protocol(
            protocol,
            kappa=ab != "A1",
            a2=ab == "A2",
            w_a=1.0 if ab == "wA1" else None,
            prior=prior,
        )
        ln_d50 = LogFeature.from_stats(stats, "D50_m") if prior != "none" else None
        return M5Hard(f, blind_idx, cfg5, backbone, ln_d50)
    return M6NN(f, blind_idx, M6Config.from_protocol(protocol), backbone)


def build_model(
    spec: ModelSpec,
    protocol: ProtocolParams,
    *,
    stats: NormStats,
    blind_idx: Tensor,
    hyper: NnHyper,
    r_bar: float | None = None,
) -> LadderNet:
    """Construct the torch rung of ``spec`` (protocol 4.1; ablation flags per P6-D12 (a)).

    ``stats`` must be the fit's ``NormStats`` (its names define ``in_features``; ``ln D50`` is
    recovered from it for M3, A3-SP and A3-SR). ``r_bar`` is required for M3 only and
    ``hyper.lambda_max`` for M4 only. The result is checked to hold no batch-coupled layer
    (Proposition 3.4 scope; ``assert_row_wise``).
    """
    model = _construct(spec, protocol, stats=stats, blind_idx=blind_idx, hyper=hyper, r_bar=r_bar)
    assert_row_wise(model)
    return model


# ----------------------------------------------------------------------------- state
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_state(directory: Path, model: LadderNet) -> dict[str, Any]:
    """Write ``state.npz`` (``numpy.savez``, no pickle) atomically; return its metadata."""
    arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    for k, a in arrays.items():
        if a.dtype == object:  # pragma: no cover - torch state dicts hold numeric tensors
            raise TypeError(f"state entry {k} is not numeric")
    buf = io.BytesIO()
    np.savez(buf, allow_pickle=False, **arrays)
    data = buf.getvalue()
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".state.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, directory / STATE_FILE)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return {
        "sha256": _sha256(data),
        "tensors": {k: {"dtype": str(a.dtype), "shape": list(a.shape)} for k, a in arrays.items()},
    }


def load_state(directory: Path, model: LadderNet, *, sha256: str | None = None) -> None:
    """Load ``state.npz`` with ``allow_pickle=False`` into ``model`` (strict key match).

    ``sha256`` (from ``model.json``) is verified when given.
    """
    data = (directory / STATE_FILE).read_bytes()
    if sha256 is not None and _sha256(data) != sha256:
        raise ValueError(f"{directory / STATE_FILE}: SHA-256 differs from model.json")
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        state = {k: torch.from_numpy(np.array(npz[k], copy=True)) for k in npz.files}
    model.load_state_dict(state, strict=True)


def save_model(
    directory: Path,
    model: LadderNet,
    *,
    spec: ModelSpec,
    stats: NormStats,
    blind_idx: Sequence[int],
    hyper: NnHyper,
    r_bar: float | None,
    params_sha256: str,
) -> dict[str, Any]:
    """Write ``state.npz``, ``normstats.json`` and ``model.json`` (P6-D6 (a)); return the JSON."""
    meta = save_state(directory, model)
    stats.save(directory / NORMSTATS_FILE)
    doc = {
        "format": _FORMAT,
        "spec": asdict(spec),
        "class": type(model).__name__,
        "features": list(stats.names),
        "in_features": len(stats.names),
        "blind_idx": [int(i) for i in blind_idx],
        "hyper": asdict(hyper),
        "r_bar": r_bar,
        "n_trainable_params": model.get_num_trainable_params(),
        "params_sha256": params_sha256,
        "state": meta,
    }
    write_json(directory / MODEL_FILE, doc)
    return doc


def load_model(
    directory: Path, protocol: ProtocolParams, *, params_sha256: str
) -> tuple[LadderNet, NormStats, dict[str, Any]]:
    """Rebuild a saved rung and load its state; refuses a different ``params_sha256``."""
    doc = read_json(directory / MODEL_FILE)
    if doc.get("format") != _FORMAT:
        raise ValueError(f"{directory}: not a model artefact")
    if doc["params_sha256"] != params_sha256:
        raise ValueError(f"{directory}: trained under params {doc['params_sha256'][:12]}...")
    stats = NormStats.load(directory / NORMSTATS_FILE)
    if list(stats.names) != doc["features"]:
        raise ValueError(f"{directory}: normstats.json and model.json features differ")
    spec = ModelSpec(**doc["spec"])
    model = build_model(
        spec,
        protocol,
        stats=stats,
        blind_idx=torch.tensor(doc["blind_idx"], dtype=torch.int64),
        hyper=NnHyper(**doc["hyper"]),
        r_bar=doc["r_bar"],
    )
    load_state(directory, model, sha256=doc["state"]["sha256"])
    model.eval()
    return model, stats, doc

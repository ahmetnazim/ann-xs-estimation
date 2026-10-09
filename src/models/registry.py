"""Model identities, ablation switches, hyperparameter grids and the checkpoint layout.

Sources: protocol section 4.1 (ladder rows; M5 ablations A1, A2, A3-SP, A3-SR, A4, ``w_A = 1``;
A4 applied to M5 and M*), section 4.2 (inner grid ``weight_decay x lr``, plus ``lambda_max`` for
M4; XGB grid), section 3.4 (nested selection), section 4.8 (ablations reuse the M5
hyperparameters); P6-D6 (a) (layout ``<model>/<analysis>/<family>/r<rep>/f<fold>/...``),
P6-D12 (a) (typed switches), P6-Q4 (a) (M3 and XGB members), Batch D Q-D2 / Q-D3.

Grid order is fixed: ``itertools.product(weight_decay, lr[, lambda_max])`` for NN models and
``product(max_depth, eta, min_child_weight)`` for XGB (``src.models.baselines.xgb_grid``). Grid
point ``g`` is the 0-based position in that order (so the P6-D9 "grid point 1" is ``g = 0``).

Torch-free and xgboost-free (OpenMP rule, ADL-051): XGB runs import this module.
Owning phase: 6.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Final, Literal

from src.config import ProtocolParams
from src.models.baselines import M1_FEATURES, XgbHyper, a4_features, m1_grid, nn_features, xgb_grid

__all__ = [
    "ABLATIONS",
    "NN_MODELS",
    "WARM_START_MODELS",
    "XGB_MODELS",
    "FitKind",
    "ModelSpec",
    "NnHyper",
    "block_dir",
    "fit_dir",
    "nn_grid",
    "read_json",
    "write_json",
    "xgb_grid_for",
]

#: Torch rungs trained by ``src.models.trainer`` (protocol 4.1; M3 = its W regressor).
NN_MODELS: Final[tuple[str, ...]] = ("M2-MLP", "M3", "M4", "M5", "M6-NN")
#: Booster models (M6-XGB needs torch for its labels and runs boosters in a worker, P6-D7).
XGB_MODELS: Final[tuple[str, ...]] = ("M1", "M2-XGB", "M6-XGB")
#: Rungs with the 20-epoch MSE warm start before the (beta-)NLL (P6-Q3 (a)).
WARM_START_MODELS: Final[frozenset[str]] = frozenset({"M2-MLP", "M4", "M5"})
#: Ablation switch -> models it applies to (protocol 4.1; F9: A4 is a generic feature drop,
#: except M3, whose Strickler branch needs D50).
ABLATIONS: Final[dict[str, frozenset[str]]] = {
    "A1": frozenset({"M5"}),
    "A2": frozenset({"M5"}),
    "A3-SP": frozenset({"M5"}),
    "A3-SR": frozenset({"M5"}),
    "wA1": frozenset({"M5"}),
    "A4": frozenset({"M2-MLP", "M4", "M5", "M6-NN", "M1", "M2-XGB", "M6-XGB"}),
}

FitKind = Literal["inner", "member"]


@dataclass(frozen=True)
class ModelSpec:
    """One trainable model: a ladder rung and at most one ablation switch (P6-D12 (a))."""

    model: str
    ablation: str | None = None

    def __post_init__(self) -> None:
        """Reject unknown models and ablations that do not apply to the model."""
        if self.model not in NN_MODELS + XGB_MODELS:
            raise ValueError(f"unknown model {self.model!r}; expected {NN_MODELS + XGB_MODELS}")
        if self.ablation is not None:
            if self.ablation not in ABLATIONS:
                raise ValueError(f"unknown ablation {self.ablation!r}")
            if self.model not in ABLATIONS[self.ablation]:
                raise ValueError(f"ablation {self.ablation} does not apply to {self.model}")

    @property
    def name(self) -> str:
        """Directory name: ``M5`` or ``M5_A1`` (P6-D6 layout)."""
        return self.model if self.ablation is None else f"{self.model}_{self.ablation}"

    @property
    def is_nn(self) -> bool:
        """``True`` for the torch rungs."""
        return self.model in NN_MODELS

    @property
    def needs_torch(self) -> bool:
        """M1 and M2-XGB are torch-free; every other model loads torch (ADL-051)."""
        return self.model not in ("M1", "M2-XGB")

    @property
    def hyper_source(self) -> ModelSpec:
        """Whose inner selection the members use: the base model for ablations (protocol 4.8)."""
        return ModelSpec(self.model) if self.ablation is not None else self

    def features(self, protocol: ProtocolParams) -> tuple[str, ...]:
        """Feature set: M1 = Zarrabi Tier-2 (19); others = the NN set (22); A4 drops D50, AI."""
        base = M1_FEATURES if self.model == "M1" else nn_features(protocol)
        return a4_features(base) if self.ablation == "A4" else base


@dataclass(frozen=True)
class NnHyper:
    """One NN grid point (protocol 4.2): learning rate, AdamW weight decay, M4 ``lambda_max``."""

    lr: float
    weight_decay: float
    lambda_max: float | None = None


def nn_grid(protocol: ProtocolParams, model: str) -> list[NnHyper]:
    """``product(weight_decay, lr[, m4_lambda_max])`` in params order (protocol 4.2, P3-09)."""
    if model not in NN_MODELS:
        raise ValueError(f"{model!r} is not an NN model")
    grid = protocol.model.inner_grid
    lams: tuple[float | None, ...] = grid.m4_lambda_max if model == "M4" else (None,)
    return [
        NnHyper(lr=lr, weight_decay=wd, lambda_max=lam)
        for wd, lr, lam in product(grid.weight_decay, grid.lr, lams)
    ]


def xgb_grid_for(protocol: ProtocolParams, model: str) -> list[XgbHyper]:
    """M1: ``m1_max_depth`` grid (Q-C2 (b)); M2-XGB and M6-XGB: the shared grid (protocol 4.2)."""
    if model not in XGB_MODELS:
        raise ValueError(f"{model!r} is not an XGB model")
    return m1_grid(protocol) if model == "M1" else xgb_grid(protocol.model.xgb)


# ----------------------------------------------------------------------------- layout
def block_dir(
    root: Path, spec: ModelSpec, analysis: str, family: str, repeat: int, fold: int
) -> Path:
    """``<root>/<model>/<analysis>/<family>/r<rep>/f<fold>`` (P6-D6 (a))."""
    return root / spec.name / analysis / family / f"r{repeat}" / f"f{fold}"


def fit_dir(
    root: Path,
    spec: ModelSpec,
    analysis: str,
    family: str,
    repeat: int,
    fold: int,
    kind: FitKind,
    index: int,
    grid: int | None = None,
) -> Path:
    """Fit directory: ``.../inner_<k>/g<NN>`` (inner fit at grid point NN) or ``.../member_<m>``.

    The grid level is added to the P6-D6 layout because one inner fold holds one fit per grid
    point (Batch D).
    """
    base = block_dir(root, spec, analysis, family, repeat, fold)
    if kind == "inner":
        if grid is None or grid < 0:
            raise ValueError("inner fits need a grid point >= 0")
        return base / f"inner_{index}" / f"g{grid:02d}"
    if kind == "member":
        return base / f"member_{index}"
    raise ValueError("kind must be 'inner' or 'member'")


def _check_finite(obj: object, where: str = "$") -> None:
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError(f"non-finite float at {where}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_finite(v, f"{where}.{k}")
    elif isinstance(obj, list | tuple):
        for i, v in enumerate(obj):
            _check_finite(v, f"{where}[{i}]")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    """Atomic JSON write (temporary file + ``os.replace``); sorted keys; NaN and inf refused."""
    _check_finite(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object written by :func:`write_json`."""
    out = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(out, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return out

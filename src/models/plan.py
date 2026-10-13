"""The deterministic Phase 8 fit plan: enumeration, order, dependencies, hash and counts.

Sources: docs/3 section 13.3 (stage layout and order; Q2 (a): XGB fits and the Event 1 inputs
before any NN fit; Q7 (a): 10 XGB members), section 13.6 (sort key, canonical JSON with a header
of ``params_sha256`` and the table SHA-256 values, ``plan_sha256``, expected counts), section 13.2
(``run_id = "p8-" + plan_sha256[:12]``), section 13.8 (SA1 for M5 only; A4 on M5 and the three
M* candidates; SA2-SA6 for M5, M2-MLP, M1, M2-XGB) and section 4.8 (ablations and SAs reuse the
primary selection of the base model, Q-D3 (a)).

The plan is a sorted tuple of :class:`PlannedFit`. Every fit id is the artefact path relative to
the run root and equals :func:`src.models.registry.fit_dir` for runner fits; a selection has the id
``<block>/select`` and its single artefact is ``<block>/selection.json``.

Torch-free and xgboost-free (OpenMP rule, ADL-051): orchestrator workers import this module.
Owning phase: 8.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd

from src.config import ProtocolParams
from src.data.io import sha256_file
from src.models.registry import NN_MODELS, XGB_MODELS, nn_grid, xgb_grid_for
from src.models.runner import BASELINES

__all__ = [
    "BASELINES",
    "EXPECTED_COUNTS_13_6",
    "NN_STAGES",
    "PLAN_SCHEMA_VERSION",
    "STAGES",
    "TABLE_KEYS",
    "PlanKind",
    "PlannedFit",
    "blocks_from_folds",
    "build_plan",
    "canonical_json",
    "count_groups",
    "dependency_map",
    "expected_counts",
    "plan_document",
    "plan_sha256",
    "run_id_for",
    "table_hashes",
    "table_paths",
]

PLAN_SCHEMA_VERSION: Final[int] = 1
#: Training stages of docs/3 13.3 in execution order (1, 2, 5a, 5b, 5c).
STAGES: Final[tuple[str, ...]] = (
    "train@baselines",
    "train@xgb",
    "train@nn_primary",
    "train@ablations",
    "train@sensitivity",
)
#: Stages refused unless ``checkpoint_1`` is in the scoring ledger (docs/3 13.3 row 5, 13.4).
NN_STAGES: Final[frozenset[str]] = frozenset(STAGES[2:])
#: Tables whose SHA-256 enters the plan header (docs/3 13.6).
TABLE_KEYS: Final[tuple[str, ...]] = (
    "site_table",
    "fold_table",
    "srnd_assignment",
    "ehydro_ind",
    "ind_features",
    "srnd_frame",
)

_MODEL_ORDER: Final[tuple[str, ...]] = (*BASELINES, *XGB_MODELS, *NN_MODELS)
_ABLATION_ORDER: Final[tuple[str | None, ...]] = (None, "A1", "A2", "A3-SP", "A3-SR", "A4", "wA1")
_ANALYSIS_ORDER: Final[tuple[str, ...]] = ("primary", "SA1", "SA2", "SA3", "SA4", "SA5", "SA6")
_FAMILY_ORDER: Final[tuple[str, ...]] = (
    "S-ID",
    "S-LRO",
    "S-TAIL-DA",
    "S-TAIL-SO",
    "S-IND",
    "S-RND",
)
_KIND_ORDER: Final[tuple[str, ...]] = ("outer", "inner", "select", "member")
_M5_ABLATIONS: Final[tuple[str, ...]] = ("A1", "A2", "A3-SP", "A3-SR", "A4", "wA1")
_SA_ALL: Final[tuple[str, ...]] = ("SA1", "SA2", "SA3", "SA4", "SA5", "SA6")
_SA_NO_PHI: Final[tuple[str, ...]] = _SA_ALL[1:]  # docs/3 13.8: SA1 needs a Phi (M5 only)
_SRND_MODELS: Final[tuple[str, ...]] = ("M1", "M2-XGB")  # P8-D8 (b)
_TORCH_FREE: Final[frozenset[str]] = frozenset({*BASELINES, "M1", "M2-XGB"})

#: The count table of docs/3 13.6, verbatim (frozen params, 26 blocks).
EXPECTED_COUNTS_13_6: Final[Mapping[str, int]] = {
    "M2-MLP": 884,
    "M3": 884,
    "M4": 2132,
    "M5": 884,
    "M6-NN": 884,
    "M5 ablations": 1560,
    "A4 on M2-MLP": 260,
    "SA on M5": 1560,
    "SA on M2-MLP": 1300,
    "NN total": 10348,
    "M1": 1196,
    "M2-XGB": 1196,
    "M6-XGB": 1196,
    "A4 on M1, M2-XGB": 520,
    "SA on M1, M2-XGB": 2600,
    "S-RND": 20,
    "XGB total": 6728,
    "selections": 208,
    "baselines": 78,
}

PlanKind = Literal["outer", "inner", "select", "member"]


@dataclass(frozen=True)
class PlannedFit:
    """One planned unit of work: a baseline, an inner fit, a selection or a member."""

    stage: str
    model: str
    ablation: str | None
    analysis: str
    family: str
    repeat: int
    fold: int
    kind: PlanKind
    index: int
    grid: int | None

    def __post_init__(self) -> None:
        """Validate the coordinates against the fixed orders."""
        checks = (
            (self.stage, STAGES, "stage"),
            (self.model, _MODEL_ORDER, "model"),
            (self.ablation, _ABLATION_ORDER, "ablation"),
            (self.analysis, _ANALYSIS_ORDER, "analysis"),
            (self.family, _FAMILY_ORDER, "family"),
            (self.kind, _KIND_ORDER, "kind"),
        )
        for value, allowed, name in checks:
            if value not in allowed:
                raise ValueError(f"unknown {name} {value!r}")
        if (self.kind == "inner") != (self.grid is not None):
            raise ValueError("exactly the inner fits carry a grid point")

    @property
    def spec_name(self) -> str:
        """Directory name of the model: ``M5`` or ``M5_A1`` (P6-D6 layout)."""
        return self.model if self.ablation is None else f"{self.model}_{self.ablation}"

    @property
    def block_id(self) -> str:
        """``<model>/<analysis>/<family>/r<rep>/f<fold>`` relative to the run root."""
        return f"{self.spec_name}/{self.analysis}/{self.family}/r{self.repeat}/f{self.fold}"

    @property
    def fit_id(self) -> str:
        """Artefact path relative to the run root (docs/3 13.7)."""
        if self.kind == "outer":
            return f"{self.block_id}/outer"
        if self.kind == "inner":
            return f"{self.block_id}/inner_{self.index}/g{self.grid:02d}"
        if self.kind == "select":
            return f"{self.block_id}/select"
        return f"{self.block_id}/member_{self.index}"

    @property
    def torch_free(self) -> bool:
        """Baselines, M1 and M2-XGB run in torch-free workers (ADL-051/-086)."""
        return self.model in _TORCH_FREE

    def sort_key(self) -> tuple[int | str, ...]:
        """Order of docs/3 13.6: stage, model, ablation, analysis, family, ..., index, grid."""
        return (
            STAGES.index(self.stage),
            _MODEL_ORDER.index(self.model),
            _ABLATION_ORDER.index(self.ablation),
            _ANALYSIS_ORDER.index(self.analysis),
            _FAMILY_ORDER.index(self.family),
            self.repeat,
            self.fold,
            _KIND_ORDER.index(self.kind),
            self.index,
            -1 if self.grid is None else self.grid,
        )

    def completion_path(self, run_dir: Path) -> Path:
        """``run.json`` of a fit directory, or ``selection.json`` of a block (docs/3 13.7)."""
        if self.kind == "select":
            return run_dir / self.block_id / "selection.json"
        return run_dir / self.fit_id / "run.json"

    def artefact_files(self, run_dir: Path) -> list[tuple[str, str]]:
        """Sorted ``(relative path, SHA-256)`` of every artefact of this fit (empty if absent).

        Paths are relative to the fit directory; for a selection, to the block directory.
        """
        if self.kind == "select":
            path = self.completion_path(run_dir)
            return [(path.name, sha256_file(path))] if path.is_file() else []
        base = run_dir / self.fit_id
        if not base.is_dir():
            return []
        files = sorted(p for p in base.rglob("*") if p.is_file())
        return [(p.relative_to(base).as_posix(), sha256_file(p)) for p in files]

    def to_json(self) -> dict[str, Any]:
        """Plain JSON object (the plan entry)."""
        return asdict(self)

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> PlannedFit:
        """Inverse of :meth:`to_json`."""
        return cls(**dict(obj))


# ----------------------------------------------------------------------------- blocks and tables
def blocks_from_folds(folds: pd.DataFrame) -> list[tuple[str, int, int]]:
    """Distinct ``(family, repeat, outer_fold)`` of the frozen fold table, in plan order."""
    keys = folds[["family", "repeat", "outer_fold"]].drop_duplicates()
    out = [(str(f), int(r), int(o)) for f, r, o in keys.itertuples(index=False)]
    unknown = sorted({f for f, _, _ in out} - set(_FAMILY_ORDER[:-1]))
    if unknown:
        raise ValueError(f"unknown fold families {unknown}")
    return sorted(out, key=lambda b: (_FAMILY_ORDER.index(b[0]), b[1], b[2]))


def table_paths(processed_dir: Path) -> dict[str, Path]:
    """Location of every table of the plan header under ``data/processed``."""
    return {
        "site_table": processed_dir / "site_table.parquet",
        "fold_table": processed_dir / "splits" / "fold_assignment.parquet",
        "srnd_assignment": processed_dir / "splits" / "srnd_assignment.parquet",
        "ehydro_ind": processed_dir / "ind" / "ehydro_ind.parquet",
        "ind_features": processed_dir / "ind" / "ehydro_ind_features.parquet",
        "srnd_frame": processed_dir / "srnd" / "srnd_frame.parquet",
    }


def table_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    """SHA-256 of every header table; raises listing the missing ones (stages 0a/0b first)."""
    if sorted(paths) != sorted(TABLE_KEYS):
        raise ValueError(f"need exactly the tables {TABLE_KEYS}")
    missing = [k for k in TABLE_KEYS if not paths[k].is_file()]
    if missing:
        raise FileNotFoundError(f"plan tables missing: {missing} (run stages 0a/0b first)")
    return {k: sha256_file(paths[k]) for k in TABLE_KEYS}


# ----------------------------------------------------------------------------- enumeration
def _block_fits(
    protocol: ProtocolParams,
    stage: str,
    model: str,
    block: tuple[str, int, int],
    *,
    ablation: str | None = None,
    analysis: str = "primary",
    inner: bool,
) -> Iterable[PlannedFit]:
    family, repeat, fold = block

    def make(kind: PlanKind, index: int, grid: int | None = None) -> PlannedFit:
        return PlannedFit(stage, model, ablation, analysis, family, repeat, fold, kind, index, grid)

    if inner:
        n_grid = len(
            nn_grid(protocol, model) if model in NN_MODELS else xgb_grid_for(protocol, model)
        )
        for k in range(protocol.splits.inner_k):
            for g in range(n_grid):
                yield make("inner", k, g)
        yield make("select", 0)
    for m in range(1, protocol.uq.ensemble_m + 1):
        yield make("member", m)


def build_plan(
    protocol: ProtocolParams, blocks: Sequence[tuple[str, int, int]]
) -> tuple[PlannedFit, ...]:
    """Every planned fit of docs/3 13.3/13.6/13.8, sorted by :meth:`PlannedFit.sort_key`."""
    fits: list[PlannedFit] = []
    for block in blocks:
        family, repeat, fold = block
        for name in BASELINES:
            fits.append(
                PlannedFit(
                    "train@baselines", name, None, "primary", family, repeat, fold, "outer", 0, None
                )
            )
        for model in XGB_MODELS:
            fits += _block_fits(protocol, "train@xgb", model, block, inner=True)
        for model in ("M1", "M2-XGB"):
            fits += _block_fits(protocol, "train@xgb", model, block, ablation="A4", inner=False)
            for sa in _SA_NO_PHI:
                fits += _block_fits(protocol, "train@xgb", model, block, analysis=sa, inner=False)
        for model in NN_MODELS:
            fits += _block_fits(protocol, "train@nn_primary", model, block, inner=True)
        for abl in _M5_ABLATIONS:
            fits += _block_fits(protocol, "train@ablations", "M5", block, ablation=abl, inner=False)
        fits += _block_fits(
            protocol, "train@ablations", "M2-MLP", block, ablation="A4", inner=False
        )
        for model, analyses in (("M5", _SA_ALL), ("M2-MLP", _SA_NO_PHI)):
            for sa in analyses:
                fits += _block_fits(
                    protocol, "train@sensitivity", model, block, analysis=sa, inner=False
                )
    for model in _SRND_MODELS:
        for m in range(1, protocol.uq.ensemble_m + 1):
            fits.append(
                PlannedFit("train@xgb", model, None, "primary", "S-RND", 0, 0, "member", m, None)
            )
    out = tuple(sorted(fits, key=PlannedFit.sort_key))
    ids = [f.fit_id for f in out]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate fit ids in the plan")
    return out


def dependency_map(fits: Sequence[PlannedFit]) -> dict[str, tuple[str, ...]]:
    """Fit id -> ids it needs complete first (inner -> select -> members; S-RND last).

    Ablation and SA members use the base model's primary selection (protocol 4.8, Q-D3 (a)).
    S-RND members need every primary S-ID selection and member of their model (docs/3 13.6).
    """
    ids = {f.fit_id for f in fits}
    inner: dict[str, list[str]] = {}
    sid: dict[str, list[str]] = {}
    for f in fits:
        if f.kind == "inner":
            inner.setdefault(f.block_id, []).append(f.fit_id)
        primary_base = f.ablation is None and f.analysis == "primary"
        if f.family == "S-ID" and primary_base and f.kind in ("select", "member"):
            sid.setdefault(f.model, []).append(f.fit_id)
    deps: dict[str, tuple[str, ...]] = {}
    for f in fits:
        if f.kind == "select":
            need = tuple(inner.get(f.block_id, ()))
            if not need:
                raise RuntimeError(f"{f.fit_id}: no inner fits planned")
        elif f.kind == "member" and f.family == "S-RND":
            need = tuple(sid.get(f.model, ()))
        elif f.kind == "member":
            need = (f"{f.model}/primary/{f.family}/r{f.repeat}/f{f.fold}/select",)
        else:
            need = ()
        absent = [d for d in need if d not in ids]
        if absent or (f.family == "S-RND" and not need):
            raise RuntimeError(f"{f.fit_id}: dependencies not in the plan {absent[:3]}")
        deps[f.fit_id] = need
    return deps


# ----------------------------------------------------------------------------- hash and identity
def canonical_json(obj: object) -> bytes:
    """Sorted keys, separators ``,`` and ``:``, UTF-8, NaN refused (docs/3 13.6)."""
    text = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return text.encode("utf-8")


def plan_document(
    fits: Sequence[PlannedFit], params_sha256: str, tables: Mapping[str, str]
) -> dict[str, Any]:
    """Return the plan as a JSON object: header (params and table hashes) and sorted fits."""
    if sorted(tables) != sorted(TABLE_KEYS):
        raise ValueError(f"the header needs exactly the tables {TABLE_KEYS}")
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "header": {"params_sha256": params_sha256, "tables": dict(tables)},
        "fits": [f.to_json() for f in fits],
    }


def plan_sha256(document: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of the plan document."""
    return hashlib.sha256(canonical_json(document)).hexdigest()


def run_id_for(sha: str) -> str:
    """``"p8-" + plan_sha256[:12]`` (docs/3 13.2)."""
    if len(sha) != 64 or set(sha) - set("0123456789abcdef"):
        raise ValueError("plan_sha256 must be 64 lower-case hex characters")
    return f"p8-{sha[:12]}"


# ----------------------------------------------------------------------------- counts
def _group(f: PlannedFit) -> str:
    if f.kind == "outer":
        return "baselines"
    if f.kind == "select":
        return "selections"
    if f.family == "S-RND":
        return "S-RND"
    xgb_pair = f.model in ("M1", "M2-XGB")
    if f.ablation is not None:
        if f.model == "M5":
            return "M5 ablations"
        return "A4 on M1, M2-XGB" if xgb_pair else f"A4 on {f.model}"
    if f.analysis != "primary":
        return "SA on M1, M2-XGB" if xgb_pair else f"SA on {f.model}"
    return f.model


def count_groups(fits: Iterable[PlannedFit]) -> dict[str, int]:
    """Count fits per row of the docs/3 13.6 table, with the NN and XGB totals."""
    out: dict[str, int] = {}
    nn_total = xgb_total = 0
    for f in fits:
        key = _group(f)
        out[key] = out.get(key, 0) + 1
        if key not in ("baselines", "selections"):
            if f.model in NN_MODELS:
                nn_total += 1
            else:
                xgb_total += 1
    out["NN total"] = nn_total
    out["XGB total"] = xgb_total
    return out


def expected_counts(protocol: ProtocolParams, n_blocks: int) -> dict[str, int]:
    """docs/3 13.6 counts by formula from the grids, ``inner_k``, ``M`` and the block count."""
    k, m, nb = protocol.splits.inner_k, protocol.uq.ensemble_m, n_blocks
    out = {name: nb * (len(nn_grid(protocol, name)) * k + m) for name in NN_MODELS}
    out["M5 ablations"] = len(_M5_ABLATIONS) * nb * m
    out["A4 on M2-MLP"] = nb * m
    out["SA on M5"] = len(_SA_ALL) * nb * m
    out["SA on M2-MLP"] = len(_SA_NO_PHI) * nb * m
    out["NN total"] = sum(out.values())
    xgb = {name: nb * (len(xgb_grid_for(protocol, name)) * k + m) for name in XGB_MODELS}
    xgb["A4 on M1, M2-XGB"] = 2 * nb * m
    xgb["SA on M1, M2-XGB"] = 2 * len(_SA_NO_PHI) * nb * m
    xgb["S-RND"] = len(_SRND_MODELS) * m
    out.update(xgb)
    out["XGB total"] = sum(xgb.values())
    out["selections"] = (len(NN_MODELS) + len(XGB_MODELS)) * nb
    out["baselines"] = len(BASELINES) * nb
    return out

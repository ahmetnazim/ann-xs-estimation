"""Run tracking: registry, RFC 6962 Merkle root, training summaries, manifest, pilot, archive.

Sources: docs/3 section 13.7 (per-fit ``run.json``; run registry
``data/processed/runs/run_registry.parquet`` with one row per planned fit; run manifest
``reports/run_manifest.json``; Merkle leaves ``SHA-256(0x00 || fit_id || 0x00 || path || 0x00 ||
file SHA-256 hex)`` sorted by (fit_id, path), nodes ``SHA-256(0x01 || left || right)`` split at the
largest power of two below n, RFC 6962 section 2.1, Laurie et al. 2013, doi:10.17487/rfc6962;
``dvc metrics``/``plots`` from training diagnostics only, never outer-test rows), section 13.2
(pilot moved unchanged to ``<run_root>/pilot_adl090/``; archive ``<dvc_store>/runs_archive/
<run_id>.tar``: POSIX ustar, entries sorted by path, mtime 0, uid/gid 0).

Nothing here reads an observed target or a prediction. The training summaries use only the
per-epoch training objective and the ``es_val`` early-stopping monitor recorded in ``run.json``
(member fits; inner fits are excluded), Xi_e (M4) and Picard counts.

Torch-free and xgboost-free (versions are read with ``importlib.metadata``). Owning phase: 8.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import tarfile
from collections.abc import Iterable, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src.data.io import sha256_file, write_parquet
from src.models.plan import STAGES, PlannedFit
from src.models.registry import read_json, write_json

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "REGISTRY_COLUMNS",
    "archive_run",
    "build_manifest",
    "build_registry",
    "host_info",
    "merkle_leaf",
    "merkle_root",
    "move_pilot",
    "registry_merkle_root",
    "rfc6962_leaf_hash",
    "training_summaries",
    "tree_digests",
    "write_registry",
    "write_training_reports",
]

MANIFEST_SCHEMA_VERSION: Final[int] = 1
REGISTRY_COLUMNS: Final[tuple[str, ...]] = (
    "fit_id",
    "stage",
    "model",
    "ablation",
    "analysis",
    "family",
    "repeat",
    "fold",
    "kind",
    "index",
    "grid",
    "status",
    "wall_s",
    "epochs",
    "mean_picard",
    "max_picard",
    "xi_final",
    "test_blanked",
    "n_files",
    "files_json",
)
_PACKAGES: Final[tuple[str, ...]] = ("numpy", "pandas", "torch", "xgboost", "pyarrow", "dvc")


# ----------------------------------------------------------------------------- Merkle (RFC 6962)
def rfc6962_leaf_hash(data: bytes) -> bytes:
    """``SHA-256(0x00 || data)`` (RFC 6962 section 2.1)."""
    return hashlib.sha256(b"\x00" + data).digest()


def merkle_leaf(fit_id: str, path: str, sha256_hex: str) -> bytes:
    """docs/3 13.7 leaf: ``SHA-256(0x00 || fit_id || 0x00 || path || 0x00 || sha256 hex)``."""
    if "\x00" in fit_id or "\x00" in path:
        raise ValueError("fit ids and paths must not contain NUL")
    data = b"\x00".join((fit_id.encode("utf-8"), path.encode("utf-8"), sha256_hex.encode("ascii")))
    return rfc6962_leaf_hash(data)


def merkle_root(leaves: Sequence[bytes]) -> str:
    """RFC 6962 Merkle tree hash of leaf hashes (hex); the empty tree is ``SHA-256("")``."""

    def mth(lo: int, hi: int) -> bytes:
        n = hi - lo
        if n == 1:
            return leaves[lo]
        k = 1 << ((n - 1).bit_length() - 1)  # largest power of two < n
        return hashlib.sha256(b"\x01" + mth(lo, lo + k) + mth(lo + k, hi)).digest()

    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    return mth(0, len(leaves)).hex()


# ----------------------------------------------------------------------------- registry
def _num(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool) and np.isfinite(value):
        return float(value)
    return None


def _registry_row(fit: PlannedFit, run_dir: Path) -> dict[str, Any]:
    marker = fit.completion_path(run_dir)
    done = marker.is_file()
    files = fit.artefact_files(run_dir) if done else []
    row: dict[str, Any] = {
        "fit_id": fit.fit_id,
        "stage": fit.stage,
        "model": fit.model,
        "ablation": fit.ablation,
        "analysis": fit.analysis,
        "family": fit.family,
        "repeat": fit.repeat,
        "fold": fit.fold,
        "kind": fit.kind,
        "index": fit.index,
        "grid": fit.grid,
        "status": "done" if done else "missing",
        "wall_s": None,
        "epochs": None,
        "mean_picard": None,
        "max_picard": None,
        "xi_final": None,
        "test_blanked": None,
        "n_files": len(files),
        "files_json": json.dumps(files, separators=(",", ":")),
    }
    if done and fit.kind != "select":
        doc = read_json(marker)
        row["wall_s"] = _num(doc.get("wall_s"))
        row["test_blanked"] = (doc.get("rows") or {}).get("test_blanked")
        body = doc.get("fit") or {}
        extras = doc.get("extras") or {}
        if "epochs_run" in body:
            row["epochs"] = int(body["epochs_run"])
            xis = [h.get("xi") for h in body.get("history", []) if h.get("xi") is not None]
            row["xi_final"] = _num(xis[-1]) if xis else None
        row["mean_picard"] = _num(extras.get("mean_picard_iterations"))
        row["max_picard"] = _num(extras.get("max_picard_iterations"))
    return row


def build_registry(fits: Sequence[PlannedFit], run_dir: Path) -> pd.DataFrame:
    """One row per planned fit, in plan order, with the SHA-256 of every artefact (13.7)."""
    frame = pd.DataFrame([_registry_row(f, run_dir) for f in fits], columns=list(REGISTRY_COLUMNS))
    for col in ("repeat", "fold", "index", "n_files"):
        frame[col] = frame[col].astype("int64")
    for col in ("grid", "epochs", "max_picard", "test_blanked"):
        frame[col] = frame[col].astype("Int64")
    for col in ("wall_s", "mean_picard", "xi_final"):
        frame[col] = frame[col].astype("Float64")
    return frame


def write_registry(frame: pd.DataFrame, path: Path, *, params_sha256: str, plan_sha256: str) -> str:
    """Write the registry Parquet (provenance metadata) and return its SHA-256."""
    write_parquet(
        frame,
        path,
        stage="run_experiment",
        params_sha256=params_sha256,
        inputs={"plan": plan_sha256},
    )
    return sha256_file(path)


def registry_merkle_root(frame: pd.DataFrame) -> str:
    """Merkle root over the artefacts of the completed fits, leaves sorted by (fit_id, path)."""
    pairs: list[tuple[str, str, str]] = []
    for fit_id, files_json in frame.loc[
        frame["status"].eq("done"), ["fit_id", "files_json"]
    ].itertuples(index=False):
        pairs += [(str(fit_id), str(p), str(s)) for p, s in json.loads(files_json)]
    pairs.sort(key=lambda t: (t[0], t[1]))
    return merkle_root([merkle_leaf(*t) for t in pairs])


# ----------------------------------------------------------------------------- training summaries
def _mean(values: Iterable[float]) -> float | None:
    arr = [v for v in values if np.isfinite(v)]
    return float(np.mean(arr)) if arr else None


def training_summaries(fits: Sequence[PlannedFit], run_dir: Path) -> dict[str, Any]:
    """Training diagnostics for ``dvc metrics``/``plots`` (docs/3 13.7; no outer-test rows).

    Returns ``summary`` (per stage planned/done; per model wall time, epochs, Picard maxima and
    the M4 final Xi_e), ``curves`` (per model and epoch: mean training objective and mean
    ``es_val`` monitor over completed primary member fits) and ``xi`` (M4 per epoch).
    """
    stages = {s: {"planned": 0, "done": 0} for s in STAGES}
    models: dict[str, dict[str, list[float]]] = {}
    curves: dict[tuple[str, int], list[tuple[float, float]]] = {}
    xi: dict[int, list[float]] = {}
    for fit in fits:
        stages[fit.stage]["planned"] += 1
        marker = fit.completion_path(run_dir)
        if not marker.is_file():
            continue
        stages[fit.stage]["done"] += 1
        if fit.kind == "select":
            continue
        doc = read_json(marker)
        acc = models.setdefault(fit.spec_name, {"wall_s": [], "epochs": [], "picard": [], "xi": []})
        acc["wall_s"].append(float(doc.get("wall_s", np.nan)))
        body = doc.get("fit") or {}
        extras = doc.get("extras") or {}
        if "max_picard_iterations" in extras:
            acc["picard"].append(float(extras["max_picard_iterations"]))
        history = body.get("history") if isinstance(body.get("history"), list) else None
        if history is None:
            continue
        acc["epochs"].append(float(body.get("epochs_run", len(history))))
        xis = [h["xi"] for h in history if h.get("xi") is not None]
        if xis:
            acc["xi"].append(float(xis[-1]))
        if fit.kind != "member" or fit.analysis != "primary":
            continue  # curves: es_val monitor of members only (inner fits monitor inner_val)
        for h in history:
            curves.setdefault((fit.spec_name, int(h["epoch"])), []).append(
                (float(h["train"]), float(h["monitor"]))
            )
            if fit.model == "M4" and fit.ablation is None and h.get("xi") is not None:
                xi.setdefault(int(h["epoch"]), []).append(float(h["xi"]))
    summary_models = {}
    for name, acc in sorted(models.items()):
        summary_models[name] = {
            "fits_done": len(acc["wall_s"]),
            "wall_s_mean": _mean(acc["wall_s"]),
            "wall_s_max": max(acc["wall_s"]) if acc["wall_s"] else None,
            "epochs_mean": _mean(acc["epochs"]),
            "epochs_max": max(acc["epochs"]) if acc["epochs"] else None,
            "picard_max": max(acc["picard"]) if acc["picard"] else None,
            "xi_final_max": max(acc["xi"]) if acc["xi"] else None,
        }
    curve_rows = [
        {
            "model": name,
            "epoch": epoch,
            "train": float(np.mean([t for t, _ in vals])),
            "es_val": float(np.mean([v for _, v in vals])),
            "n": len(vals),
        }
        for (name, epoch), vals in sorted(curves.items())
    ]
    xi_rows = [
        {"epoch": e, "xi_mean": float(np.mean(v)), "xi_max": float(np.max(v)), "n": len(v)}
        for e, v in sorted(xi.items())
    ]
    return {
        "summary": {"stages": stages, "models": summary_models},
        "curves": curve_rows,
        "xi": xi_rows,
    }


def write_training_reports(out_dir: Path, summaries: Mapping[str, Any]) -> dict[str, str]:
    """``reports/training/{summary,curves,xi}.json`` (``cache: false``); returns their SHA-256."""
    out: dict[str, str] = {}
    for name in ("summary", "curves", "xi"):
        path = out_dir / f"{name}.json"
        payload = summaries[name]
        if isinstance(payload, list):  # dvc plots: a top-level list of records
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n")
        else:
            write_json(path, dict(payload))
        out[name] = sha256_file(path)
    return out


# ----------------------------------------------------------------------------- manifest
def host_info() -> dict[str, Any]:
    """Platform, Python and package versions (``importlib.metadata``; nothing is imported)."""

    def version(pkg: str) -> str:
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return "absent"

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in _PACKAGES},
    }


def build_manifest(
    *,
    run_id: str,
    plan_sha256: str,
    plan_header: Mapping[str, Any],
    uv_lock_sha256: str,
    registry: pd.DataFrame,
    registry_sha256: str,
    stage_docs: Mapping[str, Mapping[str, Any]],
    n_workers: int,
    torch_threads_per_worker: int,
    training_reports: Mapping[str, str],
    pilot: Mapping[str, str] | None,
    archive: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the run manifest of docs/3 13.7 (counts and hashes only, no metric)."""
    invocations = [inv for s in STAGES for inv in stage_docs.get(s, {}).get("invocations", [])]
    invocations.sort(key=lambda inv: str(inv["utc_start"]))
    per_stage: dict[str, Any] = {}
    for stage in STAGES:
        rows = registry.loc[registry["stage"].eq(stage)]
        invs = stage_docs.get(stage, {}).get("invocations", [])
        per_stage[stage] = {
            "planned": len(rows),
            "done": int(rows["status"].eq("done").sum()),
            "failures": [inv["failure"] for inv in invs if inv.get("failure")],
            "reruns": [r for inv in invs for r in inv.get("reruns", [])],
            "cleanup": [c for inv in invs for c in inv.get("cleanup", [])],
            "wall_s": float(sum(float(inv.get("wall_s", 0.0)) for inv in invs)),
            "invocations": len(invs),
        }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "plan_sha256": plan_sha256,
        "params_sha256": plan_header["params_sha256"],
        "uv_lock_sha256": uv_lock_sha256,
        "tables": dict(plan_header["tables"]),
        "git": {
            "start": invocations[0]["git_start"] if invocations else None,
            "end": invocations[-1]["git_end"] if invocations else None,
        },
        "host": host_info(),
        "n_workers": n_workers,
        "torch_threads_per_worker": torch_threads_per_worker,
        "stages": per_stage,
        "registry_sha256": registry_sha256,
        "merkle_root": registry_merkle_root(registry),
        "n_leaves": int(registry["n_files"].sum()),
        "training_reports": dict(training_reports),
        "pilot": dict(pilot) if pilot is not None else None,
        "archive": dict(archive) if archive is not None else None,
    }


# ----------------------------------------------------------------------------- pilot, archive
def tree_digests(root: Path) -> dict[str, str]:
    """``relative path -> SHA-256`` of every file under ``root`` (sorted)."""
    files = sorted(p for p in root.rglob("*") if p.is_file())
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in files}


def move_pilot(source: Path, dest: Path) -> dict[str, str]:
    """Move the P6-D9 pilot directory unchanged (docs/3 13.2): copy, verify, then delete.

    Refuses an existing destination; the source is removed only after every digest matches.
    """
    if dest.exists():
        raise FileExistsError(f"{dest} exists")
    before = tree_digests(source)
    if not before:
        raise FileNotFoundError(f"no pilot files under {source}")
    shutil.copytree(source, dest)
    after = tree_digests(dest)
    if after != before:
        shutil.rmtree(dest)
        raise RuntimeError("pilot copy differs from the source; source kept")
    shutil.rmtree(source)
    return after


def archive_run(run_dir: Path, dest: Path) -> dict[str, Any]:
    """Deterministic POSIX ustar archive of the run root (docs/3 13.2); returns path and SHA-256.

    Entries are the regular files sorted by relative path, mtime 0, uid/gid 0, empty owner
    names and mode 0644; directories are implied. Refuses an existing destination.
    """
    if dest.exists():
        raise FileExistsError(f"{dest} exists; an archive is never overwritten")
    dest.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(
        (p for p in run_dir.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(run_dir).as_posix(),
    )
    tmp = dest.with_name(f".{dest.name}.tmp")
    with tarfile.open(tmp, "w", format=tarfile.USTAR_FORMAT) as tar:
        for path in files:
            info = tarfile.TarInfo(f"{run_dir.name}/{path.relative_to(run_dir).as_posix()}")
            info.size = path.stat().st_size
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as handle:
                tar.addfile(info, handle)
    tmp.replace(dest)
    return {"path": str(dest), "sha256": sha256_file(dest), "n_files": len(files)}

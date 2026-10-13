"""Phase 8 orchestrator: runs one stage of the fit plan on a spawn process pool.

Sources: docs/3 section 13.3 (stage order; NN stages refused unless ``checkpoint_1`` is in the
scoring ledger), section 13.7 (a fit is complete when its ``run.json`` exists; resume skips a
complete fit whose params hash, table hashes and coordinates match the plan; a complete fit that
does not match stops the run; an incomplete directory is removed and the fit re-run, logged under
``reruns``; ``--overwrite`` is never used; any worker exception, ``PhiConvergenceError`` or
non-finite loss stops the scheduling of new fits, running fits drain, and the failing fit, seed
and coordinates are reported), P8-D5 (a) (``ProcessPoolExecutor`` with the spawn start method,
``runtime.n_workers`` workers, 1 torch thread each, M1 / M2-XGB in torch-free workers per
ADL-051/-086, ``caffeinate -i`` on macOS), docs/3 13.2 (run root ``<run_root>/<run_id>``).

Scheduling: a fit is submitted once all its dependencies (``plan.dependency_map``) are complete;
among ready fits the lowest plan position goes first. At most ``n_workers`` fits run at once,
split over two lazily started pools: torch-free (baselines, M1, M2-XGB) and torch (NN models,
M6-XGB). Workers receive and return JSON strings only; every artefact is JSON, ``.npz``
(``allow_pickle=False``) or booster JSON, never pickle. BLAS/OpenMP pools are pinned to one
thread in every worker (``OMP_NUM_THREADS`` and friends are set before the workers start).

Importing this module never loads torch or xgboost. Owning phase: 8.
"""

from __future__ import annotations

import heapq
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Final, Literal, ParamSpec, TypeVar

from src.config import Settings, load_settings
from src.evaluation.ledger import Ledger, LedgerError
from src.models.plan import NN_STAGES, STAGES, PlannedFit, dependency_map
from src.models.registry import ModelSpec, read_json, write_json
from src.models.runner import (
    FitRequest,
    Tables,
    run_baseline,
    run_fit,
    run_select,
    run_srnd_member,
)
from src.utils.logging import get_logger, git_state
from src.utils.seeding import fit_seed

__all__ = [
    "THREAD_ENV",
    "ResumeMismatchError",
    "RunContext",
    "RunFailureError",
    "StageRefusedError",
    "StageResult",
    "execute_fit",
    "fit_status",
    "keep_awake",
    "require_checkpoint_1",
    "run_stage",
    "thread_env",
]

#: Pinned before any worker starts (one BLAS/OpenMP thread per process; ADL-051, P8-D5).
THREAD_ENV: Final[Mapping[str, str]] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_STALE_PREFIX: Final[str] = "m6xgb_"  # runner temp dirs of M6-XGB (P6-D7)
FitStatus = Literal["done", "absent", "incomplete"]
_P = ParamSpec("_P")
_T = TypeVar("_T")


class StageRefusedError(RuntimeError):
    """A stage may not start (ledger guard, unmet dependency, dirty tree)."""


class ResumeMismatchError(RuntimeError):
    """A complete fit does not match the plan (docs/3 13.7: stop the run)."""


class RunFailureError(RuntimeError):
    """A fit failed; scheduling stopped and the running fits drained (docs/3 13.7)."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        """Keep the JSON failure report (fit id, coordinates, seed, error)."""
        super().__init__(f"fit {report['fit_id']} failed: {report['error']}: {report['message']}")
        self.report = dict(report)


@dataclass(frozen=True)
class RunContext:
    """Everything a worker needs, as plain strings (sent as JSON, never pickled objects).

    ``srnd_path`` and ``srnd_sha256`` name the S-RND frame (docs/3 13.6); empty when the stage
    has no S-RND fit.
    """

    params_path: str
    site_path: str
    fold_path: str
    run_dir: str
    params_sha256: str
    site_sha256: str
    fold_sha256: str
    log_level: str = "INFO"
    srnd_path: str = ""
    srnd_sha256: str = ""

    def to_json(self) -> str:
        """JSON string for the worker initializer."""
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> RunContext:
        """Inverse of :meth:`to_json`."""
        return cls(**json.loads(text))


@dataclass
class StageResult:
    """Outcome of one :func:`run_stage` invocation (also written to ``stages/<stage>.json``)."""

    stage: str
    planned: int
    already_done: int
    executed: int = 0
    reruns: list[dict[str, str]] = field(default_factory=list)
    cleanup: list[dict[str, str]] = field(default_factory=list)
    failure: dict[str, Any] | None = None
    wall_s: float = 0.0
    workers: list[int] = field(default_factory=list)


# ----------------------------------------------------------------------------- environment
@contextmanager
def thread_env() -> Iterator[None]:
    """Set :data:`THREAD_ENV` for the duration (spawned workers inherit it), then restore."""
    saved = {k: os.environ.get(k) for k in THREAD_ENV}
    os.environ.update(THREAD_ENV)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def keep_awake(enabled: bool = True) -> Iterator[subprocess.Popen[bytes] | None]:
    """``caffeinate -i -w <pid>`` on macOS while the block runs (P8-D5); no-op elsewhere."""
    exe = shutil.which("caffeinate") if enabled and sys.platform == "darwin" else None
    proc = subprocess.Popen([exe, "-i", "-w", str(os.getpid())]) if exe else None
    try:
        yield proc
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=10)


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------- guard
def require_checkpoint_1(ledger_path: Path, run_id: str) -> None:
    """Refuse NN training unless this run's ledger records ``checkpoint_1`` (docs/3 13.3/13.4)."""
    if not ledger_path.is_file():
        raise StageRefusedError(f"NN stages need checkpoint_1; no scoring ledger at {ledger_path}")
    try:
        ledger = Ledger.load(ledger_path)
    except (LedgerError, ValueError) as exc:
        raise StageRefusedError(f"scoring ledger invalid: {exc}") from exc
    if ledger.header["run_id"] != run_id:
        raise StageRefusedError(f"ledger belongs to {ledger.header['run_id']}, not {run_id}")
    if not ledger.has("checkpoint_1"):
        raise StageRefusedError("NN stages are refused until checkpoint_1 is recorded")


# ----------------------------------------------------------------------------- completion
def _mismatch(fit: PlannedFit, what: str) -> ResumeMismatchError:
    return ResumeMismatchError(f"{fit.fit_id}: complete fit does not match the plan ({what})")


def fit_status(fit: PlannedFit, run_dir: Path, ctx: RunContext) -> FitStatus:
    """``done`` (verified against the plan), ``absent`` or ``incomplete`` (docs/3 13.7)."""
    marker = fit.completion_path(run_dir)
    if marker.is_file():
        doc = read_json(marker)
        if fit.kind == "select":
            got = (doc.get("model"), doc.get("analysis"), doc.get("family"))
            coords = (doc.get("repeat"), doc.get("outer_fold"))
            params = doc.get("params_sha256")
            want_ids: tuple[Any, ...] = (fit.model, fit.analysis, fit.family)
        else:
            head = doc.get("header", {})
            got = tuple(
                head.get(k) for k in ("spec", "analysis", "family", "kind", "index", "grid")
            )
            coords = (head.get("repeat"), head.get("outer_fold"))
            params = head.get("params_sha256")
            want_ids = (fit.spec_name, fit.analysis, fit.family, fit.kind, fit.index, fit.grid)
        if params != ctx.params_sha256:
            raise _mismatch(fit, "params_sha256")
        if (doc.get("site_table_sha256"), doc.get("fold_table_sha256")) != (
            ctx.site_sha256,
            ctx.fold_sha256,
        ):
            raise _mismatch(fit, "table hashes")
        if got != want_ids or coords != (fit.repeat, fit.fold):
            raise _mismatch(fit, "coordinates")
        return "done"
    if fit.kind != "select" and (run_dir / fit.fit_id).exists():
        return "incomplete"
    return "absent"


# ----------------------------------------------------------------------------- execution
def execute_fit(
    fit: PlannedFit, tables: Tables, settings: Settings, run_dir: Path, logger: logging.Logger
) -> None:
    """Run one planned fit through :mod:`src.models.runner` (never with ``overwrite``)."""
    if fit.kind == "outer":
        run_baseline(fit.model, fit.family, fit.repeat, fit.fold, tables, settings, run_dir)
    elif fit.family == "S-RND":
        run_srnd_member(fit.model, fit.index, tables, settings, run_dir)
    elif fit.kind == "select":
        run_select(
            ModelSpec(fit.model), fit.analysis, fit.family, fit.repeat, fit.fold, settings, run_dir
        )
    else:
        req = FitRequest(
            spec=ModelSpec(fit.model, fit.ablation),
            analysis=fit.analysis,
            family=fit.family,
            repeat=fit.repeat,
            outer_fold=fit.fold,
            kind=fit.kind,
            index=fit.index,
            grid=fit.grid,
        )
        run_fit(req, tables, settings, run_dir, logger=logger)


_WORKER: dict[str, Any] = {}


def _init_worker(ctx_json: str, torch_free: bool, file_log: bool = True) -> None:
    """Pool initializer: load and verify settings and tables once per worker process."""
    ctx = RunContext.from_json(ctx_json)
    if torch_free and "torch" in sys.modules:
        raise RuntimeError("torch-free worker started with torch loaded (ADL-051)")
    settings = load_settings(ctx.params_path)
    if settings.params_sha256 != ctx.params_sha256:
        raise RuntimeError("worker params_sha256 differs from the plan")
    tables = Tables.load(
        Path(ctx.site_path), Path(ctx.fold_path), Path(ctx.srnd_path) if ctx.srnd_path else None
    )
    if (tables.site_sha256, tables.fold_sha256) != (ctx.site_sha256, ctx.fold_sha256):
        raise RuntimeError("worker table hashes differ from the plan")
    if ctx.srnd_path and tables.srnd_sha256 != ctx.srnd_sha256:
        raise RuntimeError("worker srnd_frame hash differs from the plan")
    run_dir = Path(ctx.run_dir)
    if file_log:  # pool workers: one log file each, nothing on the terminal
        logger = logging.getLogger("annxs.worker")
        logger.setLevel(ctx.log_level)
        logger.propagate = False
        log_dir = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_dir / f"worker_{os.getpid()}.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    else:
        logger = get_logger("annxs", ctx.log_level)
    _WORKER.clear()
    _WORKER.update(settings=settings, tables=tables, run_dir=run_dir, logger=logger)


def _execute(fit_json: str) -> str:
    """Worker entry point: JSON in, JSON out; exceptions are reported, never pickled."""
    fit = PlannedFit.from_json(json.loads(fit_json))
    start = time.perf_counter()
    out: dict[str, Any] = {"fit_id": fit.fit_id, "pid": os.getpid(), "ok": True}
    try:
        execute_fit(
            fit, _WORKER["tables"], _WORKER["settings"], _WORKER["run_dir"], _WORKER["logger"]
        )
    except BaseException as exc:  # reported to the scheduler, which stops (docs/3 13.7)
        out.update(
            ok=False, error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()
        )
    out["torch_loaded"] = "torch" in sys.modules
    out["wall_s"] = time.perf_counter() - start
    return json.dumps(out)


class _InlineExecutor(Executor):
    """Synchronous executor for tests and debugging (no isolation; torch-free check skipped)."""

    def submit(self, fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs) -> Future[_T]:
        future: Future[_T] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # pragma: no cover - _execute never raises
            future.set_exception(exc)
        return future


def _seed_of(fit: PlannedFit, settings: Settings) -> int | None:
    if fit.kind not in ("inner", "member"):
        return None
    return fit_seed(settings.protocol.seeds, fit.family, fit.repeat, fit.fold, fit.kind, fit.index)


def _failure_report(
    fit: PlannedFit, result: Mapping[str, Any], settings: Settings
) -> dict[str, Any]:
    return {
        "fit_id": fit.fit_id,
        "coordinates": fit.to_json(),
        "seed": _seed_of(fit, settings),
        "error": result.get("error", "unknown"),
        "message": result.get("message", ""),
        "traceback": result.get("traceback", ""),
        "utc": _utc(),
    }


# ----------------------------------------------------------------------------- stage runner
def run_stage(
    stage: str,
    fits: Sequence[PlannedFit],
    *,
    ctx: RunContext,
    settings: Settings,
    run_id: str,
    ledger_path: Path,
    n_workers: int,
    inline: bool = False,
    logger: logging.Logger | None = None,
) -> StageResult:
    """Run every incomplete fit of ``stage`` (resume semantics of docs/3 13.7).

    Raises :class:`StageRefusedError` (guard or unmet cross-stage dependency),
    :class:`ResumeMismatchError` (a complete fit disagrees with the plan) or
    :class:`RunFailureError` after the running fits drained. The invocation record is appended to
    ``<run_dir>/stages/<stage>.json`` in every case except a refusal.
    """
    log = logger or logging.getLogger("annxs")
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected {STAGES}")
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    if stage in NN_STAGES:
        require_checkpoint_1(ledger_path, run_id)
    run_dir = Path(ctx.run_dir)
    deps = dependency_map(fits)
    by_id = {f.fit_id: f for f in fits}
    todo_all = [f for f in fits if f.stage == stage]
    status = {f.fit_id: fit_status(f, run_dir, ctx) for f in todo_all}
    external = sorted({d for f in todo_all for d in deps[f.fit_id] if by_id[d].stage != stage})
    unmet = [d for d in external if fit_status(by_id[d], run_dir, ctx) != "done"]
    if unmet:
        raise StageRefusedError(
            f"{len(unmet)} dependencies of {stage} incomplete, e.g. {unmet[:3]}"
        )
    result = StageResult(stage, len(todo_all), sum(s == "done" for s in status.values()))
    start, utc_start, git_start = time.perf_counter(), _utc(), git_state(settings.runtime.repo_root)
    for f in todo_all:
        if status[f.fit_id] == "incomplete":
            shutil.rmtree(run_dir / f.fit_id)
            result.reruns.append(
                {"fit_id": f.fit_id, "reason": "incomplete directory", "utc": _utc()}
            )
    for stale in sorted(p for p in run_dir.glob(f"{_STALE_PREFIX}*") if p.is_dir()):
        shutil.rmtree(stale)
        result.cleanup.append(
            {"path": stale.name, "reason": "stale M6-XGB work dir", "utc": _utc()}
        )
    todo = [f for f in todo_all if status[f.fit_id] != "done"]
    done = {f.fit_id for f in fits if f.stage != stage} | {
        fid for fid, s in status.items() if s == "done"
    }
    try:
        _schedule(todo, deps, done, ctx, settings, n_workers, inline, result, log)
    finally:
        result.wall_s = time.perf_counter() - start
        _record_invocation(run_dir, result, utc_start, git_start, settings, n_workers)
    if result.failure is not None:
        raise RunFailureError(result.failure)
    return result


def _schedule(
    todo: Sequence[PlannedFit],
    deps: Mapping[str, tuple[str, ...]],
    done: set[str],
    ctx: RunContext,
    settings: Settings,
    n_workers: int,
    inline: bool,
    result: StageResult,
    log: logging.Logger,
) -> None:
    pos = {f.fit_id: i for i, f in enumerate(todo)}
    waiting: dict[str, set[str]] = {}
    dependents: dict[str, list[str]] = {}
    ready: list[int] = []
    for f in todo:
        need = {d for d in deps[f.fit_id] if d not in done}
        unknown = [d for d in need if d not in pos]
        if unknown:
            raise StageRefusedError(
                f"{f.fit_id}: dependency {unknown[0]} is neither done nor planned"
            )
        waiting[f.fit_id] = need
        for d in need:
            dependents.setdefault(d, []).append(f.fit_id)
        if not need:
            heapq.heappush(ready, pos[f.fit_id])
    pools: dict[bool, Executor] = {}

    def pool(torch_free: bool) -> Executor:
        if torch_free not in pools:
            pools[torch_free] = (
                _InlineExecutor()
                if inline
                else ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=get_context("spawn"),
                    initializer=_init_worker,
                    initargs=(ctx.to_json(), torch_free),
                )
            )
        return pools[torch_free]

    if inline:
        _init_worker(ctx.to_json(), False, False)
    running: dict[Future[str], PlannedFit] = {}
    pids: set[int] = set()
    try:
        with thread_env():
            while ready or running:
                while result.failure is None and ready and len(running) < n_workers:
                    fit = todo[heapq.heappop(ready)]
                    running[pool(fit.torch_free).submit(_execute, json.dumps(fit.to_json()))] = fit
                if not running:
                    break
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for fut in sorted(finished, key=lambda x: pos[running[x].fit_id]):
                    fit = running.pop(fut)
                    out = json.loads(fut.result())
                    pids.add(int(out["pid"]))
                    if out["ok"] and fit.torch_free and out["torch_loaded"] and not inline:
                        out.update(
                            ok=False,
                            error="IsolationError",
                            message="torch loaded in a torch-free worker",
                        )
                    if not out["ok"]:
                        if result.failure is None:
                            result.failure = _failure_report(fit, out, settings)
                            log.error("fit failed, draining: %s (%s)", fit.fit_id, out["error"])
                        continue
                    result.executed += 1
                    done.add(fit.fit_id)
                    for child in dependents.get(fit.fit_id, ()):
                        waiting[child].discard(fit.fit_id)
                        if not waiting[child]:
                            heapq.heappush(ready, pos[child])
                    if result.executed % 100 == 0:
                        log.info(
                            "%s: %d/%d fits executed", result.stage, result.executed, len(todo)
                        )
    finally:
        for ex in pools.values():
            ex.shutdown(wait=True, cancel_futures=True)
        result.workers = sorted(pids)
    if result.failure is None and result.executed != len(todo):
        raise RuntimeError(
            f"{len(todo) - result.executed} fits never became ready (dependency cycle)"
        )


def _record_invocation(
    run_dir: Path,
    result: StageResult,
    utc_start: str,
    git_start: Mapping[str, Any],
    settings: Settings,
    n_workers: int,
) -> None:
    path = run_dir / "stages" / f"{result.stage}.json"
    doc: dict[str, Any] = read_json(path) if path.is_file() else {"stage": result.stage}
    invocations: list[dict[str, Any]] = list(doc.get("invocations", []))
    invocations.append(
        {
            "utc_start": utc_start,
            "utc_end": _utc(),
            "git_start": dict(git_start),
            "git_end": git_state(settings.runtime.repo_root),
            "planned": result.planned,
            "already_done": result.already_done,
            "executed": result.executed,
            "reruns": result.reruns,
            "cleanup": result.cleanup,
            "failure": result.failure,
            "wall_s": result.wall_s,
            "n_workers": n_workers,
            "torch_threads_per_worker": settings.runtime.torch_threads_per_worker,
            "worker_pids": len(result.workers),
        }
    )
    doc["invocations"] = invocations
    write_json(path, doc)

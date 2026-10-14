"""Scoring-event runtime shared by ``scripts/audit.py`` and ``scripts/evaluate.py`` (docs/3 13.4).

Sources: docs/3 section 13.4 (an event refuses unless every earlier ledger step is recorded; the
ledger header carries ``run_id``, ``plan_sha256``, ``params_sha256`` and ``uv_lock_sha256``; events
need a clean git tree and a commit; every event checks its inputs against the sealed hashes; an
event that crashes before its record is appended may be re-run on the same sealed inputs and the
crash is logged; a recorded event is never re-run; a checkpoint carries PI approval text and no
number), section 13.11 (no protocol value changes).

Order of an event:

1. :func:`open_event` loads and verifies the ledger, checks the header against the current
   plan, ``params.yaml`` and ``uv.lock``, checks that ``event`` is the next ledger step and that
   the git tree is clean, and logs the attempt in ``reports/results/event_attempts.json``.
2. :func:`sealed_entries` re-hashes every prediction input and compares it with the seal.
3. The caller computes and writes its outputs.
4. :func:`close_event` appends the record, using the git state captured in step 1 (the outputs
   written in step 3 are what make the tree dirty; they are committed with the ledger).

The attempts log stands in for the run manifest named in docs/3 13.4 because the manifest is a
DVC out of the ``track`` stage (C3b-h). This module reads no prediction or target value.
Torch-free and xgboost-free. Owning phase: 8.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd

from src.config import Settings
from src.data.io import sha256_file
from src.evaluation.ledger import CHECKPOINTS, Ledger, LedgerError
from src.models.plan import (
    EXPECTED_COUNTS_13_6,
    blocks_from_folds,
    build_plan,
    count_groups,
    expected_counts,
    plan_document,
    plan_sha256,
    run_id_for,
    table_hashes,
    table_paths,
)
from src.models.registry import write_json
from src.utils.logging import git_state

__all__ = [
    "ATTEMPTS_FILE",
    "LEDGER_FILE",
    "EventRefusedError",
    "EventSession",
    "PlanIdentity",
    "check_checkpoint_note",
    "close_event",
    "file_entry",
    "json_safe",
    "open_event",
    "plan_identity",
    "rel_path",
    "results_dir",
    "sealed_entries",
]

LEDGER_FILE: Final[str] = "scoring_ledger.json"
ATTEMPTS_FILE: Final[str] = "event_attempts.json"
#: A decimal number in a checkpoint note (a checkpoint contains no number; docs/3 13.4).
_DECIMAL: Final[re.Pattern[str]] = re.compile(r"\d[.,]\d|\d\s*%")


class EventRefusedError(RuntimeError):
    """A scoring event may not run (ledger order, header, dirty tree, unsealed input)."""


@dataclass
class EventSession:
    """An opened event: the verified ledger and the git state captured before any output."""

    event: str
    repo_root: Path
    results_dir: Path
    ledger: Ledger
    git: dict[str, Any]
    params_sha256: str
    prior_attempts: int


def results_dir(settings: Settings) -> Path:
    """``reports/results`` (ledger, event outputs, attempts log)."""
    return settings.runtime.reports_dir / "results"


@dataclass(frozen=True)
class PlanIdentity:
    """The recomputed fit plan's identity and the frozen tables it hashes (docs/3 13.6)."""

    run_id: str
    sha256: str
    paths: dict[str, Path]
    hashes: dict[str, str]


def plan_identity(settings: Settings, *, reduced: bool = False) -> PlanIdentity:
    """Recompute the plan from ``params.yaml`` and the frozen tables (as ``run_experiment``).

    An event compares it with the ledger header, so a changed input table refuses the event.
    ``reduced`` (dry runs) skips only the comparison with the docs/3 13.6 count table.
    """
    paths = table_paths(settings.runtime.processed_dir)
    hashes = table_hashes(paths)
    folds = pd.read_parquet(paths["fold_table"], columns=["family", "repeat", "outer_fold"])
    blocks = blocks_from_folds(folds)
    fits = build_plan(settings.protocol, blocks)
    counts = count_groups(fits)
    if counts != expected_counts(settings.protocol, len(blocks)):
        raise RuntimeError("plan counts differ from the formula of docs/3 13.6")
    if not reduced and counts != dict(EXPECTED_COUNTS_13_6):
        raise RuntimeError("plan counts differ from the docs/3 13.6 table")
    sha = plan_sha256(plan_document(fits, settings.params_sha256, hashes))
    return PlanIdentity(run_id_for(sha), sha, dict(paths), dict(hashes))


def rel_path(path: Path, root: Path) -> str:
    """Repository-relative POSIX path (the ledger's convention); symlinks are not resolved.

    DVC-tracked files are symlinks into the cache outside the repository, so the path is
    normalised lexically (``os.path.abspath``) instead of with :meth:`Path.resolve`.
    """
    rel = Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root)))
    return rel.as_posix()


def file_entry(path: Path, root: Path) -> dict[str, str]:
    """``{path, sha256}`` of a repository file, as listed in ledger inputs and outputs."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing")
    return {"path": rel_path(path, root), "sha256": sha256_file(path)}


def json_safe(value: object) -> object:
    """NumPy scalars to Python; non-finite floats to ``None`` (strict JSON for event outputs)."""
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, str | bytes):
        value = item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tracked(root: Path, path: Path) -> bool:
    """Whether ``path`` is tracked by git in ``root`` (an untracked ledger is not committed)."""
    try:
        subprocess.run(  # fixed argv, no shell
            ["git", "ls-files", "--error-unmatch", "--", rel_path(path, root)],
            cwd=root,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError, ValueError):
        return False
    return True


def _attempts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not (isinstance(doc, dict) and isinstance(doc.get("attempts"), list)):
        raise EventRefusedError(f"{path} is malformed")
    return list(doc["attempts"])


def open_event(settings: Settings, event: str, *, run_id: str, plan_sha256: str) -> EventSession:
    """Verify the ledger and the tree for ``event`` and log the attempt (docs/3 13.4)."""
    rt = settings.runtime
    rdir = results_dir(settings)
    path = rdir / LEDGER_FILE
    if not path.is_file():
        raise EventRefusedError(f"{event} refused: no scoring ledger at {path}; seal first")
    try:
        ledger = Ledger.load(path)
    except (LedgerError, ValueError) as exc:
        raise EventRefusedError(f"{event} refused: scoring ledger invalid: {exc}") from exc
    head = ledger.header
    want = {
        "run_id": run_id,
        "plan_sha256": plan_sha256,
        "params_sha256": settings.params_sha256,
        "uv_lock_sha256": sha256_file(rt.repo_root / "uv.lock"),
    }
    bad = sorted(k for k, v in want.items() if head.get(k) != v)
    if bad:
        raise EventRefusedError(f"{event} refused: ledger header differs in {bad}")
    nxt = ledger.next_expected()
    if event != nxt:
        done = event in ledger.recorded() and event != "seal"
        why = "it is recorded and is never re-run" if done else f"the next ledger step is {nxt!r}"
        raise EventRefusedError(f"{event} refused: {why} (docs/3 13.4)")
    git = git_state(rt.repo_root)
    clean = not git["dirty"] and git["commit"] not in ("", "unknown")
    if not (clean and _tracked(rt.repo_root, path)):
        raise EventRefusedError(
            f"{event} refused: events need a clean, committed tree (ledger included)"
        )
    log_path = rdir / ATTEMPTS_FILE
    attempts = _attempts(log_path)
    prior = sum(1 for a in attempts if a.get("event") == event and a.get("seq") is None)
    attempts.append({"event": event, "utc": _now(), "git_commit": git["commit"], "seq": None})
    write_json(log_path, {"attempts": attempts})
    return EventSession(
        event=event,
        repo_root=rt.repo_root,
        results_dir=rdir,
        ledger=ledger,
        git=dict(git),
        params_sha256=settings.params_sha256,
        prior_attempts=prior,
    )


def sealed_entries(session: EventSession, paths: Sequence[Path]) -> list[dict[str, str]]:
    """Ledger entries of prediction inputs; each must be sealed and still hash as sealed."""
    sealed = session.ledger.sealed()
    out = []
    for path in paths:
        entry = file_entry(path, session.repo_root)
        want = sealed.get(entry["path"])
        if want is None:
            raise EventRefusedError(f"{entry['path']} is not sealed (docs/3 13.4)")
        if entry["sha256"] != want:
            raise EventRefusedError(f"{entry['path']} differs from its sealed SHA-256")
        out.append(entry)
    return out


def check_checkpoint_note(note: str) -> str:
    """Check that a checkpoint note is PI approval text without a number (13.4; C3b-j)."""
    text = note.strip()
    if not text:
        raise EventRefusedError("a checkpoint needs the PI approval text (--note)")
    if _DECIMAL.search(text):
        raise EventRefusedError("a checkpoint note contains no number (docs/3 13.4)")
    return text


def close_event(
    session: EventSession,
    *,
    inputs: Sequence[Mapping[str, str]],
    outputs: Sequence[Mapping[str, str]],
    note: str,
) -> dict[str, Any]:
    """Append the event record with the git state captured by :func:`open_event`."""
    if session.event in CHECKPOINTS:
        note = check_checkpoint_note(note)
    if session.prior_attempts:
        note = f"{note} [re-run after {session.prior_attempts} unrecorded attempt(s)]".strip()
    rec = session.ledger.append(
        session.event,
        git=session.git,
        params_sha256=session.params_sha256,
        inputs=inputs,
        outputs=outputs,
        note=note,
    )
    log_path = session.results_dir / ATTEMPTS_FILE
    attempts = _attempts(log_path)
    for item in reversed(attempts):
        if item.get("event") == session.event and item.get("seq") is None:
            item["seq"] = rec["seq"]
            break
    write_json(log_path, {"attempts": attempts})
    return rec

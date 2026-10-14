"""Thin CLI: PI checkpoints and Event 2 (M* selection); Event 3 follows in C3c (docs/3 13.4).

Sources: docs/3 section 13.3 (stage 7 ``select_mstar``: ``reports/results/mstar.json``; ledger
``event_2``, then ``checkpoint_2``; stage 8 ``evaluate`` = Event 3), section 13.4 (a checkpoint
is appended by ``evaluate.py checkpoint --after event_N --note "<PI approval text>"`` only after
written PI approval and contains no number; Event 2 = ``select-mstar``: M* per ``k in {dbar, A}``
by protocol 4.4 on S-ID OOF; every event checks its inputs against the sealed hashes), protocol
4.3-4.7. Libraries: :mod:`src.evaluation.event2`, :mod:`src.evaluation.events`.

Usage::

    python scripts/evaluate.py checkpoint --after event_1 --note "<PI approval text>"
    python scripts/evaluate.py select-mstar
    python scripts/evaluate.py checkpoint --after event_2 --note "<PI approval text>"
    python scripts/evaluate.py evaluate       # Event 3: locked until C3c

Prints only paths and ledger sequence numbers; the PI reads ``mstar.json``. Owning phase: 8.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.config import PARAMS_PATH, Settings, load_settings
from src.evaluation.event2 import run_event2
from src.evaluation.events import (
    PlanIdentity,
    check_checkpoint_note,
    close_event,
    file_entry,
    json_safe,
    open_event,
    plan_identity,
    sealed_entries,
)
from src.models.predict import predictions_dir
from src.models.registry import write_json
from src.utils.logging import get_logger

LOCK_MESSAGE = "Event 3 (`evaluate.py evaluate`) is implemented in Phase 8 Batch C3c"
MSTAR_JSON = "mstar.json"


def cmd_checkpoint(settings: Settings, ident: PlanIdentity, args: argparse.Namespace) -> int:
    """Append ``checkpoint_N`` after ``event_N`` with the PI approval text (docs/3 13.4)."""
    note = check_checkpoint_note(args.note)
    event = "checkpoint_" + args.after.removeprefix("event_")
    session = open_event(settings, event, run_id=ident.run_id, plan_sha256=ident.sha256)
    rec = close_event(session, inputs=[], outputs=[], note=note)
    print(f"{event} recorded as ledger seq {rec['seq']}")
    return 0


def cmd_select_mstar(settings: Settings, ident: PlanIdentity, args: argparse.Namespace) -> int:
    """Event 2: M* per target on S-ID OOF predictions; writes ``mstar.json``."""
    rt = settings.runtime
    root = rt.repo_root
    session = open_event(settings, "event_2", run_id=ident.run_id, plan_sha256=ident.sha256)
    cands = tuple(settings.protocol.evaluation.mstar_candidates)
    pdir = predictions_dir(rt.processed_dir)
    pred_paths = {m: pdir / f"{m}.parquet" for m in cands}
    inputs = [
        *sealed_entries(session, list(pred_paths.values())),
        file_entry(ident.paths["site_table"], root),
        file_entry(ident.paths["fold_table"], root),
    ]
    site_table = pd.read_parquet(ident.paths["site_table"])
    folds = pd.read_parquet(ident.paths["fold_table"])
    frames: dict[str, pd.DataFrame] = {str(m): pd.read_parquet(p) for m, p in pred_paths.items()}
    result = run_event2(frames, site_table, folds, settings.protocol)
    doc: dict[str, Any] = {
        "run_id": ident.run_id,
        "plan_sha256": ident.sha256,
        "params_sha256": settings.params_sha256,
        "inputs": inputs,
        **result,
    }
    path = session.results_dir / MSTAR_JSON
    write_json(path, cast(dict[str, Any], json_safe(doc)))
    outputs = [file_entry(path, root)]
    rec = close_event(
        session,
        inputs=inputs,
        outputs=outputs,
        note="Event 2: M* per target on S-ID OOF (docs/3 13.4, protocol 4.4)",
    )
    print(f"event_2 recorded as ledger seq {rec['seq']}")
    print(f"  {outputs[0]['path']}  sha256 {outputs[0]['sha256']}")
    return 0


def cmd_evaluate(settings: Settings, ident: PlanIdentity, args: argparse.Namespace) -> int:
    """Event 3 (docs/3 13.4): not yet implemented (C3c)."""
    raise NotImplementedError(LOCK_MESSAGE)


_COMMANDS = {
    "checkpoint": cmd_checkpoint,
    "select-mstar": cmd_select_mstar,
    "evaluate": cmd_evaluate,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scoring events 2-3 and PI checkpoints.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in _COMMANDS:
        p = sub.add_parser(name)
        p.add_argument("--params", type=Path, default=PARAMS_PATH)
        p.add_argument("--reduced", action="store_true", help="dry run with reduced params")
        p.add_argument("--log-level", default=None)
        if name == "checkpoint":
            p.add_argument("--after", required=True, choices=("event_1", "event_2"))
            p.add_argument("--note", required=True, help="the PI approval text (no numbers)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch; return the exit status."""
    args = _parser().parse_args(argv)
    if args.command == "evaluate":
        raise NotImplementedError(LOCK_MESSAGE)
    settings = load_settings(args.params)
    get_logger("annxs", args.log_level or settings.runtime.log_level)
    ident = plan_identity(settings, reduced=args.reduced)
    return _COMMANDS[args.command](settings, ident, args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Thin CLI: Event 1, the spatial and network dependence audit on S-ID (docs/3 13.4).

Sources: docs/3 section 13.3 (stage 4 ``audit``: ``reports/results/audit.json``; ledger
``event_1``, then ``checkpoint_1`` via ``scripts/evaluate.py checkpoint``), section 13.4 (Event 1
content; refused unless the audit inputs are sealed and every earlier ledger step is recorded; a
recorded event is never re-run), protocol 3.6. The library is :mod:`src.evaluation.event1` on
top of :mod:`src.evaluation.dependence_audit` (Phase 7); the event runtime is
:mod:`src.evaluation.events`.

Usage::

    python scripts/audit.py            # production (clean, committed tree)

Writes ``reports/results/audit.json`` and ``reports/results/audit_strata.parquet``, appends
``event_1`` to ``reports/results/scoring_ledger.json`` and prints only paths and the ledger
sequence number; the PI reads the results in ``audit.json``. Commit the outputs, the ledger and
``event_attempts.json`` afterwards. Owning phase: 8.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.config import PARAMS_PATH, load_settings
from src.data.folds import analysis_frame
from src.data.io import sha256_file, unit_manifest_sha256, write_parquet
from src.data.network import read_network
from src.evaluation.event1 import run_event1
from src.evaluation.events import (
    close_event,
    file_entry,
    json_safe,
    open_event,
    plan_identity,
    rel_path,
    sealed_entries,
)
from src.models.predict import AUDIT_FILE, predictions_dir
from src.models.registry import write_json
from src.utils.logging import get_logger

AUDIT_JSON = "audit.json"
STRATA_FILE = "audit_strata.parquet"


def main(argv: Sequence[str] | None = None) -> int:
    """Run Event 1 and append ``event_1``; return the exit status."""
    parser = argparse.ArgumentParser(description="Event 1: dependence audit (docs/3 13.4).")
    parser.add_argument("--params", type=Path, default=PARAMS_PATH)
    parser.add_argument("--reduced", action="store_true", help="dry run with reduced params")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)
    settings = load_settings(args.params)
    rt = settings.runtime
    log = get_logger("annxs", args.log_level or rt.log_level)
    ident = plan_identity(settings, reduced=args.reduced)
    session = open_event(settings, "event_1", run_id=ident.run_id, plan_sha256=ident.sha256)
    root = rt.repo_root
    preds_path = predictions_dir(rt.processed_dir) / AUDIT_FILE
    nhd_dir = rt.raw_dir / "nhdplusv21"
    inputs = [
        *sealed_entries(session, [preds_path]),
        file_entry(ident.paths["site_table"], root),
        file_entry(ident.paths["fold_table"], root),
        file_entry(nhd_dir / "MANIFEST.sha256", root),
    ]
    site_table = pd.read_parquet(ident.paths["site_table"])
    folds = pd.read_parquet(ident.paths["fold_table"])
    preds = pd.read_parquet(preds_path)
    comids = analysis_frame(site_table, "primary")["comid"].astype(int).tolist()
    log.info("event_1: reading the NHDPlusV2.1 network (%d site COMIDs)", len(set(comids)))
    network = read_network(nhd_dir, comids)
    result = run_event1(site_table, folds, preds, network, settings.protocol)
    out_dir = session.results_dir
    strata_path = out_dir / STRATA_FILE
    write_parquet(
        result.strata,
        strata_path,
        stage="event_1",
        params_sha256=settings.params_sha256,
        inputs={e["path"]: e["sha256"] for e in inputs},
    )
    doc: dict[str, Any] = {
        "run_id": ident.run_id,
        "plan_sha256": ident.sha256,
        "params_sha256": settings.params_sha256,
        "nhdplusv21_manifest_sha256": unit_manifest_sha256(nhd_dir),
        "inputs": inputs,
        "strata": {"path": rel_path(strata_path, root), "sha256": sha256_file(strata_path)},
        **result.document,
    }
    audit_path = out_dir / AUDIT_JSON
    write_json(audit_path, cast(dict[str, Any], json_safe(doc)))
    outputs = [file_entry(audit_path, root), file_entry(strata_path, root)]
    rec = close_event(
        session,
        inputs=inputs,
        outputs=outputs,
        note="Event 1: spatial and network dependence audit on S-ID (docs/3 13.4, protocol 3.6)",
    )
    print(f"event_1 recorded as ledger seq {rec['seq']}")
    for item in outputs:
        print(f"  {item['path']}  sha256 {item['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

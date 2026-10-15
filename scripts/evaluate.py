"""Thin CLI: PI checkpoints, Event 2 (M* selection) and Event 3 (evaluation) (docs/3 13.4).

Sources: docs/3 section 13.3 (stage 7 ``select_mstar``: ``reports/results/mstar.json``; ledger
``event_2``, then ``checkpoint_2``; stage 8 ``evaluate`` = Event 3), section 13.4 (a checkpoint
is appended by ``evaluate.py checkpoint --after event_N --note "<PI approval text>"`` only after
written PI approval and contains no number; Event 2 = ``select-mstar``: M* per ``k in {dbar, A}``
by protocol 4.4 on S-ID OOF; Event 3 = ``evaluate``: everything else in protocol 4.3-4.7; every
event checks its inputs against the sealed hashes), section 13.13 (Event 3 computation rules),
protocol 4.3-4.7. Libraries: :mod:`src.evaluation.event2`, :mod:`src.evaluation.event3`,
:mod:`src.evaluation.event3_workers`, :mod:`src.evaluation.events`.

Usage::

    python scripts/evaluate.py checkpoint --after event_1 --note "<PI approval text>"
    python scripts/evaluate.py select-mstar
    python scripts/evaluate.py checkpoint --after event_2 --note "<PI approval text>"
    python scripts/evaluate.py evaluate       # Event 3

Prints only paths and ledger sequence numbers; the PI reads the result files. Event 3 runs
``ln kappa`` (torch) and the AOA blocks (xgboost) in spawned workers; this process imports
neither (ADL-051). Owning phase: 8.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import PARAMS_PATH, Settings, load_settings
from src.data.external import read_hyg_manning
from src.data.folds import analysis_frame
from src.data.io import write_parquet
from src.data.network import NetworkInputs, network_inputs
from src.data.nhdplus import read_nhd_table
from src.evaluation.event2 import run_event2
from src.evaluation.event3 import OUTCOME_TEXT, AoaFn, Event3Inputs, LogKappa, run_event3
from src.evaluation.event3_workers import aoa_job, kappa_job, run_jobs
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

MSTAR_JSON = "mstar.json"
#: The 20 sealed prediction files read by Event 3 (docs/3 13.9), by file stem.
EVENT3_PREDICTIONS: Final[tuple[str, ...]] = (
    "A_J",
    "M-null",
    "M0",
    "M1",
    "M1_A4",
    "M2-MLP",
    "M2-MLP_A4",
    "M2-XGB",
    "M2-XGB_A4",
    "M3",
    "M4",
    "M5",
    "M5_A1",
    "M5_A2",
    "M5_A3-SP",
    "M5_A3-SR",
    "M5_A4",
    "M5_wA1",
    "M6-NN",
    "M6-XGB",
)
#: Event 3 result documents (docs/3 13.13 s), written as ``event3_<name>.json``.
EVENT3_DOCS: Final[tuple[str, ...]] = (
    "hypotheses",
    "uq",
    "consistency",
    "dependence",
    "descriptive",
)
EVENT3_MATRIX = "event3_matrix.parquet"
EVENT3_INDEX = "event3.json"
AUDIT_JSON = "audit.json"
STRATA_FILE = "audit_strata.parquet"
HYG_FILE = "HyG_20230426.csv"
_FORBIDDEN: Final[tuple[str, ...]] = ("torch", "xgboost")


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


def _kappa_fn(settings: Settings) -> LogKappa:
    """``ln kappa`` rows via one spawned torch worker without xgboost (ADL-051)."""
    solver = settings.protocol.solver

    def log_kappa(
        ell: NDArray[np.float64], w: NDArray[np.float64], r: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        payload = {
            "ell": np.asarray(ell, dtype=np.float64).tolist(),
            "w": np.asarray(w, dtype=np.float64).tolist(),
            "r": np.asarray(r, dtype=np.float64).tolist(),
            "level": int(solver.quad_level),
            "trunc": float(solver.quad_trunc),
        }
        (res,) = run_jobs(kappa_job, [payload], n_workers=1, forbid="xgboost")
        return np.asarray(res["k"], dtype=np.float64)

    return log_kappa


def _aoa_fn(settings: Settings, ident: PlanIdentity, params_path: Path) -> AoaFn:
    """AOA blocks via spawned xgboost workers without torch (ADL-051; docs/3 13.9)."""
    rt = settings.runtime
    base = {
        "params_path": str(params_path.resolve()),
        "params_sha256": settings.params_sha256,
        "site_path": str(ident.paths["site_table"]),
        "site_sha256": ident.hashes["site_table"],
        "fold_path": str(ident.paths["fold_table"]),
        "fold_sha256": ident.hashes["fold_table"],
        "ind_path": str(ident.paths["ind_features"]),
        "ind_sha256": ident.hashes["ind_features"],
        "run_dir": str(rt.run_root / ident.run_id),
        "members": int(settings.protocol.uq.ensemble_m),
    }

    def aoa(blocks: Sequence[tuple[str, int, int]]) -> list[dict[str, Any]]:
        payloads = [
            {**base, "family": fam, "repeat": int(r), "outer_fold": int(f)} for fam, r, f in blocks
        ]
        return run_jobs(aoa_job, payloads, n_workers=rt.n_workers, forbid="torch")

    return aoa


def _vaa_maps(
    nhd_dir: Path, comids: Sequence[int]
) -> tuple[NetworkInputs, dict[int, float], dict[int, int]]:
    """Network inputs (as Event 1) and VAA ``PathLength`` / ``LevelPathI`` of the site COMIDs."""
    vaa, _ = read_nhd_table(nhd_dir, "vaa")
    plusflow, _ = read_nhd_table(nhd_dir, "plusflow")
    network = network_inputs(vaa, plusflow, comids)
    v = vaa.set_index("ComID").loc[sorted({int(c) for c in comids})]
    keys = [int(c) for c in v.index]
    pathlength = dict(zip(keys, v["PathLength"].astype(float).tolist(), strict=True))
    levelpath = dict(zip(keys, v["LevelPathI"].astype(int).tolist(), strict=True))
    return network, pathlength, levelpath


def cmd_evaluate(settings: Settings, ident: PlanIdentity, args: argparse.Namespace) -> int:
    """Event 3 (docs/3 13.4, 13.13): every remaining protocol 4 result and the 4.7 row."""
    rt = settings.runtime
    root = rt.repo_root
    log = get_logger("annxs")
    session = open_event(settings, "event_3", run_id=ident.run_id, plan_sha256=ident.sha256)
    pdir = predictions_dir(rt.processed_dir)
    pred_paths = {s: pdir / f"{s}.parquet" for s in EVENT3_PREDICTIONS}
    rdir = session.results_dir
    nhd_dir = rt.raw_dir / "nhdplusv21"
    hyg_dir = rt.raw_dir / "hyg"
    qc_path = rt.processed_dir / "qc" / "hydroswot_init.parquet"
    inputs = [
        *sealed_entries(session, list(pred_paths.values())),
        *(file_entry(ident.paths[k], root) for k in sorted(ident.paths)),
        file_entry(rdir / AUDIT_JSON, root),
        file_entry(rdir / STRATA_FILE, root),
        file_entry(rdir / MSTAR_JSON, root),
        file_entry(qc_path, root),
        file_entry(nhd_dir / "MANIFEST.sha256", root),
        file_entry(hyg_dir / "MANIFEST.sha256", root),
    ]
    site_table = pd.read_parquet(ident.paths["site_table"])
    comids = analysis_frame(site_table, "primary")["comid"].astype(int).tolist()
    log.info("event_3: reading the NHDPlusV2.1 VAA and PlusFlow (%d COMIDs)", len(set(comids)))
    network, pathlength, levelpath = _vaa_maps(nhd_dir, comids)
    hyg, hyg_counts = read_hyg_manning(hyg_dir / HYG_FILE)
    inp = Event3Inputs(
        preds={s: pd.read_parquet(p) for s, p in pred_paths.items()},
        site_table=site_table,
        folds=pd.read_parquet(ident.paths["fold_table"]),
        ehydro=pd.read_parquet(ident.paths["ehydro_ind"]),
        ind_features=pd.read_parquet(ident.paths["ind_features"]),
        srnd=pd.read_parquet(ident.paths["srnd_frame"]),
        audit_strata=pd.read_parquet(rdir / STRATA_FILE),
        audit=json.loads((rdir / AUDIT_JSON).read_text(encoding="utf-8")),
        mstar=json.loads((rdir / MSTAR_JSON).read_text(encoding="utf-8")),
        hyg=hyg,
        hyg_counts=hyg_counts,
        qc_rows=pd.read_parquet(qc_path, columns=["row_id", "chan_mat_cd", "F_eps"]),
        network=network,
        pathlength=pathlength,
        levelpath=levelpath,
    )
    result = run_event3(
        inp,
        settings.protocol,
        log_kappa=_kappa_fn(settings),
        aoa=_aoa_fn(settings, ident, args.params),
    )
    loaded = [m for m in _FORBIDDEN if m in sys.modules]
    if loaded:
        raise RuntimeError(f"the Event 3 process loaded {loaded} (ADL-051)")
    head: dict[str, Any] = {
        "run_id": ident.run_id,
        "plan_sha256": ident.sha256,
        "params_sha256": settings.params_sha256,
    }
    outputs: list[dict[str, str]] = []
    for name in EVENT3_DOCS:
        path = rdir / f"event3_{name}.json"
        doc = {**head, "inputs": inputs, **result.documents[name]}
        write_json(path, cast(dict[str, Any], json_safe(doc)))
        outputs.append(file_entry(path, root))
    matrix_path = rdir / EVENT3_MATRIX
    write_parquet(
        result.matrix,
        matrix_path,
        stage="event_3",
        params_sha256=settings.params_sha256,
        inputs={e["path"]: e["sha256"] for e in inputs},
    )
    outputs.append(file_entry(matrix_path, root))
    outcome = result.documents["hypotheses"]["outcome_4_7"]
    index = {
        **head,
        "inputs": inputs,
        "outputs": list(outputs),
        "audit_label": result.documents["hypotheses"]["audit_label"],
        "outcome_4_7": outcome,
        "outcome_table_4_7": {str(k): list(v) for k, v in OUTCOME_TEXT.items()},
    }
    index_path = rdir / EVENT3_INDEX
    write_json(index_path, cast(dict[str, Any], json_safe(index)))
    outputs.append(file_entry(index_path, root))
    rec = close_event(
        session,
        inputs=inputs,
        outputs=outputs,
        note="Event 3: protocol 4.3-4.7 on sealed predictions (docs/3 13.4, 13.13)",
    )
    print(f"event_3 recorded as ledger seq {rec['seq']}")
    for item in outputs:
        print(f"  {item['path']}  sha256 {item['sha256']}")
    return 0


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
    settings = load_settings(args.params)
    get_logger("annxs", args.log_level or settings.runtime.log_level)
    ident = plan_identity(settings, reduced=args.reduced)
    return _COMMANDS[args.command](settings, ident, args)


if __name__ == "__main__":
    raise SystemExit(main())

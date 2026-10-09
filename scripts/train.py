"""Thin CLI: train one model on one family/fold from the frozen fold table.

Sources: protocol sections 3.3, 3.4 and 4.1-4.2; P6-D1 (a), P6-D6 (a), P6-D7 (a).
Owning phase: 6. The logic lives in :mod:`src.models.runner`; this file only parses arguments.

Examples::

    train.py fit --model M5 --family S-ID --fold 0 --kind inner --index 0 --grid 0
    train.py select --model M5 --family S-ID --fold 0
    train.py fit --model M5 --ablation A1 --family S-ID --fold 0 --kind member --index 1

M1 and M2-XGB never import torch (ADL-051); run them in their own processes. No outer-test row
is predicted or scored here.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from src.config import PARAMS_PATH, load_settings
from src.models.registry import ABLATIONS, NN_MODELS, XGB_MODELS, ModelSpec
from src.models.runner import FitRequest, Tables, run_fit, run_select
from src.utils.logging import get_logger


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one fit or run the inner selection.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "select"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, choices=NN_MODELS + XGB_MODELS)
        p.add_argument("--ablation", choices=tuple(ABLATIONS), default=None)
        p.add_argument("--analysis", default="primary")
        p.add_argument("--family", required=True)
        p.add_argument("--repeat", type=int, default=0)
        p.add_argument("--fold", type=int, required=True)
        p.add_argument("--params", type=Path, default=PARAMS_PATH)
        p.add_argument("--site-table", type=Path, default=None)
        p.add_argument("--fold-table", type=Path, default=None)
        p.add_argument("--checkpoint-dir", type=Path, default=None)
        p.add_argument("--overwrite", action="store_true")
        p.add_argument("--log-level", default=None)
        if name == "fit":
            p.add_argument("--kind", required=True, choices=("inner", "member"))
            p.add_argument("--index", type=int, required=True)
            p.add_argument("--grid", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and run ``fit`` or ``select``; return the exit status."""
    args = _parser().parse_args(argv)
    settings = load_settings(args.params)
    rt = settings.runtime
    logger = get_logger("annxs", args.log_level or rt.log_level)
    root = args.checkpoint_dir or rt.checkpoint_dir
    spec = ModelSpec(args.model, args.ablation)
    logger.info("params_sha256 %s", settings.params_sha256)
    if args.command == "select":
        doc = run_select(
            spec,
            args.analysis,
            args.family,
            args.repeat,
            args.fold,
            settings,
            root,
            overwrite=args.overwrite,
        )
        logger.info("selection written for %s: %s", spec.name, doc.get("selected", doc.get("kind")))
        return 0
    tables = Tables.load(
        args.site_table or rt.processed_dir / "site_table.parquet",
        args.fold_table or rt.processed_dir / "splits" / "fold_assignment.parquet",
    )
    req = FitRequest(
        spec=spec,
        analysis=args.analysis,
        family=args.family,
        repeat=args.repeat,
        outer_fold=args.fold,
        kind=args.kind,
        index=args.index,
        grid=args.grid,
    )
    doc = run_fit(req, tables, settings, root, overwrite=args.overwrite, logger=logger)
    logger.info("done %s in %.1f s", spec.name, doc["wall_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

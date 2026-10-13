"""Phase 8 experiment CLI: plan, run one stage, status and tracking (docs/3 13.2-13.7).

Sources: P8-D5 (a) (``run_experiment.py {plan,run,status}``), docs/3 section 13.6 (every count
checked by ``plan`` before execution), section 13.7 (registry, manifest, Merkle root,
``reports/training``), section 13.2 (run root, pilot move, archive), section 13.3 (stage order;
NN stages need ``checkpoint_1``). Owning phase: 8. The logic lives in
:mod:`src.models.plan`, :mod:`src.models.orchestrator` and :mod:`src.models.tracking`.

Examples::

    run_experiment.py plan
    run_experiment.py run --stage train@baselines
    run_experiment.py status
    run_experiment.py track
    run_experiment.py move-pilot
    run_experiment.py archive

``--reduced`` (dry runs with a reduced ``params.yaml``) makes the comparison with the docs/3 13.6
count table informative instead of binding. Nothing here reads a target or a prediction; no
metric is printed.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PARAMS_PATH, Settings, load_settings
from src.data.io import sha256_file
from src.models.orchestrator import RunContext, RunFailureError, keep_awake, run_stage
from src.models.plan import (
    EXPECTED_COUNTS_13_6,
    STAGES,
    PlannedFit,
    blocks_from_folds,
    build_plan,
    canonical_json,
    count_groups,
    expected_counts,
    plan_document,
    plan_sha256,
    run_id_for,
    table_hashes,
    table_paths,
)
from src.models.registry import read_json, write_json
from src.models.tracking import (
    archive_run,
    build_manifest,
    build_registry,
    move_pilot,
    training_summaries,
    tree_digests,
    write_registry,
    write_training_reports,
)
from src.utils.logging import get_logger, git_state

PILOT_NAME = "pilot_adl090"
#: Why the docs/3 13.6 NN count exceeds the section 4.8 estimate (docs/3 13.6, verbatim reasons).
DIFFERENCES_4_8 = (
    "M3, a torch rung, is not counted in 4.8: +884",
    "M4's lambda_max triples its inner grid: +1,248",
    "there are 26 blocks, not about 25",
    "SA6 is listed in 4.9 but not in 4.8",
    "A4 on M* (Q5)",
)


@dataclass(frozen=True)
class Built:
    """The recomputed plan and its identity."""

    fits: tuple[PlannedFit, ...]
    document: dict[str, Any]
    sha256: str
    run_id: str
    paths: dict[str, Path]
    hashes: dict[str, str]
    counts: dict[str, int]

    def run_dir(self, settings: Settings) -> Path:
        """``<run_root>/<run_id>`` (docs/3 13.2)."""
        return settings.runtime.run_root / self.run_id


def _build(settings: Settings, *, reduced: bool) -> Built:
    paths = table_paths(settings.runtime.processed_dir)
    hashes = table_hashes(paths)
    folds = pd.read_parquet(paths["fold_table"], columns=["family", "repeat", "outer_fold"])
    blocks = blocks_from_folds(folds)
    fits = build_plan(settings.protocol, blocks)
    counts = count_groups(fits)
    if counts != expected_counts(settings.protocol, len(blocks)):
        raise RuntimeError("plan counts differ from the formula of docs/3 13.6")
    if not reduced and counts != dict(EXPECTED_COUNTS_13_6):
        raise RuntimeError(
            "plan counts differ from the docs/3 13.6 table (use --reduced for dry runs)"
        )
    document = plan_document(fits, settings.params_sha256, hashes)
    sha = plan_sha256(document)
    return Built(fits, document, sha, run_id_for(sha), paths, hashes, counts)


def _print_plan(built: Built) -> None:
    print(f"run_id       {built.run_id}")
    print(f"plan_sha256  {built.sha256}")
    print(f"fits         {len(built.fits)}")
    print(f"{'group':<20} {'planned':>8} {'13.6':>8}  match")
    for key, want in EXPECTED_COUNTS_13_6.items():
        got = built.counts.get(key, 0)
        print(f"{key:<20} {got:>8} {want:>8}  {'yes' if got == want else 'NO'}")
    print("NN count vs the 4.8 estimate (about 7,000-7,500):")
    for line in DIFFERENCES_4_8:
        print(f"  - {line}")


def _track(settings: Settings, built: Built) -> dict[str, Any]:
    """Registry, training reports and manifest from the run root (docs/3 13.7)."""
    rt = settings.runtime
    run_dir = built.run_dir(settings)
    registry = build_registry(built.fits, run_dir)
    reg_sha = write_registry(
        registry,
        rt.processed_dir / "runs" / "run_registry.parquet",
        params_sha256=settings.params_sha256,
        plan_sha256=built.sha256,
    )
    reports = write_training_reports(
        rt.reports_dir / "training", training_summaries(built.fits, run_dir)
    )
    stage_docs = {
        s: read_json(run_dir / "stages" / f"{s}.json")
        for s in STAGES
        if (run_dir / "stages" / f"{s}.json").is_file()
    }
    pilot_dir = rt.run_root / PILOT_NAME
    archive_rec = rt.dvc_store / "runs_archive" / f"{built.run_id}.json"
    manifest = build_manifest(
        run_id=built.run_id,
        plan_sha256=built.sha256,
        plan_header=built.document["header"],
        uv_lock_sha256=sha256_file(rt.repo_root / "uv.lock"),
        registry=registry,
        registry_sha256=reg_sha,
        stage_docs=stage_docs,
        n_workers=rt.n_workers,
        torch_threads_per_worker=rt.torch_threads_per_worker,
        training_reports=reports,
        pilot=tree_digests(pilot_dir) if pilot_dir.is_dir() else None,
        archive=read_json(archive_rec) if archive_rec.is_file() else None,
    )
    write_json(rt.reports_dir / "run_manifest.json", manifest)
    return manifest


def _check_plan_file(built: Built, settings: Settings) -> Path:
    path = built.run_dir(settings) / "plan.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run `run_experiment.py plan` first")
    if path.read_bytes() != canonical_json(built.document):
        raise RuntimeError(f"{path} differs from the recomputed plan; inputs changed")
    return path


def cmd_plan(settings: Settings, args: argparse.Namespace, log: logging.Logger) -> int:
    """Build, check and register the plan (``<run_root>/<run_id>/plan.json``)."""
    built = _build(settings, reduced=args.reduced)
    path = built.run_dir(settings) / "plan.json"
    data = canonical_json(built.document)
    if path.is_file() and path.read_bytes() != data:
        raise RuntimeError(f"{path} exists with different content")
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        log.info("plan written to %s", path)
    _print_plan(built)
    return 0


def cmd_run(settings: Settings, args: argparse.Namespace, log: logging.Logger) -> int:
    """Run one stage under ``caffeinate``, then refresh registry, reports and manifest."""
    rt = settings.runtime
    built = _build(settings, reduced=args.reduced)
    _check_plan_file(built, settings)
    git = git_state(rt.repo_root)
    if git["dirty"] and not args.allow_dirty:
        raise RuntimeError("the git tree is dirty; production fits need a clean tree")
    ctx = RunContext(
        params_path=str(settings.params_path),
        site_path=str(built.paths["site_table"]),
        fold_path=str(built.paths["fold_table"]),
        run_dir=str(built.run_dir(settings)),
        params_sha256=settings.params_sha256,
        site_sha256=built.hashes["site_table"],
        fold_sha256=built.hashes["fold_table"],
        log_level=rt.log_level,
    )
    code = 0
    with keep_awake():
        try:
            res = run_stage(
                args.stage,
                built.fits,
                ctx=ctx,
                settings=settings,
                run_id=built.run_id,
                ledger_path=rt.reports_dir / "results" / "scoring_ledger.json",
                n_workers=rt.n_workers,
                logger=log,
            )
            print(f"{args.stage}: planned {res.planned}, already done {res.already_done}, "
                  f"executed {res.executed}, reruns {len(res.reruns)}")  # fmt: skip
        except RunFailureError as exc:
            rep = exc.report
            print(f"{args.stage}: FAILED at {rep['fit_id']} (seed {rep['seed']}): {rep['error']}")
            code = 2
        finally:
            manifest = _track(settings, built)
    print(f"merkle_root  {manifest['merkle_root']}")
    return code


def cmd_status(settings: Settings, args: argparse.Namespace, log: logging.Logger) -> int:
    """Per stage: planned, complete (``run.json``/``selection.json``), incomplete directories."""
    built = _build(settings, reduced=args.reduced)
    run_dir = built.run_dir(settings)
    print(f"run_id {built.run_id}  run_dir {run_dir}")
    for stage in STAGES:
        fits = [f for f in built.fits if f.stage == stage]
        done = sum(f.completion_path(run_dir).is_file() for f in fits)
        partial = sum(
            f.kind != "select"
            and not f.completion_path(run_dir).is_file()
            and (run_dir / f.fit_id).exists()
            for f in fits
        )
        print(f"{stage:<20} planned {len(fits):>6}  done {done:>6}  incomplete {partial:>4}")
    return 0


def cmd_track(settings: Settings, args: argparse.Namespace, log: logging.Logger) -> int:
    """Rebuild the registry, training reports and manifest."""
    built = _build(settings, reduced=args.reduced)
    _check_plan_file(built, settings)
    manifest = _track(settings, built)
    print(f"merkle_root  {manifest['merkle_root']}  leaves {manifest['n_leaves']}")
    return 0


def cmd_move_pilot(settings: Settings, args: argparse.Namespace, log: logging.Logger) -> int:
    """Move ``data/checkpoints`` unchanged to ``<run_root>/pilot_adl090`` (docs/3 13.2)."""
    digests = move_pilot(settings.runtime.checkpoint_dir, settings.runtime.run_root / PILOT_NAME)
    print(f"pilot moved: {len(digests)} files verified")
    return 0


def cmd_archive(settings: Settings, args: argparse.Namespace, log: logging.Logger) -> int:
    """Archive a complete run root as ``<dvc_store>/runs_archive/<run_id>.tar`` (docs/3 13.2)."""
    built = _build(settings, reduced=args.reduced)
    _check_plan_file(built, settings)
    run_dir = built.run_dir(settings)
    missing = [f.fit_id for f in built.fits if not f.completion_path(run_dir).is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} planned fits incomplete; archive refused")
    dest = settings.runtime.dvc_store / "runs_archive" / f"{built.run_id}.tar"
    record = archive_run(run_dir, dest)
    write_json(dest.with_suffix(".json"), record)
    _track(settings, built)
    print(f"archive {dest}  sha256 {record['sha256']}")
    return 0


_COMMANDS = {
    "plan": cmd_plan,
    "run": cmd_run,
    "status": cmd_status,
    "track": cmd_track,
    "move-pilot": cmd_move_pilot,
    "archive": cmd_archive,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 8 fit plan, stages and tracking.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in _COMMANDS:
        p = sub.add_parser(name)
        p.add_argument("--params", type=Path, default=PARAMS_PATH)
        p.add_argument("--reduced", action="store_true", help="dry run with reduced params")
        p.add_argument("--log-level", default=None)
        if name == "run":
            p.add_argument("--stage", required=True, choices=STAGES)
            p.add_argument("--allow-dirty", action="store_true", help="dry runs only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch; return the exit status."""
    args = _parser().parse_args(argv)
    settings = load_settings(args.params)
    log = get_logger("annxs", args.log_level or settings.runtime.log_level)
    log.info("params_sha256 %s", settings.params_sha256)
    return _COMMANDS[args.command](settings, args, log)


if __name__ == "__main__":
    raise SystemExit(main())

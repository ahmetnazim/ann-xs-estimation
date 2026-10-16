"""Hash-verifying loader for Phase 9 results, sealed predictions, and frozen tables (P9-D5 (a)).

Verifies ``reports/results/scoring_ledger.json`` (header, 7-record sequence, clean git flags,
canonical JSON SHA-256 chain from genesis, and Merkle root), checks the SHA-256 of every file in
``reports/results/`` against ``.agents/.research_state.md`` §17 and the ledger output records,
refuses if ``torch`` or ``xgboost`` is imported (ADL-051, ADL-086), and returns tidy DataFrames
tagged with pre-registered evidence labels (P9-D6 (a)).
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

#: Forbidden modules in the Phase 9 figure/table pipeline (ADL-051, ADL-086).
FORBIDDEN_MODULES: Final[tuple[str, ...]] = ("torch", "xgboost")

#: Expected 7-record scoring ledger sequence (docs/3 §13.4).
LEDGER_SEQUENCE: Final[tuple[str, ...]] = (
    "seal",
    "event_1",
    "checkpoint_1",
    "seal",
    "event_2",
    "checkpoint_2",
    "event_3",
)

#: Expected header digests in ``reports/results/scoring_ledger.json``.
EXPECTED_PARAMS_SHA256: Final[str] = (
    "1b6c670e0ab7816247fc065d0c0fa77699d0b439edb5d66ca7efeb9e720adbde"
)
EXPECTED_PLAN_SHA256: Final[str] = (
    "da9b7303cb841f28b3ca081d657e477315238f5e773eb20575239f6476794bca"
)
EXPECTED_UV_LOCK_SHA256: Final[str] = (
    "85c69462b26510f2cd61bc4ba8f6ac0f6677a219715fbaf7220959b221da6e6d"
)
EXPECTED_RUN_ID: Final[str] = "p8-da9b7303cb84"
EXPECTED_MERKLE_ROOT: Final[str] = (
    "bb97203df74599de568bee5316d4b147d1b7dea770220d7917da464b937a1fb2"
)

#: Frozen SHA-256 digests of ``reports/results/*`` (.agents/.research_state.md §17).
RESULTS_SHA256: Final[dict[str, str]] = {
    "reports/results/scoring_ledger.json": (
        "e2fd2e3ba9ff64d2108103944306f6e5f0b68d0eddcd2c2968ddd2d5384cef71"
    ),
    "reports/results/audit.json": (
        "d8b44a5e2556e7a04f574c3a7be8f6785e58f7ab0d334dd22c3ac9b0e3c2adcf"
    ),
    "reports/results/audit_strata.parquet": (
        "c039cc16973083a79fffdd9a02df6cda02bc9fb9b0085446e034b1c7744e0e68"
    ),
    "reports/results/mstar.json": (
        "6bd7bc01aaad60d12cf350e681cf0dcaac32bfb913151d99660c2ef9ee259834"
    ),
    "reports/results/event3.json": (
        "59841fea054e366d6616f24b27a121802200c2bd0a03237f16552e71f032de17"
    ),
    "reports/results/event3_hypotheses.json": (
        "27866ceaa3ac0aec2e45c5e3f2ad4fbe0f6fcb3735669913c96033fb27469b09"
    ),
    "reports/results/event3_uq.json": (
        "e11761af22f198a2252e86693b6e4bf2180e142c29f4f6a3ab3abe50c67498b3"
    ),
    "reports/results/event3_consistency.json": (
        "d3d99ac85d3a6ccf93ee2c0071f9b8fbd9b822a0c94194eebd10d43fe3215818"
    ),
    "reports/results/event3_dependence.json": (
        "f7184e8cac7adaf77596ce746dc2e8d6d423eabdc5b01684b41869d4d343e1a3"
    ),
    "reports/results/event3_descriptive.json": (
        "17baa7c14c70b14d66bee3e58945b03e86b95bc93537529568fff45be0306591"
    ),
    "reports/results/event3_matrix.parquet": (
        "bc66633897d7d7b8b31013bd6061780a82641c609d95be09a180a0e762a8dc55"
    ),
}

#: Frozen SHA-256 digests of data tables (docs/3 §10.1, ADL-093 / ADL-099).
FROZEN_TABLES_SHA256: Final[dict[str, str]] = {
    "data/processed/site_table.parquet": (
        "8d186b07aae165eddff23119b9bc67223705c8dff5cf50d7950711d2566d5470"
    ),
    "data/processed/splits/fold_assignment.parquet": (
        "1bf628c83fd658c179243499af5d533738fefaeb15b63a71c476530c40dc872c"
    ),
    "data/processed/splits/srnd_assignment.parquet": (
        "b3c07ba313f5e3b834d85dbeb7ec1c9b17c31e7f5b2db0f1ef775a52013808a1"
    ),
    "data/processed/ind/ehydro_ind.parquet": (
        "a47bcd71e234a4dee69e8563c25b35d548f2a5030a04ca0b63ccf1ce0fff550c"
    ),
}

#: Verbatim docs/3 §4.7 row 4 strings recorded in ``reports/results/event3.json``.
VERBATIM_ROW4_OUTCOME: Final[str] = "M5 worse (S-LRO CI lower bound > 0)"
VERBATIM_ROW4_CLAIM: Final[str] = (
    "Diagnose with the pre-registered ablations (A1, A3, M4, SA1, SA5) and report. "
    "No post-hoc headline switch: any new variant is labelled exploratory"
)
VERBATIM_ROW4_VENUE: Final[str] = "ASCE; fallback per ADL-014"
VERBATIM_ROW4_STATEMENT: Final[str] = f'row 4: "{VERBATIM_ROW4_CLAIM}"'
EVENT1_AUDIT_LABEL: Final[str] = "optimistic under spatial/network dependence"
S_IND_PANEL_NOTE: Final[str] = "131 locations / 18 rivers"
S_IND_UQ_NOTE: Final[str] = "131 locations / 18 rivers (UQ: 33 single-COMID rows)"

_HEADER_KEYS: Final[tuple[str, ...]] = (
    "params_sha256",
    "plan_sha256",
    "run_id",
    "uv_lock_sha256",
)
_RECORD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "seq",
        "event",
        "utc",
        "git_commit",
        "git_dirty",
        "params_sha256",
        "inputs",
        "outputs",
        "note",
        "prev_sha256",
    }
)
_OPTIONAL_KEYS: Final[frozenset[str]] = frozenset({"merkle_root", "references"})
_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")


class ProvenanceError(RuntimeError):
    """Raised when any hash, ledger record, or import invariant is violated."""


def assert_no_forbidden_imports(modules: Mapping[str, object] | None = None) -> None:
    """Refuse execution if ``torch`` or ``xgboost`` is present in ``sys.modules``.

    Parameters
    ----------
    modules : Mapping[str, object] | None, optional
        Module mapping to inspect; defaults to ``sys.modules``.

    Raises
    ------
    ProvenanceError
        If any forbidden module or submodule is loaded.
    """
    active = sys.modules if modules is None else modules
    found = sorted(
        name
        for name in active
        if any(name == pkg or name.startswith(f"{pkg}.") for pkg in FORBIDDEN_MODULES)
    )
    if found:
        raise ProvenanceError(
            f"Forbidden module(s) imported in figure pipeline (ADL-051/ADL-086): {found}"
        )


class ForbiddenImportBlocker(importlib.abc.MetaPathFinder):
    """Meta-path finder that actively blocks ``import torch`` and ``import xgboost``."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: object | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """Raise :class:`ProvenanceError` if ``fullname`` matches a forbidden package."""
        del path, target
        if any(fullname == pkg or fullname.startswith(f"{pkg}.") for pkg in FORBIDDEN_MODULES):
            raise ProvenanceError(
                f"Import of {fullname!r} is forbidden in the figure pipeline (ADL-051/ADL-086)"
            )
        return None

    @classmethod
    @contextmanager
    def guard(cls) -> Iterator[ForbiddenImportBlocker]:
        """Context manager installing the import blocker and checking ``sys.modules``."""
        assert_no_forbidden_imports()
        blocker = cls()
        sys.meta_path.insert(0, blocker)
        try:
            yield blocker
        finally:
            if blocker in sys.meta_path:
                sys.meta_path.remove(blocker)


def sha256_file(path: Path) -> str:
    """Compute the lower-case hex SHA-256 digest of ``path``.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        64-character lower-case hexadecimal SHA-256 digest.

    Raises
    ------
    ProvenanceError
        If ``path`` does not exist or is not a regular file.
    """
    if not path.is_file():
        raise ProvenanceError(f"Required provenance file missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(obj: Mapping[str, Any]) -> str:
    text = json.dumps(
        dict(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and not (set(value) - _HEX)


def verify_ledger_dict(doc: Mapping[str, Any]) -> dict[str, str]:
    """Verify the structure, sequence, and cryptographic hash chain of the scoring ledger.

    Parameters
    ----------
    doc : Mapping[str, Any]
        Parsed JSON object from ``reports/results/scoring_ledger.json``.

    Returns
    -------
    dict[str, str]
        Mapping of ``rel_path -> sha256`` for all sealed prediction files.

    Raises
    ------
    ProvenanceError
        If the header, event order, hash chain, clean-tree flag, or output hashes are invalid.
    """
    if set(doc) != {"header", "events"}:
        raise ProvenanceError("Ledger JSON must contain exactly 'header' and 'events'")
    header = doc["header"]
    events = doc["events"]
    if not isinstance(header, Mapping) or tuple(sorted(header)) != _HEADER_KEYS:
        raise ProvenanceError(f"Ledger header must contain exactly {_HEADER_KEYS}")
    if header["params_sha256"] != EXPECTED_PARAMS_SHA256:
        raise ProvenanceError(
            f"Ledger params_sha256 {header['params_sha256']} != {EXPECTED_PARAMS_SHA256}"
        )
    if header["plan_sha256"] != EXPECTED_PLAN_SHA256:
        raise ProvenanceError(
            f"Ledger plan_sha256 {header['plan_sha256']} != {EXPECTED_PLAN_SHA256}"
        )
    if header["uv_lock_sha256"] != EXPECTED_UV_LOCK_SHA256:
        raise ProvenanceError(
            f"Ledger uv_lock_sha256 {header['uv_lock_sha256']} != {EXPECTED_UV_LOCK_SHA256}"
        )
    if header["run_id"] != EXPECTED_RUN_ID:
        raise ProvenanceError(f"Ledger run_id {header['run_id']} != {EXPECTED_RUN_ID}")
    if not isinstance(events, list) or len(events) != len(LEDGER_SEQUENCE):
        count = len(events) if isinstance(events, list) else "non-list"
        raise ProvenanceError(
            f"Ledger must have exactly {len(LEDGER_SEQUENCE)} records {LEDGER_SEQUENCE}; "
            f"got {count}"
        )

    expected_prev = _canonical_sha256(header)
    sealed: dict[str, str] = {}
    recorded_outputs: dict[str, str] = {}

    for idx, (rec, expected_event) in enumerate(zip(events, LEDGER_SEQUENCE, strict=True), start=1):
        if not isinstance(rec, Mapping):
            raise ProvenanceError(f"Ledger record {idx} is not a mapping")
        keys = set(rec)
        if not keys >= _RECORD_KEYS or keys - _RECORD_KEYS - _OPTIONAL_KEYS:
            raise ProvenanceError(f"Ledger record {idx} has invalid keys: {sorted(keys)}")
        if rec["seq"] != idx:
            raise ProvenanceError(f"Ledger record seq {rec['seq']} != expected {idx}")
        if rec["event"] != expected_event:
            raise ProvenanceError(
                f"Ledger record {idx} event {rec['event']!r} != expected {expected_event!r}"
            )
        if rec["prev_sha256"] != expected_prev:
            raise ProvenanceError(f"Ledger hash chain broken at seq {idx}")
        if (
            rec["git_dirty"] is not False
            or not isinstance(rec["git_commit"], str)
            or not rec["git_commit"]
        ):
            raise ProvenanceError(f"Ledger record {idx} has dirty or missing git state")
        if rec["params_sha256"] != header["params_sha256"]:
            raise ProvenanceError(f"Ledger record {idx} params_sha256 mismatch")

        for field_name in ("inputs", "outputs"):
            items = rec[field_name]
            if not isinstance(items, list):
                raise ProvenanceError(f"Ledger record {idx} {field_name} is not a list")
            for item in items:
                if (
                    not isinstance(item, Mapping)
                    or set(item) != {"path", "sha256"}
                    or not isinstance(item["path"], str)
                    or not _is_sha256(item["sha256"])
                ):
                    raise ProvenanceError(f"Ledger record {idx} invalid {field_name} entry: {item}")
                if rec["event"] == "seal" and field_name == "inputs":
                    sealed[item["path"]] = item["sha256"]
                if field_name == "outputs":
                    recorded_outputs[item["path"]] = item["sha256"]

        expected_prev = _canonical_sha256(rec)

    if events[3].get("merkle_root") != EXPECTED_MERKLE_ROOT:
        raise ProvenanceError(
            f"Ledger seq 4 merkle_root {events[3].get('merkle_root')} != {EXPECTED_MERKLE_ROOT}"
        )
    if len(sealed) != 21:
        raise ProvenanceError(f"Expected 21 sealed prediction files, got {len(sealed)}")

    for rel_path, expected_sha in RESULTS_SHA256.items():
        if rel_path == "reports/results/scoring_ledger.json":
            continue
        if recorded_outputs.get(rel_path) != expected_sha:
            raise ProvenanceError(
                f"Ledger output digest for {rel_path} ({recorded_outputs.get(rel_path)}) "
                f"!= §17 digest ({expected_sha})"
            )

    return sealed


@dataclass(frozen=True)
class Outcome47:
    """Verified pre-registered §4.7 decision-matrix outcome from ``event3.json``."""

    row: int
    outcome: str
    claim: str
    venue: str
    statement: str
    evidence_tier: str = "Confirmatory"
    source_file: str = "reports/results/event3.json"


@dataclass(frozen=True)
class VerifiedBundle:
    """Cryptographically verified Phase 8 evaluation results and tidy frames for Phase 9."""

    repo_root: Path
    ledger: dict[str, Any]
    sealed_hashes: dict[str, str]
    audit: dict[str, Any]
    mstar: dict[str, Any]
    event3: dict[str, Any]
    hypotheses: dict[str, Any]
    uq: dict[str, Any]
    consistency: dict[str, Any]
    dependence: dict[str, Any]
    descriptive: dict[str, Any]
    outcome_4_7: Outcome47
    matrix_df: pd.DataFrame
    audit_strata_df: pd.DataFrame
    hypotheses_df: pd.DataFrame
    uq_df: pd.DataFrame
    consistency_df: pd.DataFrame

    def load_sealed_prediction(self, filename: str) -> pd.DataFrame:
        """Verify SHA-256 against the ledger seal and load a sealed prediction file.

        Parameters
        ----------
        filename : str
            File name (e.g. ``"M5.parquet"``) or path under ``data/processed/predictions/``.

        Returns
        -------
        pd.DataFrame
            Verified prediction table (for visual display only; P9-D5 (a)).

        Raises
        ------
        ProvenanceError
            If the file is not in the ledger seal map or its on-disk SHA-256 differs.
        """
        assert_no_forbidden_imports()
        rel_path = (
            filename
            if filename.startswith("data/processed/predictions/")
            else f"data/processed/predictions/{filename}"
        )
        expected = self.sealed_hashes.get(rel_path)
        if expected is None:
            raise ProvenanceError(
                f"Prediction file {rel_path!r} is not sealed in scoring_ledger.json"
            )
        full_path = self.repo_root / rel_path
        actual = sha256_file(full_path)
        if actual != expected:
            raise ProvenanceError(
                f"Sealed prediction SHA-256 mismatch for {rel_path}: {actual} != {expected}"
            )
        return pd.read_parquet(full_path)

    def load_frozen_table(
        self, rel_path: str, expected_hashes: Mapping[str, str] | None = None
    ) -> pd.DataFrame:
        """Verify SHA-256 against ``FROZEN_TABLES_SHA256`` and load a frozen data table.

        Parameters
        ----------
        rel_path : str
            Relative path from ``repo_root`` (e.g. ``"data/processed/site_table.parquet"``).
        expected_hashes : Mapping[str, str] | None, optional
            Override mapping for testing; defaults to :data:`FROZEN_TABLES_SHA256`.

        Returns
        -------
        pd.DataFrame
            Verified frozen table.

        Raises
        ------
        ProvenanceError
            If ``rel_path`` is unknown or its on-disk SHA-256 does not match.
        """
        assert_no_forbidden_imports()
        registry = FROZEN_TABLES_SHA256 if expected_hashes is None else expected_hashes
        expected = registry.get(rel_path)
        if expected is None:
            raise ProvenanceError(
                f"Frozen table {rel_path!r} is not registered in FROZEN_TABLES_SHA256"
            )
        full_path = self.repo_root / rel_path
        actual = sha256_file(full_path)
        if actual != expected:
            raise ProvenanceError(
                f"Frozen table SHA-256 mismatch for {rel_path}: {actual} != {expected}"
            )
        return pd.read_parquet(full_path)

    def load_training_json(self, rel_path: str) -> dict[str, Any]:
        """Load a training diagnostic JSON (e.g. ``reports/training/summary.json``)."""
        assert_no_forbidden_imports()
        full_path = self.repo_root / rel_path
        if not full_path.is_file():
            raise ProvenanceError(f"Training diagnostic file missing: {full_path}")
        data = json.loads(full_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ProvenanceError(f"Training diagnostic {rel_path} must be a JSON object")
        return data

    def load_training_list(self, rel_path: str) -> list[dict[str, Any]]:
        """Load a list-valued training diagnostic JSON (e.g. ``curves.json``, ``xi.json``)."""
        assert_no_forbidden_imports()
        full_path = self.repo_root / rel_path
        if not full_path.is_file():
            raise ProvenanceError(f"Training diagnostic file missing: {full_path}")
        data = json.loads(full_path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
            raise ProvenanceError(f"Training diagnostic {rel_path} must be a JSON list of objects")
        return data


def _split_note_for_family(family: str, *, is_uq: bool = False) -> str:
    if family == "S-ID":
        return EVENT1_AUDIT_LABEL
    if family == "S-RND":
        return f"Secondary — {EVENT1_AUDIT_LABEL}"
    if family == "S-IND":
        return S_IND_UQ_NOTE if is_uq else S_IND_PANEL_NOTE
    return ""


def _enrich_matrix_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["evidence_tier"] = out["family"].map(
        lambda fam: "Secondary" if fam == "S-RND" else "Descriptive"
    )
    out["split_note"] = out["family"].map(lambda fam: _split_note_for_family(str(fam), is_uq=False))
    out["source_file"] = "reports/results/event3_matrix.parquet"
    return out


def _build_hypotheses_df(hyp: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source = "reports/results/event3_hypotheses.json"

    # H1: M5 vs M* on S-LRO and S-IND
    h1 = hyp["H1"]
    h1_verdict = "Falsified" if h1["decision"]["falsified"] else "Supported"
    for split_name in ("S-LRO", "S-IND"):
        split_blk = h1[split_name]
        n_units = split_blk.get("n_sites", split_blk.get("n_locations"))
        for target in ("dbar", "A"):
            stat = split_blk[target]
            rows.append(
                {
                    "hypothesis": "H1",
                    "contrast": f"H1_{split_name}_{target}",
                    "family": split_name,
                    "target": target,
                    "estimate": float(stat["estimate"]),
                    "ci_lo": float(stat["lo"]),
                    "ci_hi": float(stat["hi"]),
                    "ci_level": float(stat["level"]),
                    "n_units": int(n_units),
                    "verdict": h1_verdict,
                    "evidence_tier": "Confirmatory",
                    "split_note": _split_note_for_family(split_name),
                    "source_file": source,
                }
            )

    # H2: Small-watershed (Q1) signed dbar bias contrast
    h2 = hyp["H2"]
    h2_verdict = "Falsified" if h2["decision"]["falsified"] else "Supported"
    for key, fam, tier in (
        ("S-ID_Q1", "S-ID", "Confirmatory"),
        ("S-LRO_Q1_secondary", "S-LRO", "Secondary"),
    ):
        blk = h2[key]
        for sub_key in ("h2", "bias_M5", "bias_Mstar"):
            stat = blk[sub_key]
            rows.append(
                {
                    "hypothesis": "H2",
                    "contrast": f"H2_{key}_{sub_key}",
                    "family": fam,
                    "target": "dbar",
                    "estimate": float(stat["estimate"]),
                    "ci_lo": float(stat["lo"]),
                    "ci_hi": float(stat["hi"]),
                    "ci_level": float(stat["level"]),
                    "n_units": int(blk["n_sites"]),
                    "verdict": h2_verdict,
                    "evidence_tier": tier,
                    "split_note": _split_note_for_family(fam),
                    "source_file": source,
                }
            )

    # H3: OOD skill loss L(M5) - L(M*)
    h3 = hyp["H3"]
    h3_verdict = "Falsified" if h3["decision"]["falsified"] else "Supported"
    for key, fam in (
        ("H3-LRO", "S-LRO"),
        ("H3-TAIL", "S-TAIL-DA"),
        ("H3-TAIL-SO", "S-TAIL-SO"),
    ):
        blk = h3[key]
        tier = "Secondary" if blk.get("secondary", False) else "Confirmatory"
        for sub_key in ("delta", "L_M5", "L_Mstar"):
            stat = blk[sub_key]
            rows.append(
                {
                    "hypothesis": "H3",
                    "contrast": f"{key}_{sub_key}",
                    "family": fam,
                    "target": "dbar",
                    "estimate": float(stat["estimate"]),
                    "ci_lo": float(stat["lo"]),
                    "ci_hi": float(stat["hi"]),
                    "ci_level": float(stat["level"]),
                    "n_units": int(blk["n_sites"]),
                    "verdict": h3_verdict,
                    "evidence_tier": tier,
                    "split_note": _split_note_for_family(fam),
                    "source_file": source,
                }
            )

    # H4: Physical parameter plausibility (rho_n_d50, rho_n_s, rho_r_robs)
    h4 = hyp["H4"]
    h4_verdict = "Falsified" if h4["decision"]["falsified"] else "Supported"
    for split_key, fam, tier in (
        ("S-ID", "S-ID", "Confirmatory"),
        ("S-LRO_reported", "S-LRO", "Descriptive"),
    ):
        blk = h4[split_key]
        for metric_key, target_name in (
            ("n_d50", "n"),
            ("n_s", "n"),
            ("r_robs", "r"),
        ):
            stat = blk["ci"][metric_key]
            rows.append(
                {
                    "hypothesis": "H4",
                    "contrast": f"H4_{fam}_{metric_key}",
                    "family": fam,
                    "target": target_name,
                    "estimate": float(stat["estimate"]),
                    "ci_lo": float(stat["lo"]),
                    "ci_hi": float(stat["hi"]),
                    "ci_level": float(stat["level"]),
                    "n_units": int(blk["n_sites"]),
                    "verdict": h4_verdict,
                    "evidence_tier": tier,
                    "split_note": _split_note_for_family(fam),
                    "source_file": source,
                }
            )

    # S2: Stepwise physics contrasts (98.75% Bonferroni CIs)
    s2 = hyp["S2"]
    for fam in ("S-ID", "S-LRO"):
        for contrast_key in ("s2_M4_vs_M5", "s2_M5_vs_M6NN"):
            stat = s2[fam][contrast_key]
            rows.append(
                {
                    "hypothesis": "S2",
                    "contrast": f"S2_{fam}_{contrast_key}",
                    "family": fam,
                    "target": "dbar",
                    "estimate": float(stat["estimate"]),
                    "ci_lo": float(stat["lo"]),
                    "ci_hi": float(stat["hi"]),
                    "ci_level": float(stat["level"]),
                    "n_units": 3988,
                    "verdict": "Secondary contrast",
                    "evidence_tier": "Secondary",
                    "split_note": _split_note_for_family(fam),
                    "source_file": source,
                }
            )

    return pd.DataFrame(rows)


def _build_uq_df(uq: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source = "reports/results/event3_uq.json"
    alpha = float(uq["alpha"])
    band_lo, band_hi = float(uq["band"][0]), float(uq["band"][1])

    for spec_key, spec_val in sorted(uq.items()):
        if "|" not in spec_key or not isinstance(spec_val, Mapping):
            continue
        model_part, analysis_part = spec_key.split("|", 1)
        for family, fam_val in sorted(spec_val.items()):
            if not isinstance(fam_val, Mapping):
                continue
            for target, tgt_val in sorted(fam_val.items()):
                if not isinstance(tgt_val, Mapping) or "overall" not in tgt_val:
                    continue
                ov = tgt_val["overall"]
                is_band_check = spec_key == "M5|primary" and family == "S-ID" and target == "dbar"
                rows.append(
                    {
                        "spec": spec_key,
                        "model": model_part,
                        "analysis": analysis_part,
                        "family": family,
                        "target": target,
                        "stratum_kind": "overall",
                        "stratum": "all",
                        "alpha": alpha,
                        "band_lo": band_lo,
                        "band_hi": band_hi,
                        "coverage": float(ov["coverage"]),
                        "crps": float(ov["crps"]),
                        "mean_log_width": float(ov["mean_log_width"]),
                        "width_factor": float(ov["width_factor"]),
                        "in_band": ov.get("in_band"),
                        "n_rows": int(ov["n_rows"]),
                        "n_excluded": int(ov["n_excluded"]),
                        "evidence_tier": "Secondary" if is_band_check else "Descriptive",
                        "split_note": _split_note_for_family(family, is_uq=True),
                        "source_file": source,
                    }
                )
                by_map = tgt_val.get("by", {})
                if isinstance(by_map, Mapping):
                    for s_kind, s_dict in sorted(by_map.items()):
                        if not isinstance(s_dict, Mapping):
                            continue
                        for s_name, s_entry in sorted(s_dict.items()):
                            if not isinstance(s_entry, Mapping):
                                continue
                            rows.append(
                                {
                                    "spec": spec_key,
                                    "model": model_part,
                                    "analysis": analysis_part,
                                    "family": family,
                                    "target": target,
                                    "stratum_kind": str(s_kind),
                                    "stratum": str(s_name),
                                    "alpha": alpha,
                                    "band_lo": band_lo,
                                    "band_hi": band_hi,
                                    "coverage": float(s_entry["coverage"]),
                                    "crps": float(s_entry["crps"]),
                                    "mean_log_width": float(s_entry["mean_log_width"]),
                                    "width_factor": float(s_entry["width_factor"]),
                                    "in_band": s_entry.get("in_band"),
                                    "n_rows": int(s_entry["n_rows"]),
                                    "n_excluded": int(s_entry["n_excluded"]),
                                    "evidence_tier": "Descriptive",
                                    "split_note": _split_note_for_family(family, is_uq=True),
                                    "source_file": source,
                                }
                            )
    return pd.DataFrame(rows)


def _build_consistency_df(cons: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source = "reports/results/event3_consistency.json"
    models = cons.get("models", {})
    if not isinstance(models, Mapping):
        raise ProvenanceError("event3_consistency.json missing 'models' mapping")
    for model_name, fam_map in sorted(models.items()):
        if not isinstance(fam_map, Mapping):
            continue
        for family, metrics in sorted(fam_map.items()):
            if not isinstance(metrics, Mapping):
                continue
            imp_v = metrics.get("implied_V", {})
            pred_c = metrics.get("predicted_closure", {})
            back_c = metrics.get("back_calc", {})
            man_m4 = metrics.get("manning_residual_M4", {})
            rows.append(
                {
                    "model": str(model_name),
                    "family": str(family),
                    "closure_median_abs": pred_c.get("median_abs", 0.0),
                    "closure_share_gt_ln105": pred_c.get("share_gt_ln105", 0.0),
                    "back_calc_share_outside": back_c.get("share_outside"),
                    "back_calc_n_undefined": back_c.get("n_undefined"),
                    "back_calc_n_invariant4": back_c.get("n_invariant4"),
                    "solver_residual_max": metrics.get("solver_residual_max"),
                    "picard_iterations_max": metrics.get("picard_iterations_max"),
                    "manning_rms_M4": man_m4.get("rms"),
                    "manning_median_M4": man_m4.get("median"),
                    "manning_q05_M4": man_m4.get("q05"),
                    "manning_q95_M4": man_m4.get("q95"),
                    "shape_violation_share": metrics.get("shape_violation_share", 0.0),
                    "implied_V_rmse_log": imp_v.get("rmse_log"),
                    "implied_V_bias_log": imp_v.get("bias_log"),
                    "implied_V_oracle_rmse_log": imp_v.get("oracle_rmse_log"),
                    "implied_V_oracle_bias_log": imp_v.get("oracle_bias_log"),
                    "implied_V_share_A": imp_v.get("share_A"),
                    "implied_V_share_Q": imp_v.get("share_Q"),
                    "evidence_tier": "Descriptive",
                    "split_note": _split_note_for_family(str(family)),
                    "source_file": source,
                }
            )
    return pd.DataFrame(rows)


def _extract_outcome_4_7(event3_doc: Mapping[str, Any]) -> Outcome47:
    raw = event3_doc.get("outcome_4_7")
    if not isinstance(raw, Mapping):
        raise ProvenanceError("event3.json missing 'outcome_4_7'")
    matched = raw.get("matched")
    if not isinstance(matched, list) or len(matched) != 1 or not isinstance(matched[0], Mapping):
        raise ProvenanceError(f"event3.json outcome_4_7.matched must have 1 entry, got {matched}")
    entry = matched[0]
    if entry.get("row") != 4:
        raise ProvenanceError(f"Expected §4.7 row 4, got {entry.get('row')}")
    if entry.get("outcome") != VERBATIM_ROW4_OUTCOME:
        raise ProvenanceError(
            f"§4.7 row 4 outcome {entry.get('outcome')!r} != {VERBATIM_ROW4_OUTCOME!r}"
        )
    if entry.get("claim") != VERBATIM_ROW4_CLAIM:
        raise ProvenanceError(f"§4.7 row 4 claim {entry.get('claim')!r} != {VERBATIM_ROW4_CLAIM!r}")
    if entry.get("venue") != VERBATIM_ROW4_VENUE:
        raise ProvenanceError(f"§4.7 row 4 venue {entry.get('venue')!r} != {VERBATIM_ROW4_VENUE!r}")
    if raw.get("statement") != VERBATIM_ROW4_STATEMENT:
        raise ProvenanceError(
            f"§4.7 statement {raw.get('statement')!r} != {VERBATIM_ROW4_STATEMENT!r}"
        )
    return Outcome47(
        row=int(entry["row"]),
        outcome=str(entry["outcome"]),
        claim=str(entry["claim"]),
        venue=str(entry["venue"]),
        statement=str(raw["statement"]),
    )


def load_verified_bundle(
    repo_root: Path,
    *,
    expected_results_sha256: Mapping[str, str] | None = None,
) -> VerifiedBundle:
    """Verify the ledger chain and all 11 ``reports/results/*`` SHA-256 digests and load frames.

    Parameters
    ----------
    repo_root : Path
        Repository root directory containing ``reports/results/``.
    expected_results_sha256 : Mapping[str, str] | None, optional
        Override digest map for refusal tests; defaults to :data:`RESULTS_SHA256`.

    Returns
    -------
    VerifiedBundle
        Cryptographically verified dictionaries and tidy DataFrames with evidence labels.

    Raises
    ------
    ProvenanceError
        If ``torch``/``xgboost`` is loaded, any file is missing or its SHA-256 differs, or the
        scoring ledger chain or sequence is broken.
    """
    assert_no_forbidden_imports()
    digests = RESULTS_SHA256 if expected_results_sha256 is None else expected_results_sha256

    if set(digests) != set(RESULTS_SHA256):
        raise ProvenanceError(
            f"Expected digest keys {sorted(RESULTS_SHA256)}, got {sorted(digests)}"
        )

    for rel_path, expected_sha in sorted(digests.items()):
        full_path = repo_root / rel_path
        actual_sha = sha256_file(full_path)
        if actual_sha != expected_sha:
            raise ProvenanceError(
                f"SHA-256 mismatch for {rel_path}: {actual_sha} != expected {expected_sha}"
            )

    ledger_path = repo_root / "reports/results/scoring_ledger.json"
    ledger_doc = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(ledger_doc, dict):
        raise ProvenanceError("scoring_ledger.json must parse to a JSON object")
    sealed_hashes = verify_ledger_dict(ledger_doc)

    def _read_json(rel: str) -> dict[str, Any]:
        obj = json.loads((repo_root / rel).read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ProvenanceError(f"{rel} must parse to a JSON object")
        return obj

    audit_doc = _read_json("reports/results/audit.json")
    mstar_doc = _read_json("reports/results/mstar.json")
    event3_doc = _read_json("reports/results/event3.json")
    hyp_doc = _read_json("reports/results/event3_hypotheses.json")
    uq_doc = _read_json("reports/results/event3_uq.json")
    cons_doc = _read_json("reports/results/event3_consistency.json")
    dep_doc = _read_json("reports/results/event3_dependence.json")
    desc_doc = _read_json("reports/results/event3_descriptive.json")

    audit_label_obj = audit_doc.get("label")
    audit_label_text = (
        audit_label_obj.get("text") if isinstance(audit_label_obj, Mapping) else audit_label_obj
    )
    if audit_label_text != EVENT1_AUDIT_LABEL:
        raise ProvenanceError(f"audit.json label {audit_label_text!r} != {EVENT1_AUDIT_LABEL!r}")
    if event3_doc.get("audit_label") != EVENT1_AUDIT_LABEL:
        raise ProvenanceError(
            f"event3.json audit_label {event3_doc.get('audit_label')!r} != {EVENT1_AUDIT_LABEL!r}"
        )

    outcome_4_7 = _extract_outcome_4_7(event3_doc)

    raw_matrix = pd.read_parquet(repo_root / "reports/results/event3_matrix.parquet")
    matrix_df = _enrich_matrix_df(raw_matrix)

    raw_audit_strata = pd.read_parquet(repo_root / "reports/results/audit_strata.parquet")
    audit_strata_df = raw_audit_strata.copy()
    audit_strata_df["evidence_tier"] = "Descriptive"
    audit_strata_df["source_file"] = "reports/results/audit_strata.parquet"

    hypotheses_df = _build_hypotheses_df(hyp_doc)
    uq_df = _build_uq_df(uq_doc)
    consistency_df = _build_consistency_df(cons_doc)

    return VerifiedBundle(
        repo_root=repo_root,
        ledger=ledger_doc,
        sealed_hashes=sealed_hashes,
        audit=audit_doc,
        mstar=mstar_doc,
        event3=event3_doc,
        hypotheses=hyp_doc,
        uq=uq_doc,
        consistency=cons_doc,
        dependence=dep_doc,
        descriptive=desc_doc,
        outcome_4_7=outcome_4_7,
        matrix_df=matrix_df,
        audit_strata_df=audit_strata_df,
        hypotheses_df=hypotheses_df,
        uq_df=uq_df,
        consistency_df=consistency_df,
    )

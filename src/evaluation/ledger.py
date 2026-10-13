"""Append-only scoring ledger with a SHA-256 hash chain (docs/3 section 13.4, P8-D7 (a)).

Sources: docs/3 section 13.4 (file ``reports/results/scoring_ledger.json``; header ``run_id``,
``plan_sha256``, ``params_sha256``, ``uv_lock_sha256``; event fields ``seq``, ``event``, ``utc``,
``git_commit``, ``git_dirty`` (must be false), ``params_sha256``, ``inputs``, ``outputs``,
``note``, ``prev_sha256``; fixed order ``seal``, ``event_1``, ``checkpoint_1``, ``seal``,
``event_2``, ``checkpoint_2``, ``event_3``; ``erratum`` events reference the original; a broken
chain refuses every later event; a seal records the prediction-file hashes and the run-manifest
Merkle root; a checkpoint carries PI approval text and no number), section 13.3 (NN stages are
refused unless ``checkpoint_1`` is recorded).

Hash chain: ``prev_sha256`` of record ``i`` is the SHA-256 of the canonical JSON (sorted keys,
separators ``,`` and ``:``, UTF-8) of record ``i - 1``; for the first record it is the SHA-256 of
the canonical header (genesis). The file is rewritten atomically on every append; earlier records
are never modified, which :meth:`Ledger.load` verifies.

This module reads no prediction or target values. Torch-free and xgboost-free.
Owning phase: 8.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

__all__ = [
    "CHECKPOINTS",
    "ERRATUM",
    "EVENT_ORDER",
    "HEADER_KEYS",
    "Ledger",
    "LedgerError",
    "canonical_json",
    "record_sha256",
]

#: docs/3 13.4: every step refuses unless all earlier steps are recorded.
EVENT_ORDER: Final[tuple[str, ...]] = (
    "seal",
    "event_1",
    "checkpoint_1",
    "seal",
    "event_2",
    "checkpoint_2",
    "event_3",
)
ERRATUM: Final[str] = "erratum"
CHECKPOINTS: Final[frozenset[str]] = frozenset({"checkpoint_1", "checkpoint_2"})
HEADER_KEYS: Final[tuple[str, ...]] = ("params_sha256", "plan_sha256", "run_id", "uv_lock_sha256")
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


class LedgerError(RuntimeError):
    """An out-of-order, malformed or chain-breaking ledger operation."""


def canonical_json(obj: object) -> bytes:
    """Sorted keys, separators ``,`` and ``:``, UTF-8; NaN refused."""
    text = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return text.encode("utf-8")


def record_sha256(obj: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a record (or of the header, for the genesis link)."""
    return hashlib.sha256(canonical_json(dict(obj))).hexdigest()


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and not set(value) - _HEX


def _check_files(files: object, where: str) -> list[dict[str, str]]:
    if not isinstance(files, list):
        raise LedgerError(f"{where} must be a list of {{path, sha256}}")
    out = []
    for item in files:
        if not (isinstance(item, Mapping) and set(item) == {"path", "sha256"}):
            raise LedgerError(f"{where}: entries need exactly path and sha256")
        if not (isinstance(item["path"], str) and item["path"] and _is_sha(item["sha256"])):
            raise LedgerError(f"{where}: bad entry {item!r}")
        out.append({"path": item["path"], "sha256": item["sha256"]})
    return out


def _write_atomic(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@dataclass
class Ledger:
    """The ledger file: a header and the ordered, hash-chained event records."""

    path: Path
    header: dict[str, str]
    events: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------------- construction
    @classmethod
    def create(
        cls,
        path: Path,
        *,
        run_id: str,
        plan_sha256: str,
        params_sha256: str,
        uv_lock_sha256: str,
    ) -> Ledger:
        """Write a new, empty ledger; refuses to replace an existing file."""
        if path.exists():
            raise LedgerError(f"{path} exists; a ledger is never recreated")
        for name, digest in (
            ("plan_sha256", plan_sha256),
            ("params_sha256", params_sha256),
            ("uv_lock_sha256", uv_lock_sha256),
        ):
            if not _is_sha(digest):
                raise LedgerError(f"{name} must be 64 lower-case hex characters")
        if not run_id.startswith("p8-"):
            raise LedgerError("run_id must start with 'p8-' (docs/3 13.2)")
        header = {
            "params_sha256": params_sha256,
            "plan_sha256": plan_sha256,
            "run_id": run_id,
            "uv_lock_sha256": uv_lock_sha256,
        }
        ledger = cls(path=path, header=header, events=[])
        ledger._write()
        return ledger

    @classmethod
    def load(cls, path: Path) -> Ledger:
        """Read and fully verify the ledger (header, fields, order, chain)."""
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not (isinstance(doc, dict) and set(doc) == {"header", "events"}):
            raise LedgerError("a ledger holds exactly 'header' and 'events'")
        header, events = doc["header"], doc["events"]
        if not (isinstance(header, dict) and tuple(sorted(header)) == HEADER_KEYS):
            raise LedgerError(f"ledger header needs exactly {HEADER_KEYS}")
        if not isinstance(events, list):
            raise LedgerError("events must be a list")
        ledger = cls(path=path, header=dict(header), events=[])
        for rec in events:
            ledger._verify_next(rec)
            ledger.events.append(rec)
        return ledger

    # ------------------------------------------------------------------------- queries
    def recorded(self) -> list[str]:
        """Names of the recorded ordered events (errata excluded)."""
        return [e["event"] for e in self.events if e["event"] != ERRATUM]

    def next_expected(self) -> str | None:
        """Return the next ordered event, or ``None`` after ``event_3``."""
        n = len(self.recorded())
        return EVENT_ORDER[n] if n < len(EVENT_ORDER) else None

    def has(self, event: str) -> bool:
        """Return ``True`` if ``event`` (other than ``seal``) is recorded."""
        if event == "seal" or event not in EVENT_ORDER:
            raise ValueError(f"has() takes a unique ordered event, not {event!r}")
        return event in self.recorded()

    def require(self, event: str) -> None:
        """Raise :class:`LedgerError` unless ``event`` is recorded."""
        if not self.has(event):
            raise LedgerError(f"{event} is not in the scoring ledger (docs/3 13.4)")

    def sealed(self) -> dict[str, str]:
        """``path -> sha256`` of every file listed by a seal (sealed files are never rewritten)."""
        out: dict[str, str] = {}
        for rec in self.events:
            if rec["event"] == "seal":
                for item in rec["inputs"]:
                    out[item["path"]] = item["sha256"]
        return out

    # ------------------------------------------------------------------------- append
    def append(
        self,
        event: str,
        *,
        git: Mapping[str, Any],
        params_sha256: str,
        inputs: Sequence[Mapping[str, str]] = (),
        outputs: Sequence[Mapping[str, str]] = (),
        note: str = "",
        merkle_root: str | None = None,
        references: int | None = None,
        utc: str | None = None,
    ) -> dict[str, Any]:
        """Append one record after checking order, cleanliness, params and fields."""
        rec: dict[str, Any] = {
            "seq": len(self.events) + 1,
            "event": event,
            "utc": utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": str(git.get("commit", "")),
            "git_dirty": bool(git.get("dirty", True)),
            "params_sha256": params_sha256,
            "inputs": [dict(i) for i in inputs],
            "outputs": [dict(o) for o in outputs],
            "note": note,
            "prev_sha256": self._tip(),
        }
        if merkle_root is not None:
            rec["merkle_root"] = merkle_root
        if references is not None:
            rec["references"] = references
        self._verify_next(rec)
        self.events.append(rec)
        try:
            self._write()
        except BaseException:
            self.events.pop()
            raise
        return rec

    # ------------------------------------------------------------------------- internals
    def _tip(self) -> str:
        return record_sha256(self.events[-1] if self.events else self.header)

    def _verify_next(self, rec: object) -> None:
        if not isinstance(rec, dict):
            raise LedgerError("a record must be a JSON object")
        keys = set(rec)
        if not keys >= _RECORD_KEYS or keys - _RECORD_KEYS - _OPTIONAL_KEYS:
            raise LedgerError(f"record fields must be {sorted(_RECORD_KEYS)} (+ optional)")
        seq = len(self.events) + 1
        if rec["seq"] != seq:
            raise LedgerError(f"record seq {rec['seq']} != {seq}")
        if rec["prev_sha256"] != self._tip():
            raise LedgerError(f"hash chain broken at seq {seq}; every later event is refused")
        if rec["git_dirty"] is not False or not rec["git_commit"]:
            raise LedgerError("events need a clean git tree and a commit (docs/3 13.4)")
        if rec["params_sha256"] != self.header["params_sha256"]:
            raise LedgerError("params_sha256 differs from the ledger header")
        inputs = _check_files(rec["inputs"], "inputs")
        outputs = _check_files(rec["outputs"], "outputs")
        if not isinstance(rec["note"], str):
            raise LedgerError("note must be a string")
        event = rec["event"]
        if event == ERRATUM:
            ref = rec.get("references")
            if not (isinstance(ref, int) and 1 <= ref < seq):
                raise LedgerError("an erratum references an earlier seq")
            if self.events[ref - 1]["event"] in CHECKPOINTS or not rec["note"]:
                raise LedgerError("an erratum corrects an event (not a checkpoint) and says why")
        else:
            expected = self.next_expected()
            if event != expected:
                raise LedgerError(f"out-of-order event {event!r}; next is {expected!r}")
            if "references" in rec:
                raise LedgerError("only an erratum carries 'references'")
        if event == "seal":
            if not (inputs and _is_sha(rec.get("merkle_root"))):
                raise LedgerError("a seal lists the prediction files and the Merkle root")
            sealed = self.sealed()
            clash = [i["path"] for i in inputs if sealed.get(i["path"], i["sha256"]) != i["sha256"]]
            if clash:
                raise LedgerError(f"sealed files changed: {clash[:3]}")
        elif "merkle_root" in rec:
            raise LedgerError("only a seal carries 'merkle_root'")
        if event in CHECKPOINTS and (inputs or outputs or not rec["note"].strip()):
            raise LedgerError("a checkpoint has PI approval text and no inputs or outputs")

    def _write(self) -> None:
        _write_atomic(self.path, {"header": self.header, "events": self.events})

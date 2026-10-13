"""Tests for the hash-chained scoring ledger (docs/3 13.4; src.evaluation.ledger).

The ledger holds only file hashes, notes and git state; no metric enters these tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.evaluation.ledger import EVENT_ORDER, Ledger, LedgerError, record_sha256

pytestmark = pytest.mark.unit

P = "0" * 64
CLEAN = {"commit": "abc123", "dirty": False}
FILE_A = {"path": "data/processed/predictions/audit_inputs.parquet", "sha256": "a" * 64}
FILE_B = {"path": "data/processed/predictions/M5.parquet", "sha256": "b" * 64}
ROOT = "c" * 64


def new(tmp_path: Path) -> Ledger:
    return Ledger.create(
        tmp_path / "results" / "scoring_ledger.json",
        run_id="p8-0123456789ab",
        plan_sha256="1" * 64,
        params_sha256=P,
        uv_lock_sha256="2" * 64,
    )


def step(ledger: Ledger, event: str, **kw: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {"git": CLEAN, "params_sha256": P}
    if event == "seal":
        defaults.update(inputs=[FILE_A], merkle_root=ROOT)
    elif event.startswith("checkpoint"):
        defaults.update(note="PI approves continuing (written approval)")
    else:
        defaults.update(
            inputs=[FILE_A], outputs=[{"path": "reports/results/x.json", "sha256": "d" * 64}]
        )
    defaults.update(kw)
    return ledger.append(event, **defaults)


def test_full_order_and_reload(tmp_path: Path) -> None:
    led = new(tmp_path)
    assert led.next_expected() == "seal"
    for event in EVENT_ORDER:
        rec = step(led, event)
        assert rec["event"] == event
    assert led.next_expected() is None
    again = Ledger.load(led.path)
    assert again.recorded() == list(EVENT_ORDER)
    assert again.events[0]["prev_sha256"] == record_sha256(again.header)
    assert again.events[1]["prev_sha256"] == record_sha256(again.events[0])
    assert again.has("checkpoint_1")
    with pytest.raises(LedgerError, match="out-of-order"):
        step(again, "event_3")


@pytest.mark.parametrize("event", ["event_1", "checkpoint_1", "event_2", "event_3"])
def test_out_of_order_events_are_refused(tmp_path: Path, event: str) -> None:
    led = new(tmp_path)
    with pytest.raises(LedgerError, match="out-of-order"):
        step(led, event)
    assert led.events == []
    assert Ledger.load(led.path).events == []


def test_require_and_has(tmp_path: Path) -> None:
    led = new(tmp_path)
    with pytest.raises(LedgerError, match="checkpoint_1"):
        led.require("checkpoint_1")
    for event in EVENT_ORDER[:3]:
        step(led, event)
    led.require("checkpoint_1")
    assert not led.has("event_2")
    with pytest.raises(ValueError, match="unique"):
        led.has("seal")


def test_dirty_tree_params_and_fields_are_refused(tmp_path: Path) -> None:
    led = new(tmp_path)
    with pytest.raises(LedgerError, match="clean git tree"):
        step(led, "seal", git={"commit": "abc", "dirty": True})
    with pytest.raises(LedgerError, match="params_sha256"):
        step(led, "seal", params_sha256="9" * 64)
    with pytest.raises(LedgerError, match="Merkle"):
        step(led, "seal", merkle_root=None)
    with pytest.raises(LedgerError, match="bad entry"):
        step(led, "seal", inputs=[{"path": "x", "sha256": "nothex"}])
    with pytest.raises(LedgerError, match="exactly path"):
        step(led, "seal", inputs=[{"path": "x"}])
    step(led, "seal")
    with pytest.raises(LedgerError, match="only a seal"):
        step(led, "event_1", merkle_root=ROOT)
    step(led, "event_1")
    with pytest.raises(LedgerError, match="checkpoint"):
        step(led, "checkpoint_1", note="  ")
    with pytest.raises(LedgerError, match="checkpoint"):
        step(led, "checkpoint_1", inputs=[FILE_A])


def test_sealed_files_are_never_rewritten(tmp_path: Path) -> None:
    led = new(tmp_path)
    for event in EVENT_ORDER[:3]:
        step(led, event)
    changed = {"path": FILE_A["path"], "sha256": "e" * 64}
    with pytest.raises(LedgerError, match="sealed files changed"):
        step(led, "seal", inputs=[changed, FILE_B])
    step(led, "seal", inputs=[FILE_A, FILE_B])
    assert led.sealed() == {FILE_A["path"]: FILE_A["sha256"], FILE_B["path"]: FILE_B["sha256"]}


def test_erratum_rules(tmp_path: Path) -> None:
    led = new(tmp_path)
    for event in EVENT_ORDER[:3]:
        step(led, event)
    with pytest.raises(LedgerError, match="references"):
        step(led, "erratum", note="fix")
    with pytest.raises(LedgerError, match="not a checkpoint"):
        step(led, "erratum", references=3, note="fix")
    with pytest.raises(LedgerError, match="only an erratum"):
        step(led, "seal", references=1)
    rec = step(led, "erratum", references=2, note="corrected input path")
    assert rec["references"] == 2
    assert led.next_expected() == "seal"  # errata do not advance the order
    assert Ledger.load(led.path).recorded() == list(EVENT_ORDER[:3])


def test_broken_chain_refuses_everything(tmp_path: Path) -> None:
    led = new(tmp_path)
    for event in EVENT_ORDER[:3]:
        step(led, event)
    doc = json.loads(led.path.read_text())
    doc["events"][0]["note"] = "edited after the fact"
    led.path.write_text(json.dumps(doc))
    with pytest.raises(LedgerError, match="hash chain broken at seq 2"):
        Ledger.load(led.path)


def test_malformed_files(tmp_path: Path) -> None:
    led = new(tmp_path)
    with pytest.raises(LedgerError, match="never recreated"):
        new(tmp_path)
    path = led.path
    path.write_text(json.dumps({"header": led.header}))
    with pytest.raises(LedgerError, match="exactly 'header' and 'events'"):
        Ledger.load(path)
    path.write_text(json.dumps({"header": {"run_id": "p8-x"}, "events": []}))
    with pytest.raises(LedgerError, match="header"):
        Ledger.load(path)
    path.write_text(json.dumps({"header": led.header, "events": {}}))
    with pytest.raises(LedgerError, match="list"):
        Ledger.load(path)
    path.write_text(json.dumps({"header": led.header, "events": [{"seq": 1}]}))
    with pytest.raises(LedgerError, match="fields"):
        Ledger.load(path)
    path.write_text(json.dumps({"header": led.header, "events": ["x"]}))
    with pytest.raises(LedgerError, match="JSON object"):
        Ledger.load(path)
    with pytest.raises(LedgerError, match="hex"):
        Ledger.create(
            tmp_path / "x.json", run_id="p8-x", plan_sha256="z", params_sha256=P, uv_lock_sha256=P
        )
    with pytest.raises(LedgerError, match="p8-"):
        Ledger.create(
            tmp_path / "x.json", run_id="run", plan_sha256=P, params_sha256=P, uv_lock_sha256=P
        )


def test_seq_mismatch_is_refused(tmp_path: Path) -> None:
    led = new(tmp_path)
    rec = step(led, "seal")
    doc = json.loads(led.path.read_text())
    doc["events"][0]["seq"] = 5
    led.path.write_text(json.dumps(doc))
    with pytest.raises(LedgerError, match="seq 5"):
        Ledger.load(led.path)
    assert rec["seq"] == 1

"""Unit and refusal tests for the Phase 9 hash-verifying loader (Step 2; P9-D5 (a), P9-D6 (a))."""

from __future__ import annotations

import copy
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from annxs_figures import (
    EXPECTED_MERKLE_ROOT,
    EXPECTED_PARAMS_SHA256,
    EXPECTED_UV_LOCK_SHA256,
    FORBIDDEN_MODULES,
    FROZEN_TABLES_SHA256,
    LEDGER_SEQUENCE,
    RESULTS_SHA256,
    ForbiddenImportBlocker,
    ProvenanceError,
    VerifiedBundle,
    assert_no_forbidden_imports,
    load_verified_bundle,
    sha256_file,
    verify_ledger_dict,
)
from annxs_figures.loader import (
    EVENT1_AUDIT_LABEL,
    S_IND_PANEL_NOTE,
    S_IND_UQ_NOTE,
    VERBATIM_ROW4_CLAIM,
    VERBATIM_ROW4_OUTCOME,
    VERBATIM_ROW4_STATEMENT,
    VERBATIM_ROW4_VENUE,
    _extract_outcome_4_7,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def bundle() -> VerifiedBundle:
    return load_verified_bundle(REPO_ROOT)


def test_real_bundle_loads_and_verifies_all_hashes(bundle: VerifiedBundle) -> None:
    assert len(bundle.ledger["events"]) == 7
    assert tuple(e["event"] for e in bundle.ledger["events"]) == LEDGER_SEQUENCE
    assert len(bundle.sealed_hashes) == 21
    assert bundle.ledger["header"]["params_sha256"] == EXPECTED_PARAMS_SHA256
    assert bundle.ledger["header"]["uv_lock_sha256"] == EXPECTED_UV_LOCK_SHA256
    assert bundle.ledger["events"][3]["merkle_root"] == EXPECTED_MERKLE_ROOT


def test_verbatim_outcome_4_7_matches_protocol_row_4(bundle: VerifiedBundle) -> None:
    out = bundle.outcome_4_7
    assert out.row == 4
    assert out.outcome == VERBATIM_ROW4_OUTCOME
    assert out.claim == VERBATIM_ROW4_CLAIM
    assert out.venue == VERBATIM_ROW4_VENUE
    assert out.statement == VERBATIM_ROW4_STATEMENT
    assert out.evidence_tier == "Confirmatory"
    assert out.source_file == "reports/results/event3.json"


def test_tidy_frames_carry_evidence_labels_and_notes(bundle: VerifiedBundle) -> None:
    assert len(bundle.matrix_df) == 45544
    assert set(bundle.matrix_df["evidence_tier"].unique()) == {"Descriptive", "Secondary"}
    srnd_tiers = set(
        bundle.matrix_df.loc[bundle.matrix_df["family"] == "S-RND", "evidence_tier"].unique()
    )
    assert srnd_tiers == {"Secondary"}
    sid_notes = set(
        bundle.matrix_df.loc[bundle.matrix_df["family"] == "S-ID", "split_note"].unique()
    )
    assert sid_notes == {EVENT1_AUDIT_LABEL}
    sind_notes = set(
        bundle.matrix_df.loc[bundle.matrix_df["family"] == "S-IND", "split_note"].unique()
    )
    assert sind_notes == {S_IND_PANEL_NOTE}

    # Hypotheses frame checks
    hyp = bundle.hypotheses_df
    assert set(hyp["hypothesis"].unique()) == {"H1", "H2", "H3", "H4", "S2"}
    h1_rows = hyp.loc[hyp["hypothesis"] == "H1"]
    assert set(h1_rows["evidence_tier"]) == {"Confirmatory"}
    assert set(h1_rows["ci_level"]) == {0.95}
    s2_rows = hyp.loc[hyp["hypothesis"] == "S2"]
    assert set(s2_rows["evidence_tier"]) == {"Secondary"}
    assert set(s2_rows["ci_level"]) == {0.9875}

    # UQ frame checks
    uq = bundle.uq_df
    band_row = uq.loc[
        (uq["spec"] == "M5|primary")
        & (uq["family"] == "S-ID")
        & (uq["target"] == "dbar")
        & (uq["stratum_kind"] == "overall")
    ]
    assert len(band_row) == 1
    assert band_row.iloc[0]["evidence_tier"] == "Secondary"
    assert bool(band_row.iloc[0]["in_band"]) is True
    assert band_row.iloc[0]["coverage"] == pytest.approx(0.8996990972918759)
    sind_uq = uq.loc[uq["family"] == "S-IND"]
    assert set(sind_uq["split_note"].unique()) == {S_IND_UQ_NOTE}

    # Consistency frame checks
    cons = bundle.consistency_df
    assert len(cons) > 0
    assert set(cons["evidence_tier"].unique()) == {"Descriptive"}
    assert set(bundle.audit_strata_df["evidence_tier"].unique()) == {"Descriptive"}


def test_sealed_prediction_and_frozen_table_loaders(bundle: VerifiedBundle) -> None:
    pred_path = REPO_ROOT / "data/processed/predictions/M5.parquet"
    if pred_path.is_file():
        df_pred = bundle.load_sealed_prediction("M5.parquet")
        assert len(df_pred) > 0
        df_pred_rel = bundle.load_sealed_prediction("data/processed/predictions/M5.parquet")
        assert len(df_pred_rel) == len(df_pred)

    site_path = REPO_ROOT / "data/processed/site_table.parquet"
    if site_path.is_file():
        df_site = bundle.load_frozen_table("data/processed/site_table.parquet")
        assert len(df_site) == 13289

    manifest_doc = bundle.load_training_json("reports/run_manifest.json")
    assert "run_id" in manifest_doc


def test_refusal_on_wrong_results_hash() -> None:
    corrupted = dict(RESULTS_SHA256)
    corrupted["reports/results/event3.json"] = "0" * 64
    with pytest.raises(ProvenanceError, match=r"SHA-256 mismatch for reports/results/event3\.json"):
        load_verified_bundle(REPO_ROOT, expected_results_sha256=corrupted)


def test_refusal_on_missing_or_extra_digest_keys() -> None:
    incomplete = {k: v for k, v in RESULTS_SHA256.items() if k != "reports/results/mstar.json"}
    with pytest.raises(ProvenanceError, match="Expected digest keys"):
        load_verified_bundle(REPO_ROOT, expected_results_sha256=incomplete)


def test_refusal_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ProvenanceError, match="Required provenance file missing"):
        sha256_file(tmp_path / "nonexistent.json")
    with pytest.raises(ProvenanceError, match="Required provenance file missing"):
        load_verified_bundle(tmp_path)


def test_refusal_on_sealed_prediction_and_frozen_table_errors(bundle: VerifiedBundle) -> None:
    with pytest.raises(ProvenanceError, match=r"not sealed in scoring_ledger\.json"):
        bundle.load_sealed_prediction("unsealed_model.parquet")

    with pytest.raises(ProvenanceError, match="not registered in FROZEN_TABLES_SHA256"):
        bundle.load_frozen_table("data/processed/unknown.parquet")

    bad_frozen = dict(FROZEN_TABLES_SHA256)
    bad_frozen["data/processed/site_table.parquet"] = "f" * 64
    if (REPO_ROOT / "data/processed/site_table.parquet").is_file():
        with pytest.raises(ProvenanceError, match="Frozen table SHA-256 mismatch"):
            bundle.load_frozen_table(
                "data/processed/site_table.parquet", expected_hashes=bad_frozen
            )

    with pytest.raises(ProvenanceError, match="Training diagnostic file missing"):
        bundle.load_training_json("reports/training/missing.json")


def test_refusal_on_missing_ledger_record_or_broken_chain(bundle: VerifiedBundle) -> None:
    base_ledger: dict[str, Any] = copy.deepcopy(bundle.ledger)

    # 1. Missing ledger record (6 instead of 7)
    missing_rec = copy.deepcopy(base_ledger)
    missing_rec["events"] = missing_rec["events"][:-1]
    with pytest.raises(ProvenanceError, match="Ledger must have exactly 7 records"):
        verify_ledger_dict(missing_rec)

    # 2. Wrong event name in sequence
    wrong_event = copy.deepcopy(base_ledger)
    wrong_event["events"][2]["event"] = "event_2"
    with pytest.raises(ProvenanceError, match="event 'event_2' != expected 'checkpoint_1'"):
        verify_ledger_dict(wrong_event)

    # 3. Broken hash chain (prev_sha256 altered)
    broken_chain = copy.deepcopy(base_ledger)
    broken_chain["events"][4]["prev_sha256"] = "a" * 64
    with pytest.raises(ProvenanceError, match="Ledger hash chain broken at seq 5"):
        verify_ledger_dict(broken_chain)

    # 4. Dirty git state in ledger
    dirty_git = copy.deepcopy(base_ledger)
    dirty_git["events"][0]["git_dirty"] = True
    with pytest.raises(ProvenanceError, match="dirty or missing git state"):
        verify_ledger_dict(dirty_git)

    # 5. Header digest mismatches
    for field_name in ("params_sha256", "plan_sha256", "uv_lock_sha256", "run_id"):
        bad_hdr = copy.deepcopy(base_ledger)
        bad_hdr["header"][field_name] = "bad_value"
        with pytest.raises(ProvenanceError, match=f"Ledger {field_name}"):
            verify_ledger_dict(bad_hdr)

    # 6. Top-level keys invalid
    with pytest.raises(ProvenanceError, match="exactly 'header' and 'events'"):
        verify_ledger_dict({"header": base_ledger["header"]})

    # 7. Record seq mismatch or extra key
    bad_seq = copy.deepcopy(base_ledger)
    bad_seq["events"][0]["seq"] = 99
    with pytest.raises(ProvenanceError, match="Ledger record seq 99 != expected 1"):
        verify_ledger_dict(bad_seq)

    bad_keys = copy.deepcopy(base_ledger)
    bad_keys["events"][0]["extra_key"] = 1
    with pytest.raises(ProvenanceError, match="invalid keys"):
        verify_ledger_dict(bad_keys)

    bad_params = copy.deepcopy(base_ledger)
    bad_params["events"][0]["params_sha256"] = "0" * 64
    with pytest.raises(ProvenanceError, match="params_sha256 mismatch"):
        verify_ledger_dict(bad_params)


def test_refusal_on_tampered_outcome_4_7(bundle: VerifiedBundle) -> None:
    doc = copy.deepcopy(bundle.event3)
    doc["outcome_4_7"]["matched"][0]["row"] = 2
    with pytest.raises(ProvenanceError, match=r"Expected §4\.7 row 4"):
        _extract_outcome_4_7(doc)

    doc2 = copy.deepcopy(bundle.event3)
    doc2["outcome_4_7"]["matched"][0]["claim"] = "Altered claim"
    with pytest.raises(ProvenanceError, match=r"§4\.7 row 4 claim"):
        _extract_outcome_4_7(doc2)

    doc3 = copy.deepcopy(bundle.event3)
    doc3["outcome_4_7"]["matched"][0]["outcome"] = "Altered outcome"
    with pytest.raises(ProvenanceError, match=r"§4\.7 row 4 outcome"):
        _extract_outcome_4_7(doc3)

    doc4 = copy.deepcopy(bundle.event3)
    doc4["outcome_4_7"]["matched"][0]["venue"] = "Altered venue"
    with pytest.raises(ProvenanceError, match=r"§4\.7 row 4 venue"):
        _extract_outcome_4_7(doc4)

    doc5 = copy.deepcopy(bundle.event3)
    doc5["outcome_4_7"]["statement"] = "Altered statement"
    with pytest.raises(ProvenanceError, match=r"§4\.7 statement"):
        _extract_outcome_4_7(doc5)


def test_refusal_on_torch_or_xgboost_import() -> None:
    assert set(FORBIDDEN_MODULES) == {"torch", "xgboost"}
    assert_no_forbidden_imports()

    for mod_name in ("torch", "torch.nn", "xgboost", "xgboost.core"):
        fake_map = {mod_name: types.ModuleType(mod_name)}
        with pytest.raises(ProvenanceError, match="Forbidden module"):
            assert_no_forbidden_imports(fake_map)

    # Inject a fake torch module into sys.modules and verify load_verified_bundle refuses
    sys.modules["torch"] = types.ModuleType("torch")
    try:
        with pytest.raises(ProvenanceError, match="Forbidden module"):
            load_verified_bundle(REPO_ROOT)
    finally:
        sys.modules.pop("torch", None)

    # Active MetaPathFinder guard blocks `import torch` and `import xgboost`
    with ForbiddenImportBlocker.guard():
        with pytest.raises(ProvenanceError, match="Import of 'torch' is forbidden"):
            __import__("torch")
        with pytest.raises(ProvenanceError, match="Import of 'xgboost' is forbidden"):
            __import__("xgboost")


def test_make_figures_cli_verify_and_manifest() -> None:
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "make_figures.py"
    spec = importlib.util.spec_from_file_location("make_figures_cli", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rc = module.main(["--repo-root", str(REPO_ROOT), "--verify-only"])
    assert rc == 0
    manifest = module.build_provenance_manifest(REPO_ROOT)
    assert manifest["outcome_4_7"]["row"] == 4
    assert manifest["matrix_rows"] == 45544

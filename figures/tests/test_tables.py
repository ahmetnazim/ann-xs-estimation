"""Reconciliation, provenance, and determinism tests for Phase 9 diagnostic tables (T1-T9).

Verifies that every cell in ``t1_*.csv``-``t9_*.csv`` and every formatted Markdown
cell in ``docs/tables/benchmark_results.md`` equals its source value in
``reports/results/*`` (P9-D5 (a), P9-D6 (a), P9-D9 (a)).
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from annxs_figures.loader import (
    EVENT1_AUDIT_LABEL,
    S_IND_PANEL_NOTE,
    S_IND_UQ_NOTE,
    VERBATIM_ROW4_CLAIM,
    VERBATIM_ROW4_OUTCOME,
    VERBATIM_ROW4_STATEMENT,
    VERBATIM_ROW4_VENUE,
    VerifiedBundle,
    load_verified_bundle,
    sha256_file,
)
from annxs_figures.tables import (
    ROUNDING_RULES_TEXT,
    GeneratedTables,
    format_num,
    generate_all_tables,
    metric_fmt_kind,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def bundle() -> VerifiedBundle:
    """Load the cryptographically verified Phase 8 results bundle once per module."""
    return load_verified_bundle(REPO_ROOT)


@pytest.fixture(scope="module")
def generated(bundle: VerifiedBundle, tmp_path_factory: pytest.TempPathFactory) -> GeneratedTables:
    """Generate T1-T9 CSV mirrors and benchmark_results.md in a temporary directory."""
    out_dir = tmp_path_factory.mktemp("tables_out")
    return generate_all_tables(bundle, out_dir)


def _parse_markdown_tables(md_text: str) -> list[pd.DataFrame]:
    """Parse all GitHub-flavored Markdown tables in ``md_text`` into DataFrames of strings."""
    tables: list[pd.DataFrame] = []
    current_lines: list[str] = []
    for line in md_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            current_lines.append(stripped)
        else:
            if current_lines:
                tables.append(_lines_to_df(current_lines))
                current_lines = []
    if current_lines:
        tables.append(_lines_to_df(current_lines))
    return tables


def _split_md_row(line: str) -> list[str]:
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    parts: list[str] = []
    buf: list[str] = []
    escaped = False
    for ch in inner:
        if escaped:
            if ch == "|":
                buf.append("|")
            else:
                buf.append("\\" + ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == "|":
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf).strip())
    return parts


def _lines_to_df(lines: list[str]) -> pd.DataFrame:
    headers = _split_md_row(lines[0])
    rows = [_split_md_row(line) for line in lines[2:]]
    return pd.DataFrame(rows, columns=headers)


def _float_match(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-15)


def _first_row(df: pd.DataFrame) -> dict[str, object]:
    """Return the first row of ``df`` as a string-keyed dictionary."""
    rec = df.iloc[0].to_dict()
    return {str(k): v for k, v in rec.items()}


def test_format_num_and_metric_fmt_kind() -> None:
    """Test deterministic scalar formatting rules across all branches."""
    assert format_num(None) == "—"
    assert format_num(float("nan")) == "—"
    assert format_num("literal") == "literal"
    assert format_num(True) == "true"
    assert format_num(False) == "false"
    assert format_num(42) == "42"
    assert format_num(42.0, "int") == "42"
    assert format_num(1.23456, "dec4") == "1.2346"
    assert format_num(12.3456, "dec2") == "12.35"
    assert format_num(6.650446859879366e-11, "sci2") == "6.65e-11"
    assert format_num(8.358408559010364e-05, "auto_share") == "8.36e-05"
    assert format_num(0.064616, "auto_share") == "0.0646"
    assert format_num(0.0, "auto_share") == "0.0000"
    assert format_num([1, 2]) == "[1, 2]"

    assert metric_fmt_kind("mdape") == "dec2"
    assert metric_fmt_kind("apb_pct") == "dec2"
    assert metric_fmt_kind("rmse_raw") == "dec2"
    assert metric_fmt_kind("solver_residual_max") == "sci2"
    assert metric_fmt_kind("rmse_log") == "dec4"


def test_t1_reconciliation(bundle: VerifiedBundle, generated: GeneratedTables) -> None:
    """Reconcile every row of T1 against event3_hypotheses.json, mstar.json, and event3.json."""
    t1 = generated.frames["t1"]
    assert generated.csv_paths["t1"].is_file()
    csv_t1 = pd.read_csv(generated.csv_paths["t1"])
    assert len(csv_t1) == len(t1)

    # 1. Verbatim §4.7 row 4
    o_row = _first_row(t1.loc[t1["section"] == "outcome_4_7"])
    assert o_row["hypothesis"] == "§4.7 row 4"
    assert o_row["target_or_quantity"] == VERBATIM_ROW4_OUTCOME
    assert o_row["statistic"] == VERBATIM_ROW4_STATEMENT
    assert VERBATIM_ROW4_VENUE in str(o_row["verdict"])
    assert o_row["evidence_tier"] == "Confirmatory"

    # 2. Event 2 M* selection
    mstar_rows = t1.loc[t1["section"] == "event_2_mstar"]
    assert len(mstar_rows) == 6
    for _, r in mstar_rows.iterrows():
        cand, tgt = str(r["target_or_quantity"]).split(":")
        expected_mean = float(bundle.mstar["rmse_log_mean"][cand][tgt])
        assert _float_match(float(r["estimate"]), expected_mean)
        expected_verdict = "selected M*" if bundle.mstar["mstar"][tgt] == cand else "candidate"
        assert r["verdict"] == expected_verdict
        assert r["split_note"] == EVENT1_AUDIT_LABEL

    # 3. H1-H4 rows match bundle.hypotheses_df
    h_rows = t1.loc[t1["section"] == "hypotheses_H1_H4"]
    hyp_src = bundle.hypotheses_df[
        bundle.hypotheses_df["hypothesis"].isin(["H1", "H2", "H3", "H4"])
    ]
    assert len(h_rows) == len(hyp_src)
    for (_, r), (_, s) in zip(h_rows.iterrows(), hyp_src.iterrows(), strict=True):
        assert _float_match(float(r["estimate"]), float(s["estimate"]))
        assert _float_match(float(r["ci_lo"]), float(s["ci_lo"]))
        assert _float_match(float(r["ci_hi"]), float(s["ci_hi"]))
        assert _float_match(float(r["ci_level"]), float(s["ci_level"]))
        assert int(r["n_units"]) == int(s["n_units"])
        assert r["evidence_tier"] == s["evidence_tier"]

    # 4. H4 point quantities
    h4_pts = t1.loc[t1["section"] == "H4_point_quantities"]
    assert len(h4_pts) == 10
    for _, r in h4_pts.iterrows():
        split_k = "S-ID" if r["split"] == "S-ID" else "S-LRO_reported"
        pt_k = str(r["target_or_quantity"]).split("_", 2)[2]
        assert _float_match(
            float(r["estimate"]),
            float(bundle.hypotheses["H4"][split_k][pt_k]),
        )

    # 5. H4 caveats by F_S (including undefined rho_n_s on F_S)
    fs_rows = t1.loc[t1["section"] == "H4_caveats_by_F_S"]
    fs_undef = _first_row(fs_rows.loc[fs_rows["target_or_quantity"] == "F_S:rho_n_s"])
    assert format_num(fs_undef["estimate"]) == "—"
    assert "undefined: an input is constant in this stratum" in str(fs_undef["verdict"])

    # 6. Far H2 explicit "not computed"
    far_h2 = _first_row(t1.loc[(t1["section"] == "H2_far_not_nested") & (t1["stratum"] == "far")])
    assert format_num(far_h2["estimate"]) == "—"
    assert far_h2["verdict"] == "not computed: fewer than 30 sites in a repeat"


def test_t2_reconciliation(bundle: VerifiedBundle, generated: GeneratedTables) -> None:
    """Reconcile every cell of T2 against reports/results/event3_matrix.parquet."""
    t2 = generated.frames["t2"]
    assert len(t2) > 0
    assert "mae" not in set(t2["metric"].str.lower())

    mat = bundle.matrix_df
    key_cols = [
        "spec",
        "model",
        "ablation",
        "analysis",
        "family",
        "target",
        "stratum_kind",
        "stratum",
        "metric",
    ]
    merged = t2.merge(mat, on=key_cols, suffixes=("_t2", "_src"))
    assert len(merged) == len(t2)
    for _, r in merged.iterrows():
        assert _float_match(float(r["value_t2"]), float(r["value_src"]))
        assert int(r["n_rows_t2"]) == int(r["n_rows_src"])
        assert int(r["n_units_t2"]) == int(r["n_units_src"])
        assert int(r["n_excluded_t2"]) == int(r["n_excluded_src"])
        assert int(r["n_repeats_t2"]) == int(r["n_repeats_src"])


def test_t3_reconciliation(bundle: VerifiedBundle, generated: GeneratedTables) -> None:
    """Reconcile every cell of T3 (S2 98.75% contrasts and kappa quintiles) against source."""
    t3 = generated.frames["t3"]
    s2_src = bundle.hypotheses["S2"]

    ov = t3.loc[t3["section"] == "S2_overall"]
    assert len(ov) == 4
    for _, r in ov.iterrows():
        blk = s2_src[str(r["split"])][str(r["contrast"])]
        assert _float_match(float(r["estimate"]), float(blk["estimate"]))
        assert _float_match(float(r["ci_lo"]), float(blk["lo"]))
        assert _float_match(float(r["ci_hi"]), float(blk["hi"]))
        assert _float_match(float(r["ci_level"]), 0.9875)
        assert r["evidence_tier"] == "Secondary"

    far_rows = t3.loc[(t3["section"] == "S2_far_not_nested") & (t3["stratum"] == "far")]
    assert len(far_rows) == 2
    for _, r in far_rows.iterrows():
        assert pd.isna(r["estimate"])
        assert r["status"] == "not computed: fewer than 30 sites in a repeat"

    kq_rows = t3.loc[t3["section"] == "S2_kappa_quintiles_descriptive"]
    assert len(kq_rows) == 20
    for _, r in kq_rows.iterrows():
        q_name = str(r["stratum"]).replace("kappa_", "")
        expected = float(
            s2_src["kappa_quintiles_descriptive"][str(r["split"])][q_name][str(r["contrast"])]
        )
        assert _float_match(float(r["estimate"]), expected)


def test_t4_reconciliation(bundle: VerifiedBundle, generated: GeneratedTables) -> None:
    """Reconcile every row of T4 against event3_uq.json."""
    t4 = generated.frames["t4"]
    assert len(t4) == len(bundle.uq_df)

    m5_sid_dbar = _first_row(
        t4.loc[
            (t4["spec"] == "M5|primary")
            & (t4["family"] == "S-ID")
            & (t4["target"] == "dbar")
            & (t4["stratum_kind"] == "overall")
        ]
    )
    src_ov = bundle.uq["M5|primary"]["S-ID"]["dbar"]["overall"]
    assert _float_match(float(str(m5_sid_dbar["coverage"])), float(src_ov["coverage"]))
    assert _float_match(float(str(m5_sid_dbar["crps"])), float(src_ov["crps"]))
    assert _float_match(
        float(str(m5_sid_dbar["mean_log_width"])),
        float(src_ov["mean_log_width"]),
    )
    assert bool(m5_sid_dbar["in_band"]) is True
    assert m5_sid_dbar["evidence_tier"] == "Secondary"

    m5_sind_dbar = _first_row(
        t4.loc[
            (t4["spec"] == "M5|primary")
            & (t4["family"] == "S-IND")
            & (t4["target"] == "dbar")
            & (t4["stratum_kind"] == "overall")
        ]
    )
    assert int(str(m5_sind_dbar["n_rows"])) == 33
    assert m5_sind_dbar["split_note"] == S_IND_UQ_NOTE


def test_t5_reconciliation(bundle: VerifiedBundle, generated: GeneratedTables) -> None:
    """Reconcile every row of T5 against event3_consistency.json."""
    t5 = generated.frames["t5"]
    c_doc = bundle.consistency

    obs_rows = t5.loc[t5["model"] == "observed_frame"]
    assert len(obs_rows) == 6
    for _, r in obs_rows.iterrows():
        expected = float(c_doc["observed_closure"][str(r["metric"])])
        assert _float_match(float(r["value"]), expected)

    mod_rows = t5.loc[t5["model"] != "observed_frame"]
    assert len(mod_rows) > 0
    for _, r in mod_rows.iterrows():
        m_blk = c_doc["models"][str(r["model"])][str(r["split"])]
        cat = str(r["category"])
        met = str(r["metric"])
        expected = float(m_blk[met]) if cat == "scalar" else float(m_blk[cat][met])
        assert _float_match(float(r["value"]), expected)


def test_t6_reconciliation(bundle: VerifiedBundle, generated: GeneratedTables) -> None:
    """Reconcile T6 ablations, A4 roles, SA1 statuses, and ablation H4 points against source."""
    t6 = generated.frames["t6"]
    desc = bundle.descriptive

    # Check A4 primary vs supplementary role assignment
    a4_m2xgb = t6.loc[t6["spec"] == "M2-XGB_A4"]
    assert len(a4_m2xgb) > 0
    assert (a4_m2xgb["role"] == "primary realised M* (A4)").all()

    a4_supp = t6.loc[t6["spec"].isin(["M1_A4", "M2-MLP_A4"])]
    assert len(a4_supp) > 0
    assert (a4_supp["role"] == "supplementary (A4; §13.8)").all()

    # Check SA1 not-trained status rows
    sa1_status = t6.loc[t6["section"] == "sensitivity_SA1_status"]
    assert len(sa1_status) == 3
    for _, r in sa1_status.iterrows():
        expected_status = desc["sensitivity_status"][str(r["model"])]["SA1"]
        assert r["status"] == expected_status

    # Check ablation H4 points
    ab_h4 = t6.loc[t6["section"].str.startswith("ablation_H4_points")]
    assert len(ab_h4) > 0
    for _, r in ab_h4.iterrows():
        grp = str(r["section"]).removeprefix("ablation_")
        expected_val = float(desc["ablations_M5"][str(r["spec"])][grp][str(r["metric"])])
        assert _float_match(float(r["value"]), expected_val)


def test_t7_and_t8_reconciliation(bundle: VerifiedBundle, generated: GeneratedTables) -> None:
    """Reconcile T7 (S-RND) and T8 (S1 HyG consistency) against source."""
    t7 = generated.frames["t7"]
    assert len(t7) > 0
    assert set(t7["model"].unique()) == {"M1", "M2-XGB"}
    assert (t7["weights"] == "1/|I_j|").all()
    assert (t7.loc[t7["stratum_kind"] == "all", "n_rows"] == 9673).all()
    assert (t7["evidence_tier"] == "Secondary").all()

    t8 = generated.frames["t8"]
    s1 = bundle.descriptive["S1"]
    rmse_row = _first_row(t8.loc[t8["metric"] == "rmse_log"])
    rho_row = _first_row(t8.loc[t8["metric"] == "spearman_rho"])
    assert _float_match(float(str(rmse_row["value"])), float(s1["rmse_log"]))
    assert _float_match(float(str(rho_row["value"])), float(s1["spearman_rho"]))
    assert rmse_row["note"] == "consistency check, not independent validation (protocol 4.6)"


def test_t9_reconciliation(bundle: VerifiedBundle, generated: GeneratedTables) -> None:
    """Reconcile T9 frame, split, Event 1 audit, and 'not computed' items against source."""
    t9 = generated.frames["t9"]
    not_comp = t9.loc[t9["section"] == "not_computed_items"]
    assert set(not_comp["item"]) == {"L8_zarrabi_product", "H2_far_stratum", "S2_far_stratum"}
    l8_row = _first_row(not_comp.loc[not_comp["item"] == "L8_zarrabi_product"])
    assert "doi:10.5281/zenodo.13883263" in str(l8_row["status_or_note"])

    audit_rows = t9.loc[t9["section"] == "event_1_audit"]
    rho_row = _first_row(audit_rows.loc[audit_rows["item"] == "rho_hat_km"])
    assert math.isclose(
        float(str(rho_row["value"])), float(bundle.audit["rho_hat_km"]), rel_tol=0.0, abs_tol=1e-15
    )

    strata_rows = t9.loc[t9["section"] == "event_1_audit_strata"]
    assert len(strata_rows) == 1
    strata_first = _first_row(strata_rows)
    assert int(float(str(strata_first["value"]))) == len(bundle.audit_strata_df) == 11964


def test_markdown_tables_and_verbatim_strings(generated: GeneratedTables) -> None:
    """Verify benchmark_results.md structure, verbatim strings, and deterministic hashes."""
    md_text = generated.markdown_path.read_text(encoding="utf-8")

    # Rounding rules stated once
    assert md_text.count(ROUNDING_RULES_TEXT) == 1

    # Verbatim required strings
    for required in (
        VERBATIM_ROW4_OUTCOME,
        VERBATIM_ROW4_CLAIM,
        VERBATIM_ROW4_STATEMENT,
        VERBATIM_ROW4_VENUE,
        EVENT1_AUDIT_LABEL,
        S_IND_PANEL_NOTE,
        S_IND_UQ_NOTE,
        "consistency check, not independent validation (protocol 4.6)",
        "doi:10.5281/zenodo.13883263",
        "not computed: fewer than 30 sites in a repeat",
    ):
        assert required in md_text, f"Missing required verbatim text: {required!r}"

    md_tables = _parse_markdown_tables(md_text)
    assert len(md_tables) == 9
    assert len(md_tables[0]) == len(generated.frames["t1"])
    assert len(md_tables[1]) == len(generated.frames["t2"])
    assert len(md_tables[2]) == len(generated.frames["t3"])
    assert len(md_tables[4]) == len(generated.frames["t5"])
    assert len(md_tables[6]) == len(generated.frames["t7"])
    assert len(md_tables[7]) == len(generated.frames["t8"])
    assert len(md_tables[8]) == len(generated.frames["t9"])

    # Cell-level displayed-precision check on Markdown tables
    md_t1_rows = md_tables[0].to_dict(orient="records")
    for md_row, (_, r) in zip(md_t1_rows, generated.frames["t1"].iterrows(), strict=True):
        assert md_row["estimate"] == format_num(r["estimate"], "dec4")
        assert md_row["ci_lo"] == format_num(r["ci_lo"], "dec4")
        assert md_row["ci_hi"] == format_num(r["ci_hi"], "dec4")

    md_t2_rows = md_tables[1].to_dict(orient="records")
    for md_row, (_, r) in zip(md_t2_rows, generated.frames["t2"].iterrows(), strict=True):
        assert md_row["value"] == format_num(r["value"], metric_fmt_kind(str(r["metric"])))

    md_t3_rows = md_tables[2].to_dict(orient="records")
    for md_row, (_, r) in zip(md_t3_rows, generated.frames["t3"].iterrows(), strict=True):
        assert md_row["estimate"] == format_num(r["estimate"], "dec4")
        assert md_row["ci_lo"] == format_num(r["ci_lo"], "dec4")
        assert md_row["ci_hi"] == format_num(r["ci_hi"], "dec4")


def test_tables_determinism(bundle: VerifiedBundle, tmp_path: Path) -> None:
    """Verify that running generate_all_tables twice produces byte-identical outputs."""
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    out_a = generate_all_tables(bundle, dir_a)
    out_b = generate_all_tables(bundle, dir_b)

    assert sha256_file(out_a.markdown_path) == sha256_file(out_b.markdown_path)
    for key in out_a.csv_paths:
        assert sha256_file(out_a.csv_paths[key]) == sha256_file(out_b.csv_paths[key])

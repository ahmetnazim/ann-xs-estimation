"""Phase 11 FAIR archival, software citation, and metadata verification suite.

Sources: docs/3_experimental_protocol.md (§§4.7, 13, 14), P11-D1 to P11-D9, CITATION.cff,
codemeta.json, LICENSE, README.md, and reports/archival_manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.evaluation.ledger import Ledger

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_license_spdx_and_bsd3_compliance() -> None:
    """P11-D4 (a): LICENSE is non-empty, declares SPDX-License-Identifier: BSD-3-Clause, and
    includes institutional non-endorsement terms for UM/NCCHE and USDA-ARS.
    """
    lic_path = REPO_ROOT / "LICENSE"
    assert lic_path.is_file(), "LICENSE file missing at repository root"
    text = lic_path.read_text(encoding="utf-8")
    assert "SPDX-License-Identifier: BSD-3-Clause" in text
    assert "BSD 3-Clause License" in text
    assert "Ahmet Nazim Sahin" in text
    assert "University of Mississippi" in text
    assert "United States" in text
    assert "Agricultural Research Service" in text
    assert "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS" in text


def test_citation_cff_v120_schema_and_metadata() -> None:
    """P11-D3 (a), P11-D5 (a): CITATION.cff is valid YAML conforming to CFF v1.2.0 with all 6
    authors, lead ORCID, BSD-3-Clause license, Zenodo DOI placeholders, and ASCE preferred-citation.
    """
    cff_path = REPO_ROOT / "CITATION.cff"
    assert cff_path.is_file(), "CITATION.cff missing at repository root"
    cff: dict[str, Any] = yaml.safe_load(cff_path.read_text(encoding="utf-8"))

    assert cff["cff-version"] == "1.2.0"
    assert cff["type"] == "software"
    assert cff["version"] == "0.1.0"
    assert cff["license"] == "BSD-3-Clause"
    assert cff["doi"] == "10.5281/zenodo.XXXXXXX"
    assert cff["repository-code"] == "https://github.com/ansahin/ann-xs-estimation"

    authors = cff["authors"]
    assert isinstance(authors, list)
    assert len(authors) == 6
    expected_families = ["Sahin", "Yasarer", "Ozeren", "Langendoen", "Al-Hamdan", "D'Alessio"]
    assert [a["family-names"] for a in authors] == expected_families
    assert re.match(
        r"^https://orcid\.org/\d{4}-\d{4}-\d{4}-\d{3}[\dX]$",
        str(authors[0]["orcid"]),
    )
    assert authors[0]["orcid"] == "https://orcid.org/0009-0003-5171-7589"

    pref = cff["preferred-citation"]
    assert pref["type"] == "article"
    assert pref["journal"] == "Journal of Hydrologic Engineering"
    assert "6060-13000-030-00D" in str(pref["notes"])
    assert len(pref["authors"]) == 6


def test_codemeta_json_ld_consistency_with_pyproject_and_cff() -> None:
    """P11-D3 (a), P11-D8 (a): codemeta.json is valid CodeMeta v2.0 JSON-LD and is consistent
    with both pyproject.toml and CITATION.cff.
    """
    cm_path = REPO_ROOT / "codemeta.json"
    assert cm_path.is_file(), "codemeta.json missing at repository root"
    cm: dict[str, Any] = json.loads(cm_path.read_text(encoding="utf-8"))
    cff: dict[str, Any] = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    pyproj: dict[str, Any] = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert cm["@context"] == "https://doi.org/10.5063/schema/codemeta-2.0"
    assert cm["@type"] == "SoftwareSourceCode"
    assert cm["name"] == pyproj["project"]["name"] == "ann-xs-estimation"
    assert cm["version"] == pyproj["project"]["version"] == cff["version"] == "0.1.0"
    assert cm["description"] == pyproj["project"]["description"]
    assert cm["license"] == "https://spdx.org/licenses/BSD-3-Clause.html"
    assert cm["codeRepository"] == cff["repository-code"]
    assert cm["funding"] == "USDA-ARS Cooperative Agreement No. 6060-13000-030-00D"
    assert len(cm["author"]) == len(cff["authors"]) == 6
    assert cm["author"][0]["@id"] == cff["authors"][0]["orcid"]
    assert cm["softwareRequirements"] == pyproj["project"]["dependencies"]


def test_readme_completeness_and_exact_benchmark_metrics() -> None:
    """P11-D7 (a), P11-D8 (a): README.md is populated with required sections, exact locked Event 1-3
    metrics, ASCE Availability Statement, BibTeX/APA citations, and zero stale constants.
    """
    readme_path = REPO_ROOT / "README.md"
    assert readme_path.is_file(), "README.md missing at repository root"
    text = readme_path.read_text(encoding="utf-8")
    assert len(text) > 10_000

    required_tokens = (
        "10.5281/zenodo.XXXXXXX",
        "10.5281/zenodo.YYYYYYY",
        "BSD-3-Clause",
        "tanh-sinh",
        "quad_level = 6",
        "3,988",
        "131",
        "M* = M2-XGB",
        "0.4396",
        "0.6254",
        "0.4782",
        "0.7284",
        "0.8997",
        "0.9025",
        "M5 worse (S-LRO CI lower bound > 0)",
        "optimistic under spatial/network dependence",
        "Data, Models, and Code Availability Statement",
        "6060-13000-030-00D",
        "@article{Sahin2026_JHE_ANN_XS",
        "@software{Sahin2026_Zenodo_ANN_XS_Software",
    )
    for tok in required_tokens:
        assert tok in text, f"README.md missing required token: {tok!r}"

    for stale in ("Gauss-Legendre", "B = 2,000", "[0.005, 0.40]"):
        assert stale not in text, f"README.md contains stale token: {stale!r}"


def test_archival_manifest_and_frozen_artifact_digests() -> None:
    """P11-D6 (a), P11-D8 (a): reports/archival_manifest.json verifies live SHA-256 parity across
    all archival metadata, frozen results/ledger, and pipeline lockfiles, and confirms the 7-record
    scoring ledger chain remains intact.
    """
    manifest_path = REPO_ROOT / "reports" / "archival_manifest.json"
    assert manifest_path.is_file(), "reports/archival_manifest.json missing"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["package_name"] == "ann-xs-estimation"
    assert manifest["package_version"] == "0.1.0"
    assert manifest["release_tag"] == "v1.0.0"
    assert manifest["license"] == "BSD-3-Clause"

    rr = manifest["run_root_archive"]
    assert rr["run_id"] == "p8-da9b7303cb84"
    assert rr["sha256"] == "a6a88f772aa864ed24cadcc380aeb85831c02dd0e28473143ce9905e450db4b8"
    assert rr["n_files"] == 54494
    assert (
        rr["grid_merkle_root"] == "bb97203df74599de568bee5316d4b147d1b7dea770220d7917da464b937a1fb2"
    )
    assert rr["grid_n_leaves"] == 51508

    # Verify live SHA-256 digests for all git-tracked groups
    for group in (
        "archival_metadata_files",
        "frozen_results_and_ledger",
        "frozen_pipeline_and_locks",
    ):
        entries: dict[str, dict[str, Any]] = manifest[group]
        assert entries, f"Empty manifest group: {group}"
        for rel_path, meta in entries.items():
            fp = REPO_ROOT / rel_path
            assert fp.is_file(), f"Tracked file missing on disk: {rel_path}"
            assert _sha256_file(fp) == meta["sha256"], f"SHA-256 mismatch for {rel_path}"
            assert fp.stat().st_size == meta["size_bytes"], f"Size mismatch for {rel_path}"

    # Verify optional DVC-cached / local-docs groups when present on disk
    for opt_group in ("frozen_processed_data", "local_docs_artifacts"):
        for rel_path, meta in manifest[opt_group].items():
            fp = REPO_ROOT / rel_path
            if fp.is_file():
                assert _sha256_file(fp) == meta["sha256"], f"SHA-256 mismatch for {rel_path}"

    # Verify scoring_ledger.json hash chain integrity (7 events, all dirty=False)
    ledger = Ledger.load(REPO_ROOT / "reports" / "results" / "scoring_ledger.json")
    assert len(ledger.events) == 7
    assert [r["event"] for r in ledger.events] == [
        "seal",
        "event_1",
        "checkpoint_1",
        "seal",
        "event_2",
        "checkpoint_2",
        "event_3",
    ]
    assert all(r["git_dirty"] is False for r in ledger.events)


def test_joss_paper_md_and_bib_integrity() -> None:
    """P11-D2 (a), P11-D7 (a), P11-D8 (a): when docs/ is present on disk, verify docs/paper.md
    frontmatter, section structure, absence of stale constants, and 100% citation key resolution
    against docs/paper.bib.
    """
    docs_dir = REPO_ROOT / "docs"
    if not docs_dir.exists():
        pytest.skip("docs/ directory not present")

    paper_path = docs_dir / "paper.md"
    bib_path = docs_dir / "paper.bib"
    assert paper_path.is_file(), "docs/paper.md missing"
    assert bib_path.is_file(), "docs/paper.bib missing"

    paper_text = paper_path.read_text(encoding="utf-8")
    bib_text = bib_path.read_text(encoding="utf-8")

    assert paper_text.startswith("---\n")
    parts = paper_text.split("---\n", 2)
    assert len(parts) == 3
    frontmatter: dict[str, Any] = yaml.safe_load(parts[1])
    assert "title" in frontmatter
    assert frontmatter["bibliography"] == "paper.bib"
    assert len(frontmatter["authors"]) == 6
    assert frontmatter["authors"][0]["orcid"] == "0009-0003-5171-7589"

    for heading in (
        "# Summary",
        "# Statement of Need",
        "# Mathematical & Physical Architecture",
        "# Empirical Benchmark Summary & Negative-Result Disclosure",
        "# Reproducibility & FAIR Architecture",
        "# Data, Models, and Code Availability Statement",
        "# Acknowledgements",
        "# References",
    ):
        assert heading in paper_text, f"docs/paper.md missing section: {heading}"

    for stale in ("Gauss-Legendre", "B = 2,000", "[0.005, 0.40]"):
        assert stale not in paper_text, f"docs/paper.md contains stale token: {stale!r}"

    cited_keys = set(re.findall(r"@([A-Za-z0-9_]+)", parts[2]))
    defined_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))
    assert cited_keys, "No citations found in docs/paper.md body"
    missing = cited_keys - defined_keys
    assert not missing, f"Unresolved citation keys in docs/paper.md: {sorted(missing)}"

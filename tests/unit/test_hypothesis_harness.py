"""Hypothesis harness and test-source contract (PHASE_07 refinement items 4 and 5; P7-D6 (a)).

Sources: P7-D6 (a) (ADL-101): ``ci`` is derandomized with no example database; ``thorough``
(5000 examples) keeps its database outside Google Drive (P4-D2 (a)) and is "used only by slow
tests"; the D strategies include the boundary atoms ``r = 1/2``, ``r = 10``, ``S = S_floor``
and logits ``z = +/-40`` (docs/2 1.3, 7.6). Refinement item 4: no ``HealthCheck`` is suppressed
without a written reason. Item 5: every Phase 7 test docstring cites docs/2, docs/3, a P7
decision, PHASE_07 or an ADL row. Expected values are those decision texts.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import torch
from hypothesis import HealthCheck, find, settings
from hypothesis.database import DirectoryBasedExampleDatabase

import hypothesis_profiles as hp
from domain_strategies import (
    LN_S_FLOOR,
    LOGIT_ATOM,
    R_MAX,
    R_MIN,
    TR,
    LogitRows,
    Rows,
    logit_rows,
    phi_rows,
)

pytestmark = pytest.mark.unit

TESTS = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS.parent
#: Phase 7 harness files (PHASE_07 Step 2-5 deliverables and Batch E).
PHASE7_FILES = (
    *sorted((TESTS / "physics").glob("test_*.py")),
    *(
        TESTS / "unit" / name
        for name in (
            "test_gradients.py",
            "test_architecture.py",
            "test_data.py",
            "test_evaluation.py",
            "test_ci_config.py",
            "test_hypothesis_harness.py",
        )
    ),
    TESTS / "integration" / "test_frozen_artifacts.py",
)
CITATION = re.compile(r"docs/[23]|P7-[A-Z]?\d|PHASE_07|ADL-\d+")
FIND = settings(database=None, derandomize=True, max_examples=2000)


def test_profiles_match_p7_d6() -> None:
    """P7-D6 (a): ci derandomized, no database, 200 examples; thorough 5000 outside Drive."""
    ci, th = settings.get_profile(hp.CI), settings.get_profile(hp.THOROUGH)
    got_ci = (ci.derandomize, ci.database, ci.max_examples, ci.deadline)
    assert got_ci == (True, None, 200, None), f"ci profile {got_ci} != (True, None, 200, None)"
    assert th.max_examples == 5000, f"thorough max_examples {th.max_examples} != 5000 (P7-D6)"
    assert isinstance(th.database, DirectoryBasedExampleDatabase), f"thorough db {th.database!r}"
    db = Path(th.database.path).resolve()
    assert not db.is_relative_to(REPO_ROOT), f"thorough database {db} inside the repo (P4-D2)"
    assert "CloudStorage" not in db.parts, f"thorough database {db} inside Google Drive (P4-D2)"
    for name, prof in ((hp.CI, ci), (hp.THOROUGH, th)):
        assert not prof.suppress_health_check, f"{name} suppresses {prof.suppress_health_check}"


@pytest.mark.parametrize(
    ("profile", "markexpr", "allowed"),
    [
        ("ci", "", True),
        ("ci", "not integration and not slow", True),
        ("ci", "slow", True),
        ("thorough", "slow", True),
        ("thorough", " slow ", True),
        ("thorough", "", False),
        ("thorough", "not integration and not slow", False),
        ("thorough", "slow or physics", False),
        ("dev", "", False),
    ],
)
def test_thorough_runs_only_under_slow(profile: str, markexpr: str, allowed: bool) -> None:
    """P7-D6: thorough is "used only by slow tests"; unknown profiles are refused (conftest)."""
    error = hp.check_profile_selection(profile, markexpr)
    assert (error is None) == allowed, f"({profile!r}, -m {markexpr!r}): {error!r} (P7-D6)"


@pytest.mark.parametrize(
    "atom",
    ["r = r_min", "r = r_max", "S = S_floor", "z_n = -40", "z_n = +40"],
)
def test_phi_rows_reach_boundary_atoms(atom: str) -> None:
    """P7-D6 (a), docs/2 1.3 and 7.6: phi_rows draws each boundary atom exactly."""
    nu_lo = float(TR.nu(torch.tensor([-LOGIT_ATOM], dtype=torch.float64))[0])
    nu_hi = float(TR.nu(torch.tensor([LOGIT_ATOM], dtype=torch.float64))[0])
    hit = {
        "r = r_min": lambda rows: bool((rows.r == R_MIN).any()),
        "r = r_max": lambda rows: bool((rows.r == R_MAX).any()),
        "S = S_floor": lambda rows: bool((rows.s == LN_S_FLOOR).any()),
        "z_n = -40": lambda rows: bool((rows.nu == nu_lo).any()),
        "z_n = +40": lambda rows: bool((rows.nu == nu_hi).any()),
    }[atom]
    found: Rows = find(phi_rows(), hit, settings=FIND)
    assert hit(found), f"phi_rows never drew {atom} in {FIND.max_examples} examples (P7-D6)"


@pytest.mark.parametrize("coord", ["z_n", "z_r"])
@pytest.mark.parametrize("z", [-LOGIT_ATOM, LOGIT_ATOM])
def test_logit_rows_reach_logit_atoms(coord: str, z: float) -> None:
    """P7-D6 (a), docs/2 7.6 and Theorem 8.2: logit_rows draws z_n and z_r = +/-40 exactly."""

    def hit(lr: LogitRows) -> bool:
        return bool((getattr(lr, coord) == z).any())

    found = find(logit_rows(), hit, settings=FIND)
    assert hit(found), f"logit_rows never drew {coord} = {z} (P7-D6)"


def test_no_health_check_suppressed_without_reason() -> None:
    """PHASE_07 refinement 4: every ``suppress_health_check=`` has a ``reason:`` comment."""
    keyword = re.compile(r"suppress_health_check\s*=")
    offenders = []
    for path in sorted(TESTS.rglob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if keyword.search(line.split("#", 1)[0]):
                context = " ".join(lines[max(0, i - 1) : i + 1])
                if "reason:" not in context:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{i + 1}")
    assert not offenders, f"HealthCheck suppressed without a written reason: {offenders}"
    assert list(HealthCheck), "hypothesis HealthCheck enum is empty"


def test_phase7_test_docstrings_cite_their_source() -> None:
    """PHASE_07 refinement 5: every Phase 7 test has a docstring citing docs/2, docs/3, a P7
    decision, PHASE_07 or an ADL row (the legacy Phase 4-6 tests are a Phase 8 carry-over).
    """
    missing = []
    for path in PHASE7_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                doc = ast.get_docstring(node) or ""
                if not CITATION.search(doc):
                    missing.append(f"{path.relative_to(REPO_ROOT)}::{node.name}")
    assert not missing, f"{len(missing)} Phase 7 tests lack a cited source: {missing}"

"""Hypothesis profiles of the Phase 7 harness (P7-D6 (a); PHASE_07 refinement item 4; ADL-101).

- ``ci`` (default): derandomized, no example database, 200 examples, no deadline.
- ``thorough``: 5000 examples, randomized, example database under
  ``HYPOTHESIS_STORAGE_DIRECTORY`` (default ``~/.cache/hypothesis/ANN_XS_Estimation``, outside
  Google Drive per P4-D2 (a)). P7-D6: "Selected by ``HYPOTHESIS_PROFILE`` and used only by
  ``slow`` tests", so :func:`check_profile_selection` refuses it unless the run is ``-m slow``.

No ``HealthCheck`` is suppressed anywhere in tests/ (enforced by
``tests/unit/test_hypothesis_harness.py``). Kept outside ``conftest.py`` so that the harness test
can import it (``--import-mode=importlib`` gives conftest modules private names).
"""

from __future__ import annotations

import os
from pathlib import Path

from hypothesis import configuration, settings
from hypothesis.database import DirectoryBasedExampleDatabase

HYPOTHESIS_HOME = Path(
    os.environ.setdefault(
        "HYPOTHESIS_STORAGE_DIRECTORY",
        str(Path.home() / ".cache" / "hypothesis" / "ANN_XS_Estimation"),
    )
)
configuration.set_hypothesis_home_dir(HYPOTHESIS_HOME)

CI = "ci"
THOROUGH = "thorough"
#: P7-D6: the only marker expression under which the thorough profile may run.
THOROUGH_MARKEXPR = "slow"

settings.register_profile(
    CI,
    derandomize=True,
    database=None,
    max_examples=200,
    deadline=None,
    print_blob=True,
)
settings.register_profile(
    THOROUGH,
    max_examples=5000,
    deadline=None,
    print_blob=True,
    database=DirectoryBasedExampleDatabase(HYPOTHESIS_HOME / "examples"),
)


def selected_profile() -> str:
    """The profile named by ``HYPOTHESIS_PROFILE`` (default ``ci``)."""
    return os.environ.get("HYPOTHESIS_PROFILE", CI)


def check_profile_selection(profile: str, markexpr: str) -> str | None:
    """Return an error message if ``profile`` may not run under ``markexpr`` (P7-D6), else None."""
    if profile not in (CI, THOROUGH):
        return f"unknown HYPOTHESIS_PROFILE {profile!r}; expected {CI!r} or {THOROUGH!r} (P7-D6)"
    if profile == THOROUGH and markexpr.strip() != THOROUGH_MARKEXPR:
        return (
            f"HYPOTHESIS_PROFILE={THOROUGH} runs only under -m {THOROUGH_MARKEXPR} (P7-D6); "
            f"got -m {markexpr.strip() or '<none>'!r}"
        )
    return None

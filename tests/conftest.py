"""Shared pytest configuration for the Phase 7 harness (P7-D4 (a), P7-D6 (a); ADL-101).

- Hypothesis storage lives outside the Drive-synced tree: ``HYPOTHESIS_STORAGE_DIRECTORY``
  defaults to ``~/.cache/hypothesis/ANN_XS_Estimation`` (P4-D2 (a) cache rule). Hypothesis
  resolves its home directory lazily, so setting it here precedes any storage access.
- Profiles: ``ci`` (default; derandomized, no example database, 200 examples, no deadline) and
  ``thorough`` (5000 examples, database under the same cache directory; for ``-m slow`` runs).
  Select with ``HYPOTHESIS_PROFILE=thorough``.
- Session-wide torch determinism: ``use_deterministic_algorithms(True)`` and one intra-op
  thread, restored at session end (torch is imported lazily, inside the fixture).

Strategies and oracles live in ``tests/domain_strategies.py`` and ``tests/reference_oracles.py``
(importable through ``pythonpath = ["tests"]``) because ``@given`` needs them at import time.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import configuration, settings
from hypothesis.database import DirectoryBasedExampleDatabase

HYPOTHESIS_HOME = Path(
    os.environ.setdefault(
        "HYPOTHESIS_STORAGE_DIRECTORY",
        str(Path.home() / ".cache" / "hypothesis" / "ANN_XS_Estimation"),
    )
)
configuration.set_hypothesis_home_dir(HYPOTHESIS_HOME)

settings.register_profile(
    "ci",
    derandomize=True,
    database=None,
    max_examples=200,
    deadline=None,
    print_blob=True,
)
settings.register_profile(
    "thorough",
    max_examples=5000,
    deadline=None,
    print_blob=True,
    database=DirectoryBasedExampleDatabase(HYPOTHESIS_HOME / "examples"),
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))


@pytest.fixture(scope="session", autouse=True)
def _torch_determinism() -> Iterator[None]:
    """Deterministic torch kernels and one thread for the whole session (P7-D6 (a))."""
    import torch

    prev = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_num_threads(),
    )
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    yield
    torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])
    torch.set_num_threads(prev[2])

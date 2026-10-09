"""Shared pytest configuration for the Phase 7 harness (P7-D4 (a), P7-D6 (a); ADL-101).

- Hypothesis profiles and storage live in ``tests/hypothesis_profiles.py``: ``ci`` (default;
  derandomized, no example database, 200 examples, no deadline) and ``thorough`` (5000
  examples, database outside the Drive-synced tree). Select with ``HYPOTHESIS_PROFILE``;
  ``thorough`` is refused unless the run is ``-m slow`` (P7-D6: "used only by slow tests").
- Session-wide torch determinism: ``use_deterministic_algorithms(True)`` and one intra-op
  thread, restored at session end (torch is imported lazily, inside the fixture).

Strategies and oracles live in ``tests/domain_strategies.py`` and ``tests/reference_oracles.py``
(importable through ``pythonpath = ["tests"]``) because ``@given`` needs them at import time.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from hypothesis import settings

import hypothesis_profiles as hp

settings.load_profile(hp.selected_profile())


def pytest_configure(config: pytest.Config) -> None:
    """Refuse an unknown profile, or ``thorough`` outside ``-m slow`` (P7-D6 (a))."""
    error = hp.check_profile_selection(hp.selected_profile(), str(config.getoption("markexpr")))
    if error is not None:
        raise pytest.UsageError(error)


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

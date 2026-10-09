"""Deterministic seeding of Python, NumPy and torch from params seeds.

Sources: protocol sections 1.5 and 3.2 (``seeds.global`` for S-IND, S-RND and every non-S-ID
family; ``seeds.sid_repeats[r]`` for S-ID repeat ``r``), section 10.8 (fold-table ES splits use
``SeedSequence([seed, family_idx, repeat, outer_fold, m])``), section 4.2 and ADL-016 / ADL-051
(one torch thread per worker process); P6-D5 (a), Batch D Q-D1 (common random numbers: a fit's
seed depends on its fold coordinates, part and index, **not** on the model, the ablation or the
grid point, so paired contrasts share initialization streams and loader orders).

Every derived seed is ``numpy.random.SeedSequence([base, *tokens]).generate_state(1)[0]``. String
tokens are the first four bytes of their SHA-256, so seeds do not depend on ``PYTHONHASHSEED``.

Torch is imported only by :func:`seed_torch` (OpenMP rule, ADL-051: XGB processes never load it).
Owning phase: 5 (implemented in Phase 6, Batch D).
"""

from __future__ import annotations

import hashlib
import random
from typing import Final, Literal

import numpy as np

from src.config import SeedsParams

__all__ = ["block_seed", "derive_seed", "fit_seed", "seed_python_numpy", "seed_torch"]

_UINT32: Final[int] = 2**32


def _token(value: int | str) -> int:
    """Map a key to ``[0, 2^32)``: integers as themselves, strings by SHA-256 (first 4 bytes)."""
    if isinstance(value, bool):
        raise TypeError("bool is not a valid seed key")
    if isinstance(value, int):
        if not 0 <= value < _UINT32:
            raise ValueError(f"integer seed keys must lie in [0, 2^32), got {value}")
        return value
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "big")


def derive_seed(base: int, *keys: int | str) -> int:
    """``SeedSequence([base, *tokens(keys)]).generate_state(1)[0]`` as a Python ``int``.

    Deterministic across processes and platforms (NumPy's SeedSequence hashing is specified).
    """
    entropy = [_token(base), *(_token(k) for k in keys)]
    state = np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)
    return int(state[0])


def block_seed(seeds: SeedsParams, family: str, repeat: int) -> int:
    """Return the block seed (protocol 3.2): ``sid_repeats[repeat]`` for S-ID, else global."""
    if family == "S-ID":
        if not 0 <= repeat < len(seeds.sid_repeats):
            raise ValueError(f"S-ID repeat {repeat} outside 0..{len(seeds.sid_repeats) - 1}")
        return int(seeds.sid_repeats[repeat])
    if repeat != 0:
        raise ValueError(f"family {family!r} has a single repeat (protocol 3.2)")
    return int(seeds.global_seed)


def fit_seed(
    seeds: SeedsParams,
    family: str,
    repeat: int,
    outer_fold: int,
    kind: Literal["inner", "member"],
    index: int,
) -> int:
    """Seed of one fit (Q-D1): ``derive_seed(block_seed, "fit", family, repeat, fold, kind, k)``.

    Independent of the model, ablation, analysis and grid point (common random numbers).
    """
    if kind not in ("inner", "member"):
        raise ValueError("kind must be 'inner' or 'member'")
    base = block_seed(seeds, family, repeat)
    return derive_seed(base, "fit", family, repeat, outer_fold, kind, index)


def seed_python_numpy(seed: int) -> None:
    """Seed ``random`` and NumPy's legacy global generator (code here uses explicit Generators)."""
    random.seed(seed)
    np.random.seed(seed % _UINT32)  # noqa: NPY002 - seeds third-party users of the global RNG


def seed_torch(seed: int, *, threads: int) -> None:
    """``torch.manual_seed(seed)`` and ``torch.set_num_threads(threads)`` (ADL-051: 1 thread).

    Call it immediately before building a model, so parameter initialization is reproducible.
    """
    import torch  # lazy: torch-free processes never load it (ADL-051)

    if threads < 1:
        raise ValueError("threads must be >= 1")
    torch.manual_seed(seed)
    torch.set_num_threads(threads)

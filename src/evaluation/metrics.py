"""Primary, secondary and descriptive metrics and the physical-consistency analogues.

Sources: protocol sections 4.3.1 and 4.3.2.
Owning phase: 6 for :func:`rmse_log` only (P6-D1 (a): the inner-selection statistic of protocol
3.4 item 2); every other metric of section 4.3 is re-tagged to Phase 7 and not implemented yet.
Torch-free and xgboost-free (usable in XGB workers, ADL-051).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["rmse_log"]


def rmse_log(pred_log: np.ndarray, obs_log: np.ndarray) -> float:
    """``RMSE_log = [(1/J) sum_j (ln p_j - ln o_j)^2]^(1/2)`` (protocol 4.3.1; one row per site).

    Both inputs are already natural logs. Used for inner selection (protocol 3.4 item 2): the
    inner-fold mean of this statistic for ``dbar`` (NN models) or the model's own targets (XGB).
    Raises on shape mismatch, empty input or non-finite values.
    """
    p = np.asarray(pred_log, dtype=np.float64)
    o = np.asarray(obs_log, dtype=np.float64)
    if p.shape != o.shape or p.ndim != 1 or p.size == 0:
        raise ValueError(f"need equal non-empty 1-D arrays, got {p.shape} and {o.shape}")
    if not (bool(np.isfinite(p).all()) and bool(np.isfinite(o).all())):
        raise ValueError("non-finite log predictions or observations")
    return math.sqrt(float(np.mean((p - o) ** 2)))

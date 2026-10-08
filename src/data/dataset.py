"""Float64 CPU PyTorch dataset and loader over one fold view (P5-D9; protocol 4.1, 4.2, 3.4).

Sources: protocol section 4.2 (CPU, float64, batch 256; O-8), section 4.1 (information routing:
under I_blind the n-head receives every feature except Q_E, S, F_S and V_E), section 2.6
(one row per site, so omega_i = 1/|J_tr|), section 3.4 (scaling fitted on the training part
only), Propositions 1.2 and 3.5 (targets and Phi inputs stay unscaled SI logs), ADL-016 and
ADL-051 (``num_workers=0``, one torch thread per process; this is the **only** module in
``src/data`` that imports torch).

Batch contract (draft; Phase 6 may extend it by ADL). Every tensor is float64 unless noted:

* ``x``: (B, F) standardized features (:class:`~src.data.normstats.NormStats`);
* ``x_blind_idx``: (F_blind,) int64 column indices of ``x`` for the n-head (I_blind);
* ``phys``: (B, 2) unscaled ``[ln Q, ln S]`` (m3/s, m/m); Q is ``Q_E`` (``Q_C`` for SA1);
* ``y``: (B, 4) unscaled ``[ln W, ln dbar, t_obs = ln(dmax/dbar), ln A]`` (m, m, -, m2);
* ``omega``: (B,) ``1 / |J_tr|`` (the training-part size passed in);
* ``row``: (B,) int64 position in the analysis frame.

Owning phase: 5.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from src.data.normstats import NormStats

__all__ = ["BLIND_EXCLUDED", "TARGETS", "SiteDataset", "make_loader"]

#: Features withheld from the n-head under I_blind (protocol 4.1; V_E is never a feature).
BLIND_EXCLUDED: Final[frozenset[str]] = frozenset({"Q_E", "S", "F_S", "V_E"})
#: Target columns and their log map: y = [ln W, ln dbar, t_obs, ln A].
TARGETS: Final[tuple[str, ...]] = ("W", "dbar", "t_obs", "A")

Batch = dict[str, torch.Tensor]


class SiteDataset(Dataset[Batch]):
    """Rows ``rows`` of an analysis frame as float64 tensors (batch-indexed).

    ``__getitem__`` takes a **list** of positions and returns a whole batch; use
    :func:`make_loader`, which pairs it with a ``BatchSampler`` and ``batch_size=None``.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        rows: np.ndarray,
        features: Sequence[str],
        stats: NormStats,
        n_train: int,
        discharge: str = "Q_E",
    ) -> None:
        if tuple(features) != stats.names:
            raise ValueError("feature order must match the fitted NormStats")
        if discharge not in ("Q_E", "Q_C"):
            raise ValueError("discharge must be 'Q_E' or 'Q_C' (SA1)")
        if n_train < 1:
            raise ValueError("n_train must be positive")
        sub = frame.iloc[np.asarray(rows, dtype=np.int64)]
        raw_x = sub[list(features)].to_numpy(dtype=np.float64)
        phys = np.column_stack([np.log(sub[discharge]), np.log(sub["S"])]).astype(np.float64)
        y = np.column_stack(
            [np.log(sub["W"]), np.log(sub["dbar"]), sub["t_obs"], np.log(sub["A"])]
        ).astype(np.float64)
        for name, arr in (("x", raw_x), ("phys", phys), ("y", y)):
            if not np.isfinite(arr).all():
                raise ValueError(f"non-finite values in {name} (no imputation, protocol 2.7)")
        self.x = torch.from_numpy(stats.transform(raw_x))
        self.phys = torch.from_numpy(phys)
        self.y = torch.from_numpy(y)
        self.row = torch.from_numpy(np.asarray(rows, dtype=np.int64).copy())
        self.omega_value = 1.0 / n_train
        blind = [j for j, name in enumerate(features) if name not in BLIND_EXCLUDED]
        self.x_blind_idx = torch.tensor(blind, dtype=torch.int64)
        self.features = tuple(features)

    def __len__(self) -> int:
        return int(self.row.shape[0])

    def __getitem__(self, index: list[int]) -> Batch:
        idx = torch.as_tensor(index, dtype=torch.int64)
        return {
            "x": self.x[idx],
            "x_blind_idx": self.x_blind_idx,
            "phys": self.phys[idx],
            "y": self.y[idx],
            "omega": torch.full((len(index),), self.omega_value, dtype=torch.float64),
            "row": self.row[idx],
        }


def make_loader(
    dataset: SiteDataset, batch_size: int, *, train: bool, seed: int
) -> DataLoader[Batch]:
    """DataLoader: seeded shuffling for training only, ``num_workers=0``, no pinning, no drop."""
    generator = torch.Generator().manual_seed(seed)
    base = RandomSampler(dataset, generator=generator) if train else SequentialSampler(dataset)
    sampler = BatchSampler(base, batch_size=batch_size, drop_last=False)
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=0,
        pin_memory=False,
    )

"""Split-then-scale feature normalization with a JSON artefact (protocol 3.4, L5; P5-D9).

Sources: protocol section 3.4 item 1 (every transformation that learns parameters is fitted on
the outer, or inner, **training** set only), section 3.5 L5, P5-D5 (per-feature transform
manifest ``params.transforms``), P5-D9 (``NormStats`` saved as JSON; no pickle or joblib;
features only, because targets and the physics inputs of Phi stay unscaled SI logs,
Propositions 1.2 and 3.5).

Transform kinds (``params.transforms.features``):

* ``identity``: ``g(x) = x``, then standardized.
* ``log``: ``g(x) = ln x``; every value must be > 0 (``ValueError`` otherwise), then standardized.
* ``log_offset``: ``g(x) = ln(x + c)`` with ``c`` = ``params.transforms.log_offset_frac`` x the
  smallest positive **training** value; values must be >= 0. ``c`` is stored in the artefact.
* ``passthrough``: binary flags; not centred or scaled (mean 0, std 1 stored).

No global scaler exists by design: Phase 6 fits one :class:`NormStats` per fit and saves it next
to that fit's checkpoint. Every saved artefact carries its **fit provenance**
(:data:`PROVENANCE_KEYS`: analysis, fold block, training part and index, and the SHA-256 of
``site_table.parquet`` and ``fold_assignment.parquet``; built by
:func:`src.data.folds.fit_provenance`), and the part must be a training part (L5).
Torch-free and xgboost-free (OpenMP rule, ADL-051).

Owning phase: 5.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np

__all__ = ["FIT_PARTS", "KINDS", "PROVENANCE_KEYS", "NormStats"]

Kind = Literal["identity", "log", "log_offset", "passthrough"]
KINDS: Final[tuple[str, ...]] = ("identity", "log", "log_offset", "passthrough")
#: Keys of the fit provenance stored in the artefact (all values are strings).
PROVENANCE_KEYS: Final[tuple[str, ...]] = (
    "analysis",
    "family",
    "fold_table_sha256",
    "index",
    "outer_fold",
    "part",
    "repeat",
    "site_table_sha256",
)
#: Fold-view parts a scaler may be fitted on (protocol 3.4 item 1, L5).
FIT_PARTS: Final[tuple[str, ...]] = ("train", "inner_train", "es_train")
_FORMAT: Final[str] = "annxs-normstats-v1"


def _provenance_items(provenance: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    if provenance is None:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in provenance.items()))


def _forward(x: np.ndarray, kind: str, offset: float) -> np.ndarray:
    if kind in ("identity", "passthrough"):
        return x
    if kind == "log":
        if not np.all(x > 0):
            raise ValueError("'log' transform needs strictly positive values")
        return np.log(x)
    if not np.all(x >= 0):
        raise ValueError("'log_offset' transform needs non-negative values")
    return np.log(x + offset)


def _inverse(g: np.ndarray, kind: str, offset: float) -> np.ndarray:
    if kind in ("identity", "passthrough"):
        return g
    if kind == "log":
        return np.exp(g)
    return np.exp(g) - offset


@dataclass(frozen=True)
class NormStats:
    """Fitted per-feature transform and standardization (float64).

    Attributes
    ----------
    names, kinds
        Feature names and transform kinds, in column order.
    offsets, mean, std
        Per-feature ``log_offset`` constants (0 otherwise), and the mean and population
        standard deviation of the transformed training values.
    n_fit
        Number of training rows used.
    provenance
        Sorted ``(key, value)`` pairs over :data:`PROVENANCE_KEYS`, or empty (in-memory use
        only; :meth:`save` requires it).
    """

    names: tuple[str, ...]
    kinds: tuple[str, ...]
    offsets: tuple[float, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    n_fit: int
    provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Validate field lengths, transform kinds and the fit provenance (L5)."""
        n = len(self.names)
        if not all(len(v) == n for v in (self.kinds, self.offsets, self.mean, self.std)):
            raise ValueError("NormStats fields must have one entry per feature")
        bad = sorted(set(self.kinds) - set(KINDS))
        if bad:
            raise ValueError(f"unknown transform kinds: {bad}")
        if self.provenance:
            prov = dict(self.provenance)
            if tuple(sorted(prov)) != PROVENANCE_KEYS:
                raise ValueError(f"provenance keys must be {PROVENANCE_KEYS}, got {sorted(prov)}")
            if prov["part"] not in FIT_PARTS:
                raise ValueError(f"NormStats may be fitted on {FIT_PARTS} only (L5)")

    @property
    def provenance_dict(self) -> dict[str, str]:
        """The fit provenance as a dict (empty if none)."""
        return dict(self.provenance)

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(
        cls,
        x_train: np.ndarray,
        names: Sequence[str],
        kinds: Mapping[str, str],
        log_offset_frac: float,
        provenance: Mapping[str, str] | None = None,
    ) -> NormStats:
        """Fit on training rows only (``x_train``: shape (n, F), raw feature values).

        ``provenance`` identifies the fitting part (:func:`src.data.folds.fit_provenance`).
        Raises ``ValueError`` on non-finite input, a missing transform kind, a domain violation,
        a zero-variance (non-passthrough) feature or an invalid provenance.
        """
        x = np.asarray(x_train, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(names) or x.shape[0] < 2:
            raise ValueError(f"x_train must be (n >= 2, {len(names)}), got {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("x_train contains non-finite values (no imputation, protocol 2.7)")
        missing = [n for n in names if n not in kinds]
        if missing:
            raise ValueError(f"no transform kind for {missing}")
        k_list, offsets, means, stds = [], [], [], []
        for j, name in enumerate(names):
            kind = kinds[name]
            col = x[:, j]
            offset = 0.0
            if kind == "log_offset":
                positive = col[col > 0]
                if positive.size == 0:
                    raise ValueError(f"{name}: no positive training value for the log offset")
                offset = float(log_offset_frac * positive.min())
            g = _forward(col, kind, offset)
            if kind == "passthrough":
                mu, sd = 0.0, 1.0
            else:
                mu, sd = float(g.mean()), float(g.std())
                if not sd > 0:
                    raise ValueError(f"{name}: zero variance in the training rows")
            k_list.append(kind)
            offsets.append(offset)
            means.append(mu)
            stds.append(sd)
        return cls(
            tuple(names),
            tuple(k_list),
            tuple(offsets),
            tuple(means),
            tuple(stds),
            len(x),
            _provenance_items(provenance),
        )

    # ------------------------------------------------------------------ apply
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Return standardized features, shape (n, F), float64."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.names):
            raise ValueError(f"expected (n, {len(self.names)}), got {x.shape}")
        out = np.empty_like(x)
        for j, (kind, c, mu, sd) in enumerate(
            zip(self.kinds, self.offsets, self.mean, self.std, strict=True)
        ):
            out[:, j] = (_forward(x[:, j], kind, c) - mu) / sd
        return out

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        """Invert :meth:`transform` (raw feature units)."""
        z = np.asarray(z, dtype=np.float64)
        out = np.empty_like(z)
        for j, (kind, c, mu, sd) in enumerate(
            zip(self.kinds, self.offsets, self.mean, self.std, strict=True)
        ):
            out[:, j] = _inverse(z[:, j] * sd + mu, kind, c)
        return out

    # ------------------------------------------------------------------ JSON
    def to_json(self) -> str:
        """Serialize; floats as ``float.hex`` so the round trip is exact."""
        payload = {
            "format": _FORMAT,
            "names": list(self.names),
            "kinds": list(self.kinds),
            "offsets": [float(v).hex() for v in self.offsets],
            "mean": [float(v).hex() for v in self.mean],
            "std": [float(v).hex() for v in self.std],
            "n_fit": self.n_fit,
            "provenance": dict(self.provenance),
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> NormStats:
        """Inverse of :meth:`to_json`."""
        raw = json.loads(text)
        if raw.get("format") != _FORMAT:
            raise ValueError(f"not a NormStats artefact (format={raw.get('format')!r})")
        return cls(
            names=tuple(raw["names"]),
            kinds=tuple(raw["kinds"]),
            offsets=tuple(float.fromhex(v) for v in raw["offsets"]),
            mean=tuple(float.fromhex(v) for v in raw["mean"]),
            std=tuple(float.fromhex(v) for v in raw["std"]),
            n_fit=int(raw["n_fit"]),
            provenance=_provenance_items(raw["provenance"]),
        )

    def save(self, path: Path) -> None:
        """Write the JSON artefact; refuses one without fit provenance."""
        if not self.provenance:
            raise ValueError("NormStats.save needs the fit provenance (folds.fit_provenance)")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> NormStats:
        """Read a JSON artefact written by :meth:`save`."""
        return cls.from_json(path.read_text(encoding="utf-8"))

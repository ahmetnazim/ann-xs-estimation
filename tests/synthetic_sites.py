"""Synthetic site and fold tables shared by the runner and orchestrator tests (no real data).

``site_frame`` and ``fold_table`` moved verbatim from ``tests/unit/test_models_trainer_runner.py``
(Phase 6, Batch D) so that the Phase 8 orchestrator tests reuse the same synthetic frame.
``fold_table`` builds one S-ID block (repeat 0, fold 0): every 5th site is test, inner fold =
position mod 3, ES split by position mod 7. Importable through ``pythonpath = ["tests"]``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import load_settings

PROTOCOL = load_settings().protocol
NN_FEATURES = tuple(PROTOCOL.transforms.features)


def site_frame(n: int = 60, seed: int = 11) -> pd.DataFrame:
    """Primary-variant site table with every NN, M1, target and physics column."""
    rng = np.random.default_rng(seed)
    da = np.exp(rng.normal(5.0, 1.0, n))
    q = 0.02 * da**0.9 * np.exp(rng.normal(0, 0.2, n))
    s = np.where(
        np.arange(n) % 4 == 0, np.exp(rng.normal(-5.0, 0.3, n)), np.exp(rng.normal(-8, 0.8, n))
    )
    w = 4.0 * q**0.5 * np.exp(rng.normal(0, 0.2, n))
    dbar = 0.3 * q**0.35 * np.exp(rng.normal(0, 0.2, n))
    t = rng.uniform(0.2, 0.9, n)
    df = pd.DataFrame(
        {
            "site_no": [f"{i:08d}" for i in range(n)],
            "variant": "primary",
            "F_DA": False,
            "F_Qsel": False,
            "F_S": np.arange(n) % 9 == 0,
            "division": np.where(np.arange(n) % 2 == 0, "A", "B"),
            "DA_km2": da,
            "stream_order": rng.integers(1, 8, n).astype(float),
            "z_min_m": rng.normal(200, 80, n),
            "S": s,
            "Q_E": q,
            "Q_C": q * np.exp(rng.normal(0, 0.1, n)),
            "V_E": np.where(np.arange(n) % 10 == 3, np.nan, q / (w * dbar)),
            "W": w,
            "dbar": dbar,
            "t_obs": t,
            "dmax": dbar * np.exp(t),
            "A": w * dbar,
            "D50_m": np.exp(rng.normal(math.log(0.01), 1.0, n)),
            "AI": np.exp(rng.normal(0.0, 0.4, n)),
            "damdensws": np.where(np.arange(n) % 3 == 0, 0.0, rng.uniform(0.01, 0.4, n)),
        }
    )
    for name in NN_FEATURES:
        if name not in df:
            kind = PROTOCOL.transforms.features[name]
            df[name] = np.exp(rng.normal(1.0, 0.5, n)) if kind == "log" else rng.normal(10, 3, n)
    m1 = (
        "ACC_NDAMS2010",
        "CAT_POPDENS10",
        "CAT_EVI_AMJ_2012",
        "CAT_EVI_JAS_2012",
        "CAT_EVI_JFM_2012",
        "CAT_EVI_OND_2011",
        "CAT_CLAYAVE",
        "CAT_SILTAVE",
        "CAT_SANDAVE",
        "m1_urban",
        "m1_forest",
        "m1_agriculture",
    )
    for name in m1:
        df[name] = rng.uniform(0.0, 50.0, n)
    missing = np.arange(n) % 11 == 5
    df.loc[missing, "CAT_EVI_AMJ_2012"] = np.nan
    df["F_M1_missing"] = missing
    return df


def fold_table(sites: pd.Series) -> pd.DataFrame:
    """One S-ID block: every 5th site is test; inner fold = position mod 3; ES by position mod 7."""
    n = len(sites)
    is_test = np.arange(n) % 5 == 0
    out = pd.DataFrame(
        {
            "family": "S-ID",
            "repeat": np.int8(0),
            "outer_fold": np.int8(0),
            "site_no": sites.to_numpy(),
            "is_test": is_test,
            "inner_fold": np.where(is_test, -1, np.arange(n) % 3).astype("int8"),
        }
    )
    for m in range(1, 11):
        out[f"es_member_{m}"] = ~is_test & (np.arange(n) % 7 == m % 7)
    return out

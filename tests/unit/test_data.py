"""DataLoader, missing-value and scaler contracts (Phase 7, Step 1; P7-D4 (a)). Synthetic data.

Sources: docs/3 section 4.2 and the ADL-077 batch contract of ``src.data.dataset`` (keys ``x``,
``x_blind_idx``, ``phys``, ``y``, ``omega``, ``row``; float64 CPU; batch 256), section 2.6
(one row per site, so omega_i = 1/|J_tr|), section 4.1 (I_blind withholds Q_E, S, F_S and V_E;
A_J = Q_E/V_E is training-free), section 2.7 (no imputation in any model; R12/R14 exclude the
site), section 2.3 and features stage R14 note (M1-only predictors set ``F_M1_missing``), section
3.4 item 1 and 10.10 (every scaler is fitted on a training part only and saved with its fit
provenance), section 3.5 L5. ADL-016 / ADL-051: ``num_workers = 0``.

Expected values come from those statements (numpy logs of the frame columns, ``ceil(n/B)``
batches, the docs/3 lists of parts and exclusions), not from the modules under test. The
existing Phase 5 tests (``test_data_folds_normstats_dataset.py``) stay unchanged; this module
adds the batch-size matrix, last-batch handling, every fold part and the missing-value rules.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from domain_strategies import PROTOCOL
from src.data import folds as fv
from src.data.dataset import SiteDataset, make_loader
from src.data.features import M1_PREDICTORS, build_site_table
from src.data.normstats import FIT_PARTS, NormStats
from src.models.baselines import aj_log_area, nn_features
from src.models.registry import FitKind, ModelSpec
from src.models.runner import FitRequest, Tables, _resolve_rows

pytestmark = pytest.mark.unit

FEATURES = nn_features(PROTOCOL)
KINDS = PROTOCOL.transforms.features
BATCH_KEYS = {"x", "x_blind_idx", "phys", "y", "omega", "row"}  # ADL-077
BATCH_SIZES = (1, 16, 64, 256)
#: docs/3 4.1: I_blind withholds these from the n-head (V_E is not a feature at all).
BLIND_OUT = ("Q_E", "S", "F_S", "V_E")
#: docs/3 3.4 item 1, 10.10: the parts a scaler may be fitted on.
TRAINING_PARTS = ("train", "inner_train", "es_train")
SHA_SITE, SHA_FOLD = "a" * 64, "b" * 64
N_FRAME, N_ROWS = 330, 300


def site_frame(n: int = N_FRAME, seed: int = 0) -> pd.DataFrame:
    """One row per site with every NN feature (kind-consistent values), targets and flags."""
    rng = np.random.default_rng(seed)
    cols: dict[str, object] = {
        "site_no": [f"{i:08d}" for i in range(n)],
        "variant": "primary",
        "F_DA": False,
        "F_Qsel": False,
    }
    for j, name in enumerate(FEATURES):
        kind = KINDS[name]
        if kind == "log":
            cols[name] = np.exp(rng.normal(0.5 * j - 3.0, 1.0, n))
        elif kind == "log_offset":
            cols[name] = np.where(np.arange(n) % 3 == 0, 0.0, rng.uniform(0.01, 0.4, n))
        elif kind == "passthrough":
            cols[name] = np.arange(n) % 4 == 0
        else:
            cols[name] = rng.normal(10.0 * j, 3.0, n)
    cols |= {
        "Q_C": np.exp(rng.normal(2.0, 1.0, n)),
        "V_E": rng.uniform(0.2, 2.0, n),
        "W": np.exp(rng.normal(3.0, 0.5, n)),
        "dbar": np.exp(rng.normal(0.0, 0.5, n)),
        "t_obs": rng.uniform(0.1, 1.0, n),
        "A": np.exp(rng.normal(3.0, 0.5, n)),
    }
    return pd.DataFrame(cols)


def view_rows(seed: int = 1) -> np.ndarray:
    """A sorted, non-contiguous training view of ``N_ROWS`` frame positions."""
    return np.sort(np.random.default_rng(seed).choice(N_FRAME, N_ROWS, replace=False))


def fitted(frame: pd.DataFrame, rows: np.ndarray) -> NormStats:
    x = frame.iloc[rows][list(FEATURES)].to_numpy(dtype=np.float64)
    return NormStats.fit(x, FEATURES, KINDS, PROTOCOL.transforms.log_offset_frac)


def epoch_rows(loader: torch.utils.data.DataLoader) -> list[int]:
    return [int(v) for b in loader for v in b["row"]]


# ----------------------------------------------------------------------------- batch contract
@pytest.mark.parametrize("train", [True, False], ids=["train", "eval"])
@pytest.mark.parametrize("b", BATCH_SIZES)
def test_batch_contract_and_last_batch(b: int, train: bool) -> None:
    """ADL-077 / docs/3 4.2: exact keys; x (B, F), phys (B, 2), y (B, 4), omega (B,) float64;
    x_blind_idx and row int64; ceil(n/B) batches, all of size B except a last one of n mod B
    (no drop); every view row exactly once; phys = [ln Q_E, ln S] and y = [ln W, ln dbar, t_obs,
    ln A] of the frame row (unscaled SI logs, docs/2 Props 1.2, 3.5); omega = 1/|J_tr| (2.6).
    """
    frame, rows = site_frame(), view_rows()
    stats = fitted(frame, rows)
    ds = SiteDataset(frame, rows, FEATURES, stats, n_train=N_ROWS)
    batches = list(make_loader(ds, b, train=train, seed=11))
    sizes = [int(bt["row"].shape[0]) for bt in batches]
    want = [b] * (N_ROWS // b) + ([N_ROWS % b] if N_ROWS % b else [])
    assert sizes == want, f"B={b}: batch sizes {sizes[-3:]} (last 3) != {want[-3:]}"
    assert len(batches) == math.ceil(N_ROWS / b)
    blind = [j for j, name in enumerate(FEATURES) if name not in BLIND_OUT]
    raw_x = frame[list(FEATURES)].to_numpy(dtype=np.float64)
    seen: list[int] = []
    for k, bt in enumerate(batches):
        assert set(bt) == BATCH_KEYS, f"batch {k} keys {sorted(bt)} != {sorted(BATCH_KEYS)}"
        n = sizes[k]
        shapes = {key: tuple(bt[key].shape) for key in ("x", "phys", "y", "omega", "row")}
        assert shapes == {
            "x": (n, len(FEATURES)), "phys": (n, 2), "y": (n, 4), "omega": (n,), "row": (n,)
        }, f"B={b} batch {k}: shapes {shapes}"  # fmt: skip
        for key in ("x", "phys", "y", "omega"):
            assert bt[key].dtype == torch.float64, f"{key} dtype {bt[key].dtype}"
            assert bt[key].device.type == "cpu", f"{key} device {bt[key].device}"
        assert bt["row"].dtype == torch.int64, f"row dtype {bt['row'].dtype}"
        assert bt["x_blind_idx"].dtype == torch.int64, f"x_blind_idx {bt['x_blind_idx'].dtype}"
        assert bt["x_blind_idx"].tolist() == blind, f"x_blind_idx {bt['x_blind_idx'].tolist()}"
        r = bt["row"].numpy()
        sub = frame.iloc[r]
        np.testing.assert_array_equal(bt["x"].numpy(), stats.transform(raw_x[r]))
        phys = np.column_stack([np.log(sub["Q_E"]), np.log(sub["S"])])
        y = np.column_stack([np.log(sub["W"]), np.log(sub["dbar"]), sub["t_obs"], np.log(sub["A"])])
        np.testing.assert_array_equal(bt["phys"].numpy(), phys)
        np.testing.assert_array_equal(bt["y"].numpy(), y)
        np.testing.assert_array_equal(bt["omega"].numpy(), np.full(n, 1.0 / N_ROWS))
        seen += r.tolist()
    assert sorted(seen) == rows.tolist(), "an epoch must visit every view row exactly once"
    if not train:
        assert seen == rows.tolist(), "the evaluation loader must be sequential"


def test_seeded_shuffling_is_reproducible() -> None:
    """docs/3 4.2 (seeded runs, ADL-016): the same seed gives the same order over several epochs;
    another seed gives another order; each epoch is a permutation of the view; the evaluation
    loader is sequential whatever the seed.
    """
    frame, rows = site_frame(), view_rows()
    ds = SiteDataset(frame, rows, FEATURES, fitted(frame, rows), n_train=N_ROWS)

    def epochs(seed: int, train: bool = True) -> list[list[int]]:
        loader = make_loader(ds, 64, train=train, seed=seed)
        return [epoch_rows(loader) for _ in range(3)]

    first, again, other = epochs(5), epochs(5), epochs(6)
    assert first == again, "same seed, different batch order"
    assert first != other, "seeds 5 and 6 gave the same order"
    for e in first:
        assert sorted(e) == rows.tolist(), "an epoch is not a permutation of the view"
        assert e != rows.tolist(), "a training epoch kept the sequential order"
    assert epochs(0, train=False) == epochs(99, train=False) == [rows.tolist()] * 3


# ----------------------------------------------------------------------------- missing values
NON_FINITE = [
    ("DA_km2", math.nan, "Q_E", "x"),
    ("DA_km2", math.inf, "Q_E", "x"),
    ("DA_km2", -math.inf, "Q_E", "x"),
    ("Q_E", math.nan, "Q_E", "x"),
    ("S", math.inf, "Q_E", "x"),
    ("Q_C", math.nan, "Q_C", "phys"),
    ("Q_C", math.inf, "Q_C", "phys"),
    ("W", math.nan, "Q_E", "y"),
    ("t_obs", -math.inf, "Q_E", "y"),
    ("A", math.inf, "Q_E", "y"),
]


@pytest.mark.parametrize(("column", "value", "discharge", "tensor"), NON_FINITE)
def test_non_finite_values_raise_and_are_never_imputed(
    column: str, value: float, discharge: str, tensor: str
) -> None:
    """docs/3 2.7 (imputation: none, in any model): a non-finite feature, physics input (the
    SA1 discharge Q_C included) or target inside the view raises; the same missing value
    outside the view is never read (the view builds and its tensors are finite).
    """
    frame, rows = site_frame(), view_rows()
    stats = fitted(frame, rows)
    frame.loc[rows[3], column] = value
    with pytest.raises(ValueError, match=rf"non-finite values in {tensor} \(no imputation"):
        SiteDataset(frame, rows, FEATURES, stats, n_train=N_ROWS, discharge=discharge)
    others = np.delete(rows, 3)
    ds = SiteDataset(frame, others, FEATURES, stats, n_train=N_ROWS, discharge=discharge)
    for name in ("x", "phys", "y"):
        assert bool(torch.isfinite(getattr(ds, name)).all()), f"{name} not finite"


def test_a_j_is_masked_never_imputed() -> None:
    """docs/3 4.1 (A_J = Q_E/V_E, training-free) and 2.7: rows with a missing V_E are unavailable
    and stay NaN (no imputation); available rows equal ln(Q_E/V_E) to 1e-13 absolute; a
    non-positive or infinite V_E, or a non-positive Q_E, raises.
    """
    rng = np.random.default_rng(4)
    q, v = np.exp(rng.normal(2.0, 2.0, 50)), rng.uniform(0.05, 3.0, 50)
    v[[0, 7, 31]] = np.nan
    out, available = aj_log_area(q, v)
    np.testing.assert_array_equal(available, ~np.isnan(v))
    assert bool(np.isnan(out[~available]).all()), "missing V_E rows must stay NaN"
    err = float(np.abs(out[available] - np.log(q[available] / v[available])).max())
    assert err <= 1e-13, f"ln A_J vs ln(Q_E/V_E): {err:.3e} > 1e-13 (docs/3 4.1)"
    for bad_q, bad_v, match in (
        (q, np.where(np.arange(50) == 5, 0.0, v), "V_E must be > 0"),
        (q, np.where(np.arange(50) == 5, np.inf, v), "V_E must be > 0"),
        (np.where(np.arange(50) == 5, -1.0, q), v, "Q_E must be finite and > 0"),
    ):
        with pytest.raises(ValueError, match=match):
            aj_log_area(bad_q, bad_v)


def test_site_table_missing_value_rules() -> None:
    """docs/3 2.7, R12/R14 and the M1-only rule (2.3): a missing slope (R12) or core predictor
    (R14) excludes the site; a missing M1-only predictor keeps the site with F_M1_missing = True
    and the value still missing; a missing V_E keeps the site with A_J missing (A_J = Q_E/V_E
    elsewhere); a banned predictor (V_E, L4) in either list raises. No value is imputed.
    """
    sites = [f"{i:08d}" for i in range(5)]
    core = pd.DataFrame(
        {
            "site_no": sites,
            "variant": "primary",
            "comid": [11, 12, 13, 14, 15],
            "division": "D1",
            "Q_E": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    per_site = pd.DataFrame(
        {
            "comid": [11, 12, 13, 14, 15],
            "DA_km2": [10.0, 20.0, 30.0, 40.0, np.nan],  # site 4: R14
            "S": [1e-3, 2e-3, 3e-3, np.nan, 5e-3],  # site 3: R12
            "F_S": False,
            "V_E": [0.5, np.nan, 0.6, 0.7, 0.8],  # site 1: A_J missing
            "CAT_POPDENS10": [1.0, 2.0, np.nan, 4.0, 5.0],  # site 2: M1-only missing
        }
    )
    table, counts = build_site_table(
        core, per_site, ("DA_km2", "S", "F_S", "Q_E"), ("CAT_POPDENS10",)
    )
    assert table["site_no"].tolist() == sites[:3], f"kept {table['site_no'].tolist()}"
    assert (counts["R12_slope_missing"], counts["R14_core_missing"]) == (1, 1), counts
    assert table["F_M1_missing"].tolist() == [False, False, True]
    assert counts["F_M1_missing_sites_out"] == 1, counts
    assert counts["A_J_missing_rows"] == 1, counts
    assert math.isnan(table.loc[2, "CAT_POPDENS10"]), "M1-only value was imputed"
    assert table.loc[0, "A_J"] == 1.0 / 0.5
    assert math.isnan(table.loc[1, "A_J"]), "A_J imputed for a missing V_E"
    assert table.loc[2, "A_J"] == 3.0 / 0.6
    for core_list, m1_list in ((("DA_km2", "V_E"), ()), (("DA_km2",), ("V_E",))):
        with pytest.raises(ValueError, match=r"banned predictors \(L4\)"):
            build_site_table(core, per_site, core_list, m1_list)


def test_m1_flag_cannot_remove_an_nn_row() -> None:
    """docs/3 2.3 (F_M1_missing concerns M1 only): no M1-only predictor and no flag column is an
    NN feature, so the flag can never exclude a row from M2, M4, M5 or M6.
    """
    feats = set(FEATURES)
    for model in ("M2-MLP", "M3", "M4", "M5", "M6-NN"):
        feats |= set(ModelSpec(model).features(PROTOCOL))
    assert not feats & {*M1_PREDICTORS, "F_M1_missing"}, sorted(feats & set(M1_PREDICTORS))


# ----------------------------------------------------------------------------- scaler
def _provenance(part: str) -> dict[str, str]:
    index = None if part in ("train", "test") else 1
    return fv.fit_provenance(
        "primary", "S-ID", 0, 0, part, index, site_table_sha256=SHA_SITE, fold_table_sha256=SHA_FOLD
    )


@pytest.mark.parametrize("part", fv.PARTS)
def test_normstats_fit_parts_only(part: str, tmp_path: Path) -> None:
    """docs/3 3.4 item 1, 10.10 and L5: a scaler may carry the provenance of a training part
    only (train, inner_train, es_train); any other part (test, inner_val, es_val) raises; an
    artefact without provenance cannot be saved; a saved one round-trips exactly.
    """
    assert set(FIT_PARTS) == set(TRAINING_PARTS), f"FIT_PARTS {FIT_PARTS}"
    assert set(FIT_PARTS) < set(fv.PARTS), f"FIT_PARTS {FIT_PARTS} vs PARTS {fv.PARTS}"
    frame, rows = site_frame(), view_rows()
    x = frame.iloc[rows][list(FEATURES)].to_numpy(dtype=np.float64)
    frac = PROTOCOL.transforms.log_offset_frac
    if part not in TRAINING_PARTS:
        with pytest.raises(ValueError, match="fitted on"):
            NormStats.fit(x, FEATURES, KINDS, frac, provenance=_provenance(part))
        return
    stats = NormStats.fit(x, FEATURES, KINDS, frac, provenance=_provenance(part))
    stats.save(tmp_path / "normstats.json")
    assert NormStats.load(tmp_path / "normstats.json") == stats
    with pytest.raises(ValueError, match="needs the fit provenance"):
        fitted(frame, rows).save(tmp_path / "bare.json")


def fold_block(sites: pd.Series) -> pd.DataFrame:
    """One S-ID block (docs/3 10.8 columns): every 5th site is test; inner fold = position mod 3;
    member m's ES split = training sites with position mod 7 == m mod 7.
    """
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


@pytest.mark.parametrize(
    ("kind", "index", "grid"), [("inner", 0, 0), ("member", 1, None)], ids=["inner", "member"]
)
def test_runner_fits_on_a_training_part_and_blanks_the_test_fold(
    kind: FitKind, index: int, grid: int | None
) -> None:
    """docs/3 3.4 items 1 and 4, L5, L6 (P6-D1): the runner's fitting part is a training part;
    it is disjoint from the outer test fold and inside the outer training set; every feature,
    physics input and target of a test row is NaN before any scaler or dataset is built, so a
    scaler over all rows or a dataset over the test rows raises.
    """
    table = site_frame()
    tables = Tables(table, fold_block(table["site_no"]), SHA_SITE, SHA_FOLD)
    req = FitRequest(ModelSpec("M5"), "primary", "S-ID", 0, 0, kind, index, grid)
    rows = _resolve_rows(tables, req, FEATURES)
    frame = fv.analysis_frame(table, "primary")
    test = fv.get_view(frame, tables.folds, "S-ID", 0, 0, "test")
    outer = fv.get_view(frame, tables.folds, "S-ID", 0, 0, "train")
    assert rows.train_part in TRAINING_PARTS, f"fitting part {rows.train_part!r}"
    assert not np.intersect1d(rows.train, test).size, "fitting part overlaps the test fold"
    assert not np.setdiff1d(rows.train, outer).size, "fitting part leaves the outer training set"
    blanked = [*FEATURES, "Q_C", "W", "dbar", "t_obs", "A"]
    assert bool(rows.frame.iloc[test][blanked].isna().all().all()), "test rows still readable"
    assert not rows.frame.iloc[rows.train][blanked].isna().any().any()
    all_x = rows.frame[list(FEATURES)].to_numpy(dtype=np.float64)
    with pytest.raises(ValueError, match="non-finite"):
        NormStats.fit(all_x, FEATURES, KINDS, PROTOCOL.transforms.log_offset_frac)
    stats = fitted(rows.frame, rows.train)
    with pytest.raises(ValueError, match="non-finite values in x"):
        SiteDataset(rows.frame, test, FEATURES, stats, n_train=len(rows.train))

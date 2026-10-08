"""Unit tests for src/data/folds.py, normstats.py and dataset.py (Phase 5, Batch E). Synthetic."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import load_settings
from src.data import folds as fv
from src.data.dataset import BLIND_EXCLUDED, SiteDataset, make_loader
from src.data.features import CORE_NHD_PREDICTORS
from src.data.normstats import NormStats
from src.data.validate_raw import streamcat_fields

pytestmark = pytest.mark.unit

SETTINGS = load_settings()
FEATURES = ("DA_km2", "z_min_m", "S", "F_S", "Q_E", "damdensws")
KINDS = {
    "DA_km2": "log",
    "z_min_m": "identity",
    "S": "log",
    "F_S": "passthrough",
    "Q_E": "log",
    "damdensws": "log_offset",
}


def frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "site_no": [f"{i:08d}" for i in range(n)],
            "variant": "primary",
            "F_DA": [i % 10 == 9 for i in range(n)],
            "F_Qsel": False,
            "F_S": [i % 4 == 0 for i in range(n)],
            "DA_km2": np.exp(rng.normal(5, 1, n)),
            "z_min_m": rng.normal(200, 50, n),
            "S": np.exp(rng.normal(-7, 1, n)),
            "Q_E": np.exp(rng.normal(2, 1, n)),
            "Q_C": np.exp(rng.normal(2, 1, n)),
            "damdensws": np.where(np.arange(n) % 3 == 0, 0.0, rng.uniform(0.01, 0.4, n)),
            "W": np.exp(rng.normal(3, 0.5, n)),
            "dbar": np.exp(rng.normal(0, 0.5, n)),
            "t_obs": rng.uniform(0.1, 1.0, n),
            "A": np.exp(rng.normal(3, 0.5, n)),
        }
    )


def fold_table(sites: pd.Series) -> pd.DataFrame:
    """One S-ID block: every 5th site is test; inner fold = position mod 3; ES on member 1."""
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


# ----------------------------------------------------------------------------- folds
def test_analysis_frames_filter_flags() -> None:
    table = frame()
    assert len(fv.analysis_frame(table, "primary")) == 36  # F_DA dropped
    assert len(fv.analysis_frame(table, "SA6")) == 40
    sa5 = fv.analysis_frame(table, "SA5")
    assert not sa5["F_S"].any()
    assert fv.ANALYSES["SA1"].discharge == "Q_C"


def test_views_partition_the_block() -> None:
    table = frame()
    fr = fv.analysis_frame(table, "SA6")
    ft = fold_table(table["site_no"])
    train = fv.get_view(fr, ft, "S-ID", 0, 0, "train")
    test = fv.get_view(fr, ft, "S-ID", 0, 0, "test")
    assert len(train) + len(test) == len(fr)
    assert not set(train) & set(test)
    for k in range(3):
        itr = fv.get_view(fr, ft, "S-ID", 0, 0, "inner_train", k)
        iva = fv.get_view(fr, ft, "S-ID", 0, 0, "inner_val", k)
        assert sorted([*itr, *iva]) == sorted(train)
    etr = fv.get_view(fr, ft, "S-ID", 0, 0, "es_train", 3)
    eva = fv.get_view(fr, ft, "S-ID", 0, 0, "es_val", 3)
    assert sorted([*etr, *eva]) == sorted(train)
    assert len(eva) > 0
    with pytest.raises(ValueError, match="needs an index"):
        fv.get_view(fr, ft, "S-ID", 0, 0, "inner_val")
    with pytest.raises(KeyError):
        fv.get_view(fr, ft, "S-LRO", 0, 0, "train")


# ----------------------------------------------------------------------------- normstats
def fitted() -> tuple[pd.DataFrame, np.ndarray, NormStats]:
    fr = fv.analysis_frame(frame(), "SA6")
    train = fv.get_view(fr, fold_table(fr["site_no"]), "S-ID", 0, 0, "train")
    x = fr.loc[train, list(FEATURES)].to_numpy(dtype=np.float64)
    return fr, train, NormStats.fit(x, FEATURES, KINDS, 0.5)


def test_test_rows_cannot_change_normstats() -> None:
    fr, train, stats = fitted()
    test = np.setdiff1d(np.arange(len(fr)), train)
    perturbed = fr.copy()
    perturbed.loc[test, "DA_km2"] *= 1000.0
    perturbed.loc[test, "z_min_m"] = -9.0
    x = perturbed.loc[train, list(FEATURES)].to_numpy(dtype=np.float64)
    assert NormStats.fit(x, FEATURES, KINDS, 0.5) == stats


def test_round_trip_and_standardization() -> None:
    fr, train, stats = fitted()
    x = fr[list(FEATURES)].to_numpy(dtype=np.float64)
    z = stats.transform(x)
    assert z.dtype == np.float64
    np.testing.assert_allclose(stats.inverse_transform(z), x, rtol=1e-12, atol=1e-12)
    zt = stats.transform(x[train])
    scaled = [j for j, n in enumerate(FEATURES) if KINDS[n] != "passthrough"]
    np.testing.assert_allclose(zt[:, scaled].mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(zt[:, scaled].std(axis=0), 1.0, atol=1e-12)
    j = FEATURES.index("F_S")
    np.testing.assert_array_equal(z[:, j], x[:, j])
    c = stats.offsets[FEATURES.index("damdensws")]
    xd = x[train, FEATURES.index("damdensws")]
    assert c == 0.5 * xd[xd > 0].min()


def test_json_round_trip_is_exact(tmp_path: Path) -> None:
    _, _, stats = fitted()
    path = tmp_path / "normstats.json"
    stats.save(path)
    assert NormStats.load(path) == stats
    assert path.read_text(encoding="utf-8").endswith("\n")


@pytest.mark.parametrize(
    ("column", "kind", "message"),
    [
        (np.full(5, 2.0), "identity", "zero variance"),
        (np.array([1.0, 0.0, 2.0, 3.0, 4.0]), "log", "strictly positive"),
        (np.array([1.0, -1.0, 2.0, 3.0, 4.0]), "log_offset", "non-negative"),
        (np.array([1.0, np.nan, 2.0, 3.0, 4.0]), "identity", "non-finite"),
    ],
)
def test_normstats_rejects_bad_training_columns(
    column: np.ndarray, kind: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        NormStats.fit(column[:, None], ["f"], {"f": kind}, 0.5)


# ----------------------------------------------------------------------------- dataset
def test_dataset_batch_contract() -> None:
    fr, train, stats = fitted()
    ds = SiteDataset(fr, train, FEATURES, stats, n_train=len(train))
    batch = next(iter(make_loader(ds, 8, train=True, seed=1)))
    assert batch["x"].shape == (8, len(FEATURES))
    assert batch["phys"].shape == (8, 2)
    assert batch["y"].shape == (8, 4)
    for key in ("x", "phys", "y", "omega"):
        assert batch[key].dtype == torch.float64
    assert batch["row"].dtype == torch.int64
    blind = [FEATURES[j] for j in batch["x_blind_idx"].tolist()]
    assert blind == [f for f in FEATURES if f not in BLIND_EXCLUDED]
    assert torch.allclose(batch["omega"], torch.full((8,), 1.0 / len(train), dtype=torch.float64))
    r = batch["row"].numpy()
    np.testing.assert_allclose(batch["phys"][:, 0].numpy(), np.log(fr.loc[r, "Q_E"]))
    np.testing.assert_allclose(batch["y"][:, 2].numpy(), fr.loc[r, "t_obs"])


def test_loader_order_is_seeded_and_test_loader_is_sequential() -> None:
    fr, train, stats = fitted()
    ds = SiteDataset(fr, train, FEATURES, stats, n_train=len(train))

    def rows(loader: torch.utils.data.DataLoader) -> list[int]:
        return [int(v) for b in loader for v in b["row"]]

    assert rows(make_loader(ds, 5, train=True, seed=7)) == rows(
        make_loader(ds, 5, train=True, seed=7)
    )
    assert rows(make_loader(ds, 5, train=True, seed=7)) != rows(
        make_loader(ds, 5, train=True, seed=8)
    )
    assert rows(make_loader(ds, 5, train=False, seed=0)) == train.tolist()
    sa1 = SiteDataset(fr, train, FEATURES, stats, n_train=len(train), discharge="Q_C")
    assert torch.allclose(sa1.phys[:, 0], torch.from_numpy(np.log(fr.loc[train, "Q_C"].to_numpy())))
    with pytest.raises(ValueError, match="feature order"):
        SiteDataset(fr, train, FEATURES[::-1], stats, n_train=len(train))


# ----------------------------------------------------------------------------- contracts
def test_folds_and_normstats_do_not_import_torch() -> None:
    code = "import sys, src.data.folds, src.data.normstats; print('torch' in sys.modules)"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    assert out.stdout.strip() == "False"


def test_transform_manifest_covers_the_nn_feature_set() -> None:
    feat = SETTINGS.protocol.features
    classes = {c for g in feat.landcover_groups.values() for c in g}
    streamcat = [c for c in streamcat_fields(SETTINGS) if c not in classes]
    expected = {*CORE_NHD_PREDICTORS, *streamcat, *feat.landcover_groups, "D50_m", "AI"}
    assert set(SETTINGS.protocol.transforms.features) == expected

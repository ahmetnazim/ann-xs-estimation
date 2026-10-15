"""Tests for Events 1-2, PI checkpoints and the event runtime (docs/3 13.4; C3b). Synthetic only.

Sources: docs/3 section 13.4 (event order; clean tree; sealed inputs; a recorded event is never
re-run; checkpoints carry PI text and no number; Event 1 content; Event 2 = M* per k in {dbar, A}
on S-ID OOF, mean over repeats, M1's A = W-hat * dbar-hat, exact tie raises), protocol 3.6.1
(residuals averaged over the repeats per site), 3.6.5 (nested pairs, psi >= 0.5), 3.6.7 (D, N),
4.3.1 (RMSE_log) and 4.4 (M*); C3b decisions a-m (review 8).

Expected values are written out by hand or computed directly from the definitions in each test
(never by the function under test): residual means, nested classes of a chain network and
per-repeat RMSE_log values. No real table, prediction or target is read.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import load_settings
from src.data.io import sha256_file
from src.data.network import NetworkInputs
from src.evaluation import dependence_audit as da
from src.evaluation import event1, event2, event3, events
from src.evaluation.events import EventRefusedError
from src.evaluation.ledger import Ledger
from synthetic_sites import site_frame

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = load_settings().protocol
#: Key columns of a primary S-ID row as ``src.models.predict._base`` writes them (``spec`` is set to
#: the model name per row; no ablation is ``""``). Pinned by the F-C3b-1 contract test below.
BLOCK = {"spec": None, "ablation": "", "analysis": "primary", "family": "S-ID", "unit": "site"}


# ----------------------------------------------------------------------------- synthetic tables
def sites(n: int = 60, seed: int = 3, span_m: float = 3.0e5) -> pd.DataFrame:
    """Primary site table with COMIDs and EPSG:5070 coordinates inside a ``span_m`` square."""
    rng = np.random.default_rng(seed)
    df = site_frame(n, seed)
    df["comid"] = 1000 + np.arange(n)
    df["x5070"] = rng.uniform(0.0, span_m, n)
    df["y5070"] = rng.uniform(0.0, span_m, n)
    return df


def folds(site_nos: pd.Series, repeats: int = 3, k: int = 5) -> pd.DataFrame:
    """S-ID fold table: site ``i`` is the test site of fold ``(i + r) % k`` in repeat ``r``."""
    rows = []
    for r in range(repeats):
        for f in range(k):
            for i, s in enumerate(site_nos):
                rows.append((s, r, f, (i + r) % k == f))
    out = pd.DataFrame(rows, columns=["site_no", "repeat", "outer_fold", "is_test"])
    out.insert(0, "family", "S-ID")
    out["repeat"] = out["repeat"].astype("int8")
    out["outer_fold"] = out["outer_fold"].astype("int8")
    out["inner_fold"] = np.where(out["is_test"], -1, 0).astype("int8")
    return out


def oof_preds(
    frame: pd.DataFrame,
    fold_table: pd.DataFrame,
    model: str,
    delta: Callable[[int, int, str], float],
) -> pd.DataFrame:
    """OOF rows of one model: ``mu_k = ln o_k + delta(site index, repeat, k)`` (t: + t_obs)."""
    pos = {s: i for i, s in enumerate(frame["site_no"])}
    test = fold_table.loc[fold_table["is_test"]]
    rows = []
    for s, r, f in test[["site_no", "repeat", "outer_fold"]].itertuples(index=False):
        i = pos[s]
        row = {**BLOCK, "model": model, "spec": model, "repeat": int(r), "outer_fold": int(f)}
        row.update(site_no=s, comid=int(frame["comid"].iloc[i]))
        for k, col in event1.RESIDUAL_TARGETS.items():
            obs = float(frame[col].iloc[i])
            base = obs if k == "t" else math.log(obs)
            row[f"mu_{k}"] = base + delta(i, int(r), k)
        rows.append(row)
    return pd.DataFrame(rows)


def chain_network(frame: pd.DataFrame) -> tuple[NetworkInputs, dict[int, float]]:
    """Pairs (2j -> 2j+1) on one terminal path; psi = 0.8 for even j, 0.2 for odd j."""
    comids = frame["comid"].astype(int).tolist()
    edges, area, psi = [], {}, {}
    for j in range(len(comids) // 2):
        a, b = comids[2 * j], comids[2 * j + 1]
        p = 0.8 if j % 2 == 0 else 0.2
        edges.append((a, b))
        area[a], area[b] = 100.0, 100.0 / p
        psi[a] = psi[b] = p
    if len(comids) % 2:
        area[comids[-1]] = 50.0
    net = NetworkInputs(
        edges=np.asarray(edges, dtype=np.int64),
        terminal_path=dict.fromkeys(comids, 7),
        drainage_km2=area,
        missing=(),
    )
    return net, psi


# ----------------------------------------------------------------------------- Event 1 library
def test_oof_residuals_are_the_mean_over_repeats() -> None:
    """Protocol 3.6.1 (b) / C3b-b: e_site = mean_r (mu - ln o); misplaced rows raise."""
    frame = event1.audit_frame(sites(10))
    ft = folds(frame["site_no"], repeats=2, k=5)

    def delta(i: int, r: int, k: str) -> float:
        return 0.1 * i + (0.5 if r == 1 else -0.3) + (1.0 if k == "A" else 0.0)

    preds = pd.concat(
        [oof_preds(frame, ft, "M-null", delta), oof_preds(frame, ft, "M2-XGB", delta)]
    )
    got = event1.oof_residuals(preds, frame, ft)
    for i in range(10):
        want = 0.1 * i + (0.5 - 0.3) / 2  # hand: mean over repeats of the two offsets
        assert got["M2-XGB:dbar"].iloc[i] == pytest.approx(want, abs=1e-12)
        assert got["M-null:A"].iloc[i] == pytest.approx(want + 1.0, abs=1e-12)
    with pytest.raises(ValueError, match="missing S-ID OOF"):
        event1.oof_residuals(preds.iloc[1:], frame, ft)
    with pytest.raises(ValueError, match="duplicated"):
        event1.oof_residuals(pd.concat([preds, preds.iloc[:1]]), frame, ft)
    moved = preds.copy()
    moved.loc[moved.index[0], "outer_fold"] = (int(moved["outer_fold"].iloc[0]) + 1) % 5
    with pytest.raises(ValueError, match="outer-test block"):
        event1.oof_residuals(moved, frame, ft)
    extra = preds.iloc[:1].copy()
    extra["site_no"] = "99999999"
    with pytest.raises(ValueError, match="outside the audited frame"):
        event1.oof_residuals(pd.concat([preds, extra]), frame, ft)
    nan = preds.copy()
    nan.loc[nan.index[3], "mu_t"] = np.nan
    with pytest.raises(ValueError, match="non-finite t"):
        event1.oof_residuals(nan, frame, ft)
    with pytest.raises(ValueError, match="once per S-ID repeat"):
        event1.oof_residuals(preds, frame, ft.loc[~(ft["is_test"] & ft["site_no"].eq("00000003"))])
    other = preds.copy()
    other["analysis"] = "SA2"  # only primary rows count
    with pytest.raises(ValueError, match="missing"):
        event1.oof_residuals(other, frame, ft)


def _event1(frame_raw: pd.DataFrame, field: Callable[[np.ndarray, np.ndarray], np.ndarray]):
    frame = event1.audit_frame(frame_raw)
    ft = folds(frame["site_no"])
    x, y = frame["x_km"].to_numpy(), frame["y_km"].to_numpy()
    f = field(x, y)
    noise = np.random.default_rng(5).normal(0.0, 0.05, (len(frame), 3))

    def delta(i: int, r: int, k: str) -> float:
        return float(f[i] + noise[i, r])

    preds = pd.concat(
        [
            oof_preds(frame, ft, "M-null", lambda i, r, k: 0.3 * delta(i, r, k)),
            oof_preds(frame, ft, "M2-XGB", delta),
        ]
    )
    net, psi = chain_network(frame)
    return frame, ft, net, psi, event1.run_event1(frame_raw, ft, preds, net, PROTOCOL)


def test_event1_planted_dependence_nesting_and_strata() -> None:
    """Protocol 3.6.2-3.6.7: a smooth residual field sets D; chain nesting gives N by hand."""
    raw = sites(80)
    frame, ft, _net, psi, res = _event1(raw, lambda x, y: np.sin(x / 60.0) + np.cos(y / 60.0))
    doc, strata = res.document, res.strata
    assert doc["n_sites"] == 80
    assert doc["n_blocks"] == 15
    assert doc["repeats"] == [0, 1, 2]
    assert len(strata) == 80 * 3  # one row per (repeat, test site)
    assert set(doc["series"]) == {
        "lnW",
        "lndbar",
        "t",
        *(f"{m}:{k}" for m in ("M-null", "M2-XGB") for k in event1.RESIDUAL_TARGETS),
    }
    lab = doc["label"]
    m1 = doc["series"]["M2-XGB:dbar"]["moran_W1"]
    assert m1["I"] >= PROTOCOL.audit.moran_material
    assert m1["pseudo_p"] < PROTOCOL.audit_rules.moran_pseudo_p
    assert lab["D_moran"] is True
    assert lab["D"] is True
    assert lab["optimistic"] is True
    assert doc["rho_star_km"] == min(doc["rho_hat_km"], PROTOCOL.audit.rho_cap_km)
    assert doc["rho_hat_km"] == doc["series"]["M2-XGB:dbar"]["variogram"]["rho_hat_km"]
    # nested classes by hand: a test site's partner is strong (psi 0.8) or weak (psi 0.2) iff
    # the partner is a training site of the same block.
    pos = {s: i for i, s in enumerate(frame["site_no"])}
    comid = frame["comid"].to_numpy()
    want = []
    for r, f, s in strata[["repeat", "outer_fold", "site_no"]].itertuples(index=False):
        i = pos[s]
        j = i + 1 if i % 2 == 0 else i - 1
        partner_test = (j + int(r)) % 5 == int(f)
        p = psi[int(comid[i])]
        want.append("none" if partner_test else ("strong" if p >= 0.5 else "weak"))
    assert strata["nested"].tolist() == want
    share = float(np.mean(np.asarray(want) == "strong"))
    assert doc["nested"]["share_strong"] == pytest.approx(share, abs=1e-15)
    assert lab["N"] is (share >= PROTOCOL.audit.nested_share_material)
    # far stratum and d~_NN by brute force
    xy = frame[["x_km", "y_km"]].to_numpy()
    nn = []
    for r, f, s in strata[["repeat", "outer_fold", "site_no"]].itertuples(index=False):
        block = ft.loc[ft["repeat"].eq(r) & ft["outer_fold"].eq(f)].set_index("site_no")
        train = [pos[t] for t in frame["site_no"] if not block.loc[t, "is_test"]]
        nn.append(float(np.min(np.linalg.norm(xy[train] - xy[pos[s]], axis=1))))
    assert np.allclose(strata["nn_km"].to_numpy(), nn, rtol=0, atol=1e-9)
    assert doc["d_nn_median_km"] == pytest.approx(float(np.median(nn)), abs=1e-9)
    assert strata["far"].tolist() == [d > doc["rho_star_km"] for d in nn]
    assert lab["D_range"] is (doc["rho_hat_km"] > doc["d_nn_median_km"])
    lisa = doc["series"]["M2-XGB:dbar"]["lisa_significant_by_division"]
    assert sorted(lisa) == ["A", "B"]
    assert sum(sum(v.values()) for v in lisa.values()) == 80
    assert len(doc["per_block"]) == 15


def test_event1_iid_residuals_do_not_trigger_the_moran_part_of_d() -> None:
    """Protocol 3.6.7: iid residuals (no spatial structure) leave the Moran part of D false."""
    rng = np.random.default_rng(17)
    iid = rng.normal(0.0, 1.0, 80)
    _, _, _, _, res = _event1(sites(80), lambda x, y: iid)
    lab = res.document["label"]
    assert lab["D_moran"] is False
    assert lab["D"] is (lab["D_moran"] or lab["D_range"])


def test_event1_refuses_missing_vaa_and_label_series() -> None:
    """docs/3 13.9: every site COMID needs VAA values; D needs the M2-XGB d-bar residuals."""
    raw = sites(30)
    frame = event1.audit_frame(raw)
    ft = folds(frame["site_no"])
    net, _ = chain_network(frame)
    preds = oof_preds(frame, ft, "M-null", lambda i, r, k: 0.0)
    gone = NetworkInputs(net.edges, net.terminal_path, net.drainage_km2, missing=(5,))
    with pytest.raises(ValueError, match="not in the NHDPlus VAA"):
        event1.run_event1(raw, ft, preds, gone, PROTOCOL)
    with pytest.raises(ValueError, match="M2-XGB: missing"):
        event1.run_event1(raw, ft, preds, net, PROTOCOL)
    bad = raw.copy()
    bad.loc[0, "x5070"] = np.nan
    with pytest.raises(ValueError, match="finite EPSG:5070"):
        event1.audit_frame(bad)
    with pytest.raises(ValueError, match="lacks"):
        event1.audit_frame(raw.drop(columns=["x5070"]))
    with pytest.raises(ValueError, match="no S-ID block"):
        event1.sid_blocks(ft.assign(family="S-LRO"))
    neg = raw.copy()
    neg.loc[0, "W"] = -1.0
    with pytest.raises(ValueError, match="positive"):
        event1.run_event1(neg, ft, preds, net, PROTOCOL)


def test_series_stats_uses_the_phase7_statistics() -> None:
    """3.6.2/3.6.4: the series block equals the library calls (seed = seeds.global, C3b-d)."""
    frame = event1.audit_frame(sites(40))
    xy = frame[["x_km", "y_km"]].to_numpy()
    z = np.sin(xy[:, 0] / 50.0)
    w1 = da.knn_w(xy, PROTOCOL.audit.knn_k)
    w2, _ = da.idw_weights(xy, PROTOCOL.audit.idw_band_km)
    divs = frame["division"].astype(str).to_numpy(dtype=np.str_)
    doc, sel, m1 = event1.series_stats(z, xy, w1, w2, divs, PROTOCOL)
    seed = PROTOCOL.seeds.global_seed
    ref = da.moran_global(z, w1.sparse, PROTOCOL.audit.permutations, seed)
    assert m1 == ref
    assert doc["moran_W1"]["I"] == ref.i
    vg = da.empirical_variogram(z, xy, PROTOCOL.audit)
    ref_sel = da.select_variogram(da.fit_variogram(vg, PROTOCOL.audit_rules), PROTOCOL.audit_rules)
    assert sel == ref_sel
    assert len(doc["correlogram"]) == len(PROTOCOL.audit.dist_classes_km)


# ----------------------------------------------------------------------------- Event 2 library
def _event2_frames(frame: pd.DataFrame, ft: pd.DataFrame, offsets: dict[str, float]):
    out = {}
    for m, c in offsets.items():
        p = oof_preds(frame, ft, m, lambda i, r, k, c=c: c * (1 if (i + r) % 2 else -1))
        out[m] = p
    out["M1"] = out["M1"].assign(mu_A=99.0)  # ignored: M1's ln A = mu_W + mu_dbar (C3b-m)
    return out


def test_event2_mstar_by_hand_and_m1_area() -> None:
    """Protocol 4.3.1/4.4: |e| = c everywhere -> RMSE_log = c for dbar; M1's A uses W + dbar."""
    raw = sites(20)
    frame = event1.audit_frame(raw)
    ft = folds(frame["site_no"])
    offsets = {"M1": 0.30, "M2-MLP": 0.20, "M2-XGB": 0.25}
    doc = event2.run_event2(_event2_frames(frame, ft, offsets), raw, ft, PROTOCOL)
    for m, c in offsets.items():
        assert doc["rmse_log_mean"][m]["dbar"] == pytest.approx(c, abs=1e-12)
        for r in ("0", "1", "2"):
            assert doc["rmse_log_by_repeat"][m]["dbar"][r] == pytest.approx(c, abs=1e-12)
    # M1: e_A = e_W + e_dbar = 2c * (+/-1) -> 0.60; M2: e_A = c -> 0.20, 0.25 (hand)
    assert doc["rmse_log_mean"]["M1"]["A"] == pytest.approx(0.60, abs=1e-12)
    assert doc["rmse_log_mean"]["M2-MLP"]["A"] == pytest.approx(0.20, abs=1e-12)
    assert doc["mstar"] == {"dbar": "M2-MLP", "A": "M2-MLP"}
    tie = event2.run_event2
    with pytest.raises(ValueError, match="exact tie"):
        tie(_event2_frames(frame, ft, {"M1": 0.2, "M2-MLP": 0.2, "M2-XGB": 0.3}), raw, ft, PROTOCOL)
    with pytest.raises(ValueError, match="missing for"):
        event2.run_event2({"M1": pd.DataFrame()}, raw, ft, PROTOCOL)
    frames = _event2_frames(frame, ft, offsets)
    frames["M2-XGB"] = frames["M2-XGB"].iloc[2:]
    with pytest.raises(ValueError, match="M2-XGB: missing"):
        event2.run_event2(frames, raw, ft, PROTOCOL)
    frames = _event2_frames(frame, ft, offsets)
    frames["M2-MLP"] = pd.concat([frames["M2-MLP"], frames["M2-MLP"].iloc[:1]])
    with pytest.raises(ValueError, match="duplicated"):
        event2.run_event2(frames, raw, ft, PROTOCOL)
    frames = _event2_frames(frame, ft, offsets)
    moved = frames["M1"].copy()
    moved.loc[moved.index[0], "outer_fold"] = (int(moved["outer_fold"].iloc[0]) + 1) % 5
    frames["M1"] = moved
    with pytest.raises(ValueError, match="outer-test block"):
        event2.run_event2(frames, raw, ft, PROTOCOL)
    frames = _event2_frames(frame, ft, offsets)
    extra = frames["M1"].iloc[:1].assign(site_no="99999999")
    frames["M1"] = pd.concat([frames["M1"], extra])
    with pytest.raises(ValueError, match="outside the evaluated frame"):
        event2.run_event2(frames, raw, ft, PROTOCOL)


# ----------------------------------------------------------------------------- runtime helpers
def test_runtime_helpers() -> None:
    """json_safe makes strict JSON; checkpoint notes refuse numbers; rel_path keeps symlinks."""
    assert events.json_safe({"a": np.float64(1.5), "b": [math.nan, np.int64(2)]}) == {
        "a": 1.5,
        "b": [None, 2],
    }
    assert events.check_checkpoint_note("  approve Event 1 as recorded ") == (
        "approve Event 1 as recorded"
    )
    for bad in ("", "   ", "approve; I = 0.12", "share 25 %", "ok 0,5"):
        with pytest.raises(EventRefusedError):
            events.check_checkpoint_note(bad)
    assert events.rel_path(Path("/r/a/../b/c.json"), Path("/r")) == "b/c.json"


# ----------------------------------------------------------------------------- CLI end to end
def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _commit(root: Path, msg: str) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", msg)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    proc = root / "data" / "processed"
    (proc / "splits").mkdir(parents=True)
    raw = sites(40)
    raw.to_parquet(proc / "site_table.parquet")
    folds(raw["site_no"]).to_parquet(proc / "splits" / "fold_assignment.parquet")
    for rel in (
        "splits/srnd_assignment.parquet",
        "ind/ehydro_ind.parquet",
        "ind/ehydro_ind_features.parquet",
        "srnd/srnd_frame.parquet",
    ):
        (proc / rel).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"table": [rel]}).to_parquet(proc / rel)  # read only by a faked Event 3
    (proc / "qc").mkdir()
    pd.DataFrame({"row_id": [0], "chan_mat_cd": ["SAND"], "F_eps": [False]}).to_parquet(
        proc / "qc" / "hydroswot_init.parquet"
    )
    nhd = root / "data" / "raw" / "nhdplusv21"
    nhd.mkdir(parents=True)
    (nhd / "MANIFEST.sha256").write_text("synthetic\n")
    hyg = root / "data" / "raw" / "hyg"
    hyg.mkdir(parents=True)
    (hyg / "MANIFEST.sha256").write_text("synthetic hyg\n")
    (root / "uv.lock").write_text("lock")
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text())
    params["model"]["inner_grid"].update(weight_decay=[0.0], lr=[1.0e-3], m4_lambda_max=[10.0])
    params["model"]["xgb"].update(max_depth=[2], eta=[0.3], min_child_weight=[1])
    params["model"]["baselines"]["m1_max_depth"] = [2]
    params["uq"]["ensemble_m"] = 2
    ppath = tmp_path / "params.yaml"
    ppath.write_text(yaml.safe_dump(params, sort_keys=False))
    monkeypatch.setenv("ANNXS_REPO_ROOT", str(root))
    monkeypatch.setenv("ANNXS_DVC_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("ANNXS_RUN_ROOT", str(tmp_path / "store" / "runs"))
    return root, ppath


def _seal(root: Path, ppath: Path, rels: list[str]) -> None:
    settings = load_settings(ppath)
    ident = events.plan_identity(settings, reduced=True)
    path = root / "reports" / "results" / "scoring_ledger.json"
    led = (
        Ledger.load(path)
        if path.is_file()
        else Ledger.create(
            path,
            run_id=ident.run_id,
            plan_sha256=ident.sha256,
            params_sha256=settings.params_sha256,
            uv_lock_sha256=sha256_file(root / "uv.lock"),
        )
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    led.append(
        "seal",
        git={"commit": head, "dirty": False},
        params_sha256=settings.params_sha256,
        inputs=[{"path": r, "sha256": sha256_file(root / r)} for r in rels],
        merkle_root="0" * 64,
        note="synthetic seal",
    )


def test_cli_events_end_to_end_and_refusals(
    repo: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """docs/3 13.4: seal -> event_1 -> checkpoint_1 -> seal -> event_2 -> checkpoint_2, refusals."""
    root, ppath = repo
    audit, evaluate = _load("audit"), _load("evaluate")
    base = ["--params", str(ppath), "--reduced"]
    raw = pd.read_parquet(root / "data" / "processed" / "site_table.parquet")
    frame = event1.audit_frame(raw)
    ft = pd.read_parquet(root / "data" / "processed" / "splits" / "fold_assignment.parquet")
    net, _ = chain_network(frame)
    monkeypatch.setattr(audit, "read_network", lambda unit_dir, comids: net)
    pdir = root / "data" / "processed" / "predictions"
    pdir.mkdir()
    audit_rel = "data/processed/predictions/audit_inputs.parquet"
    pd.concat(
        [
            oof_preds(frame, ft, "M-null", lambda i, r, k: 0.1),
            oof_preds(frame, ft, "M2-XGB", lambda i, r, k: 0.05 * ((i + r) % 3)),
        ]
    ).to_parquet(root / audit_rel)

    with pytest.raises(EventRefusedError, match="no scoring ledger"):
        audit.main(base)
    _git(root, "init", "-q")
    _commit(root, "tables")
    _seal(root, ppath, [audit_rel])
    with pytest.raises(EventRefusedError, match="clean, committed tree"):
        audit.main(base)  # the ledger itself is uncommitted
    _commit(root, "seal 1")
    with pytest.raises(EventRefusedError, match="next ledger step is 'event_1'"):
        evaluate.main(["checkpoint", "--after", "event_1", "--note", "approve", *base])
    with pytest.raises(EventRefusedError, match="next ledger step is 'event_1'"):
        evaluate.main(["select-mstar", *base])
    sealed = (root / audit_rel).read_bytes()
    (root / audit_rel).write_bytes(sealed + b"x")
    _commit(root, "tamper")
    with pytest.raises(EventRefusedError, match="differs from its sealed SHA-256"):
        audit.main(base)
    (root / audit_rel).write_bytes(sealed)
    _commit(root, "restore")

    assert audit.main(base) == 0
    out = capsys.readouterr().out
    assert "event_1 recorded as ledger seq 2" in out
    assert "0." not in out.replace("sha256", "")  # no number printed besides hashes and seq
    led = Ledger.load(root / "reports" / "results" / "scoring_ledger.json")
    rec = led.events[-1]
    assert rec["event"] == "event_1"
    assert rec["inputs"][0] == {"path": audit_rel, "sha256": sha256_file(root / audit_rel)}
    assert [i["path"] for i in rec["inputs"][1:]] == [
        "data/processed/site_table.parquet",
        "data/processed/splits/fold_assignment.parquet",
        "data/raw/nhdplusv21/MANIFEST.sha256",
    ]
    outs = {o["path"]: o["sha256"] for o in rec["outputs"]}
    for rel in ("reports/results/audit.json", "reports/results/audit_strata.parquet"):
        assert outs[rel] == sha256_file(root / rel)
    doc = json.loads((root / "reports" / "results" / "audit.json").read_text())
    assert doc["label"]["N"] in (True, False)
    assert doc["strata"]["sha256"] == outs["reports/results/audit_strata.parquet"]
    attempts = json.loads((root / "reports" / "results" / "event_attempts.json").read_text())
    assert attempts["attempts"][-1]["seq"] == 2
    assert [a["seq"] for a in attempts["attempts"]].count(None) == 1  # the tampered attempt

    with pytest.raises(EventRefusedError, match="clean, committed tree"):
        evaluate.main(["checkpoint", "--after", "event_1", "--note", "approve", *base])
    _commit(root, "event 1")
    with pytest.raises(EventRefusedError, match="recorded and is never re-run"):
        audit.main(base)
    with pytest.raises(EventRefusedError, match="no number"):
        evaluate.main(["checkpoint", "--after", "event_1", "--note", "I = 0.4 ok", *base])
    with pytest.raises(EventRefusedError, match="next ledger step is 'checkpoint_1'"):
        evaluate.main(["checkpoint", "--after", "event_2", "--note", "approve", *base])
    assert (
        evaluate.main(["checkpoint", "--after", "event_1", "--note", "PI: approve Event 1", *base])
        == 0
    )
    assert "checkpoint_1 recorded as ledger seq 3" in capsys.readouterr().out
    _commit(root, "checkpoint 1")

    offsets = {"M1": 0.30, "M2-MLP": 0.20, "M2-XGB": 0.25}
    rels = []
    for m, p in _event2_frames(frame, ft, offsets).items():
        rel = f"data/processed/predictions/{m}.parquet"
        p.to_parquet(root / rel)
        rels.append(rel)
    for stem in evaluate.EVENT3_PREDICTIONS:  # the other Event 3 inputs (content faked below)
        rel = f"data/processed/predictions/{stem}.parquet"
        if rel not in rels:
            pd.DataFrame({"stem": [stem]}).to_parquet(root / rel)
            rels.append(rel)
    _commit(root, "predictions")
    with pytest.raises(EventRefusedError, match="next ledger step is 'seal'"):
        evaluate.main(["select-mstar", *base])
    _seal(root, ppath, rels[:2])  # M2-XGB left out of the seal
    _commit(root, "seal 2")
    with pytest.raises(EventRefusedError, match=r"M2-XGB\.parquet is not sealed"):
        evaluate.main(["select-mstar", *base])
    led = Ledger.load(root / "reports" / "results" / "scoring_ledger.json")
    assert led.next_expected() == "event_2"
    # an erratum-free re-seal is impossible (order), so the test seals afresh in a new ledger
    path = root / "reports" / "results" / "scoring_ledger.json"
    doc_led = json.loads(path.read_text())
    doc_led["events"] = doc_led["events"][:-1]
    path.write_text(json.dumps(doc_led))
    _commit(root, "drop partial seal (test only)")
    _seal(root, ppath, rels)
    _commit(root, "seal 2 complete")
    assert evaluate.main(["select-mstar", *base]) == 0
    assert "event_2 recorded as ledger seq 5" in capsys.readouterr().out
    mstar = json.loads((root / "reports" / "results" / "mstar.json").read_text())
    assert mstar["mstar"] == {"dbar": "M2-MLP", "A": "M2-MLP"}
    _commit(root, "event 2")
    assert (
        evaluate.main(["checkpoint", "--after", "event_2", "--note", "PI: approve M star", *base])
        == 0
    )
    with pytest.raises(EventRefusedError, match="clean, committed tree"):
        evaluate.main(["evaluate", *base])
    _commit(root, "checkpoint 2")

    # Event 3 plumbing; the computation is tested in test_evaluation_event3.py
    seen: dict[str, object] = {}

    def fake_run_event3(inp, protocol, *, log_kappa, aoa, n_boot=None):
        seen.update(preds=sorted(inp.preds), mstar=inp.mstar, n_strata=len(inp.audit_strata))
        seen.update(callables=callable(log_kappa) and callable(aoa), n_boot=n_boot)
        docs = {name: {"section": name} for name in evaluate.EVENT3_DOCS}
        docs["hypotheses"].update(
            audit_label="label text",
            outcome_4_7={"rows": [], "statement": "no row of protocol 4.7 matches"},
        )
        return event3.Event3Result(documents=docs, matrix=pd.DataFrame({"metric": ["rmse_log"]}))

    monkeypatch.setattr(evaluate, "run_event3", fake_run_event3)
    monkeypatch.setattr(evaluate, "_FORBIDDEN", ())  # pytest itself has loaded torch/xgboost
    monkeypatch.setattr(evaluate, "_vaa_maps", lambda nhd_dir, comids: (net, {}, {}))
    monkeypatch.setattr(
        evaluate,
        "read_hyg_manning",
        lambda path: (pd.DataFrame({"site_no": [], "manning_n": []}), {"rows": 0}),
    )
    assert evaluate.main(["evaluate", *base]) == 0
    out = capsys.readouterr().out
    assert "event_3 recorded as ledger seq 7" in out
    assert seen["preds"] == sorted(evaluate.EVENT3_PREDICTIONS)
    assert seen["mstar"]["mstar"] == {"dbar": "M2-MLP", "A": "M2-MLP"}  # type: ignore[index]
    assert seen["callables"] is True
    assert seen["n_boot"] is None  # production B from params
    led = Ledger.load(path)
    rec = led.events[-1]
    assert rec["event"] == "event_3"
    in_paths = [i["path"] for i in rec["inputs"]]
    order = list(evaluate.EVENT3_PREDICTIONS)
    assert in_paths[:20] == sorted(rels, key=lambda r: order.index(Path(r).stem))
    assert in_paths[20:] == [
        "data/processed/ind/ehydro_ind.parquet",
        "data/processed/splits/fold_assignment.parquet",
        "data/processed/ind/ehydro_ind_features.parquet",
        "data/processed/site_table.parquet",
        "data/processed/splits/srnd_assignment.parquet",
        "data/processed/srnd/srnd_frame.parquet",
        "reports/results/audit.json",
        "reports/results/audit_strata.parquet",
        "reports/results/mstar.json",
        "data/processed/qc/hydroswot_init.parquet",
        "data/raw/nhdplusv21/MANIFEST.sha256",
        "data/raw/hyg/MANIFEST.sha256",
    ]
    res_dir = root / "reports" / "results"
    outs = {o["path"]: o["sha256"] for o in rec["outputs"]}
    want = [f"reports/results/event3_{n}.json" for n in evaluate.EVENT3_DOCS]
    want += ["reports/results/event3_matrix.parquet", "reports/results/event3.json"]
    assert list(outs) == want
    for rel, sha in outs.items():
        assert sha == sha256_file(root / rel)
    index = json.loads((res_dir / "event3.json").read_text())
    assert index["outcome_4_7"]["statement"] == "no row of protocol 4.7 matches"
    assert [o["path"] for o in index["outputs"]] == want[:-1]
    assert index["outcome_table_4_7"]["4"][2] == "ASCE; fallback per ADL-014"
    hyp = json.loads((res_dir / "event3_hypotheses.json").read_text())
    assert hyp["run_id"] == led.header["run_id"]
    assert hyp["section"] == "hypotheses"
    _commit(root, "event 3")
    with pytest.raises(EventRefusedError, match="recorded and is never re-run"):
        evaluate.main(["evaluate", *base])
    led = Ledger.load(path)
    assert led.recorded() == [
        "seal",
        "event_1",
        "checkpoint_1",
        "seal",
        "event_2",
        "checkpoint_2",
        "event_3",
    ]


def test_open_event_header_checks(repo: tuple[Path, Path]) -> None:
    """docs/3 13.4: the ledger header must match the plan, params.yaml and uv.lock."""
    root, ppath = repo
    _git(root, "init", "-q")
    _commit(root, "tables")
    rel = "data/processed/site_table.parquet"
    _seal(root, ppath, [rel])
    _commit(root, "seal")
    settings = load_settings(ppath)
    ident = events.plan_identity(settings, reduced=True)
    with pytest.raises(EventRefusedError, match=r"header differs in \['plan_sha256', 'run_id'\]"):
        events.open_event(settings, "event_1", run_id="p8-000000000000", plan_sha256="1" * 64)
    (root / "uv.lock").write_text("other lock")
    _commit(root, "lock change")
    with pytest.raises(EventRefusedError, match="uv_lock_sha256"):
        events.open_event(settings, "event_1", run_id=ident.run_id, plan_sha256=ident.sha256)
    (root / "uv.lock").write_text("lock")
    _commit(root, "lock back")
    path = root / "reports" / "results" / "scoring_ledger.json"
    good = path.read_text()
    path.write_text(good.replace('"seq": 1', '"seq": 2'))
    _commit(root, "break")
    with pytest.raises(EventRefusedError, match="ledger invalid"):
        events.open_event(settings, "event_1", run_id=ident.run_id, plan_sha256=ident.sha256)
    path.write_text(good)
    (root / "reports" / "results" / "event_attempts.json").write_text("[]")
    _commit(root, "bad attempts log")
    with pytest.raises(EventRefusedError, match="malformed"):
        events.open_event(settings, "event_1", run_id=ident.run_id, plan_sha256=ident.sha256)
    with pytest.raises(RuntimeError, match=r"13\.6 table"):
        events.plan_identity(settings)
    with pytest.raises(FileNotFoundError):
        events.file_entry(root / "nope.json", root)


def test_event_modules_import_neither_torch_nor_xgboost() -> None:
    """ADL-051: the event modules and both CLIs load no torch or xgboost."""
    code = (
        "import sys, runpy; import src.evaluation.events, src.evaluation.event1, "
        "src.evaluation.event2, src.evaluation.event3, src.evaluation.event3_sets, "
        "src.evaluation.event3_stats, src.evaluation.event3_workers; "
        "runpy.run_path('scripts/audit.py', run_name='x'); "
        "runpy.run_path('scripts/evaluate.py', run_name='x'); "
        "print('torch' in sys.modules, 'xgboost' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    ).stdout.strip()
    assert out == "False False", f"ADL-051: torch/xgboost imported: {out}"


def test_fixture_keys_match_the_prediction_writer() -> None:
    """F-C3b-1 contract: the synthetic OOF rows carry the key values ``predict._base`` writes."""
    from src.models import predict

    keys = pd.DataFrame({"site_no": ["s0"], "comid": [1000]})
    empty = pd.DataFrame()
    bf = predict.BlockFrame(empty, np.array([], dtype=int), empty, np.array([0]), keys)
    for model in event1.AUDIT_MODELS:
        job = predict.PredictJob(model, None, "primary", "S-ID", 2, 3, 1)
        written = predict._base(job, bf).iloc[0]
        raw = sites(5)
        frame = event1.audit_frame(raw)
        ours = oof_preds(frame, folds(frame["site_no"]), model, lambda i, r, k: 0.0).iloc[0]
        for col in ("spec", "model", "ablation", "analysis", "family", "unit"):
            assert ours[col] == written[col], f"F-C3b-1: fixture {col}={ours[col]!r}"


def test_primary_rows_follow_the_writer_ablation_contract() -> None:
    """F-C3b-1 regression: ``ablation == ''`` rows are primary; a null ablation is refused."""
    raw = sites(10)
    frame = event1.audit_frame(raw)
    ft = folds(frame["site_no"])
    preds = oof_preds(frame, ft, "M-null", lambda i, r, k: 0.25)
    assert preds["ablation"].eq("").all()
    abl = preds.assign(ablation="A1")
    mask = event1.primary_sid_rows(pd.concat([preds, abl], ignore_index=True))
    assert int(mask.sum()) == len(preds)
    res = event1.oof_residuals(preds, frame, ft, models=("M-null",))
    assert np.allclose(res["M-null:dbar"].to_numpy(), 0.25, rtol=0.0, atol=1e-12)
    with pytest.raises(ValueError, match="null ablation"):
        event1.primary_sid_rows(preds.assign(ablation=None))

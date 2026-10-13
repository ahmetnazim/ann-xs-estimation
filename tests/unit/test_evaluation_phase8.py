"""Known-answer tests for the Phase 8 evaluation additions (docs/3 13.6, 13.9). Synthetic only.

Sources: docs/3 section 13.6 (weighted MdAPE), section 13.9 (V-error variance shares, Manning
back-calculation consistency, W2 with co-located sites, nested-basin inputs from the raw unit,
AOA permutation importance, S1), section 13.5 (Zarrabi product unit STOP); protocol 4.3.2, 3.6.2,
3.6.5, 3.6.6, 4.6; proofs Prop 6.2 and Prop 7.4.

Expected values are hand computations written out in each test or independent oracles
(``numpy.cov``, ``numpy.median``, ``scipy.stats.spearmanr``, a dense W2 built by hand, and a
replay of the pre-registered ``SeedSequence([member_seed, feature])`` draws); never the function
under test. No real prediction is read.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.config import load_settings
from src.data import external as ex
from src.data.network import network_inputs
from src.data.readers import RawSchemaError
from src.evaluation import dependence_audit as da
from src.evaluation import metrics as mt

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = load_settings().protocol
AUDIT = PROTOCOL.audit
TIGHT = 1e-12
NU_BOUNDS = (math.log(PROTOCOL.domain.n_bounds[0]), math.log(PROTOCOL.domain.n_bounds[1]))


def close(got: float, want: float, tol: float, ref: str) -> None:
    """Assert ``|got - want| <= tol`` with the magnitude, tolerance and reference in the message."""
    err = abs(got - want)
    assert err <= tol, f"{ref}: got {got!r}, want {want!r}, |err| = {err:.3e} > tol {tol:.1e}"


# ============================================================================ 13.6 weighted MdAPE
APE = np.array([0.1, 0.5, 0.2, 0.4])


@pytest.mark.parametrize(
    ("weights", "want"),
    [
        (None, 0.2),  # sorted .1 .2 .4 .5, cumulative .25 .5: the lower middle value
        ([1.0, 1.0, 1.0, 5.0], 0.4),  # sorted weights 1, 1, 5, 1 (/8): cumulative .125 .25 .875
        ([5.0, 1.0, 1.0, 1.0], 0.1),  # 0.1 alone carries 5/8 >= 1/2
    ],
)
def test_weighted_mdape_hand_values(weights, want) -> None:
    """docs/3 13.6: smallest |p/o - 1| whose cumulative normalized weight reaches 1/2; x 100."""
    got = mt.weighted_mdape(1.0 + APE, np.ones(4), weights)
    close(got, 100.0 * want, 1e-12, "13.6 weighted MdAPE")


def test_weighted_mdape_srnd_weights_and_rounding_guard() -> None:
    """docs/3 13.6 with S-RND weights 1/|I_j| (site_weights) and ten weights of 0.1."""
    w = mt.site_weights(["a", "a", "b"])  # 1/4, 1/4, 1/2
    got = mt.weighted_mdape(1.0 + np.array([0.3, 0.1, 0.2]), np.ones(3), w)
    close(got, 20.0, 1e-12, "13.6 sorted .1 (1/4), .2 (1/2): cumulative .75 at .2")
    ape = np.arange(1, 11) / 100.0  # cumulative weight reaches 1/2 at the 5th value (0.05)
    got = mt.weighted_mdape(1.0 + ape, np.ones(10), np.full(10, 0.1))
    close(got, 5.0, 1e-12, "13.6 ten weights of 0.1: the 5th value, not the 6th")


def test_weighted_mdape_uniform_odd_count_equals_the_median() -> None:
    """With uniform weights and an odd count the weighted median is the ordinary median."""
    rng = np.random.default_rng(3)
    p, o = rng.uniform(0.5, 2.0, 101), rng.uniform(0.5, 2.0, 101)
    want = 100.0 * float(np.median(np.abs(p / o - 1.0)))
    close(mt.weighted_mdape(p, o), want, TIGHT, "13.6 odd n, uniform weights")


# ============================================================================ 13.9 V-error shares
def test_v_error_shares_hand_values() -> None:
    """dq = (1,-1,0,0), e_A = (0,0,1,-1): e_V = (1,-1,-1,1); Var 1; both covariances 1/2."""
    s_q, s_a = mt.v_error_shares([1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0])
    close(s_q, 0.5, TIGHT, "4.3.2 s_Q")
    close(s_a, 0.5, TIGHT, "4.3.2 s_A")


def test_v_error_shares_weighted_oracle_and_sum_to_one() -> None:
    """Population moments with site weights (numpy.cov, aweights, bias=True); s_Q + s_A = 1."""
    rng = np.random.default_rng(11)
    dq, ea = rng.normal(size=40), rng.normal(size=40)
    w = mt.site_weights(rng.integers(0, 15, size=40))
    s_q, s_a = mt.v_error_shares(dq, ea, w)
    ev = dq - ea
    c = np.cov(np.vstack([dq, ea, ev]), aweights=w, bias=True)
    close(s_q, c[0, 2] / c[2, 2], 1e-12, "4.3.2 s_Q vs numpy.cov")
    close(s_a, -c[1, 2] / c[2, 2], 1e-12, "4.3.2 s_A vs numpy.cov")
    close(s_q + s_a, 1.0, 1e-12, "13.9 s_Q + s_A = 1")
    with pytest.raises(ValueError, match="zero variance"):
        mt.v_error_shares([1.0, 2.0], [1.0, 2.0])


# ============================================================================ 13.9 back-calculation
def test_depth_ratio_r_and_invariant4() -> None:
    """r = dbar/(dmax - dbar); dmax <= dbar is an invariant-4 violation (nan, not ok)."""
    r, ok = mt.depth_ratio_r([1.0, 2.0, 2.0, 2.0], [2.0, 2.0, 3.0, 1.5])
    assert ok.tolist() == [True, False, True, False]
    close(float(r[0]), 1.0, TIGHT, "r = 1/(2-1)")
    close(float(r[2]), 2.0, TIGHT, "r = 2/(3-2)")
    assert bool(np.isnan(r[[1, 3]]).all())
    t = math.log(3.0 / 2.0)  # proofs: t = ln(1 + 1/r)
    close(float(r[2]), 1.0 / math.expm1(t), 1e-12, "r = 1/(e^t - 1)")
    with pytest.raises(ValueError, match="dbar must be > 0"):
        mt.depth_ratio_r([0.0], [1.0])


def test_back_calc_consistency_hand_values() -> None:
    """Prop 7.4: dist(nu_back, [nu_min, nu_max]) > 0; the bound itself is inside."""
    nu = np.log([0.004, 0.01, 1.0, 0.7, PROTOCOL.domain.n_bounds[1]])
    nu[2] = np.nan  # an invariant-4 row carries no nu_back
    ok = np.array([True, True, False, True, True])
    res = mt.back_calc_consistency(nu, ok, NU_BOUNDS)
    assert (res.n_rows, res.n_invariant4, res.n_used) == (5, 1, 4)
    close(res.share_outside, 0.5, TIGHT, "0.004 and 0.7 outside [0.005, 0.6]; 0.6 is inside")
    weighted = mt.back_calc_consistency(nu, ok, NU_BOUNDS, [1.0, 1.0, 100.0, 1.0, 3.0])
    close(weighted.share_outside, 2.0 / 6.0, TIGHT, "weights renormalized over used rows")


@pytest.mark.parametrize(
    ("nu", "ok", "bounds", "match"),
    [
        ([0.0], [False], NU_BOUNDS, "invariant 4"),
        ([np.nan], [True], NU_BOUNDS, "non-finite"),
        ([0.0], [True], (1.0, 0.0), "lo < hi"),
        ([0.0, 1.0], [True], NU_BOUNDS, "equal shape"),
    ],
)
def test_back_calc_consistency_contracts(nu, ok, bounds, match) -> None:
    with pytest.raises(ValueError, match=match):
        mt.back_calc_consistency(nu, ok, bounds)


def test_back_calc_matches_phi_inverse_nu_identity() -> None:
    """Prop 6.2 end to end: nu_back of the Phi solution equals the nu that produced it."""
    import torch

    from src.physics.manning_layer import PhiConfig, phi, phi_inverse_nu

    cfg = PhiConfig.from_solver(PROTOCOL.solver)
    w = torch.tensor([3.0, 4.0], dtype=torch.float64)
    r = torch.tensor([2.0, 5.0], dtype=torch.float64)
    nu_true = torch.tensor([math.log(0.004), math.log(0.03)], dtype=torch.float64)
    q = torch.tensor([2.0, 3.0], dtype=torch.float64)
    s = torch.tensor([math.log(1e-3), math.log(5e-4)], dtype=torch.float64)
    ell = phi(w, r, nu_true, q, s, cfg).ell
    nu_back = phi_inverse_nu(ell, w, r, q, s, cfg).detach().numpy()
    worst = float(np.max(np.abs(nu_back - nu_true.numpy())))
    assert worst <= 1e-9, f"Prop 6.2 round trip: max |err| {worst:.3e}"
    res = mt.back_calc_consistency(nu_back, [True, True], NU_BOUNDS)
    close(res.share_outside, 0.5, TIGHT, "n = 0.004 is below n_min = 0.005")


# ============================================================================ 13.9 S1
def test_s1_consistency_hand_values() -> None:
    """Ranks (1,2,4,3) vs (1,2,3,4): rho = 1 - 6*2/(4*15) = 0.8; log-RMSE by hand."""
    n_hat = np.array([0.02, 0.03, 0.05, 0.04])
    n_ref = np.array([0.025, 0.03, 0.04, 0.05])
    rho, lr = mt.s1_consistency(n_hat, n_ref)
    close(rho, 0.8, TIGHT, "S1 Spearman")
    close(rho, float(stats.spearmanr(n_hat, n_ref).statistic), TIGHT, "S1 vs scipy")
    want = math.sqrt((math.log(0.8) ** 2 + 0.0 + math.log(1.25) ** 2 + math.log(0.8) ** 2) / 4.0)
    close(lr, want, TIGHT, "S1 log-RMSE")


# ============================================================================ 13.9 W2 co-located
def test_w2_colocated_pairs_get_zero_weight_and_are_counted() -> None:
    """Sites 0 and 1 coincide; site 2 is 30 km away; site 3 is an isolate (dense oracle)."""
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [30.0, 0.0], [500.0, 500.0]])
    w2, iso = da.idw_weights(xy, AUDIT.idw_band_km)
    want = np.array(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    worst = float(np.max(np.abs(w2.toarray() - want)))
    assert worst <= TIGHT, f"13.9 W2 with co-located sites: max |err| {worst:.3e}"
    assert iso.tolist() == [3]
    assert da.colocated_pairs(xy) == 1
    assert da.colocated_pairs(np.array([[0.0, 0.0], [1e-9, 0.0]])) == 0
    three = np.zeros((3, 2))
    assert da.colocated_pairs(three) == 3  # unordered pairs among three coincident sites


# ============================================================================ 13.9 nested-basin BFS
def test_network_inputs_feed_nested_status() -> None:
    """PlusFlow rows with a 0 end are dropped; TerminalPa/TotDASqKm for requested COMIDs only."""
    vaa = pd.DataFrame(
        {
            "ComID": [1, 2, 3, 4, 5, 7, 8, 9],
            "TerminalPa": [1, 1, 1, 1, 1, 2, 2, 3],
            "TotDASqKm": [10.0, 30.0, 100.0, 120.0, 40.0, 5.0, 50.0, 20.0],
        }
    )
    flow = pd.DataFrame(
        {"FromComID": [1, 2, 5, 3, 4, 7, 8, 0], "ToComID": [2, 3, 3, 4, 0, 8, 3, 9]}
    )
    net = network_inputs(vaa, flow, [1, 5, 7, 4, 3, 2, 8, 9, 99])
    assert net.edges.tolist() == [[1, 2], [2, 3], [5, 3], [3, 4], [7, 8], [8, 3]]
    assert net.missing == (99,)
    assert net.terminal_path[7] == 2
    assert net.drainage_km2[4] == 120.0
    r = da.nested_status(
        [1, 5, 7, 4], [3, 2, 8, 9], net.edges, net.terminal_path, net.drainage_km2, AUDIT
    )
    want = np.array([100.0 / 120.0, 10.0 / 30.0, 0.1, 0.0])  # as test_nested_basin_hand_graph
    worst = float(np.max(np.abs(r.psi_max - want)))
    assert worst <= TIGHT, f"3.6.5 psi_max {r.psi_max}"
    with pytest.raises(RawSchemaError, match="not unique"):
        network_inputs(pd.concat([vaa, vaa.iloc[:1]]), flow, [1])


def test_read_network_reads_vaa_and_plusflow_from_the_raw_unit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """docs/3 13.9: the inputs come from the raw nhdplusv21 unit (VAA and PlusFlow tables)."""
    from src.data import network, nhdplus

    calls: list[tuple[Path, str]] = []
    tables = {
        "vaa": pd.DataFrame({"ComID": [1, 2], "TerminalPa": [9, 9], "TotDASqKm": [1.0, 2.0]}),
        "plusflow": pd.DataFrame({"FromComID": [1, 0], "ToComID": [2, 1]}),
    }

    def fake(unit_dir: Path, name: str) -> tuple[pd.DataFrame, dict[str, int]]:
        calls.append((unit_dir, name))
        return tables[name], {}

    monkeypatch.setattr(nhdplus, "read_nhd_table", fake)
    net = network.read_network(tmp_path, [2])
    assert calls == [(tmp_path, "vaa"), (tmp_path, "plusflow")]
    assert net.edges.tolist() == [[1, 2]]
    assert net.terminal_path == {2: 9}


# ============================================================================ 13.9 AOA importance
def _replay(x: np.ndarray, y: np.ndarray, coef: np.ndarray, seed: int, n_perm: int) -> np.ndarray:
    """Oracle: the pre-registered draws replayed with a linear predictor ``x @ coef``."""
    base = math.sqrt(float(np.mean((x @ coef - y) ** 2)))
    out = np.zeros(x.shape[1])
    for f in range(x.shape[1]):
        rng = np.random.default_rng(np.random.SeedSequence([seed, f]))
        for _ in range(n_perm):
            xp = x.copy()
            xp[:, f] = x[rng.permutation(x.shape[0]), f]
            out[f] += math.sqrt(float(np.mean((xp @ coef - y) ** 2))) - base
    return out / n_perm


def test_permutation_importance_replays_the_preregistered_draws() -> None:
    """docs/3 13.9: SeedSequence([member_seed, f]); mean over perms, then members; clip at 0."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 3))
    y = x[:, 0] + 0.1 * rng.normal(size=30)
    good, bad = np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.5])

    def lin(c: np.ndarray):
        return lambda m: m @ c

    members = [
        da.ImportanceMember(lin(good), x, y, 101),
        da.ImportanceMember(lin(bad), x, y, 202),
    ]
    got = da.permutation_importance(members)
    a = _replay(x, y, good, 101, da.AOA_PERMUTATIONS)
    b = _replay(x, y, bad, 202, da.AOA_PERMUTATIONS)
    want = np.maximum((a + b) / 2.0, 0.0)
    worst = float(np.max(np.abs(got - want)))
    assert worst <= 1e-12, f"13.9 importance vs replay: {got} vs {want}"
    assert got[1] == 0.0, "a feature the predictors ignore has importance 0"
    assert b[0] < 0.0, "the anti-correlated member lowers feature 0 before averaging"
    assert got[0] > 0.0, "clipping is applied after the member average"
    assert da.AOA_PERMUTATIONS == 10


def test_permutation_importance_contracts() -> None:
    x = np.zeros((3, 2))
    ok = da.ImportanceMember(lambda m: m[:, 0], x, np.zeros(3), 1)
    with pytest.raises(ValueError, match="at least one"):
        da.permutation_importance([])
    with pytest.raises(ValueError, match=">= 1"):
        da.permutation_importance([ok], permutations=0)
    with pytest.raises(ValueError, match="disagree"):
        da.permutation_importance(
            [ok, da.ImportanceMember(ok.predict, np.zeros((3, 3)), ok.y_log, 2)]
        )
    with pytest.raises(ValueError, match="non-finite"):
        da.permutation_importance([da.ImportanceMember(ok.predict, x, np.array([0, 0, np.nan]), 1)])
    with pytest.raises(ValueError, match="one per row"):
        da.permutation_importance([da.ImportanceMember(lambda m: m[:2, 0], x, np.zeros(3), 1)])
    with pytest.raises(ValueError, match="N >= 2"):
        da.permutation_importance([da.ImportanceMember(ok.predict, x[:1], np.zeros(1), 1)])


# ============================================================================ external readers
def test_read_hyg_manning_pads_ids_and_counts_drops(tmp_path: Path) -> None:
    """Numeric Gage_ID loses leading zeros: all-digit IDs shorter than 8 are left-padded."""
    path = tmp_path / "hyg.csv"
    path.write_text(
        "Gage_ID,lat,Manning_n\n"
        "2381090,1,0.04\n"  # padded -> 02381090
        "12345678,1,0.03\n"
        "ABC123,1,0.05\n"  # non-digit
        "123456789012345,1,0.02\n"  # long, used as delivered
        "1234567,1,0.06\n"  # pads to 01234567 ...
        "01234567,1,0.07\n"  # ... which also occurs here: both dropped
        "7654321,1,\n"  # missing n
        "87654321,1,0\n"  # non-positive n
    )
    frame, counts = ex.read_hyg_manning(path)
    assert frame["site_no"].tolist() == ["02381090", "12345678", "123456789012345"]
    assert frame["manning_n"].tolist() == [0.04, 0.03, 0.02]
    assert counts == {
        "rows": 8,
        "non_digit_id": 1,
        "padded_id": 3,
        "duplicated_id_rows": 2,
        "manning_n_missing_or_nonpositive": 2,
        "kept": 3,
    }


def test_read_zarrabi_product_refuses_without_an_asserted_unit(tmp_path: Path) -> None:
    """docs/3 13.5: no unit in the Zenodo record -> STOP; with a recorded unit, SI metres."""
    path = tmp_path / "Bankfull_Meanflow_CONUS.txt"
    path.write_text(
        ",COMID,REACHCODE,TotDASqKM,StreamOrde,bnk_depth,bnk_width,mf_depth,mf_width\n"
        "0,30,1,1.0,1.0,2.0,80.0,1.5,70.0\n"
        "1,10,1,1.0,1.0,2.0,80.0,1.0,50.0\n"
        "2,20,1,1.0,1.0,2.0,80.0,2.0,60.0\n"
    )
    assert ex.ZARRABI_RECORD_UNIT is None
    with pytest.raises(ex.UnitNotAssertedError, match="STOP"):
        ex.read_zarrabi_product(path, [10, 30])
    m = ex.read_zarrabi_product(path, [10, 30, 99], unit="m")
    assert m["comid"].tolist() == [10, 30]
    assert m["mf_width_m"].tolist() == [50.0, 70.0]
    assert m["mf_depth_m"].tolist() == [1.0, 1.5]
    ft = ex.read_zarrabi_product(path, [20], unit="ft")
    close(float(ft["mf_width_m"].iloc[0]), 60.0 * 0.3048, TIGHT, "ft -> m")
    with pytest.raises(ValueError, match="unknown unit"):
        ex.read_zarrabi_product(path, [20], unit="cm")
    dup = tmp_path / "dup.txt"
    dup.write_text(path.read_text() + "3,20,1,1.0,1.0,2.0,80.0,2.0,61.0\n")
    with pytest.raises(ValueError, match="more than once"):
        ex.read_zarrabi_product(dup, [20], unit="m")


def test_phase8_additions_import_neither_torch_nor_xgboost() -> None:
    """ADL-051: the new evaluation and reader code stays torch-free and xgboost-free."""
    code = (
        "import sys; import src.evaluation.metrics, src.evaluation.dependence_audit, "
        "src.data.external, src.data.network; "
        "print('torch' in sys.modules, 'xgboost' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    ).stdout.strip()
    assert out == "False False"

"""Known-answer tests for src/evaluation (Phase 7, Step 5; P7-D1 (a), P7-D11 (c)). Synthetic only.

Sources: docs/3 sections 3.6.2-3.6.7 (dependence audit), 4.3.1-4.3.2 (metrics), 4.4 (M*, the
paired cluster bootstrap, H1-H4 decision table), 4.5 (S2 Bonferroni 1 - 0.05/4), 4.6 (UQ band),
4.7 (outcome matrix); docs/2 sections 9.3-9.5 (mixture quantile, coverage, PIT, closed-form
CRPS, site-weighted statistics, paired and two-sample bootstrap).

Expected values are hand computations written out in each test, closed forms from those
sections, or independent numerical oracles (``scipy.integrate.quad``, ``scipy.optimize.brentq``,
``scipy.stats``, exhaustive permutation enumeration, ``esda.Moran`` and dense NumPy formulas);
never the function under test. Thresholds come from params.yaml through ``load_settings``.
No real predictions are read and no H1-H4 verdict on real data is produced (P7-D1 (a)).
"""

from __future__ import annotations

import itertools
import math
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from esda.moran import Moran
from libpysal.weights import W
from scipy import integrate, optimize, sparse, stats

from src.config import load_settings
from src.evaluation import bootstrap as bs
from src.evaluation import dependence_audit as da
from src.evaluation import hypotheses as hy
from src.evaluation import metrics as mt
from src.evaluation import uq

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = load_settings().protocol
EVAL = PROTOCOL.evaluation
AUDIT = PROTOCOL.audit
RULES = PROTOCOL.audit_rules
LN2 = math.log(2.0)
TIGHT = 1e-12


def close(got: float, want: float, tol: float, ref: str) -> None:
    """Assert ``|got - want| <= tol`` with the magnitude, tolerance and reference in the message."""
    err = abs(got - want)
    assert err <= tol, f"{ref}: got {got!r}, want {want!r}, |err| = {err:.3e} > tol {tol:.1e}"


# ============================================================================ 4.3.1 metrics
O4 = np.array([1.0, 2.0, 4.0, 8.0])
P4 = np.array([2.0, 2.0, 2.0, 2.0])


def test_primary_log_metrics_hand_values() -> None:
    """docs/3 4.3.1: e = ln2 (1, 0, -1, -2); RMSE_log = ln2 sqrt(1.5); B_log = -ln2/2."""
    lp, lo = np.log(P4), np.log(O4)
    close(mt.rmse_log(lp, lo), LN2 * math.sqrt(1.5), TIGHT, "4.3.1 RMSE_log")
    close(mt.bias_log(lp, lo), -LN2 / 2.0, TIGHT, "4.3.1 B_log")
    close(mt.multiplicative_bias(-LN2 / 2.0), 2.0**-0.5 - 1.0, TIGHT, "9.5 e^B - 1")
    close(mt.nse_log(lp, lo), 1.0 - 6.0 / 5.0, TIGHT, "4.3.1 NSE_log = 1 - 6/5")


def test_rmse_log_unweighted_path_is_the_phase6_expression() -> None:
    """docs/3 3.4 item 2: the inner-selection statistic stays bit-identical to Phase 6."""
    rng = np.random.default_rng(3)
    p, o = rng.normal(size=257), rng.normal(size=257)
    want = math.sqrt(float(np.mean((p - o) ** 2)))
    got = mt.rmse_log(p, o)
    assert got == want, f"3.4: rmse_log {got!r} != Phase 6 expression {want!r}"


def test_secondary_raw_metrics_hand_values() -> None:
    """docs/3 4.3.1: NSE_raw, RMSE_raw, MdAPE and APB % on o = (1, 2, 4, 8), p = 2."""
    close(mt.nse_raw(P4, O4), 1.0 - 41.0 / 28.75, TIGHT, "4.3.1 NSE_raw = 1 - 41/28.75")
    close(mt.rmse_raw(P4, O4), math.sqrt(41.0 / 4.0), TIGHT, "4.3.1 RMSE_raw")
    close(mt.mdape(P4, O4), 62.5, TIGHT, "4.3.1 MdAPE = 100 median(1, 0, .5, .75)")
    close(mt.apb_pct(P4, O4), 60.0, TIGHT, "4.3.1 APB = 100 * 9 / 15")


def test_r2_and_kge_on_a_doubled_prediction() -> None:
    """docs/3 4.3.1: p = 2o gives r^2 = 1, alpha = beta = 2, so KGE = 1 - sqrt(2)."""
    p = 2.0 * O4
    close(mt.r2_pearson_raw(p, O4), 1.0, TIGHT, "4.3.1 r^2")
    close(mt.kge(p, O4), 1.0 - math.sqrt(2.0), TIGHT, "4.3.1 KGE")
    close(mt.nse_raw(p, O4), 1.0 - 85.0 / 28.75, TIGHT, "4.3.1 NSE_raw (p = 2o)")


def test_t_space_shape_metrics() -> None:
    """docs/3 4.3.1 (shape): RMSE_t and B_t are plain RMSE and bias in t units."""
    t_hat, t_obs = np.array([0.2, 0.4, 0.9]), np.array([0.3, 0.4, 0.6])
    close(mt.rmse(t_hat, t_obs), math.sqrt((0.01 + 0.0 + 0.09) / 3.0), TIGHT, "4.3.1 RMSE_t")
    close(mt.bias(t_hat, t_obs), (-0.1 + 0.0 + 0.3) / 3.0, TIGHT, "4.3.1 B_t")


def test_spearman_is_pearson_of_mid_ranks() -> None:
    """docs/2 9.5 H4 (b): rho_S = Pearson of mid-ranks; 1 - 6*2/60 = 0.8; ties vs scipy."""
    close(mt.spearman_rho([1, 2, 3, 4], [1, 3, 2, 4]), 0.8, TIGHT, "9.5 rho_S hand")
    x, y = np.array([1.0, 2.0, 2.0, 3.0, 5.0, 5.0]), np.array([2.0, 1.0, 4.0, 4.0, 6.0, 3.0])
    want = float(stats.spearmanr(x, y).statistic)
    close(mt.spearman_rho(x, y), want, TIGHT, "9.5 rho_S with ties vs scipy.stats.spearmanr")


def test_site_weights_and_weighted_rmse_follow_proofs_9_5() -> None:
    """docs/2 9.5 / docs/3 4.3.1: S-RND rows weighted 1/(J |I_j|); the weights sum to 1."""
    ids = np.array(["a", "a", "b", "c", "c", "c"])
    want = np.array([1 / 6, 1 / 6, 1 / 3, 1 / 9, 1 / 9, 1 / 9])
    got = mt.site_weights(ids)
    worst = float(np.max(np.abs(got - want)))
    assert worst <= TIGHT, f"9.5 site weights: max |err| {worst:.3e}; got {got}"
    e = np.array([1.0, 3.0, 2.0, 0.0, 0.0, 3.0])
    # site means of e^2: a -> 5, b -> 4, c -> 3; RMSE = sqrt(12 / 3) = 2.
    close(mt.rmse(e, np.zeros(6), got), 2.0, TIGHT, "9.5 site-weighted RMSE")
    close(bs.site_weighted_rmse(e, ids), 2.0, TIGHT, "9.5 site_weighted_rmse")
    close(bs.site_weighted_bias(e, ids), (2.0 + 2.0 + 1.0) / 3.0, TIGHT, "9.5 site bias")


def test_uniform_weights_equal_the_unweighted_metrics() -> None:
    """docs/3 4.3.1: one row per site is the uniform-weight case."""
    w = np.full(4, 7.0)
    for name, fn in (("rmse", mt.rmse), ("bias", mt.bias), ("nse", mt.nse)):
        close(fn(P4, O4, w), fn(P4, O4), 1e-14, f"4.3.1 {name} uniform weights")


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: mt.rmse_log(np.zeros(2), np.zeros(3)), "equal shapes"),
        (lambda: mt.rmse_log(np.array([np.nan]), np.array([0.0])), "non-finite"),
        (lambda: mt.rmse_log(np.array([]), np.array([])), "non-empty"),
        (lambda: mt.nse_raw(np.array([1.0, -1.0]), np.array([1.0, 2.0])), "strictly positive"),
        (lambda: mt.nse(np.array([1.0, 2.0]), np.array([3.0, 3.0])), "zero variance"),
        (lambda: mt.rmse(P4, O4, np.array([1.0, -1.0, 1.0, 1.0])), "non-negative"),
        (lambda: mt.rmse(P4, O4, np.ones(3)), "expected 4"),
        (lambda: mt.r2_pearson_raw(P4, O4), "zero variance"),
        (lambda: mt.multiplicative_bias(math.inf), "non-finite"),
    ],
)
def test_metric_domain_errors(call, match) -> None:
    """docs/3 4.3.1: invalid inputs raise; nothing is clipped or imputed (docs/3 2.7)."""
    with pytest.raises(ValueError, match=match):
        call()


# ============================================================================ 4.3.2 consistency
def test_closure_residual_and_summary() -> None:
    """docs/3 4.3.2: eps_A = ln A - ln W - ln dbar; share uses |eps| > ln 1.05 (strict)."""
    ln_w, ln_d = np.zeros(4), np.zeros(4)
    ln_a = np.array([0.0, math.log(1.05), -0.2, 0.01])
    eps = mt.closure_residual(ln_a, ln_w, ln_d)
    worst = float(np.max(np.abs(eps - ln_a)))
    assert worst == 0.0, f"4.3.2 eps_A: max |err| {worst:.3e}"
    med, share = mt.closure_summary(eps)
    close(med, (0.01 + math.log(1.05)) / 2.0, TIGHT, "4.3.2 median |eps_A|")
    close(share, 0.25, TIGHT, "4.3.2 share |eps_A| > ln 1.05 (boundary excluded)")


def test_shape_violation_share_counts_equality() -> None:
    """docs/3 4.3.2: invariant 4 is violated when d_max <= dbar (equality included)."""
    got = mt.shape_violation_share(np.log([2.0, 1.0, 0.5, 3.0]), np.log([1.0, 1.0, 1.0, 1.0]))
    close(got, 0.5, TIGHT, "4.3.2 shape-invariant violation share")


def test_implied_velocity_and_oracle() -> None:
    """docs/3 4.3.2: ln V_hat = q_E - ln A_hat; oracle uses q_obs; error = dq - d ln A."""
    q_e, q_obs = np.array([1.0, 2.0]), np.array([1.5, 1.0])
    ln_a_hat, ln_a_obs = np.array([0.2, 0.5]), np.array([0.1, 0.7])
    ln_v_obs = q_obs - ln_a_obs
    err = mt.implied_log_velocity(q_e, ln_a_hat) - ln_v_obs
    want = (q_e - q_obs) - (ln_a_hat - ln_a_obs)
    worst = float(np.max(np.abs(err - want)))
    assert worst <= TIGHT, f"4.3.2 / docs/2 7.3 V-error identity: max |err| {worst:.3e}"
    orc = mt.implied_log_velocity(q_obs, ln_a_hat) - ln_v_obs
    worst = float(np.max(np.abs(orc + (ln_a_hat - ln_a_obs))))
    assert worst <= TIGHT, f"4.3.2 oracle V error: max |err| {worst:.3e}"


# ============================================================================ 4.4 bootstrap
def test_cluster_design_groups_and_strata() -> None:
    """docs/3 4.4: clusters resampled within division; a cluster cannot span two strata."""
    design = bs.cluster_design(["s1", "s2", "s1", "s3"], ["A", "B", "A", "B"])
    got = [m.tolist() for m in design.members]
    assert got == [[0, 2], [1], [3]], f"4.4 cluster members {got}"
    assert [s.tolist() for s in design.strata] == [[0], [1, 2]]
    with pytest.raises(ValueError, match="spans 2 strata"):
        bs.cluster_design(["s1", "s1"], ["A", "B"])


def test_bootstrap_b_and_seed_come_from_params_and_are_reproducible() -> None:
    """docs/3 4.4: B = 10,000 and seed 20261008 from params; same seed -> identical replicates."""
    cfg = EVAL.bootstrap
    assert (cfg.b, cfg.seed, cfg.alpha_b) == (10_000, 20261008, 0.05)
    x = np.array([0.3, -1.2, 2.5, 0.7, -0.4, 1.9])
    clusters = np.arange(6)
    strata = np.array([0, 0, 0, 1, 1, 1])

    def stat(idx, groups):
        return float(x[idx].mean())

    a = bs.paired_cluster_bootstrap(stat, clusters, strata, cfg)
    b = bs.paired_cluster_bootstrap(stat, clusters, strata, cfg)
    assert a.shape == (10_000,), f"4.4: B = {a.shape[0]}, expected 10,000"
    assert np.array_equal(a, b), "4.4: same seed must give bit-identical replicates"
    c = bs.paired_cluster_bootstrap(stat, clusters, strata, cfg.model_copy(update={"seed": 1}))
    assert not np.array_equal(a, c), "4.4: a different seed must change the replicates"


def test_stratified_draw_keeps_stratum_sizes_and_whole_clusters() -> None:
    """docs/2 9.5 / docs/3 4.4: n_s clusters per stratum; all observations of a drawn cluster."""
    clusters = np.array([0, 0, 1, 2, 2, 2, 3, 4])
    strata = np.array([7, 7, 7, 9, 9, 9, 9, 9])
    members = {0: {0, 1}, 1: {2}, 2: {3, 4, 5}, 3: {6}, 4: {7}}
    seen: list[str] = []

    def stat(idx, groups):
        drawn = []
        for g in np.unique(groups):
            obs = set(idx[groups == g].tolist())
            match = [k for k, m in members.items() if m == obs]
            assert len(match) == 1, f"9.5: group {g} is not one whole cluster: {sorted(obs)}"
            drawn.append(match[0])
        n7 = sum(k in (0, 1) for k in drawn)
        assert (n7, len(drawn) - n7) == (2, 3), f"4.4: stratum sizes drawn {(n7, len(drawn) - n7)}"
        seen.append("ok")
        return 0.0

    bs.paired_cluster_bootstrap(stat, clusters, strata, EVAL.bootstrap, n_boot=300)
    assert len(seen) == 300


def test_two_cluster_bootstrap_distribution_is_exact_in_the_limit() -> None:
    """docs/2 9.5: with 2 clusters the mean takes x1, (x1+x2)/2, x2 w.p. 1/4, 1/2, 1/4."""
    x = np.array([0.0, 1.0])

    def stat(idx, groups):
        return float(x[idx].mean())

    reps = bs.paired_cluster_bootstrap(stat, [0, 1], None, EVAL.bootstrap)
    for value, prob in ((0.0, 0.25), (0.5, 0.5), (1.0, 0.25)):
        freq = float(np.mean(reps == value))
        tol = 4.0 * math.sqrt(prob * (1 - prob) / reps.size)
        close(freq, prob, tol, f"9.5 P(mean* = {value}) (4 binomial SE, B = 10,000)")


def test_paired_statistic_cancels_shared_noise() -> None:
    """docs/2 9.5: both models on the same draw; a constant shift c gives the CI [c, c]."""
    rng = np.random.default_rng(11)
    err_star = rng.normal(size=40)
    err_m5 = err_star - 0.125
    sites = np.repeat(np.arange(20), 2)
    divs = np.repeat(np.arange(4), 10)

    def delta_bias(idx, groups):
        return bs.site_weighted_bias(err_m5[idx], groups) - bs.site_weighted_bias(
            err_star[idx], groups
        )

    reps = bs.paired_cluster_bootstrap(delta_bias, sites, divs, EVAL.bootstrap, n_boot=500)
    lo, hi = bs.percentile_ci(reps, 1 - EVAL.bootstrap.alpha_b)
    close(lo, -0.125, 1e-12, "9.5 paired CI lower bound")
    close(hi, -0.125, 1e-12, "9.5 paired CI upper bound")


def test_duplicate_draws_count_as_separate_sites() -> None:
    """docs/2 9.5: a site drawn twice enters the site average twice."""
    # sites: A = (1, 1), B = (3); draw (A, A, B) -> site means of e^2 (1, 1, 9) -> sqrt(11/3)
    err = np.array([1.0, 1.0, 1.0, 1.0, 3.0])
    groups = np.array([0, 0, 1, 1, 2])
    close(bs.site_weighted_rmse(err, groups), math.sqrt(11.0 / 3.0), TIGHT, "9.5 duplicates")


def test_percentile_ci_and_bonferroni_level() -> None:
    """docs/2 9.5 percentile CI; docs/3 4.5 Bonferroni level 1 - 0.05/4 = 0.9875 (O-7)."""
    s = np.arange(1001, dtype=np.float64)  # linear quantile of 0..1000 is 1000 q
    lo, hi = bs.percentile_ci(s, 0.95)
    close(lo, 25.0, 1e-9, "9.5 q_0.025")
    close(hi, 975.0, 1e-9, "9.5 q_0.975")
    level = bs.bonferroni_level(EVAL.s2)
    close(level, 1.0 - 0.05 / 4.0, TIGHT, "4.5 Bonferroni level")
    close(level, EVAL.s2.level, TIGHT, "4.5 level vs S2Params.level")
    lo, hi = bs.percentile_ci(s, level)
    close(lo, 6.25, 1e-9, "4.5 q_0.00625")
    close(hi, 993.75, 1e-9, "4.5 q_0.99375")


def test_two_sample_bootstrap_resamples_each_set_independently() -> None:
    """docs/2 9.5 H3: T_ID and T_OOD resampled independently, paired across models within."""
    a_vals, b_vals = np.array([0.0, 1.0]), np.array([5.0, 5.0, 5.0])

    def stat(ia, ga, ib, gb):
        assert ia.max() < 2, "H3: set-A indices out of range"
        assert ib.max() < 3, "H3: set-B indices out of range"
        return float(a_vals[ia].mean() - b_vals[ib].mean())

    reps = bs.two_sample_cluster_bootstrap(stat, [0, 1], None, [0, 1, 2], None, EVAL.bootstrap)
    again = bs.two_sample_cluster_bootstrap(stat, [0, 1], None, [0, 1, 2], None, EVAL.bootstrap)
    assert np.array_equal(reps, again), "9.5 two-sample bootstrap must be reproducible"
    close(float(np.mean(reps == -5.0)), 0.25, 4.0 * math.sqrt(0.1875 / reps.size), "9.5 H3")


def test_average_over_repeats_and_nonfinite_statistic() -> None:
    """docs/3 4.4: the S-ID statistic is averaged over the three repeats in each draw."""
    fns = [lambda i, g, k=k: float(k) for k in (1, 2, 6)]
    close(
        bs.average_over_repeats(fns)(np.zeros(1, np.int64), np.zeros(1, np.int64)),
        3.0,
        TIGHT,
        "4.4 repeat average",
    )
    with pytest.raises(ValueError, match="non-finite"):
        bs.paired_cluster_bootstrap(lambda i, g: math.nan, [0, 1], None, EVAL.bootstrap, n_boot=2)


# ============================================================================ 4.4 hypotheses
def test_select_mstar_lowest_rmse_and_tie() -> None:
    """docs/3 4.4: M* = lowest S-ID OOF RMSE_log among {M1, M2-MLP, M2-XGB}; ties raise."""
    assert EVAL.mstar_candidates == ("M1", "M2-MLP", "M2-XGB")
    got = hy.select_mstar({"M1": 0.41, "M2-MLP": 0.39, "M2-XGB": 0.40, "M5": 0.1}, EVAL)
    assert got == "M2-MLP", f"4.4 M*: {got}"
    with pytest.raises(ValueError, match="exact tie"):
        hy.select_mstar({"M1": 0.39, "M2-MLP": 0.39, "M2-XGB": 0.40}, EVAL)
    with pytest.raises(ValueError, match="missing"):
        hy.select_mstar({"M1": 0.39}, EVAL)


@pytest.mark.parametrize(
    ("lro", "ind", "want"),
    [
        ((-0.20, -0.05), (-0.10, 0.30), (True, False, False)),
        ((-0.20, 0.00), (-0.10, 0.30), (False, True, False)),  # includes 0 -> falsified
        ((0.01, 0.20), (-0.10, 0.30), (False, True, False)),
        ((-0.20, -0.01), (0.01, 0.20), (False, False, True)),  # S-IND contradiction
        ((-0.20, -0.01), (0.00, 0.20), (True, False, False)),  # lo = 0 is not entirely above
    ],
)
def test_h1_decision_table(lro, ind, want) -> None:
    """docs/3 4.4 H1: supported iff S-LRO hi < 0 and S-IND not entirely above 0."""
    r = hy.decide_h1(hy.CI(*lro), hy.CI(*ind))
    got = (r.supported, r.falsified, r.ind_contradiction)
    assert got == want, f"4.4 H1 lro={lro} ind={ind}: got {got}, want {want}"


@pytest.mark.parametrize(("ci", "supported"), [((-0.3, -1e-9), True), ((-0.3, 0.0), False)])
def test_h2_decision(ci, supported) -> None:
    """docs/3 4.4 H2: supported iff the Q1 CI upper bound < 0 (strict)."""
    r = hy.decide_h2(hy.CI(*ci))
    assert (r.supported, r.falsified) == (supported, not supported), f"4.4 H2 ci={ci}: {r}"


@pytest.mark.parametrize(
    ("lro", "tail", "want"),
    [
        ((-0.2, -0.1), (-0.3, -0.01), (True, False, False)),
        ((-0.2, -0.1), (-0.3, 0.02), (False, True, True)),
        ((-0.2, 0.1), (-0.3, 0.02), (False, True, False)),
    ],
)
def test_h3_decision_and_partial(lro, tail, want) -> None:
    """docs/3 4.4 H3: both splits hi < 0; a one-split pass is reported as partial."""
    r = hy.decide_h3(hy.CI(*lro), hy.CI(*tail))
    got = (r.supported, r.falsified, r.partial)
    assert got == want, f"4.4 H3 lro={lro} tail={tail}: got {got}, want {want}"


def test_h4_boundary_band_matches_the_protocol_text() -> None:
    """docs/3 4.4 H4 (a): vartheta = 0.05 on nu gives n < 0.0064 or n > 0.47."""
    nu_lo, nu_hi = PROTOCOL.domain.nu_bounds
    band = EVAL.h4.vartheta * (nu_hi - nu_lo)
    n_low, n_high = math.exp(nu_lo + band), math.exp(nu_hi - band)
    assert round(n_low, 4) == 0.0064, f"4.4 lower band edge n = {n_low:.6f}"
    assert round(n_high, 2) == 0.47, f"4.4 upper band edge n = {n_high:.6f}"
    nu = np.log([n_low * 0.999, n_low * 1.001, 0.05, n_high * 1.001, n_high * 0.999])
    close(hy.boundary_share(nu, (nu_lo, nu_hi), EVAL.h4.vartheta), 0.4, TIGHT, "9.5 pi_partial")


def test_boundary_share_includes_the_band_edge() -> None:
    """docs/2 9.5 H4 (a): min(x - lo, hi - x) <= vartheta Delta counts the edge (exact binary)."""
    got = hy.boundary_share([0.25, 0.75, 0.5, 0.2500001], (0.0, 1.0), 0.25)
    close(got, 0.5, TIGHT, "9.5 pi_partial at the band edge")


def test_literature_share_is_inclusive() -> None:
    """docs/3 4.4 H4 (a): pi_lit = share of n_hat in N_lit = [0.012, 0.33]."""
    got = hy.literature_share([0.012, 0.33, 0.0119, 0.331, 0.05], PROTOCOL.domain)
    close(got, 0.6, TIGHT, "4.4 pi_lit")


RHO_OK = {"n_d50": hy.CI(0.1, 0.4), "n_s": hy.CI(0.05, 0.3), "r_robs": hy.CI(0.2, 0.5)}


@pytest.mark.parametrize(
    ("pi_nu", "pi_r", "pi_lit", "rho", "reasons"),
    [
        (0.10, 0.10, 0.80, RHO_OK, ()),  # thresholds themselves do not falsify
        (0.1001, 0.0, 0.9, RHO_OK, ("pi_nu",)),
        (0.0, 0.1001, 0.9, RHO_OK, ("pi_r",)),
        (0.0, 0.0, 0.7999, RHO_OK, ("pi_lit",)),
        (0.0, 0.0, 0.9, {**RHO_OK, "n_s": hy.CI(0.0, 0.3)}, ("rho_n_s",)),
        (0.0, 0.0, 0.9, {**RHO_OK, "r_robs": hy.CI(-0.2, -0.1)}, ("rho_r_robs",)),
    ],
)
def test_h4_decision_thresholds_from_params(pi_nu, pi_r, pi_lit, rho, reasons) -> None:
    """docs/3 4.4 H4: falsified iff pi > 0.10, pi_lit < 0.80 or a rho CI has lo <= 0."""
    r = hy.decide_h4(pi_nu, pi_r, pi_lit, rho, EVAL.h4)
    assert r.reasons == reasons, f"4.4 H4: reasons {r.reasons}, want {reasons}"
    assert r.supported is (not reasons)
    assert r.falsified is bool(reasons)


def _h(supported: bool):
    return hy.H2Result(supported=supported, falsified=not supported)


@pytest.mark.parametrize(
    ("h1", "h2", "h3", "h4", "ci", "rows"),
    [
        ((True, False), False, True, False, (-0.2, -0.1), (1,)),
        ((False, True), False, False, True, (-0.01, 0.04), (2,)),
        ((False, True), False, False, True, (0.01, 0.04), (2, 4)),  # rows co-occur
        ((False, True), False, False, True, (-0.01, 0.05), ()),  # margin is strict
        ((False, True), True, False, False, (-0.01, 0.2), (3,)),
        ((False, True), True, False, True, (-0.01, 0.2), ()),  # H4 also supported -> not "only"
        ((False, True), False, False, False, (0.02, 0.3), (4,)),
    ],
)
def test_outcome_matrix_rows(h1, h2, h3, h4, ci, rows) -> None:
    """docs/3 4.7 (PI decision 2026-10-09): all matching rows; margin +0.05 from params."""
    assert EVAL.no_loss_margin_log == 0.05
    r1 = hy.H1Result(supported=h1[0], falsified=h1[1], ind_contradiction=False)
    r3 = hy.H3Result(supported=h3, falsified=not h3, partial=False)
    r4 = hy.H4Result(supported=h4, falsified=not h4, reasons=())
    got = hy.outcome_rows(r1, _h(h2), r3, r4, hy.CI(*ci), EVAL)
    assert got == rows, f"4.7 outcome rows {got}, want {rows} (ci={ci})"


def test_ci_validation() -> None:
    """docs/2 9.5: a percentile CI is finite with lo <= hi."""
    with pytest.raises(ValueError, match="invalid CI"):
        hy.CI(0.2, 0.1)
    with pytest.raises(ValueError, match="invalid CI"):
        hy.CI(math.nan, 0.1)


# ============================================================================ 4.6 / 9.3-9.4 UQ
def test_single_member_interval_is_mu_pm_z_sigma() -> None:
    """docs/2 9.3: for M = 1 the exact interval is mu -/+ z_{0.95} sigma (alpha = 0.10)."""
    alpha = PROTOCOL.uq.alpha
    mu, sigma = np.array([[0.3, -1.0]]), np.array([[0.5, 2.0]])
    lo, hi = uq.central_interval(mu, sigma, alpha)
    z = float(stats.norm.ppf(1 - alpha / 2))
    for got, want, tag in ((lo, mu[0] - z * sigma[0], "L"), (hi, mu[0] + z * sigma[0], "U")):
        worst = float(np.max(np.abs(got - want)))
        assert worst <= 1e-10, f"9.3 M = 1 {tag}: max |err| {worst:.3e} > 1e-10"


def test_mixture_quantile_against_brentq() -> None:
    """docs/2 9.3: Q_p is the unique root of Psi(x) = p; oracle = brentq on scipy.stats.norm."""
    mu = np.array([[-1.0, 0.0, 2.0], [1.0, 0.5, 2.5], [0.2, 3.0, 2.2]])
    sigma = np.array([[1.0, 0.3, 0.1], [1.0, 0.6, 0.4], [0.5, 1.5, 0.2]])
    for p in (0.05, 0.5, 0.95):
        got = uq.mixture_quantile(p, mu, sigma)
        for k in range(3):

            def f(x, k=k, p=p):
                return float(np.mean(stats.norm.cdf(x, mu[:, k], sigma[:, k]))) - p

            want = optimize.brentq(f, -20.0, 20.0, xtol=1e-14)
            close(float(got[k]), want, 1e-10, f"9.3 Q_{p} column {k}")
    sym = uq.mixture_quantile(0.5, np.array([[-1.0], [1.0]]), np.ones((2, 1)))
    close(float(sym[0]), 0.0, 1e-11, "9.3 symmetric mixture median")


def test_coverage_and_width_hand_values() -> None:
    """docs/2 9.4: C = sum w 1{L <= ln y <= U} (closed interval); W = sum w (U - L)."""
    lo, hi = np.zeros(4), np.array([1.0, 1.0, 1.0, 2.0])
    y = np.array([-0.1, 0.0, 1.0, 2.5])
    close(uq.coverage(lo, hi, y), 0.5, TIGHT, "9.4 coverage (endpoints included)")
    close(uq.coverage(lo, hi, y, [0, 1, 3, 0]), 1.0, TIGHT, "9.4 weighted coverage")
    close(uq.mean_width(lo, hi), 1.25, TIGHT, "9.4 mean log-width")
    close(uq.mean_width(lo, hi, [1, 1, 1, 3]), 1.5, TIGHT, "9.4 weighted mean log-width")


def test_coverage_band_from_params_is_inclusive() -> None:
    """docs/3 4.6: acceptance band [0.85, 0.95] (quality target, not a hypothesis)."""
    band = PROTOCOL.uq.coverage_band
    assert band == (0.85, 0.95)
    hits = [uq.coverage_in_band(c, band) for c in (0.8499, 0.85, 0.90, 0.95, 0.9501)]
    assert hits == [False, True, True, True, False], f"4.6 band membership {hits}"


def test_pit_is_half_at_the_median_and_uniform_under_calibration() -> None:
    """docs/2 9.4: u = Psi(ln y) is uniform under calibration (KS at the 1 % level)."""
    close(float(uq.pit([0.7], [[0.7]], [[0.3]])[0]), 0.5, TIGHT, "9.4 PIT at the median")
    rng = np.random.default_rng(20261008)
    n = 20_000
    mu = np.vstack([np.full(n, -1.0), np.full(n, 1.5)])
    sigma = np.vstack([np.full(n, 0.5), np.full(n, 1.0)])
    comp = rng.integers(0, 2, size=n)
    y = rng.normal(mu[comp, np.arange(n)], sigma[comp, np.arange(n)])
    ks = float(stats.kstest(uq.pit(y, mu, sigma), "uniform").statistic)
    crit = 1.628 / math.sqrt(n)
    assert ks <= crit, f"9.4 PIT uniformity: KS {ks:.4f} > 1 % critical value {crit:.4f}"


def _crps_quad(y: float, mu: np.ndarray, sigma: np.ndarray) -> float:
    def cdf(x: float) -> float:
        return float(np.mean(stats.norm.cdf(x, mu, sigma)))

    lo, hi = float((mu - 12 * sigma).min()), float((mu + 12 * sigma).max())
    left = integrate.quad(lambda x: cdf(x) ** 2, lo, y, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
    right = integrate.quad(
        lambda x: (1.0 - cdf(x)) ** 2, y, hi, epsabs=1e-13, epsrel=1e-13, limit=200
    )[0]
    return left + right


def test_gaussian_crps_closed_form() -> None:
    """docs/2 9.4: M = 1 CRPS = sigma[z(2Phi - 1) + 2phi - pi^-1/2]; z = 0: sigma(rt2 - 1)/rtpi."""
    close(
        float(uq.crps_gaussian([1.0], [1.0], [2.0])[0]),
        2.0 * (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi),
        TIGHT,
        "9.4 CRPS at z = 0",
    )
    for y in (-2.0, 0.4, 3.0):
        got = float(uq.crps_gaussian([y], [0.5], [0.7])[0])
        close(got, _crps_quad(y, np.array([0.5]), np.array([0.7])), 1e-9, f"9.4 CRPS y={y}")


def test_mixture_crps_against_quadrature_and_m1_reduction() -> None:
    """docs/2 9.4: Gaussian-mixture closed form vs the integral of (Psi - 1{x >= y})^2."""
    mu = np.array([[-1.0], [0.5], [2.0]])
    sigma = np.array([[0.4], [1.0], [0.3]])
    for y in (-1.5, 0.0, 2.2, 5.0):
        got = float(uq.crps_mixture([y], mu, sigma)[0])
        close(got, _crps_quad(y, mu[:, 0], sigma[:, 0]), 1e-9, f"9.4 mixture CRPS y={y}")
    y = np.array([0.1, -0.3])
    m1 = uq.crps_mixture(y, [[0.2, 0.4]], [[0.5, 1.1]])
    g = uq.crps_gaussian(y, [0.2, 0.4], [0.5, 1.1])
    worst = float(np.max(np.abs(m1 - g)))
    assert worst <= 1e-14, f"9.4 M = 1 reduction: max |err| {worst:.3e}"


def test_folded_normal_mean_against_quadrature() -> None:
    """docs/2 9.4: A(mu, s^2) = E|Y| for Y ~ N(mu, s^2)."""
    for m, s in ((0.0, 1.0), (1.3, 0.4), (-2.0, 3.0)):
        want = integrate.quad(
            lambda x, m=m, s=s: abs(x) * stats.norm.pdf(x, m, s),
            m - 15 * s,
            m + 15 * s,
            points=[0.0],
            epsabs=1e-13,
        )[0]
        close(float(uq.folded_normal_mean(m, s * s)), want, 1e-10, f"9.4 A({m}, {s}^2)")


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: uq.mixture_cdf([0.0], [[0.0]], [[0.0]]), "> 0"),
        (lambda: uq.mixture_quantile(1.0, [[0.0]], [[1.0]]), r"\(0, 1\)"),
        (lambda: uq.coverage([1.0], [0.0], [0.5]), "lower bound above"),
        (lambda: uq.mixture_cdf([0.0, 1.0], [[0.0]], [[1.0]]), "finite"),
    ],
)
def test_uq_domain_errors(call, match) -> None:
    """docs/2 8.5 / 9.3: sigma_m > 0, p in (0, 1), L <= U; nothing is clamped."""
    with pytest.raises(ValueError, match=match):
        call()


# ============================================================================ 3.6 audit
def _line_w(n: int) -> sparse.csr_array:
    """Row-standardized binary contiguity on a line of n sites (hand lattice)."""
    a = np.zeros((n, n))
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    return sparse.csr_array(a / a.sum(axis=1, keepdims=True))


def test_global_moran_line_lattice_hand_value() -> None:
    """docs/3 3.6.2: z = (1,2,3,4) on a line, row-standardized: I = (4/4) * 2/5 = 0.4."""
    r = da.moran_global([1.0, 2.0, 3.0, 4.0], _line_w(4), AUDIT.permutations, 5)
    close(r.i, 0.4, TIGHT, "3.6.2 Moran's I hand lattice")
    close(r.expected, -1.0 / 3.0, TIGHT, "3.6.2 E[I] = -1/(N-1)")
    close(r.s0, 4.0, TIGHT, "3.6.2 S0")


def test_global_moran_pseudo_p_matches_exact_permutation_distribution() -> None:
    """docs/3 3.6.2: one-sided pseudo-p (1 + #{I* >= I})/1000 vs exact enumeration of 5! perms."""
    z = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
    w = _line_w(5)
    zc = z - z.mean()
    i_obs = 5.0 / float(w.sum()) * float(zc @ (w @ zc)) / float(zc @ zc)
    sims = []
    for perm in itertools.permutations(zc):
        v = np.array(perm)
        sims.append(5.0 / float(w.sum()) * float(v @ (w @ v)) / float(zc @ zc))
    p_exact = float(np.mean(np.array(sims) >= i_obs - 1e-12))
    r = da.moran_global(z, w, AUDIT.permutations, 20261008)
    close(r.i, i_obs, TIGHT, "3.6.2 I (dense formula)")
    want = (1.0 + AUDIT.permutations * p_exact) / (AUDIT.permutations + 1.0)
    tol = 4.0 * math.sqrt(p_exact * (1 - p_exact) / AUDIT.permutations) + 1e-3
    close(r.pseudo_p, want, tol, "3.6.2 one-sided pseudo-p vs exact permutation law")


def test_global_moran_counts_ties_in_the_pseudo_p() -> None:
    """docs/3 3.6.2: #{I* >= I} includes ties; z = 1..4 on a line has exact P(I* >= I) = 2/24.

    The observed order and its reversal both attain the maximum I, so a strict ``>`` would give
    the minimum pseudo-p 1/1000 instead of about (1 + 999/12)/1000.
    """
    r = da.moran_global([1.0, 2.0, 3.0, 4.0], _line_w(4), AUDIT.permutations, 20261008)
    p_exact = 2.0 / 24.0
    want = (1.0 + AUDIT.permutations * p_exact) / (AUDIT.permutations + 1.0)
    tol = 4.0 * math.sqrt(p_exact * (1 - p_exact) / AUDIT.permutations)
    close(r.pseudo_p, want, tol, "3.6.2 pseudo-p with tied permutations (4 binomial SE)")


def test_global_moran_matches_esda_on_knn8() -> None:
    """docs/3 3.6.2 W1 (kNN-8, row-standardized): own I equals esda.Moran.I (P7-D11 (c))."""
    rng = np.random.default_rng(7)
    xy = rng.uniform(0.0, 400.0, size=(80, 2))
    z = rng.normal(size=80) + 0.01 * xy[:, 0]
    w1 = da.knn_w(xy, AUDIT.knn_k)
    assert AUDIT.knn_k == 8
    card = np.asarray((w1.sparse != 0).sum(axis=1)).ravel()
    assert bool((card == 8).all()), f"3.6.2 W1 cardinalities {np.unique(card)}"
    rows = np.asarray(w1.sparse.sum(axis=1)).ravel()
    worst = float(np.max(np.abs(rows - 1.0)))
    assert worst <= 1e-14, f"3.6.2 W1 row sums: max |err| {worst:.3e}"
    got = da.moran_global(z, w1.sparse, 99, 1).i
    close(got, float(Moran(z, w1, permutations=0).I), 1e-12, "3.6.2 I vs esda.Moran")


def test_idw_weights_hand_values_and_isolates() -> None:
    """docs/3 3.6.2 W2: 1/d within 100 km, row-standardized; isolates keep a zero row."""
    xy = np.array([[0.0, 0.0], [30.0, 0.0], [0.0, 40.0], [500.0, 500.0]])
    w2, iso = da.idw_weights(xy, AUDIT.idw_band_km)
    dense = w2.toarray()
    want0 = np.array([0.0, 4.0 / 7.0, 3.0 / 7.0, 0.0])  # (1/30, 1/40) normalized
    worst = float(np.max(np.abs(dense[0] - want0)))
    assert worst <= TIGHT, f"3.6.2 W2 row 0: max |err| {worst:.3e}; row {dense[0]}"
    assert iso.tolist() == [3], f"3.6.2 isolates {iso.tolist()}"
    assert float(dense[3].sum()) == 0.0
    # docs/3 13.9 (supersedes the Phase 7 refusal): co-located pairs get weight 0 and are counted
    w2c, isoc = da.idw_weights(np.array([[0.0, 0.0], [0.0, 0.0]]), 100.0)
    assert w2c.nnz == 0
    assert isoc.tolist() == [0, 1]
    assert da.colocated_pairs(np.array([[0.0, 0.0], [0.0, 0.0]])) == 1


def test_correlogram_classes_are_lower_open_upper_closed() -> None:
    """docs/3 3.6.2: classes (0,10], (10,25], (25,50], ...; d = 10 belongs to (0, 10]."""
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [35.0, 0.0]])
    nnz = [da.band_weights(xy, lo, hi).nnz for lo, hi in ((0, 10), (10, 25), (25, 50))]
    assert nnz == [2, 2, 2], f"3.6.2 class membership (symmetric nnz) {nnz}"
    res = da.correlogram([1.0, 2.0, 4.0], xy, AUDIT, 3)
    assert [(lo, hi) for lo, hi, _ in res] == [
        (0.0, 10.0),
        (10.0, 25.0),
        (25.0, 50.0),
        (50.0, 100.0),
        (100.0, 250.0),
        (250.0, 500.0),
    ]
    assert [r is None for _, _, r in res] == [False, False, False, True, True, True]


def test_lisa_protocol_scale_hand_lattice() -> None:
    """docs/3 3.6.3: z = 1..5 on a line, m2 = 2: I_i = (1, 0.5, 0, 0.5, 1); quadrants."""
    w = W({0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]})
    r = da.lisa(np.arange(1.0, 6.0), w, AUDIT, 9)
    want = np.array([1.0, 0.5, 0.0, 0.5, 1.0])
    worst = float(np.max(np.abs(r.local_i - want)))
    assert worst <= TIGHT, f"3.6.3 I_i: max |err| {worst:.3e}; got {r.local_i}"
    assert r.quadrant.tolist() == ["LL", "LL", "LL", "HH", "HH"], r.quadrant.tolist()


def test_lisa_matches_dense_formula_and_is_reproducible() -> None:
    """docs/3 3.6.3 on W1: I_i = (z_i - zbar)/m2 sum_j w_ij (z_j - zbar); seeded pseudo-p."""
    rng = np.random.default_rng(5)
    xy = rng.uniform(0.0, 300.0, size=(60, 2))
    z = rng.normal(size=60)
    w1 = da.knn_w(xy, AUDIT.knn_k)
    r = da.lisa(z, w1, AUDIT, 20261008)
    zc = z - z.mean()
    lag = w1.sparse @ zc
    want = zc / float(np.mean(zc * zc)) * lag
    worst = float(np.max(np.abs(r.local_i - want)))
    assert worst <= 1e-12, f"3.6.3 I_i vs dense formula: max |err| {worst:.3e}"
    quad = np.where(zc > 0, np.where(lag > 0, "HH", "HL"), np.where(lag > 0, "LH", "LL"))
    assert r.quadrant.tolist() == quad.tolist(), "3.6.3 quadrant labels"
    assert bool(((r.pseudo_p >= 1 / 1000) & (r.pseudo_p <= 1)).all())
    assert np.array_equal(r.significant, r.pseudo_p < AUDIT.lisa_alpha)
    again = da.lisa(z, w1, AUDIT, 20261008)
    assert np.array_equal(r.pseudo_p, again.pseudo_p), "3.6.3 LISA must be reproducible"


def test_empirical_variogram_hand_values() -> None:
    """docs/3 3.6.4: 4 sites on a line (0,10,20,30 km), e = (0,1,3,6), 3 equal-count classes."""
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    vg = da.empirical_variogram([0.0, 1.0, 3.0, 6.0], xy, AUDIT.model_copy(update={"vgm_bins": 3}))
    # pairs sorted by (d, i, j): (0,1) .5, (1,2) 2 | (2,3) 4.5, (0,2) 4.5 | (1,3) 12.5, (0,3) 18
    for got, want, tag in ((vg.h, [10, 15, 25], "h"), (vg.gamma, [1.25, 4.5, 15.25], "gamma")):
        worst = float(np.max(np.abs(got - np.array(want))))
        assert worst <= TIGHT, f"3.6.4 {tag}: got {got}, want {want}"
    assert vg.n_pairs.tolist() == [2, 2, 2]


def test_empirical_variogram_tie_rule_on_a_square() -> None:
    """docs/3 3.6.4 (ADL-106 tie rule): equal distances ordered by (i, j) before equal-count split.

    Unit square of side 10 km, e = (0, 1, 3, 6): the four sides tie at 10 km. Order (d, i, j):
    (0,1) .5, (0,3) 18 | (1,2) 2, (2,3) 4.5 | (0,2) 4.5, (1,3) 12.5.
    """
    xy = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]])
    vg = da.empirical_variogram([0.0, 1.0, 3.0, 6.0], xy, AUDIT.model_copy(update={"vgm_bins": 3}))
    want = np.array([9.25, 3.25, 8.5])
    worst = float(np.max(np.abs(vg.gamma - want)))
    assert worst <= TIGHT, f"3.6.4 gamma {vg.gamma}, want {want}"
    close(float(vg.h[2]), 10.0 * math.sqrt(2.0), TIGHT, "3.6.4 diagonal lag")


def _vg(name: str, c0: float, c: float, a: float) -> da.Variogram:
    h = np.linspace(12.5, 487.5, AUDIT.vgm_bins)
    if name == "spherical":
        u = np.minimum(h / a, 1.0)
        g = c0 + c * (1.5 * u - 0.5 * u**3)
    else:
        g = c0 + c * (1.0 - np.exp(-h / a))
    return da.Variogram(h=h, gamma=g, n_pairs=np.full(h.size, 100, dtype=np.int64))


def test_variogram_fit_recovers_spherical_and_practical_range() -> None:
    """docs/3 3.6.4: WLS fit; range at 95 % of the total sill: 1.5u - 0.5u^3 = 0.94 for c0=.2."""
    fits = da.fit_variogram(_vg("spherical", 0.2, 1.0, 150.0), RULES)
    sel = da.select_variogram(fits, RULES)
    assert sel.best.name == "spherical", f"3.6.4 selected {sel.best.name}"
    for got, want, tag in (
        (sel.best.c0, 0.2, "c0"),
        (sel.best.c, 1.0, "c"),
        (sel.best.a, 150.0, "a"),
    ):
        close(got, want, 1e-6 * max(1.0, want), f"3.6.4 spherical {tag}")
    q = (0.95 * 1.2 - 0.2) / 1.0
    roots = np.roots([-0.5, 0.0, 1.5, -q])
    u = float(next(r.real for r in roots if abs(r.imag) < 1e-12 and 0 < r.real < 1))
    close(sel.rho_hat, 150.0 * u, 1e-4, "3.6.4 practical range (total sill)")
    close(sel.nugget_sill, 0.2 / 1.2, 1e-6, "3.6.4 nugget/sill ratio")
    assert not sel.unstructured


def test_exponential_practical_range_closed_forms() -> None:
    """docs/3 3.6.4 (PI decision): exp range = a ln(c / (0.05 (c0 + c))); no nugget: a ln 20."""
    frac = RULES.range_sill_frac
    f0 = da.VgmFit("exponential", 0.0, 1.0, 40.0, 0.0)
    close(da.practical_range(f0, frac), 40.0 * math.log(20.0), 1e-10, "3.6.4 a ln 20")
    f1 = da.VgmFit("exponential", 0.2, 1.0, 40.0, 0.0)
    close(da.practical_range(f1, frac), 40.0 * math.log(1.0 / 0.06), 1e-10, "3.6.4 with nugget")
    f2 = da.VgmFit("spherical", 0.0, 1.0, 100.0, 0.0)
    close(da.practical_range(f2, frac), 100.0 * 0.8114, 1e-2, "3.6.4 spherical ~0.811 a")
    assert da.practical_range(da.VgmFit("spherical", 1.0, 0.01, 100.0, 0.0), frac) == 0.0
    fits = da.fit_variogram(_vg("exponential", 0.0, 2.0, 60.0), RULES)
    close(fits["exponential"].a, 60.0, 1e-5, "3.6.4 exponential fit a")


@pytest.mark.parametrize(("nugget_sse", "unstructured"), [(1.05, True), (1.0500001, False)])
def test_nugget_rule_boundary(nugget_sse, unstructured) -> None:
    """docs/3 3.6.4: pure nugget within 5 % of the best weighted SSE -> rho_hat = 0."""
    fits = {
        "spherical": da.VgmFit("spherical", 0.0, 1.0, 100.0, 1.0),
        "nugget": da.VgmFit("nugget", 0.5, 0.0, math.nan, nugget_sse),
    }
    sel = da.select_variogram(fits, RULES)
    assert sel.unstructured is unstructured, f"3.6.4 nugget SSE {nugget_sse}: {sel}"
    assert (sel.rho_hat == 0.0) is unstructured


def test_dependence_radius_and_far_stratum() -> None:
    """docs/3 3.6.4: rho* = min(rho_hat, 250 km); far iff no training site within rho*."""
    assert da.dependence_radius(300.0, AUDIT) == 250.0
    assert da.dependence_radius(80.0, AUDIT) == 80.0
    train = np.array([[0.0, 0.0]])
    test = np.array([[50.0, 0.0], [50.0001, 0.0]])
    assert da.far_mask(test, train, 50.0).tolist() == [False, True]
    close(da.median_nn_distance(test, train), 50.00005, 1e-9, "3.6.7 d~_NN")


@pytest.mark.parametrize(
    ("i", "p", "rho", "dnn", "share", "want"),
    [
        (0.10, 0.009, 0.0, 10.0, 0.0, (True, False)),
        (0.0999, 0.009, 0.0, 10.0, 0.0, (False, False)),
        (0.30, 0.01, 0.0, 10.0, 0.0, (False, False)),  # p must be < 0.01
        (0.0, 0.5, 10.0, 10.0, 0.0, (False, False)),  # rho_hat must exceed d~_NN
        (0.0, 0.5, 10.1, 10.0, 0.0, (True, False)),
        (0.0, 0.5, 0.0, 10.0, 0.20, (False, True)),  # at least 20 %
        (0.0, 0.5, 0.0, 10.0, 0.1999, (False, False)),
    ],
)
def test_audit_label_rule_boundaries(i, p, rho, dnn, share, want) -> None:
    """docs/3 3.6.7: D = (p < 0.01 and I >= 0.10) or rho_hat > d~_NN; N = share >= 0.20."""
    m = da.MoranResult(i=i, expected=-0.01, z_perm=0.0, pseudo_p=p, s0=1.0)
    lab = da.audit_label(m, rho, dnn, share, AUDIT, RULES)
    assert (lab.d, lab.n) == want, f"3.6.7 (D, N) = {(lab.d, lab.n)}, want {want}"
    assert lab.optimistic is (want[0] or want[1])


def test_nested_basin_hand_graph() -> None:
    """docs/3 3.6.5: reachability both ways on the flow graph, same terminal path, psi >= 0.5."""
    edges = np.array([[1, 2], [2, 3], [5, 3], [3, 4], [4, 0], [7, 8], [8, 3]])
    terminal = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 7: 2, 8: 2, 9: 3}
    area = {1: 10.0, 2: 30.0, 3: 100.0, 4: 120.0, 5: 40.0, 7: 5.0, 8: 50.0, 9: 20.0}
    r = da.nested_status([1, 5, 7, 4], [3, 2, 8, 9], edges, terminal, area, AUDIT)
    # 3: max(10/100, 40/100, 100/120) (7 excluded: other terminal path); 2: max(10/30, 30/120)
    # 8: 5/50 (4 excluded: other terminal path); 9: isolated
    want = np.array([100.0 / 120.0, 10.0 / 30.0, 0.1, 0.0])
    worst = float(np.max(np.abs(r.psi_max - want)))
    assert worst <= TIGHT, f"3.6.5 psi_max {r.psi_max}, want {want}"
    assert r.strong.tolist() == [True, False, False, False]
    assert r.weak.tolist() == [False, True, True, False]
    close(r.share_strong, 0.25, TIGHT, "3.6.5 strongly nested share")


def test_aoa_dissimilarity_index_hand_values() -> None:
    """docs/3 3.6.6: DI = nearest-training distance / mean pairwise training distance."""
    tr = np.array([[0.0], [1.0], [2.0], [3.0]])  # mean pairwise distance 10/6 (scale cancels)
    close(float(da.dissimilarity_index(tr, [[5.0]], [1.0])[0]), 1.2, TIGHT, "3.6.6 DI")
    cv = da.training_cv_di(tr, [1.0], [0, 0, 1, 1])
    worst = float(np.max(np.abs(cv - np.array([1.2, 0.6, 0.6, 1.2]))))
    assert worst <= TIGHT, f"3.6.6 CV DI {cv}"
    tr2 = np.array([[0.0, 0.0], [1.0, 5.0], [2.0, 1.0]])
    di = da.dissimilarity_index(tr2, [[1.0, 100.0]], [1.0, 0.0])
    close(float(di[0]), 0.0, TIGHT, "3.6.6 zero importance removes a predictor")
    close(da.aoa_threshold([1.0, 2.0, 3.0, 4.0, 100.0], RULES), 7.0, TIGHT, "3.6.6 Q3 + 1.5 IQR")


# ============================================================================ input contracts
XY3 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
CFG_B = EVAL.bootstrap


@pytest.mark.parametrize(
    ("call", "exc", "match"),
    [
        (lambda: bs.cluster_design([]), ValueError, "non-empty"),
        (lambda: bs.cluster_design([0, 1], [0]), ValueError, "strata shape"),
        (
            lambda: bs.paired_cluster_bootstrap(lambda i, g: 0.0, [0], None, CFG_B, n_boot=0),
            ValueError,
            ">= 1",
        ),
        (
            lambda: bs.two_sample_cluster_bootstrap(
                lambda a, b, c, d: 0.0, [0], None, [0], None, CFG_B, n_boot=0
            ),
            ValueError,
            ">= 1",
        ),
        (lambda: bs.percentile_ci([], 0.95), ValueError, "non-empty"),
        (lambda: bs.percentile_ci([1.0], 1.0), ValueError, r"\(0, 1\)"),
        (lambda: bs.site_weighted_rmse([1.0, 2.0], [0]), ValueError, "equal non-empty"),
        (lambda: bs.site_weighted_bias([math.nan], [0]), ValueError, "non-finite"),
        (lambda: bs.average_over_repeats([]), ValueError, "at least one"),
        (lambda: da.knn_w(np.zeros((2, 3)), 1), ValueError, r"\(N, 2\)"),
        (lambda: da.knn_w(XY3, 3), ValueError, "1 <= k < N"),
        (lambda: da.band_weights(XY3, 5.0, 5.0), ValueError, "lower < upper"),
        (lambda: da.moran_global([1.0, 2.0], _line_w(3), 9, 0), ValueError, r"\(3,\)"),
        (lambda: da.moran_global([1.0, 2.0], sparse.csr_array((2, 2)), 9, 0), ValueError, "S0"),
        (lambda: da.moran_global([1.0, 1.0, 1.0], _line_w(3), 9, 0), ValueError, "zero var"),
        (lambda: da.moran_global([1.0, 2.0, 4.0], _line_w(3), 0, 0), ValueError, ">= 1"),
        (
            lambda: da.lisa([1.0, 1.0, 1.0], W({0: [1], 1: [0, 2], 2: [1]}), AUDIT, 0),
            ValueError,
            "zero var",
        ),
        (lambda: da.empirical_variogram([0.0, 1.0, 2.0], XY3, AUDIT), ValueError, "pairs within"),
        (lambda: da.vgm_model("spherical", [1.0], 0.0, 1.0, 0.0), ValueError, "a must be"),
        (lambda: da.vgm_model("gaussian", [1.0], 0.0, 1.0, 1.0), ValueError, "unknown"),
        (lambda: da.select_variogram({}, RULES), ValueError, "no fitted"),
        (lambda: da.dependence_radius(-1.0, AUDIT), ValueError, ">= 0"),
        (
            lambda: da.nested_status(
                [1], [2], np.array([1, 2]), {1: 1, 2: 1}, {1: 1.0, 2: 1.0}, AUDIT
            ),
            ValueError,
            r"\(E, 2\)",
        ),
        (
            lambda: da.nested_status([1], [2], np.array([[1, 2]]), {1: 1}, {1: 1.0, 2: 1.0}, AUDIT),
            ValueError,
            "COMID 2",
        ),
        (lambda: da.dissimilarity_index([[0.0]], [[1.0]], [1.0]), ValueError, "N >= 2"),
        (
            lambda: da.dissimilarity_index([[0.0], [math.nan]], [[1.0]], [1.0]),
            ValueError,
            "non-finite",
        ),
        (
            lambda: da.dissimilarity_index([[0.0], [1.0]], [[1.0]], [-1.0]),
            ValueError,
            "non-negative",
        ),
        (lambda: da.dissimilarity_index([[0.0], [0.0]], [[1.0]], [1.0]), ValueError, "constant"),
        (
            lambda: da.dissimilarity_index([[0.0], [1.0]], [[1.0, 2.0]], [1.0]),
            ValueError,
            "wrong shape",
        ),
        (lambda: da.training_cv_di([[0.0], [1.0]], [1.0], [0, 0]), ValueError, "two folds"),
        (lambda: da.aoa_threshold([], RULES), ValueError, "non-empty"),
        (
            lambda: hy.select_mstar({"M1": math.nan, "M2-MLP": 0.1, "M2-XGB": 0.2}, EVAL),
            ValueError,
            "non-finite",
        ),
        (lambda: hy.boundary_share([], (0.0, 1.0), 0.05), ValueError, "non-empty"),
        (lambda: hy.boundary_share([0.5], (1.0, 0.0), 0.05), ValueError, "lo < hi"),
        (lambda: hy.literature_share([math.nan], PROTOCOL.domain), ValueError, "finite"),
        (
            lambda: hy.decide_h4(0.0, 0.0, 1.0, {"n_d50": hy.CI(0.1, 0.2)}, EVAL.h4),
            ValueError,
            "keys",
        ),
        (lambda: hy.decide_h4(1.5, 0.0, 1.0, RHO_OK, EVAL.h4), ValueError, "share"),
        (lambda: mt.site_weights([]), ValueError, "non-empty"),
        (lambda: mt.closure_residual([0.0], [0.0], [0.0, 1.0]), ValueError, "equal shapes"),
        (lambda: uq.mixture_cdf([0.0], [0.0], [1.0]), ValueError, r"\(M, n\)"),
        (lambda: uq.mixture_cdf([0.0], [[math.inf]], [[1.0]]), ValueError, "non-finite"),
        (lambda: uq.coverage([0.0], [1.0], [0.5], [-1.0]), ValueError, "non-negative"),
        (lambda: uq.coverage([0.0], [1.0], [0.5], [0.0]), ValueError, "positive sum"),
        (
            lambda: uq.mixture_quantile(0.3, [[0.0], [5.0]], [[1.0], [1.0]], max_iter=1),
            RuntimeError,
            "did not converge",
        ),
        (lambda: uq.central_interval([[0.0]], [[1.0]], 0.0), ValueError, r"\(0, 1\)"),
        (lambda: uq.mean_width([1.0], [0.0]), ValueError, "lower bound above"),
        (lambda: uq.folded_normal_mean(0.0, 0.0), ValueError, "> 0"),
    ],
)
def test_evaluation_input_contracts(call, exc, match) -> None:
    """docs/3 2.7 / 4.3-4.6 and docs/2 9.3-9.5: invalid inputs raise; nothing is clamped."""
    with pytest.raises(exc, match=match):
        call()


def test_nugget_model_and_range_and_cluster_count() -> None:
    """docs/3 3.6.4: the pure-nugget model is flat at c0 and has practical range 0."""
    flat = da.vgm_model("nugget", [5.0, 50.0], 0.3, 0.0, math.nan)
    assert flat.tolist() == [0.3, 0.3]
    assert da.practical_range(da.VgmFit("nugget", 0.3, 0.0, math.nan, 0.0), 0.95) == 0.0
    assert bs.cluster_design(["a", "b", "a"]).n_clusters == 2


# ============================================================================ locks and imports
def test_cli_event_3_is_wired_to_the_event_3_library() -> None:
    """docs/3 13.4: ``evaluate.py evaluate`` dispatches to Event 3 (the C3b lock lifted in C3c).

    Supersedes the C3b lock test. Event 3 is never executed here: in this repository it is a
    real, one-time scoring event. Its refusals and CLI plumbing run on a synthetic repository in
    ``tests/unit/test_evaluation_events.py``; the computation in ``test_evaluation_event3.py``.
    """
    ns = runpy.run_path(str(REPO_ROOT / "scripts" / "evaluate.py"), run_name="not_main")
    assert ns["_COMMANDS"]["evaluate"] is ns["cmd_evaluate"]
    assert "LOCK_MESSAGE" not in ns
    assert len(set(ns["EVENT3_PREDICTIONS"])) == 20


def test_evaluation_package_imports_neither_torch_nor_xgboost() -> None:
    """ADL-051 / P7-D11 (c): src.evaluation (esda, libpysal included) loads no torch/xgboost."""
    code = (
        "import sys; import src.evaluation.metrics, src.evaluation.bootstrap, "
        "src.evaluation.hypotheses, src.evaluation.uq, src.evaluation.dependence_audit; "
        "print('torch' in sys.modules, 'xgboost' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    ).stdout.strip()
    assert out == "False False", f"ADL-051: torch/xgboost imported by src.evaluation: {out}"

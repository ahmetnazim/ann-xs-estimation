"""Event 3: hypotheses, metrics matrix, UQ, consistency, dependence and descriptives (docs/3 13.4).

Sources: docs/3 section 13.4 (Event 3 = ``scripts/evaluate.py evaluate``: everything else in
protocol 4.3-4.6, 4.9, 3.6.1 (c), the 3.6.5 strata, 3.6.6, S1 and S-RND, then the 4.7 row computed
mechanically; anything outside protocol 4 and docs/3 13 is labelled exploratory), sections 13.5
(S-IND), 13.6 (S-RND), 13.8 (ablations and SAs), 13.9 (Event 3 inputs), 13.12 (Event 1 strata and
the |Delta PathLength| / LevelPathI descriptives) and 13.13 (Event 3 computation rules, Batch C3c);
protocol 3.6.5-3.6.7, 4.1-4.7; proofs 9.4-9.5 and Proposition 7.6.

Decisions are taken only by :mod:`src.evaluation.hypotheses` (Phase 7 known-answer tested) on the
intervals computed here. Two numerics run in isolated workers supplied by the caller: ``ln kappa``
(torch; S2 aspect strata and the A1 map) and the AOA blocks (xgboost); see
:mod:`src.evaluation.event3_workers`. Torch-free and xgboost-free. Owning phase: 8.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order

from src.config import ProtocolParams
from src.data.folds import analysis_frame, get_view
from src.data.network import NetworkInputs
from src.evaluation import dependence_audit as da
from src.evaluation import event1
from src.evaluation import event3_sets as es
from src.evaluation import event3_stats as st
from src.evaluation import hypotheses as hy
from src.evaluation import metrics as mt
from src.evaluation.bootstrap import bonferroni_level

__all__ = [
    "A4_MODELS",
    "LADDER",
    "M5_ABLATIONS",
    "OUTCOME_TEXT",
    "SA_MODELS",
    "SIGMA_MODELS",
    "STRATUM_MIN_SITES",
    "AoaFn",
    "Event3Inputs",
    "Event3Result",
    "LogKappa",
    "nested_partners",
    "run_event3",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
LogKappa = Callable[[FloatArray, FloatArray, FloatArray], FloatArray]
AoaFn = Callable[[Sequence[tuple[str, int, int]]], list[dict[str, Any]]]

#: Protocol 4.1 ladder (primary analysis, no ablation).
LADDER: Final[tuple[str, ...]] = (
    "M-null",
    "A_J",
    "M0",
    "M1",
    "M2-MLP",
    "M2-XGB",
    "M3",
    "M4",
    "M5",
    "M6-NN",
    "M6-XGB",
)
#: Models with a predictive distribution (protocol 4.2 UQ; docs/3 13.3: XGB models have none).
SIGMA_MODELS: Final[tuple[str, ...]] = ("M2-MLP", "M4", "M5")
M5_ABLATIONS: Final[tuple[str, ...]] = ("A1", "A2", "A3-SP", "A3-SR", "A4", "wA1")
#: docs/3 13.8: A4 on all three M* candidates.
A4_MODELS: Final[tuple[str, ...]] = ("M1", "M2-MLP", "M2-XGB")
#: docs/3 13.8: SA1 for M5 only; SA2-SA6 for M5, M2-MLP, M1 and M2-XGB.
SA_MODELS: Final[Mapping[str, tuple[str, ...]]] = {
    "M5": ("SA1", "SA2", "SA3", "SA4", "SA5", "SA6"),
    "M2-MLP": ("SA2", "SA3", "SA4", "SA5", "SA6"),
    "M1": ("SA2", "SA3", "SA4", "SA5", "SA6"),
    "M2-XGB": ("SA2", "SA3", "SA4", "SA5", "SA6"),
}
UQ_TARGETS: Final[tuple[str, ...]] = ("dbar", "W", "t", "A")
#: docs/3 13.13 k: a far / not-nested stratum CI needs this many sites in every repeat.
STRATUM_MIN_SITES: Final[int] = 30
#: Protocol 4.7, verbatim (outcome, claim, venue).
OUTCOME_TEXT: Final[Mapping[int, tuple[str, str, str]]] = {
    1: (
        "H1 and H3 supported (M5 ≈ M2 in-distribution, better OOD and on S-IND)",
        "Physics constraints improve transferability of national channel geometry",
        "escalate to WRR / HESS permitted",
    ),
    2: (
        "H1 falsified, H4 supported, and accuracy not worse, defined as an S-LRO 95 % CI upper "
        "bound of Δ^H1_dbar below +0.05 log units (≈ 5 %)",
        "Hydraulically consistent geometry, shape, roughness and velocity at no loss of accuracy",
        "ASCE primary",
    ),
    3: (
        "H2 supported only",
        "Physics constraints reduce small-stream depth bias",
        "ASCE / JAWRA",
    ),
    4: (
        "M5 worse (S-LRO CI lower bound > 0)",
        "Diagnose with the pre-registered ablations (A1, A3, M4, SA1, SA5) and report. No "
        "post-hoc headline switch: any new variant is labelled exploratory",
        "ASCE; fallback per ADL-014",
    ),
}


@dataclass(frozen=True)
class Event3Inputs:
    """Frozen tables, sealed predictions (by file stem) and Event 1/2 outputs."""

    preds: Mapping[str, pd.DataFrame]
    site_table: pd.DataFrame
    folds: pd.DataFrame
    ehydro: pd.DataFrame
    ind_features: pd.DataFrame
    srnd: pd.DataFrame
    audit_strata: pd.DataFrame
    audit: Mapping[str, Any]
    mstar: Mapping[str, Any]
    hyg: pd.DataFrame
    hyg_counts: Mapping[str, int]
    qc_rows: pd.DataFrame
    network: NetworkInputs
    pathlength: Mapping[int, float]
    levelpath: Mapping[int, int]


@dataclass(frozen=True)
class Event3Result:
    """Event 3 documents (one per output JSON) and the long-format metrics matrix."""

    documents: dict[str, dict[str, Any]]
    matrix: pd.DataFrame


# ----------------------------------------------------------------------------- nested partners
class _Reach:
    """Up- and downstream reachability on the PlusFlow graph (as ``nested_status``), cached."""

    def __init__(self, network: NetworkInputs, comids: Iterable[int]) -> None:
        e = np.asarray(network.edges, dtype=np.int64)
        e = e[(e[:, 0] != 0) & (e[:, 1] != 0)]
        sites = np.asarray(sorted({int(c) for c in comids}), dtype=np.int64)
        self.nodes = np.unique(np.concatenate([e.ravel(), sites]))
        rows = np.searchsorted(self.nodes, e[:, 0])
        cols = np.searchsorted(self.nodes, e[:, 1])
        n = self.nodes.size
        self.graph = sparse.csr_array((np.ones(rows.size), (rows, cols)), shape=(n, n))
        self.graph_t = sparse.csr_array(self.graph.T)
        self._cache: dict[int, IntArray] = {}

    def index(self, comids: Sequence[int]) -> IntArray:
        out: IntArray = np.searchsorted(self.nodes, np.asarray(comids, dtype=np.int64))
        return out

    def __call__(self, node: int) -> IntArray:
        if node not in self._cache:
            down = breadth_first_order(self.graph, node, directed=True, return_predecessors=False)
            up = breadth_first_order(self.graph_t, node, directed=True, return_predecessors=False)
            self._cache[node] = np.unique(np.concatenate([down, up])).astype(np.int64)
        return self._cache[node]


def nested_partners(
    train_comids: Sequence[int],
    test_comids: Sequence[int],
    reach: _Reach,
    network: NetworkInputs,
    pathlength: Mapping[int, float],
    levelpath: Mapping[int, int],
    psi_strong: float,
) -> pd.DataFrame:
    """Per test site: ``psi_max`` (as ``nested_status``), the partner, |Delta PathLength|, mainstem.

    The partner is the nested training site attaining ``psi_max``; ties go to the smallest
    |Delta PathLength|, then the lowest COMID (docs/3 13.13 p).
    """
    tr = np.asarray(train_comids, dtype=np.int64)
    tr_node = reach.index(tr.tolist())
    tr_tp = np.array([network.terminal_path[int(c)] for c in tr], dtype=np.int64)
    tr_da = np.array([network.drainage_km2[int(c)] for c in tr], dtype=np.float64)
    tr_pl = np.array([pathlength[int(c)] for c in tr], dtype=np.float64)
    out: list[dict[str, Any]] = []
    for tc in (int(c) for c in test_comids):
        node = int(reach.index([tc])[0])
        cand = np.isin(tr_node, reach(node)) & (tr_tp == network.terminal_path[tc])
        rec: dict[str, Any] = {
            "comid": tc,
            "psi_max": 0.0,
            "partner": None,
            "dpath_km": None,
            "same_levelpath": None,
        }
        if cand.any():
            a, b = tr_da[cand], network.drainage_km2[tc]
            psi = np.minimum(a, b) / np.maximum(a, b)
            dpl = np.abs(tr_pl[cand] - pathlength[tc])
            order = np.lexsort((tr[cand], dpl, -psi))
            k = int(order[0])
            partner = int(tr[cand][k])
            rec.update(
                psi_max=float(psi.max()),
                partner=partner,
                dpath_km=float(dpl[k]),
                same_levelpath=bool(levelpath[partner] == levelpath[tc]),
            )
        out.append(rec)
    frame = pd.DataFrame(out)
    frame["nested"] = np.where(
        frame["psi_max"] >= psi_strong, "strong", np.where(frame["psi_max"] > 0, "weak", "none")
    )
    return frame


def _blocks(folds: pd.DataFrame, family: str) -> list[tuple[int, int]]:
    sub = folds.loc[folds["family"].eq(family), ["repeat", "outer_fold"]].drop_duplicates()
    return sorted((int(r), int(f)) for r, f in sub.itertuples(index=False))


def _nested_tables(
    inp: Event3Inputs, frame: pd.DataFrame, protocol: ProtocolParams
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S-ID partners (checked against Event 1) and S-LRO nested classes (docs/3 13.13 p)."""
    comids = frame["comid"].to_numpy(dtype=np.int64)
    reach = _Reach(inp.network, comids.tolist())
    parts: dict[str, list[pd.DataFrame]] = {"S-ID": [], "S-LRO": []}
    for family in parts:
        for r, f in _blocks(inp.folds, family):
            test = get_view(frame, inp.folds, family, r, f, "test")
            train = get_view(frame, inp.folds, family, r, f, "train")
            tab = nested_partners(
                comids[train].tolist(),
                comids[test].tolist(),
                reach,
                inp.network,
                inp.pathlength,
                inp.levelpath,
                protocol.audit.nested_psi,
            )
            tab.insert(0, "site_no", frame["site_no"].astype(str).to_numpy()[test])
            tab.insert(1, "repeat", r)
            tab.insert(2, "outer_fold", f)
            parts[family].append(tab)
    sid = pd.concat(parts["S-ID"], ignore_index=True)
    lro = pd.concat(parts["S-LRO"], ignore_index=True)
    ref = inp.audit_strata.astype({"repeat": int, "outer_fold": int}).copy()
    ref["site_no"] = ref["site_no"].astype(str)
    chk = sid.merge(ref, on=["site_no", "repeat", "outer_fold"], how="outer", indicator=True)
    if not chk["_merge"].eq("both").all():
        raise ValueError("S-ID test instances differ from audit_strata.parquet")
    if not np.array_equal(chk["psi_max_x"].to_numpy(), chk["psi_max_y"].to_numpy()):
        raise ValueError("nested psi_max differs from Event 1 (audit_strata.parquet)")
    if not chk["nested_x"].astype(str).equals(chk["nested_y"].astype(str)):
        raise ValueError("nested class differs from Event 1 (audit_strata.parquet)")
    return sid, lro


def _nested_doc(tab: pd.DataFrame) -> dict[str, Any]:
    doc: dict[str, Any] = {"n_test_instances": len(tab)}
    for cls in ("strong", "weak"):
        sub = tab.loc[tab["nested"].eq(cls)]
        entry: dict[str, Any] = {"n": len(sub), "share": float(len(sub) / max(len(tab), 1))}
        if len(sub):
            d = sub["dpath_km"].to_numpy(dtype=np.float64)
            entry["abs_dpathlength_km"] = {
                k: float(v)
                for k, v in zip(
                    ("q25", "median", "q75"), np.quantile(d, [0.25, 0.5, 0.75]), strict=True
                )
            }
            entry["share_same_levelpath"] = float(sub["same_levelpath"].astype(bool).mean())
        doc[cls] = entry
    blocks: list[dict[str, Any]] = []
    for key, g in tab.groupby(["repeat", "outer_fold"]):
        r, f = cast(tuple[int, int], key)
        blocks.append(
            {
                "repeat": int(r),
                "outer_fold": int(f),
                "n_test": len(g),
                "n_strong": int(g["nested"].eq("strong").sum()),
                "n_weak": int(g["nested"].eq("weak").sum()),
            }
        )
    doc["per_block"] = blocks
    return doc


# ----------------------------------------------------------------------------- set building
def _stem(model: str, ablation: str) -> str:
    return f"{model}_{ablation}" if ablation else model


@dataclass
class _Sets:
    """Evaluation sets keyed by ``(model, ablation, analysis, family)``."""

    rows: dict[tuple[str, str, str, str], pd.DataFrame]
    dropped: dict[str, int]

    def get(
        self, model: str, family: str, *, ablation: str = "", analysis: str = "primary"
    ) -> pd.DataFrame:
        return self.rows[(model, ablation, analysis, family)]


def _build_sets(
    inp: Event3Inputs,
    cuts: Sequence[float],
    strata_sid: pd.DataFrame,
    strata_lro: pd.DataFrame,
) -> _Sets:
    rows: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    dropped: dict[str, int] = {}
    variants: list[tuple[str, str, str]] = [(m, "", "primary") for m in LADDER]
    variants += [("M5", a, "primary") for a in M5_ABLATIONS]
    variants += [(m, "A4", "primary") for m in A4_MODELS]
    variants += [(m, "", sa) for m, sas in SA_MODELS.items() for sa in sas]
    for model, abl, analysis in variants:
        preds = inp.preds[_stem(model, abl)]
        primary = not abl and analysis == "primary"
        for family in es.SITE_FAMILIES:
            strata = None
            if primary and family == "S-ID":
                strata = strata_sid
            elif primary and family == "S-LRO":
                strata = strata_lro
            rows[(model, abl, analysis, family)] = es.site_rows(
                preds,
                inp.site_table,
                inp.folds,
                cuts,
                model=model,
                family=family,
                ablation=abl,
                analysis=analysis,
                strata=strata,
            )
        ind, n_drop = es.ind_rows(
            preds, inp.ehydro, inp.ind_features, cuts, model=model, ablation=abl, analysis=analysis
        )
        rows[(model, abl, analysis, "S-IND")] = ind
        dropped[f"{_stem(model, abl)}|{analysis}|S-IND"] = n_drop
    for model in ("M1", "M2-XGB"):
        srnd, n_drop = es.srnd_rows(inp.preds[model], inp.srnd, cuts, model=model)
        rows[(model, "", "primary", "S-RND")] = srnd
        dropped[f"{model}|primary|S-RND"] = n_drop
    return _Sets(rows=rows, dropped=dropped)


# ----------------------------------------------------------------------------- paired designs
def _errors(rows: pd.DataFrame, target: str) -> FloatArray:
    pair = st.log_pair(rows, target)
    if pair is None:
        raise ValueError(f"no {target} prediction")
    out: FloatArray = pair[0] - pair[1]
    return out


def _paired(
    named: Mapping[str, pd.DataFrame], targets: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, FloatArray]]:
    """Join several models' rows on ``(unit_id, repeat)``; return keys and errors per model:k."""
    names = list(named)
    base = named[names[0]]
    keys = ["unit_id", "repeat"]
    labels = [c for c in ("cluster", "boot_stratum", "da_q", "far", "nested") if c in base.columns]
    out = base[keys + labels].copy()
    errs: dict[str, FloatArray] = {}
    for name in names:
        rows = named[name]
        if len(rows) != len(base) or not rows[keys].reset_index(drop=True).equals(
            base[keys].reset_index(drop=True)
        ):
            raise ValueError(f"{name}: evaluation rows differ from {names[0]}")
        for t in targets:
            e = _errors(rows, t)
            if not bool(np.isfinite(e).all()):
                raise ValueError(f"{name}: non-finite {t} errors on hypothesis rows")
            errs[f"{name}:{t}"] = e
    return out, errs


def _design(keys: pd.DataFrame, *, stratified: bool = True) -> st.Design:
    return st.Design(
        cluster=keys["cluster"].astype(str).to_numpy(dtype=np.str_),
        stratum=keys["boot_stratum"].astype(str).to_numpy(dtype=np.str_) if stratified else None,
        repeat=keys["repeat"].to_numpy(dtype=np.int64),
    )


def _rmse(e: FloatArray) -> Callable[[Sequence[IntArray]], float]:
    def f(parts: Sequence[IntArray]) -> float:
        return float(np.mean([math.sqrt(float(np.mean(e[p] ** 2))) for p in parts]))

    return f


def _bias(e: FloatArray) -> Callable[[Sequence[IntArray]], float]:
    def f(parts: Sequence[IntArray]) -> float:
        return float(np.mean([float(np.mean(e[p])) for p in parts]))

    return f


def _drmse(e1: FloatArray, e2: FloatArray) -> Callable[[Sequence[IntArray]], float]:
    r1, r2 = _rmse(e1), _rmse(e2)
    return lambda parts: r1(parts) - r2(parts)


def _dabsbias(e1: FloatArray, e2: FloatArray) -> Callable[[Sequence[IntArray]], float]:
    def f(parts: Sequence[IntArray]) -> float:
        return float(np.mean([abs(np.mean(e1[p])) - abs(np.mean(e2[p])) for p in parts]))

    return f


def _ci(doc: Mapping[str, Any]) -> hy.CI:
    return hy.CI(lo=float(doc["lo"]), hi=float(doc["hi"]))


def _enough(keys: pd.DataFrame, minimum: int) -> tuple[bool, dict[str, int]]:
    counts = {str(r): int(g["unit_id"].nunique()) for r, g in keys.groupby("repeat")}
    return bool(counts) and min(counts.values()) >= minimum, counts


def _opposite(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """docs/3 13.13 k: one CI entirely below 0 and the other entirely above 0."""
    return bool((a["hi"] < 0 and b["lo"] > 0) or (a["lo"] > 0 and b["hi"] < 0))


# ----------------------------------------------------------------------------- H4 helpers
def _site_level(rows: pd.DataFrame, nu_col: str = "nu_mean") -> pd.DataFrame:
    g = rows.groupby("unit_id", sort=True)
    out = pd.DataFrame(
        {
            "nu_hat": g[nu_col].mean(),
            "r_hat": g["r_mean"].mean(),
            "o_t": g["o_t"].first(),
            "D50_m": g["D50_m"].first(),
            "S": g["S"].first(),
            "F_S": g["F_S"].first().astype(bool),
            "boot_stratum": g["boot_stratum"].first(),
            "row_id": g["row_id"].first(),
        }
    ).reset_index()
    out["cluster"] = out["unit_id"]
    out["repeat"] = 0
    return out


def _h4_points(site: pd.DataFrame, protocol: ProtocolParams) -> dict[str, Any]:
    dom, cfg = protocol.domain, protocol.evaluation.h4
    nu_b = (math.log(dom.n_bounds[0]), math.log(dom.n_bounds[1]))
    r_b = (math.log(dom.r_bounds[0]), math.log(dom.r_bounds[1]))
    nu = site["nu_hat"].to_numpy(dtype=np.float64)
    lr = np.log(site["r_hat"].to_numpy(dtype=np.float64))
    n_hat = np.exp(nu)
    r_obs = 1.0 / np.expm1(site["o_t"].to_numpy(dtype=np.float64))
    return {
        "n_sites": len(site),
        "pi_nu": hy.boundary_share(nu, nu_b, cfg.vartheta),
        "pi_r": hy.boundary_share(lr, r_b, cfg.vartheta),
        "pi_lit": hy.literature_share(n_hat, dom),
        "rho_n_d50": mt.spearman_rho(n_hat, site["D50_m"].to_numpy(dtype=np.float64)),
        "rho_n_s": mt.spearman_rho(n_hat, site["S"].to_numpy(dtype=np.float64)),
        "rho_r_robs": mt.spearman_rho(site["r_hat"].to_numpy(dtype=np.float64), r_obs),
    }


def _rho_design(
    site: pd.DataFrame, protocol: ProtocolParams, n_boot: int | None, keys: Sequence[str]
) -> dict[str, dict[str, Any]]:
    n_hat = np.exp(site["nu_hat"].to_numpy(dtype=np.float64))
    cols = {
        "n_d50": (n_hat, site["D50_m"].to_numpy(dtype=np.float64)),
        "n_s": (n_hat, site["S"].to_numpy(dtype=np.float64)),
        "r_robs": (
            site["r_hat"].to_numpy(dtype=np.float64),
            1.0 / np.expm1(site["o_t"].to_numpy(dtype=np.float64)),
        ),
    }
    level = 1.0 - protocol.evaluation.bootstrap.alpha_b

    def rho(x: FloatArray, y: FloatArray) -> Callable[[Sequence[IntArray]], float]:
        return lambda parts: mt.spearman_rho(x[parts[0]], y[parts[0]])

    stats = [st.BootStat(k, rho(*cols[k]), level) for k in keys]
    return st.run_design(_design(site), stats, protocol.evaluation.bootstrap, n_boot=n_boot)


def _h4_extra(site: pd.DataFrame, protocol: ProtocolParams, qc: pd.DataFrame) -> dict[str, Any]:
    """H4 caveats: F_S strata, z_n distribution, n-hat by ``chan_mat_cd`` (protocol 4.4, 7.6)."""
    dom = protocol.domain
    nu_min, nu_max = math.log(dom.n_bounds[0]), math.log(dom.n_bounds[1])
    frac = (site["nu_hat"].to_numpy(dtype=np.float64) - nu_min) / (nu_max - nu_min)
    with np.errstate(divide="ignore"):
        z = np.log(frac) - np.log1p(-frac)
    finite = np.isfinite(z)
    by_fs: dict[str, Any] = {}
    for flag, sub in site.groupby("F_S"):
        by_fs["F_S" if flag else "not_F_S"] = (
            _h4_points(sub, protocol) if len(sub) > 2 else {"n_sites": len(sub)}
        )
    codes = qc.set_index("row_id")["chan_mat_cd"]
    chan = codes.reindex(site["row_id"]).astype("string").fillna("missing").to_numpy()
    n_hat = np.exp(site["nu_hat"].to_numpy(dtype=np.float64))
    chan_doc = {}
    for code in sorted(set(chan.tolist())):
        v = n_hat[chan == code]
        q = np.quantile(v, [0.25, 0.5, 0.75])
        chan_doc[str(code)] = {"n": int(v.size), "q25": q[0], "median": q[1], "q75": q[2]}
    zq = np.quantile(z[finite], [0.05, 0.25, 0.5, 0.75, 0.95]) if finite.any() else []
    return {
        "by_F_S": by_fs,
        "z_n": {
            "quantiles": dict(zip(("q05", "q25", "median", "q75", "q95"), zq, strict=False)),
            "n_infinite": int((~finite).sum()),
        },
        "n_hat_by_chan_mat_cd": chan_doc,
    }


# ----------------------------------------------------------------------------- sections
def _hypotheses(
    sets: _Sets,
    inp: Event3Inputs,
    protocol: ProtocolParams,
    log_kappa: LogKappa,
    n_boot: int | None,
) -> dict[str, Any]:
    ev = protocol.evaluation
    cfg = ev.bootstrap
    level = 1.0 - cfg.alpha_b
    s2_level = bonferroni_level(ev.s2)
    mstar = {k: str(v) for k, v in inp.mstar["mstar"].items()}
    md, ma = mstar["dbar"], mstar["A"]

    def named(family: str, models: Sequence[str]) -> dict[str, pd.DataFrame]:
        return {m: sets.get(m, family) for m in dict.fromkeys(models)}

    doc: dict[str, Any] = {"mstar": mstar, "bootstrap": cfg.model_dump(), "s2_level": s2_level}
    # H1: S-LRO pooled (+ S2 on S-LRO) and S-IND on river clusters
    k_lro, e_lro = _paired(named("S-LRO", ["M5", md, ma, "M6-NN", "M4"]), ["dbar", "A"])
    lro = st.run_design(
        _design(k_lro),
        [
            st.BootStat("h1_dbar", _drmse(e_lro["M5:dbar"], e_lro[f"{md}:dbar"]), level),
            st.BootStat("h1_A", _drmse(e_lro["M5:A"], e_lro[f"{ma}:A"]), level),
            st.BootStat("s2_M5_vs_M6NN", _drmse(e_lro["M5:dbar"], e_lro["M6-NN:dbar"]), s2_level),
            st.BootStat("s2_M4_vs_M5", _drmse(e_lro["M4:dbar"], e_lro["M5:dbar"]), s2_level),
        ],
        cfg,
        n_boot=n_boot,
    )
    ind_sets = named("S-IND", ["M5", md, ma])
    common = set.intersection(*(set(r["unit_id"]) for r in ind_sets.values()))
    ind_sets = {
        m: r.loc[r["unit_id"].isin(common)].reset_index(drop=True) for m, r in ind_sets.items()
    }
    k_ind, e_ind = _paired(ind_sets, ["dbar", "A"])
    ind = st.run_design(
        _design(k_ind, stratified=False),
        [
            st.BootStat("h1_dbar", _drmse(e_ind["M5:dbar"], e_ind[f"{md}:dbar"]), level),
            st.BootStat("h1_A", _drmse(e_ind["M5:A"], e_ind[f"{ma}:A"]), level),
        ],
        cfg,
        n_boot=n_boot,
    )
    h1 = hy.decide_h1(_ci(lro["h1_dbar"]), _ci(ind["h1_dbar"]))
    doc["H1"] = {
        "statistic": "RMSE_dbar(M5) - RMSE_dbar(M*); k = A alongside",
        "S-LRO": {"dbar": lro["h1_dbar"], "A": lro["h1_A"], "n_sites": len(k_lro)},
        "S-IND": {
            "dbar": ind["h1_dbar"],
            "A": ind["h1_A"],
            "n_locations": len(k_ind),
            "n_clusters": int(k_ind["cluster"].nunique()),
        },
        "decision": {
            "supported": h1.supported,
            "falsified": h1.falsified,
            "ind_contradiction": h1.ind_contradiction,
        },
    }
    # S-ID: S2 and H2 (Q1), with the far / not-nested strata under the audit label
    k_sid, e_sid = _paired(named("S-ID", ["M5", md, "M6-NN", "M4"]), ["dbar"])

    def s2_stats(e: Mapping[str, FloatArray]) -> list[st.BootStat]:
        return [
            st.BootStat("s2_M5_vs_M6NN", _drmse(e["M5:dbar"], e["M6-NN:dbar"]), s2_level),
            st.BootStat("s2_M4_vs_M5", _drmse(e["M4:dbar"], e["M5:dbar"]), s2_level),
        ]

    def h2_stats(e: Mapping[str, FloatArray]) -> list[st.BootStat]:
        return [
            st.BootStat("h2", _dabsbias(e["M5:dbar"], e[f"{md}:dbar"]), level),
            st.BootStat("bias_M5", _bias(e["M5:dbar"]), level),
            st.BootStat("bias_Mstar", _bias(e[f"{md}:dbar"]), level),
        ]

    def subset(
        keys: pd.DataFrame, errs: Mapping[str, FloatArray], mask: NDArray[np.bool_]
    ) -> tuple[pd.DataFrame, dict[str, FloatArray]]:
        return keys.loc[mask].reset_index(drop=True), {k: v[mask] for k, v in errs.items()}

    sid = st.run_design(_design(k_sid), s2_stats(e_sid), cfg, n_boot=n_boot)
    q1 = k_sid["da_q"].to_numpy() == 1
    kq, eq = subset(k_sid, e_sid, q1)
    h2_full = st.run_design(_design(kq), h2_stats(eq), cfg, n_boot=n_boot)
    h2 = hy.decide_h2(_ci(h2_full["h2"]))
    k_lq, e_lq = _paired(named("S-LRO", ["M5", md]), ["dbar"])
    lq = k_lq["da_q"].to_numpy() == 1
    kl, el = subset(k_lq, e_lq, lq)
    h2_lro = st.run_design(_design(kl), h2_stats(el), cfg, n_boot=n_boot)
    optimistic = bool(inp.audit["label"]["optimistic"])
    strata_doc: dict[str, Any] = {"label": inp.audit["label"]["text"], "applies": optimistic}
    if optimistic:
        masks = {
            "far": k_sid["far"].astype(bool).to_numpy(),
            "not_nested": k_sid["nested"].astype(str).eq("none").to_numpy(),
        }
        for name, mask in masks.items():
            entry: dict[str, Any] = {}
            for stat_name, m, maker, full in (
                ("H2", mask & q1, h2_stats, h2_full),
                ("S2", mask, s2_stats, sid),
            ):
                ks, ee = subset(k_sid, e_sid, m)
                ok, counts = _enough(ks, STRATUM_MIN_SITES)
                if not ok:
                    entry[stat_name] = {
                        "not_computed": f"fewer than {STRATUM_MIN_SITES} sites in a repeat",
                        "sites_per_repeat": counts,
                    }
                    continue
                res = st.run_design(_design(ks), maker(ee), cfg, n_boot=n_boot)
                for key, val in res.items():
                    if key in full:
                        val["opposite_sign_to_full_set"] = _opposite(val, full[key])
                entry[stat_name] = {**res, "sites_per_repeat": counts}
            strata_doc[name] = entry
    doc["H2"] = {
        "statistic": "mean over repeats of |B_dbar(M5)(Q1)| - |B_dbar(M*)(Q1)|",
        "S-ID_Q1": {**h2_full, "n_sites": int(kq["unit_id"].nunique())},
        "S-LRO_Q1_secondary": {**h2_lro, "n_sites": len(kl)},
        "decision": {"supported": h2.supported, "falsified": h2.falsified},
    }
    # H3: two-sample bootstrap, ID = S-ID OOF of the same sites
    doc["H3"] = _h3(sets, md, protocol, n_boot)
    h3 = hy.decide_h3(_ci(doc["H3"]["H3-LRO"]["delta"]), _ci(doc["H3"]["H3-TAIL"]["delta"]))
    doc["H3"]["decision"] = {
        "supported": h3.supported,
        "falsified": h3.falsified,
        "partial": h3.partial,
    }
    # H4
    h4_doc, h4 = _h4(sets, inp, protocol, n_boot)
    doc["H4"] = h4_doc
    # S2 and its descriptive aspect-ratio strata
    doc["S2"] = {
        "statistic": "RMSE_dbar(first) - RMSE_dbar(second)",
        "S-ID": sid,
        "S-LRO": {k: lro[k] for k in ("s2_M5_vs_M6NN", "s2_M4_vs_M5")},
        "kappa_quintiles_descriptive": {
            fam: _kappa_strata(sets, fam, log_kappa, protocol) for fam in ("S-ID", "S-LRO")
        },
    }
    doc["far_not_nested_strata"] = strata_doc
    rows = hy.outcome_rows(h1, h2, h3, h4, _ci(lro["h1_dbar"]), ev)
    doc["outcome_4_7"] = {
        "rows": list(rows),
        "matched": [
            {
                "row": k,
                "outcome": OUTCOME_TEXT[k][0],
                "claim": OUTCOME_TEXT[k][1],
                "venue": OUTCOME_TEXT[k][2],
            }
            for k in rows
        ],
        "statement": (
            "no row of protocol 4.7 matches"
            if not rows
            else "; ".join(f'row {k}: "{OUTCOME_TEXT[k][1]}"' for k in rows)
        ),
    }
    return doc


Loss2 = Callable[[Sequence[IntArray], Sequence[IntArray]], float]


def _loss(e_id: FloatArray, e_ood: FloatArray) -> Loss2:
    """Proofs 9.5 H3: ``L = -RMSE(T_ID) + RMSE(T_OOD)`` on the two samples' draws."""
    ri, ro = _rmse(e_id), _rmse(e_ood)
    return lambda pa, pb: -ri(pa) + ro(pb)


def _diff(a: Loss2, b: Loss2) -> Loss2:
    return lambda pa, pb: a(pa, pb) - b(pa, pb)


def _h3_one(
    sets: _Sets, md: str, family: str, protocol: ProtocolParams, n_boot: int | None
) -> dict[str, Any]:
    cfg = protocol.evaluation.bootstrap
    level = 1.0 - cfg.alpha_b
    k_ood, e_ood = _paired({m: sets.get(m, family) for m in ("M5", md)}, ["dbar"])
    sites = set(k_ood["unit_id"])
    id_sets = {}
    for m in ("M5", md):
        rows = sets.get(m, "S-ID")
        id_sets[m] = rows.loc[rows["unit_id"].isin(sites)].reset_index(drop=True)
    k_id, e_id = _paired(id_sets, ["dbar"])
    l5 = _loss(e_id["M5:dbar"], e_ood["M5:dbar"])
    lm = _loss(e_id[f"{md}:dbar"], e_ood[f"{md}:dbar"])
    res = st.run_two_sample(
        st.TwoSampleDesign(a=_design(k_id), b=_design(k_ood)),
        [("delta", _diff(l5, lm), level), ("L_M5", l5, level), ("L_Mstar", lm, level)],
        cfg,
        n_boot=n_boot,
    )
    return {**res, "n_sites": len(k_ood), "secondary": family == "S-TAIL-SO"}


def _h3(sets: _Sets, md: str, protocol: ProtocolParams, n_boot: int | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "statistic": "L(M5) - L(M*), L = -RMSE_dbar(T_ID) + RMSE_dbar(T_OOD)",
    }
    for name, family in (
        ("H3-LRO", "S-LRO"),
        ("H3-TAIL", "S-TAIL-DA"),
        ("H3-TAIL-SO", "S-TAIL-SO"),
    ):
        out[name] = _h3_one(sets, md, family, protocol, n_boot)
    return out


def _h4(
    sets: _Sets, inp: Event3Inputs, protocol: ProtocolParams, n_boot: int | None
) -> tuple[dict[str, Any], hy.H4Result]:
    keys = ("n_d50", "n_s", "r_robs")
    site = _site_level(sets.get("M5", "S-ID"))
    pts = _h4_points(site, protocol)
    cis = _rho_design(site, protocol, n_boot, keys)
    res = hy.decide_h4(
        pts["pi_nu"],
        pts["pi_r"],
        pts["pi_lit"],
        {k: _ci(cis[k]) for k in keys},
        protocol.evaluation.h4,
    )
    m5_sid = sets.get("M5", "S-ID")
    shape = {
        "rmse_t": st.per_repeat(m5_sid, lambda d: mt.rmse(d["mu_t"], d["o_t"])),
        "bias_t": st.per_repeat(m5_sid, lambda d: mt.bias(d["mu_t"], d["o_t"])),
    }
    fs_ci = {}
    for flag, sub in site.groupby("F_S"):
        fs_ci["F_S" if flag else "not_F_S"] = _rho_design(
            sub.reset_index(drop=True), protocol, n_boot, ("n_s",)
        )
    lro_site = _site_level(sets.get("M5", "S-LRO"))
    lro = {
        **_h4_points(lro_site, protocol),
        "ci": _rho_design(lro_site, protocol, n_boot, keys),
        "rmse_t": mt.rmse(sets.get("M5", "S-LRO")["mu_t"], sets.get("M5", "S-LRO")["o_t"]),
        "bias_t": mt.bias(sets.get("M5", "S-LRO")["mu_t"], sets.get("M5", "S-LRO")["o_t"]),
    }
    doc = {
        "site_level": "nu-hat = mean over repeats of nu_mean; r-hat = mean over repeats of r_mean",
        "S-ID": {**pts, "ci": cis, **shape},
        "decision": {
            "supported": res.supported,
            "falsified": res.falsified,
            "reasons": list(res.reasons),
        },
        "caveats": {
            **_h4_extra(site, protocol, inp.qc_rows),
            "rho_n_s_ci_by_F_S": fs_ci,
            "C2": "slope error alone induces a positive n-S association (weak evidence for (b) S)",
            "L10": "D50 is modelled",
        },
        "S-LRO_reported": lro,
    }
    return doc, res


def _kappa_strata(
    sets: _Sets, family: str, log_kappa: LogKappa, protocol: ProtocolParams
) -> dict[str, Any]:
    """Descriptive S2 strata by quintiles of kappa-hat from M5's ensemble geometry (13.13 j)."""
    m5 = sets.get("M5", family)
    kappa = np.exp(
        log_kappa(
            m5["mu_dbar"].to_numpy(dtype=np.float64),
            m5["mu_W"].to_numpy(dtype=np.float64),
            1.0 / np.expm1(m5["mu_t"].to_numpy(dtype=np.float64)),
        )
    )
    cuts = np.quantile(kappa, [0.2, 0.4, 0.6, 0.8])
    cls = np.searchsorted(cuts, kappa, side="right") + 1
    others = {m: sets.get(m, family) for m in ("M6-NN", "M4")}
    keys = m5[["unit_id", "repeat"]].reset_index(drop=True)
    for m, d in others.items():
        if not d[["unit_id", "repeat"]].reset_index(drop=True).equals(keys):
            raise ValueError(f"{m} {family}: evaluation rows differ from M5")

    def rmse_log(x: pd.DataFrame) -> float:
        return mt.rmse_log(x["mu_dbar"], np.log(x["o_dbar"]))

    out: dict[str, Any] = {"cuts": cuts.tolist()}
    for q in range(1, 6):
        mask = cls == q
        sub = {m: d.loc[mask] for m, d in {"M5": m5, **others}.items()}
        rm = {m: st.per_repeat(d, rmse_log) for m, d in sub.items()}
        out[f"Q{q}"] = {
            "n_rows": int(mask.sum()),
            "s2_M5_vs_M6NN": rm["M5"] - rm["M6-NN"],
            "s2_M4_vs_M5": rm["M4"] - rm["M5"],
        }
    return out


def _uq(
    sets: _Sets, inp: Event3Inputs, protocol: ProtocolParams, cuts: Sequence[float]
) -> dict[str, Any]:
    band = protocol.uq.coverage_band
    out: dict[str, Any] = {
        "alpha": protocol.uq.alpha,
        "band": list(band),
        "pit_bins": st.PIT_BINS,
        "models_without_distribution": [m for m in LADDER if m not in SIGMA_MODELS],
    }
    specs: list[tuple[str, str, str]] = [(m, "", "primary") for m in SIGMA_MODELS]
    specs += [("M5", a, "primary") for a in M5_ABLATIONS] + [("M2-MLP", "A4", "primary")]
    specs += [(m, "", sa) for m in ("M5", "M2-MLP") for sa in SA_MODELS[m]]
    for model, abl, analysis in specs:
        primary = not abl and analysis == "primary"
        tag = f"{_stem(model, abl)}|{analysis}"
        entry: dict[str, Any] = {}
        for family in es.SITE_FAMILIES:
            rows = sets.get(model, family, ablation=abl, analysis=analysis)
            entry[family] = st.uq_summary(
                rows, UQ_TARGETS, band, strata=("division", "da_q") if primary else ()
            )
        entry["S-IND"] = _uq_ind(inp, model, abl, analysis, cuts, band)
        out[tag] = entry
    return out


def _uq_ind(
    inp: Event3Inputs,
    model: str,
    abl: str,
    analysis: str,
    cuts: Sequence[float],
    band: tuple[float, float],
) -> dict[str, Any]:
    """S-IND UQ on locations with ``n_comids`` = 1 (docs/3 13.5)."""
    comid = es.model_rows(
        inp.preds[_stem(model, abl)],
        model=model,
        ablation=abl,
        analysis=analysis,
        family="S-IND",
        unit="comid",
    )
    single = comid.loc[comid["n_comids"].astype(int).eq(1)].copy()
    single["loc_id"] = single["loc_id"].astype(str)
    kept = inp.ehydro.loc[inp.ehydro["dedupe_status"].eq("kept")].copy()
    kept["loc_id"] = kept["loc_id"].astype(str)
    obs = kept[["loc_id", "W", "dbar", "A"]].rename(
        columns={"W": "o_W", "dbar": "o_dbar", "A": "o_A"}
    )
    rows = single.merge(obs, on="loc_id", how="inner")
    rows["repeat"] = 0
    n_multi = int(comid.loc[comid["n_comids"].astype(int).gt(1), "loc_id"].nunique())
    res = st.uq_summary(rows, ("dbar", "W", "A"), band, strata=()) if len(rows) else {}
    return {"n_locations_single": len(rows), "not_computed_multi_comid": n_multi, **res}


def _consistency(sets: _Sets, inp: Event3Inputs, protocol: ProtocolParams) -> dict[str, Any]:
    dom = protocol.domain
    nu_b = (math.log(dom.n_bounds[0]), math.log(dom.n_bounds[1]))
    frame = analysis_frame(inp.site_table, "primary")
    eps = frame["eps_A"].to_numpy(dtype=np.float64)
    med, share = mt.closure_summary(eps)
    f_eps = inp.qc_rows["F_eps"].astype(bool)
    out: dict[str, Any] = {
        "observed_closure": {
            "n_sites": int(eps.size),
            "median_abs": med,
            "share_gt_ln105": share,
            "qc_R8_rows_flagged": int(f_eps.sum()),
            "qc_rows": len(f_eps),
            "share_removed": float(f_eps.mean()),
        },
        "continuity": "V-hat = Q_E / A-hat for every model: no continuity residual exists",
        "models": {},
    }
    for model in LADDER:
        out["models"][model] = {
            fam: st.consistency_summary(sets.get(model, fam), model, nu_b)
            for fam in es.SITE_FAMILIES
        }
    return out


def _dependence(
    sets: _Sets,
    inp: Event3Inputs,
    protocol: ProtocolParams,
    nested_sid: pd.DataFrame,
    nested_lro: pd.DataFrame,
    aoa: AoaFn,
) -> dict[str, Any]:
    frame = event1.audit_frame(inp.site_table)
    coords = frame[["x_km", "y_km"]].to_numpy(dtype=np.float64)
    divisions = frame["division"].astype(str).to_numpy(dtype=np.str_)
    cfg = protocol.audit
    w1 = da.knn_w(coords, cfg.knn_k)
    w2, _ = da.idw_weights(coords, cfg.idw_band_km)
    resid = event1.oof_residuals(inp.preds["M5"], frame, inp.folds, models=("M5",))
    series = {}
    for col in resid.columns:
        doc, _, _ = event1.series_stats(
            resid[col].to_numpy(dtype=np.float64), coords, w1, w2, divisions, protocol
        )
        series[col] = doc
    blocks = [
        (fam, r, f)
        for fam in ("S-LRO", "S-TAIL-DA", "S-TAIL-SO", "S-IND")
        for r, f in _blocks(inp.folds, fam)
    ]
    aoa_docs = aoa(blocks)
    return {
        "M5_residual_structure_3_6_1_c": {"descriptive": True, "series": series},
        "nested_3_6_5": {"S-ID": _nested_doc(nested_sid), "S-LRO": _nested_doc(nested_lro)},
        "aoa_3_6_6": _aoa_doc(sets, aoa_docs),
    }


def _aoa_doc(sets: _Sets, results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    inside: dict[str, dict[str, bool]] = {}
    blocks: list[dict[str, Any]] = []
    for res in results:
        fam = str(res["family"])
        thr = float(res["threshold"])
        di = np.asarray(res["di"], dtype=np.float64)
        flags = di <= thr
        target = inside.setdefault(fam, {})
        for unit, ok in zip(res["units"], flags.tolist(), strict=True):
            if fam == "S-IND":
                loc = str(unit).split("|", 1)[0]
                target[loc] = target.get(loc, True) and bool(ok)
            else:
                target[str(unit)] = bool(ok)
        blocks.append(
            {
                "family": fam,
                "repeat": int(res["repeat"]),
                "outer_fold": int(res["outer_fold"]),
                "threshold": thr,
                "n_train": int(res["n_train"]),
                "n_test_units": int(di.size),
                "n_inside": int(flags.sum()),
                "importance": dict(zip(res["features"], res["importance"], strict=True)),
            }
        )
    metrics: dict[str, Any] = {}
    for fam, flags_by_unit in inside.items():
        metrics[fam] = {}
        for model in LADDER:
            rows = sets.get(model, fam)
            lab = rows["unit_id"].map(flags_by_unit)
            if lab.isna().any():
                raise ValueError(f"{model} {fam}: evaluation units without an AOA flag")
            entry = {}
            for name, mask in (("inside", lab.astype(bool)), ("outside", ~lab.astype(bool))):
                sub = rows.loc[mask.to_numpy()]
                cells = {}
                for t in ("dbar", "W", "A"):
                    res2 = st.repeat_cells(sub, t) if len(sub) else None
                    if res2 is not None:
                        cells[t] = {
                            "rmse_log": res2[0]["rmse_log"],
                            "bias_log": res2[0]["bias_log"],
                            **res2[1],
                        }
                entry[name] = cells
            metrics[fam][model] = entry
    return {
        "descriptive": True,
        "inside_rule": "DI <= threshold; S-IND: every member COMID",
        "blocks": blocks,
        "metrics": metrics,
    }


def _descriptive(
    sets: _Sets, inp: Event3Inputs, protocol: ProtocolParams, log_kappa: LogKappa
) -> dict[str, Any]:
    mstar = str(inp.mstar["mstar"]["dbar"])
    variants: dict[str, Any] = {}
    for abl in M5_ABLATIONS:
        rows = sets.get("M5", "S-ID", ablation=abl)
        site = _site_level(rows)
        pts = _h4_points(site, protocol)
        if abl == "A2":  # excluded from H4 (c) (protocol 4.1, proofs 9.5)
            pts.pop("rho_r_robs")
        entry: dict[str, Any] = {"H4_points_S-ID": pts}
        if abl == "A1":  # Proposition 7.6: nu(full kappa) = nu(A1) + (2/3) ln kappa
            k = log_kappa(
                rows["mu_dbar"].to_numpy(dtype=np.float64),
                rows["mu_W"].to_numpy(dtype=np.float64),
                1.0 / np.expm1(rows["mu_t"].to_numpy(dtype=np.float64)),
            )
            mapped = rows.assign(nu_mapped=rows["nu_mean"].to_numpy() + (2.0 / 3.0) * k)
            m_site = _site_level(mapped, "nu_mapped")
            m_pts = _h4_points(m_site, protocol)
            entry["H4_points_S-ID_prop_7_6_mapped"] = {
                k2: m_pts[k2] for k2 in ("pi_nu", "pi_lit", "rho_n_d50", "rho_n_s")
            }
        variants[f"M5_{abl}"] = entry
    no_sa1 = "identical to primary by construction (no Phi); not trained"
    sa_status = {
        m: {
            "trained": list(sas),
            "SA1": "trained" if "SA1" in sas else no_sa1,
            "role": (
                "realised M*" if m == mstar else ("primary model" if m == "M5" else "supplementary")
            ),
        }
        for m, sas in SA_MODELS.items()
    }
    n_sites = analysis_frame(inp.site_table, "primary")["site_no"].isin(inp.hyg["site_no"])
    site = _site_level(sets.get("M5", "S-ID")).merge(
        inp.hyg.assign(site_no=inp.hyg["site_no"].astype(str)),
        left_on="unit_id",
        right_on="site_no",
        how="inner",
    )
    s1: dict[str, Any] = {"hyg_reader_counts": dict(inp.hyg_counts), "n_matched": len(site)}
    if len(site) > 2:
        rho, rmse = mt.s1_consistency(
            np.exp(site["nu_hat"].to_numpy()), site["manning_n"].to_numpy()
        )
        s1.update(spearman_rho=rho, rmse_log=rmse)
    s1["n_frame_sites_in_hyg"] = int(n_sites.sum())
    return {
        "descriptive": True,
        "ablations_M5": variants,
        "A4_realised_mstar": f"{mstar}_A4",
        "A4_supplementary": [f"{m}_A4" for m in A4_MODELS if m != mstar],
        "sensitivity_status": sa_status,
        "S1": {**s1, "note": "consistency check, not independent validation (protocol 4.6)"},
        "S_RND": {
            "models": ["M1", "M2-XGB"],
            "weights": "1/|I_j|",
            "rows": {m: len(sets.get(m, "S-RND")) for m in ("M1", "M2-XGB")},
            "secondary": True,
        },
        "L8_zarrabi_product": {
            "status": "not computed",
            "reason": "the Zenodo record (doi:10.5281/zenodo.13883263) states no unit for "
            "mf_width/mf_depth; docs/3 13.5 requires the unit to be asserted from the record",
        },
        "dropped_rows": dict(sets.dropped),
    }


def _matrix(sets: _Sets) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (model, abl, analysis, family), rows in sets.rows.items():
        primary = not abl and analysis == "primary"
        meta = {
            "spec": _stem(model, abl),
            "model": model,
            "ablation": abl,
            "analysis": analysis,
            "family": family,
        }
        kinds = st.STRATA_KINDS if primary else ("all",)
        records += st.matrix_records(
            rows, meta, es.TARGETS, weighted=family == "S-RND", kinds=kinds
        )
    frame = pd.DataFrame.from_records(records)
    return frame.sort_values(
        ["spec", "analysis", "family", "target", "stratum_kind", "stratum", "metric"],
        kind="mergesort",
    ).reset_index(drop=True)


# ----------------------------------------------------------------------------- driver
def run_event3(
    inp: Event3Inputs,
    protocol: ProtocolParams,
    *,
    log_kappa: LogKappa,
    aoa: AoaFn,
    n_boot: int | None = None,
) -> Event3Result:
    """Assemble Event 3 (docs/3 13.4, 13.13) from sealed inputs; ``n_boot`` for tests only."""
    frame = analysis_frame(inp.site_table, "primary")
    cuts = es.da_cuts(frame)
    nested_sid, nested_lro = _nested_tables(inp, frame, protocol)
    strata_sid = inp.audit_strata[["site_no", "repeat", "outer_fold", "far", "nested"]]
    strata_lro = nested_lro[["site_no", "repeat", "outer_fold", "nested"]]
    sets = _build_sets(inp, cuts, strata_sid, strata_lro)
    common = {"da_quintile_cuts_km2": list(cuts), "audit_label": inp.audit["label"]["text"]}
    docs = {
        "hypotheses": {**common, **_hypotheses(sets, inp, protocol, log_kappa, n_boot)},
        "uq": {**common, **_uq(sets, inp, protocol, cuts)},
        "consistency": {**common, **_consistency(sets, inp, protocol)},
        "dependence": {
            **common,
            **_dependence(sets, inp, protocol, nested_sid, nested_lro, aoa),
        },
        "descriptive": {**common, **_descriptive(sets, inp, protocol, log_kappa)},
    }
    return Event3Result(documents=docs, matrix=_matrix(sets))

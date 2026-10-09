"""Architecture contract of the torch ladder (Phase 7, Step 1; P7-D4 (a)).

Sources: docs/3 section 4.1 (ladder rows and heads; ablations A1, A2, A3-SP, A3-SR, A4, w_A = 1;
I_blind routing: the n-head receives every feature except Q_E, S, F_S and V_E; the W- and r-heads
receive the full feature set), section 4.2 (backbone MLP 3 x 128, SiLU, LayerNorm, no BatchNorm;
CPU float64; sigma_min = 0.01), section 3.5 L4 (banned predictors; V_E only in A_J); docs/2
Prop 3.4 (no batch-coupled layer upstream of Phi, so the model acts row-wise), section 7.8 and
section 7.9 (M6-NN and M3 squared losses: no sigma head), sections 8.5 and 9.1 (sigma = sigma_min
+ softplus; sigma_A = sqrt(sigma_W^2 + sigma_dbar^2)).

Expected values (the ladder list, head lists, widths, parameter counts, routed-out columns) are
derived by hand from those sections, never read from the modules under test. Two layouts are
used: the real NN feature set of ``params.transforms.features`` (F = 22) and its A4 reduction
(F = 20). Statistics are synthetic (no data are read).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch
from torch import Tensor, nn

from domain_strategies import LN_S_FLOOR, PROTOCOL
from src.data.dataset import BLIND_EXCLUDED as DATASET_BLIND_EXCLUDED
from src.data.normstats import NormStats
from src.data.schemas import BANNED_PREDICTORS
from src.models.backbone import Backbone
from src.models.baselines import BLIND_EXCLUDED as BASELINE_BLIND_EXCLUDED
from src.models.baselines import nn_features
from src.models.checkpoint import build_model
from src.models.heads import LadderNet, ModelOutput, RoutedHeads
from src.models.registry import ABLATIONS, NN_MODELS, XGB_MODELS, ModelSpec, nn_grid

pytestmark = pytest.mark.unit

#: docs/3 section 4.1: every torch ladder configuration (5 rungs, 5 M5 ablations, A4 on the
#: other NN rungs except M3, whose Strickler branch needs D50).
LADDER_SPECS: tuple[tuple[str, str | None], ...] = (
    ("M2-MLP", None),
    ("M3", None),
    ("M4", None),
    ("M5", None),
    ("M6-NN", None),
    ("M5", "A1"),
    ("M5", "A2"),
    ("M5", "A3-SP"),
    ("M5", "A3-SR"),
    ("M5", "wA1"),
    ("M2-MLP", "A4"),
    ("M4", "A4"),
    ("M5", "A4"),
    ("M6-NN", "A4"),
)
SPEC_IDS = [m if a is None else f"{m}_{a}" for m, a in LADDER_SPECS]
BATCH_SIZES = (1, 16, 64, 256)  # PHASE_07 requirement 3

#: docs/3 section 4.2: MLP 3 x 128 (SiLU, LayerNorm); sigma_min = 0.01 (P3-13).
HIDDEN_LAYERS, HIDDEN_WIDTH, SIGMA_MIN = 3, 128, 0.01
#: docs/3 section 4.1 (routing, P3-11): withheld from the n-head; V_E is never a feature at all.
ROUTED_OUT = ("Q_E", "S", "F_S")
#: docs/3 section 4.1 (ablation A4): without D50 and AI.
A4_DROPPED = ("D50_m", "AI")
#: Raw outputs per rung as (full-trunk outputs, blind-trunk outputs), from docs/3 section 4.1 and
#: docs/2 section 9.1 (three direct sigma heads sigma_W, sigma_dbar, sigma_t; sigma_A derived).
HEAD_OUTPUTS: dict[str, tuple[int, int]] = {
    "M2-MLP": (8, 0),  # four independent log means + one Gaussian sigma per target (4.1 M2 row)
    "M3": (1, 0),  # the W regressor only (4.1 M3 row; docs/2 7.9)
    "M4": (6, 1),  # w, ell, r + 3 sigma heads; nu on I_blind (4.1 M4 row)
    "M5": (5, 1),  # w, r + 3 sigma heads; nu on I_blind (4.1 M5 row)
    "M6-NN": (2, 1),  # g_w, g_t; g_nu on I_blind (4.1 M6 row)
}
#: Rungs trained with a squared loss and no sigma head (docs/2 7.8 stage 2, 7.9).
NO_SIGMA = frozenset({"M3", "M6-NN"})
#: docs/3 section 3.5 L4 (target leakage) and ADL-005: never a predictor.
L4_BANNED = frozenset(
    {"V_E", "A_J", "q_va", "meas_q_va", "drain_area_va", "chan_mat_cd", "mean_vel_va",
     "max_vel_va", "W", "dbar", "dmax", "A", "t_obs"}
)  # fmt: skip
MODEL_TOL = 1e-12  # docs/3 section 12.1: Prop 3.4, whole model


# ----------------------------------------------------------------------------- layouts
@dataclass(frozen=True)
class Layout:
    """A feature layout: names, synthetic ``NormStats`` and the docs/3 4.1 ``I_blind``."""

    features: tuple[str, ...]
    stats: NormStats
    blind: Tensor

    @property
    def n(self) -> int:
        return len(self.features)

    def column(self, name: str) -> int:
        return self.features.index(name)


def _layout(features: tuple[str, ...]) -> Layout:
    kinds = PROTOCOL.transforms.features
    d50 = [name == "D50_m" for name in features]
    stats = NormStats(
        names=features,
        kinds=tuple(kinds[name] for name in features),
        offsets=(0.0,) * len(features),
        mean=tuple(math.log(0.02) if d else 0.0 for d in d50),
        std=tuple(1.5 if d else 1.0 for d in d50),
        n_fit=100,
    )
    blind = [j for j, name in enumerate(features) if name not in (*ROUTED_OUT, "V_E")]
    return Layout(features, stats, torch.tensor(blind, dtype=torch.int64))


FULL = _layout(nn_features(PROTOCOL))
A4 = _layout(tuple(f for f in FULL.features if f not in A4_DROPPED))


def layout_of(ablation: str | None) -> Layout:
    return A4 if ablation == "A4" else FULL


def build(model: str, ablation: str | None) -> LadderNet:
    """Seeded rung on its layout (the grid point is irrelevant to the architecture)."""
    torch.manual_seed(0)
    lay = layout_of(ablation)
    return build_model(
        ModelSpec(model, ablation),
        PROTOCOL,
        stats=lay.stats,
        blind_idx=lay.blind,
        hyper=nn_grid(PROTOCOL, model)[0],
        r_bar=2.0 if model == "M3" else None,
    )


def batch(b: int, lay: Layout, seed: int) -> dict[str, Tensor]:
    """Seeded ADL-077 batch with physics inputs inside D (docs/2 section 1.3)."""
    g = torch.Generator().manual_seed(seed)

    def u(lo: float, hi: float) -> Tensor:
        return torch.empty(b, dtype=torch.float64).uniform_(lo, hi, generator=g)

    q, s, w = u(math.log(0.01), math.log(1e4)), u(LN_S_FLOOR, math.log(0.1)), u(0.0, 6.0)
    y = torch.stack([w, w - 3.0, torch.full_like(w, 0.4), 2.0 * w - 3.0], dim=1)
    return {
        "x": torch.randn(b, lay.n, dtype=torch.float64, generator=g),
        "x_blind_idx": lay.blind,
        "phys": torch.stack([q, s], dim=1),
        "y": y,
        "omega": torch.full((b,), 1.0 / b, dtype=torch.float64),
        "row": torch.arange(b),
    }


def _linear(n_in: int, n_out: int) -> int:
    return n_in * n_out + n_out


def _trunk(n_in: int) -> int:
    """Hand count of ``Linear -> LayerNorm -> SiLU`` x 3 at width 128 (docs/3 4.2)."""
    total, width = 0, n_in
    for _ in range(HIDDEN_LAYERS):
        total += _linear(width, HIDDEN_WIDTH) + 2 * HIDDEN_WIDTH  # LayerNorm weight and bias
        width = HIDDEN_WIDTH
    return total


def hand_count(model: str, lay: Layout) -> int:
    n_full, n_blind = HEAD_OUTPUTS[model]
    total = _trunk(lay.n) + _linear(HIDDEN_WIDTH, n_full)
    if n_blind:
        total += _trunk(int(lay.blind.numel())) + _linear(HIDDEN_WIDTH, n_blind)
    return total


# ----------------------------------------------------------------------------- protocol values
def test_ladder_and_protocol_values_match_docs() -> None:
    """docs/3 sections 4.1-4.2: the registry's torch configurations are exactly the 14 ladder
    specs; backbone 3 x 128 SiLU LayerNorm; sigma_min = 0.01; F = 22 with 19 I_blind columns.
    """
    registry = [(m, None) for m in NN_MODELS] + [
        (m, a) for a, models in ABLATIONS.items() for m in NN_MODELS if m in models
    ]
    assert sorted(registry, key=str) == sorted(LADDER_SPECS, key=str), (
        f"registry ladder {registry} != docs/3 4.1 ladder {list(LADDER_SPECS)}"
    )
    bb = PROTOCOL.model.backbone
    got = (bb.hidden_layers, bb.hidden_width, bb.activation, bb.norm, PROTOCOL.uq.sigma_min)
    want = (HIDDEN_LAYERS, HIDDEN_WIDTH, "silu", "layernorm", SIGMA_MIN)
    assert got == want, f"params backbone/sigma_min {got} != docs/3 4.2 {want}"
    assert (FULL.n, int(FULL.blind.numel()), A4.n, int(A4.blind.numel())) == (22, 19, 20, 17), (
        f"layouts F/F_blind = {FULL.n}/{FULL.blind.numel()}, A4 {A4.n}/{A4.blind.numel()}"
    )


# ----------------------------------------------------------------------------- shapes
@pytest.mark.parametrize("b", BATCH_SIZES)
@pytest.mark.parametrize(("model", "ablation"), LADDER_SPECS, ids=SPEC_IDS)
def test_output_contract(model: str, ablation: str | None, b: int) -> None:
    """PHASE_07 requirement 3 and docs/3 4.2: ``mu`` (B, 4) = [ln W, ln dbar, t, ln A], float64,
    CPU, finite; ``sigma`` (B, 4) >= sigma_min (docs/2 8.5, 9.1) or ``None`` for the squared-loss
    rungs M3 and M6-NN (docs/2 7.8, 7.9); M4/M5 sigma_A = sqrt(sigma_W^2 + sigma_dbar^2) (9.1).
    """
    net = build(model, ablation)
    with torch.no_grad():
        out: ModelOutput = net(batch(b, layout_of(ablation), seed=b))
    where = f"{model} {ablation or ''} B={b}"
    mu = out.mu
    assert tuple(mu.shape) == (b, 4), f"mu shape {tuple(mu.shape)} != {(b, 4)} at {where}"
    assert mu.dtype == torch.float64, f"mu dtype {mu.dtype} at {where}"
    assert mu.device.type == "cpu", f"mu device {mu.device} at {where}"
    bad = (~torch.isfinite(mu)).nonzero()
    assert bad.numel() == 0, f"non-finite mu at (row, col) {bad[:5].tolist()} at {where}"
    if model in NO_SIGMA:
        assert out.sigma is None, f"{where}: squared-loss rung must not emit sigma (docs/2 7.8-7.9)"
        return
    sig = out.sigma
    assert sig is not None, f"{where}: heteroscedastic rung lacks sigma (docs/3 4.1-4.2)"
    assert tuple(sig.shape) == (b, 4), f"sigma shape {tuple(sig.shape)} at {where}"
    assert sig.dtype == torch.float64, f"sigma dtype {sig.dtype} at {where}"
    assert bool(torch.isfinite(sig).all()), f"non-finite sigma at {where}"
    low = float(sig.min())
    assert low >= SIGMA_MIN, (
        f"min sigma = {low!r} < sigma_min = {SIGMA_MIN} (docs/2 8.5) at {where}"
    )
    if model in ("M4", "M5"):
        want = torch.sqrt(sig[:, 0] ** 2 + sig[:, 1] ** 2)
        err = float(((sig[:, 3] - want).abs() / want).max())
        assert err <= 4.0 * torch.finfo(torch.float64).eps, (
            f"sigma_A vs sqrt(sigma_W^2 + sigma_dbar^2): relative {err:.3e} > 4 eps "
            f"(docs/2 9.1) at {where}"
        )


# ----------------------------------------------------------------------------- parameters
@pytest.mark.parametrize(("model", "ablation"), LADDER_SPECS, ids=SPEC_IDS)
def test_trainable_parameter_count(model: str, ablation: str | None) -> None:
    """``get_num_trainable_params()`` equals the hand count of docs/3 4.1-4.2: one 3 x 128
    LayerNorm trunk on x plus a linear head over the rung's raw outputs, and, for rungs with an
    n-head, a second trunk on x[:, I_blind] with its own head (P6-D4 (a)). Worked example:
    M5 on F = 22 is 36,736 + 645 + 36,352 + 129 = 73,862. Every parameter is trainable.
    """
    net = build(model, ablation)
    want = hand_count(model, layout_of(ablation))
    got = net.get_num_trainable_params()
    total = sum(int(p.numel()) for p in net.parameters())
    assert got == want, f"{model} {ablation or ''}: {got} trainable parameters != hand count {want}"
    assert total == got, f"{model} {ablation or ''}: {total - got} frozen parameters"


# ----------------------------------------------------------------------------- batch coupling
ALLOWED_MODULES = (
    LadderNet,
    RoutedHeads,
    Backbone,
    nn.Sequential,
    nn.Linear,
    nn.LayerNorm,
    nn.SiLU,
)
BATCH_STAT_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


@pytest.mark.parametrize(("model", "ablation"), LADDER_SPECS, ids=SPEC_IDS)
def test_no_batch_statistic_upstream_of_phi(model: str, ablation: str | None) -> None:
    """docs/2 Prop 3.4 scope and docs/3 4.2 (LayerNorm, no BatchNorm): only Linear, LayerNorm
    over the 128 hidden features of its own row, SiLU and containers occur; no running-statistic
    buffer exists; training and evaluation mode give bit-identical outputs; in training mode
    each row's mu equals its single-row evaluation to 1e-12 (docs/3 12.1, whole model).
    """
    net = build(model, ablation)
    where = f"{model} {ablation or ''}"
    foreign = [
        f"{n}: {type(m).__name__}"
        for n, m in net.named_modules()
        if not isinstance(m, ALLOWED_MODULES)
    ]
    assert not foreign, f"{where}: modules outside the docs/3 4.2 backbone: {foreign}"
    stats_buffers = [n for n, _ in net.named_buffers() if n.split(".")[-1] in BATCH_STAT_BUFFERS]
    assert not stats_buffers, f"{where}: batch-statistic buffers {stats_buffers}"
    norms = [m for m in net.modules() if isinstance(m, nn.LayerNorm)]
    shapes = {tuple(m.normalized_shape) for m in norms}
    assert len(norms) >= HIDDEN_LAYERS, f"{where}: {len(norms)} LayerNorm layers"
    assert shapes == {(HIDDEN_WIDTH,)}, f"{where}: LayerNorm shapes {shapes}"

    lay = layout_of(ablation)
    full = batch(16, lay, seed=3)
    with torch.no_grad():
        net.train()
        mu_train = net(full).mu
        net.eval()
        mu_eval = net(full).mu
        net.train()
        diff = float((mu_train - mu_eval).abs().max())
        assert diff == 0.0, f"{where}: train vs eval mode differ by {diff:.3e} (batch statistic?)"
        for i in range(16):
            row = {k: (v if k == "x_blind_idx" else v[i : i + 1]) for k, v in full.items()}
            row["omega"] = torch.ones(1, dtype=torch.float64)
            err = float((net(row).mu[0] - mu_train[i]).abs().max())
            assert err <= MODEL_TOL, (
                f"{where}: row {i} alone vs in a batch of 16 differs by {err:.3e} > {MODEL_TOL} "
                "(docs/2 Prop 3.4; docs/3 12.1)"
            )


# ----------------------------------------------------------------------------- routing
ROUTED_SPECS = tuple((m, a) for m, a in LADDER_SPECS if HEAD_OUTPUTS[m][1])
ROUTED_IDS = [m if a is None else f"{m}_{a}" for m, a in ROUTED_SPECS]


@pytest.mark.parametrize(("model", "ablation"), ROUTED_SPECS, ids=ROUTED_IDS)
def test_n_head_routing_zero_gradient(model: str, ablation: str | None) -> None:
    """docs/3 4.1 (P3-11, O-4): d nu / d x_j == 0 exactly for j in {Q_E, S, F_S}; the gradient on
    every I_blind column is live (non-zero somewhere in the batch); the W head sees the full set,
    so d w / d x_j is live on the routed-out columns too. (M6-NN: nu = g_nu, w = g_w.)
    """
    net = build(model, ablation)
    lay = layout_of(ablation)
    data = batch(64, lay, seed=5)
    x = data["x"].clone().requires_grad_()
    data["x"] = x
    out = net(data)
    nu = out.extras["g_nu" if model == "M6-NN" else "nu"]
    w = out.extras["g_w"] if model == "M6-NN" else out.mu[:, 0]
    (g_nu,) = torch.autograd.grad(nu.sum(), x, retain_graph=True)
    (g_w,) = torch.autograd.grad(w.sum(), x)
    where = f"{model} {ablation or ''}"
    routed = [lay.column(name) for name in ROUTED_OUT]
    for name, j in zip(ROUTED_OUT, routed, strict=True):
        leak = float(g_nu[:, j].abs().max())
        assert leak == 0.0, f"{where}: |d nu / d {name}| = {leak:.3e}; required exactly 0 (4.1)"
        live = float(g_w[:, j].abs().max())
        assert live > 0.0, f"{where}: d w / d {name} is identically 0; W sees all features (4.1)"
    dead = [lay.features[j] for j in lay.blind.tolist() if float(g_nu[:, j].abs().max()) == 0.0]
    assert not dead, f"{where}: I_blind columns with zero n-head gradient: {dead}"


# ----------------------------------------------------------------------------- features
def _all_specs() -> list[ModelSpec]:
    specs = [ModelSpec(m) for m in NN_MODELS + XGB_MODELS]
    specs += [ModelSpec(m, a) for a, models in ABLATIONS.items() for m in sorted(models)]
    return specs


def test_v_e_is_never_a_feature_and_l4_holds() -> None:
    """docs/3 4.1 (V_E is never a feature; used only in A_J) and 3.5 L4: no feature set of any
    ladder model or ablation contains V_E or a banned predictor; I_blind withholds exactly
    Q_E, S, F_S and V_E in both modules that restate it.
    """
    for spec in _all_specs():
        feats = set(spec.features(PROTOCOL))
        assert "V_E" not in feats, f"{spec.name}: V_E is a feature (docs/3 4.1)"
        bad = sorted(feats & (L4_BANNED | BANNED_PREDICTORS))
        assert not bad, f"{spec.name}: banned predictors {bad} (docs/3 3.5 L4)"
    assert "V_E" not in PROTOCOL.transforms.features, "V_E in params.transforms.features"
    want = frozenset({*ROUTED_OUT, "V_E"})
    assert want == DATASET_BLIND_EXCLUDED, f"dataset I_blind exclusions {DATASET_BLIND_EXCLUDED}"
    assert want == BASELINE_BLIND_EXCLUDED, (
        f"baselines I_blind exclusions {BASELINE_BLIND_EXCLUDED}"
    )

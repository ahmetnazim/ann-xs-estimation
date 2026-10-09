"""Smoke tests for src/models/{backbone,heads}.py (Phase 6, Batch B). Synthetic; no data.

Sources: docs/3 4.1 (routing I_blind), 4.2 (backbone, sigma_min);
docs/2 7.5 (A3-SR prior), 7.6 (bounded transforms), 8.5 (sigma floor).
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from src.config import load_settings
from src.models.backbone import Backbone
from src.models.heads import BoundedTransforms, RoutedHeads, sigma_from_raw, strickler_logit

pytestmark = pytest.mark.unit

PROTOCOL = load_settings().protocol
BACKBONE = PROTOCOL.model.backbone
DOMAIN = PROTOCOL.domain
SIGMA_MIN = PROTOCOL.uq.sigma_min
VARPI = PROTOCOL.priors.strickler_residual.varpi
TR = BoundedTransforms.from_domain(DOMAIN)
F = 9
BLIND = [0, 1, 2, 5, 6, 7]  # features 3, 4, 8 play Q_E, S, F_S (routed out)


def rand_x(b: int, seed: int = 0, f: int = F) -> Tensor:
    return torch.randn(b, f, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


def m5_like(seed: int = 0) -> RoutedHeads:
    torch.manual_seed(seed)
    return RoutedHeads(F, ["z_w", "z_r", "s_w", "s_dbar", "s_t"], ["z_n"], BLIND, BACKBONE)


# --- backbone -------------------------------------------------------------------------------


def test_backbone_shape_dtype_and_layers() -> None:
    torch.manual_seed(0)
    net = Backbone(F, BACKBONE)
    out = net(rand_x(5))
    assert tuple(out.shape) == (5, BACKBONE.hidden_width)
    assert out.dtype == torch.float64
    assert all(p.dtype == torch.float64 for p in net.parameters())
    kinds = [type(m) for m in net.modules()]
    assert kinds.count(nn.LayerNorm) == BACKBONE.hidden_layers
    assert kinds.count(nn.SiLU) == BACKBONE.hidden_layers
    assert not any(issubclass(k, nn.modules.batchnorm._BatchNorm) for k in kinds)
    h = BACKBONE.hidden_width
    n_params = (
        (F * h + h) + (BACKBONE.hidden_layers - 1) * (h * h + h) + BACKBONE.hidden_layers * 2 * h
    )
    assert sum(p.numel() for p in net.parameters()) == n_params


def test_backbone_is_row_wise_in_train_and_eval() -> None:
    torch.manual_seed(1)
    net = Backbone(F, BACKBONE)
    x = rand_x(16, seed=1)
    net.train()
    full = net(x)
    rows = torch.cat([net(x[i : i + 1]) for i in range(16)])
    assert torch.allclose(full, rows, rtol=0, atol=1e-13)
    net.eval()
    assert torch.allclose(net(x), full, rtol=0, atol=0)
    leaf = x.clone().requires_grad_()
    (g,) = torch.autograd.grad(net(leaf)[3].sum(), leaf)
    assert bool((g[torch.arange(16) != 3] == 0).all())


@pytest.mark.parametrize(
    ("x", "error"),
    [
        (torch.zeros(2, F, dtype=torch.float32), TypeError),
        (torch.zeros(2, F + 1, dtype=torch.float64), ValueError),
        (torch.zeros(F, dtype=torch.float64), ValueError),
        (torch.full((2, F), float("nan"), dtype=torch.float64), ValueError),
    ],
)
def test_backbone_rejects_bad_input(x: Tensor, error: type) -> None:
    with pytest.raises(error):
        Backbone(F, BACKBONE)(x)


# --- bounded transforms, sigma floor, Strickler logit ---------------------------------------


def test_transforms_from_domain() -> None:
    assert TR.nu_min == math.log(0.005)
    assert math.isclose(TR.delta_n, math.log(120.0), rel_tol=1e-15)
    assert TR.ln_r_min == math.log(0.5)
    assert math.isclose(TR.delta_r, math.log(20.0), rel_tol=1e-15)
    z0 = torch.zeros(1, dtype=torch.float64)
    assert math.isclose(float(TR.nu(z0)[0]), TR.nu_min + TR.delta_n / 2, rel_tol=1e-15)
    assert math.isclose(float(TR.r(z0)[0]), math.sqrt(0.5 * 10.0), rel_tol=1e-14)


def test_transforms_bounds_and_derivatives_proofs_7_6() -> None:
    z = torch.linspace(-12.0, 12.0, 97, dtype=torch.float64).requires_grad_()
    nu = TR.nu(z)
    nu_max = TR.nu_min + TR.delta_n
    assert bool((nu > TR.nu_min).all())
    assert bool((nu < nu_max).all())
    (g,) = torch.autograd.grad(nu.sum(), z)
    want = (nu - TR.nu_min) * (nu_max - nu) / TR.delta_n
    assert torch.allclose(g, want.detach(), rtol=1e-9, atol=0)  # formula cancels at large |z|
    assert float(g.max()) <= TR.delta_n / 4 + 1e-15
    r = TR.r(z)
    assert bool((r > 0.5).all())
    assert bool((r < 10.0).all())
    (gr,) = torch.autograd.grad(r.sum(), z)
    s = torch.sigmoid(z.detach())
    assert torch.allclose(gr, (r * TR.delta_r * s * (1 - s)).detach(), rtol=1e-12, atol=1e-300)


def test_sigma_floor_proofs_8_5() -> None:
    z = torch.linspace(-50.0, 50.0, 101, dtype=torch.float64).requires_grad_()
    sigma = sigma_from_raw(z, SIGMA_MIN)
    assert bool((sigma >= SIGMA_MIN).all())
    (g,) = torch.autograd.grad(sigma.sum(), z)
    assert torch.allclose(
        g, torch.sigmoid(z.detach()), rtol=1e-8, atol=0
    )  # softplus is linear past 20
    with pytest.raises(ValueError, match="sigma_min"):
        sigma_from_raw(z, 0.0)


def test_strickler_logit_a3_sr() -> None:
    mid = torch.tensor([TR.nu_min + TR.delta_n / 2], dtype=torch.float64)
    assert float(strickler_logit(mid, TR, VARPI)[0]) == 0.0
    nu_s = torch.log(torch.tensor([0.02, 0.05, 0.1], dtype=torch.float64))
    base = strickler_logit(nu_s, TR, VARPI)
    assert torch.allclose(TR.nu(base), nu_s, rtol=0, atol=1e-14)  # Delta = 0 -> nu = nu_S
    outside = torch.log(torch.tensor([0.004, 0.7], dtype=torch.float64))
    clipped = TR.nu(strickler_logit(outside, TR, VARPI))
    want = torch.tensor([VARPI, 1 - VARPI], dtype=torch.float64) * TR.delta_n + TR.nu_min
    assert torch.allclose(clipped, want, rtol=0, atol=1e-14)
    # Proposition 7.2: nu - nu_S = Delta_n pi (1 - pi) Delta + O(Delta^2).
    delta = 1e-6
    pi = (nu_s - TR.nu_min) / TR.delta_n
    lin = TR.delta_n * pi * (1 - pi) * delta
    assert torch.allclose(TR.nu(base + delta) - nu_s, lin, rtol=1e-5, atol=0)
    with pytest.raises(ValueError, match="varpi"):
        strickler_logit(nu_s, TR, 0.5)


# --- routed heads (P6-D4) -------------------------------------------------------------------


def test_routed_heads_outputs() -> None:
    heads = m5_like()
    out = heads(rand_x(7))
    assert list(out) == ["z_w", "z_r", "s_w", "s_dbar", "s_t", "z_n"]
    assert all(tuple(v.shape) == (7,) and v.dtype == torch.float64 for v in out.values())
    full = {id(p) for p in heads.full_trunk.parameters()}
    assert heads.blind_trunk is not None
    blind = {id(p) for p in heads.blind_trunk.parameters()}
    assert full.isdisjoint(blind)
    assert heads.blind_trunk.in_features == len(BLIND)


def test_blind_output_has_exactly_zero_gradient_on_routed_out_features() -> None:
    heads = m5_like(seed=2)
    x = rand_x(32, seed=2).requires_grad_()
    out = heads(x)
    routed_out = [j for j in range(F) if j not in BLIND]
    (g_n,) = torch.autograd.grad(out["z_n"].sum(), x, retain_graph=True)
    assert bool((g_n[:, routed_out] == 0).all())
    assert bool((g_n[:, BLIND] != 0).any())
    (g_w,) = torch.autograd.grad(out["z_w"].sum(), x)
    assert bool((g_w[:, routed_out] != 0).all())
    x2 = x.detach().clone()
    x2[:, routed_out] += 3.0
    assert torch.equal(heads(x2)["z_n"], out["z_n"].detach())


def test_full_only_heads_and_row_independence() -> None:
    torch.manual_seed(3)
    heads = RoutedHeads(F, ["mu_w", "mu_d", "mu_dmax", "mu_a"], [], None, BACKBONE)
    assert heads.blind_trunk is None
    x = rand_x(10, seed=3)
    out = heads(x)
    one = heads(x[4:5])
    for name in out:
        assert torch.allclose(out[name][4:5], one[name], rtol=0, atol=1e-13)


@pytest.mark.parametrize(
    ("full", "blind", "idx"),
    [
        (["a"], ["b"], None),
        (["a"], [], [0, 1]),
        (["a"], ["a"], [0]),
        ([], ["b"], [0]),
        (["a"], ["b"], [0, 0, 1]),
        (["a"], ["b"], [0, F]),
        (["a"], ["b"], list(range(F))),
        (["a"], ["b"], []),
    ],
)
def test_routed_heads_rejects_bad_configuration(
    full: list[str], blind: list[str], idx: list[int] | None
) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - several distinct configuration errors
        RoutedHeads(F, full, blind, idx, BACKBONE)

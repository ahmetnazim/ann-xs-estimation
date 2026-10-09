"""Single-fit training loop: AdamW, cosine decay, early stopping with the warm-start restart.

Sources: protocol section 4.2 (AdamW, learning rate from the inner grid, cosine decay, batch 256,
at most 500 epochs, early stopping with patience 50 on the member's grouped 10 % split; 20-epoch
MSE warm start), section 3.4 (inner fits stop on ``inner_val(k)``, members on ``es_val(m)``);
proofs section 7.0 (``omega_i = 1/|J_tr|``: one epoch's mini-batch sums give the full
objective), section 7.7.1 and (A7.3) (``Xi_e = lambda_M(e) ||grad P|| / ||grad L_data||``, logged
per epoch for M4), Proposition 3.4 (Phi non-convergence raises); P6-D5 (a) with sub-rules
(i) early-stopping tracking and patience restart at the warm-start switch and (ii) the M4
monitor at ``lambda_max`` (implemented in ``M4Soft.objective(monitor=True)``).

Conventions (Batch D):

* AdamW with decoupled ``weight_decay`` on **all** parameters (Q-D5 (a)); PyTorch defaults
  ``betas = (0.9, 0.999)``, ``eps = 1e-8``; no gradient clipping (none is pre-registered).
* Cosine decay ``CosineAnnealingLR(T_max = max_epochs, eta_min = 0)``, stepped once per epoch.
* Monitor = the model's own objective with ``monitor=True`` on the validation part, summed over
  its rows with ``omega = 1/|J_val|`` (a mean), evaluated after each epoch under ``no_grad``.
* Improvement is strict (``<``); the best state is copied and restored at the end.
* ``Xi_e`` is computed after epoch ``e``'s updates, on the whole training part as one batch.
* ``PhiConvergenceError`` and non-finite losses propagate; nothing is caught or clipped.

Torch-only, float64, CPU. Owning phase: 6.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor

from src.config import ProtocolParams
from src.data.dataset import SiteDataset, make_loader
from src.models.heads import Batch, LadderNet, ModelOutput

__all__ = [
    "EarlyStopping",
    "EpochRecord",
    "FitResult",
    "TrainConfig",
    "evaluate_monitor",
    "fit",
    "predict",
    "xi_ratio",
]


@dataclass(frozen=True)
class TrainConfig:
    """Optimizer and stopping settings of one fit (protocol 4.2; P6-D5 (a))."""

    lr: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    seed: int
    restart_epoch: int | None = None

    def __post_init__(self) -> None:
        """Validate ranges; ``restart_epoch`` must lie in ``1 .. max_epochs - 1`` when given."""
        if not (self.lr > 0 and self.weight_decay >= 0):
            raise ValueError("need lr > 0 and weight_decay >= 0")
        if min(self.batch_size, self.max_epochs, self.patience) < 1:
            raise ValueError("batch_size, max_epochs and patience must be >= 1")
        if self.restart_epoch is not None and not 0 < self.restart_epoch < self.max_epochs:
            raise ValueError("restart_epoch must lie in 1 .. max_epochs - 1")

    @classmethod
    def from_protocol(
        cls,
        protocol: ProtocolParams,
        *,
        lr: float,
        weight_decay: float,
        seed: int,
        warm_start: bool,
    ) -> TrainConfig:
        """``model.optimizer`` values; restart at ``uq.warm_start_epochs`` if ``warm_start``."""
        opt = protocol.model.optimizer
        warm = protocol.uq.warm_start_epochs
        return cls(
            lr=lr,
            weight_decay=weight_decay,
            batch_size=opt.batch_size,
            max_epochs=opt.max_epochs,
            patience=opt.patience,
            seed=seed,
            restart_epoch=warm if warm_start and warm > 0 else None,
        )


@dataclass
class EarlyStopping:
    """Patience rule with a restart epoch (P6-D5 (i)).

    At ``restart_epoch`` the best value and the patience counter are reset, so the restored
    state always comes from the (beta-)NLL phase and the warm-start MSE values are never compared
    with NLL values.
    """

    patience: int
    restart_epoch: int | None = None
    best: float = math.inf
    best_epoch: int = -1
    wait: int = 0

    def update(self, epoch: int, value: float) -> tuple[bool, bool]:
        """Record ``value`` at ``epoch``; return ``(improved, stop)``. Non-finite values raise."""
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite monitor {value} at epoch {epoch}")
        if self.restart_epoch is not None and epoch == self.restart_epoch:
            self.best, self.best_epoch, self.wait = math.inf, -1, 0
        if value < self.best:
            self.best, self.best_epoch, self.wait = value, epoch, 0
            return True, False
        self.wait += 1
        return False, self.wait >= self.patience


@dataclass(frozen=True)
class EpochRecord:
    """One epoch: training objective, monitor, learning rate, parts, ``Xi_e``, Picard counts."""

    epoch: int
    train: float
    monitor: float
    lr: float
    parts: dict[str, float]
    xi: float | None
    mean_iterations: float | None
    max_iterations: int | None
    seconds: float


@dataclass(frozen=True)
class FitResult:
    """Outcome of :func:`fit`; the model holds the restored best state."""

    best_epoch: int
    best_monitor: float
    epochs_run: int
    stopped_early: bool
    wall_s: float
    history: tuple[EpochRecord, ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable dict (history included)."""
        out = asdict(self)
        out["history"] = [asdict(r) for r in self.history]
        return out


def _scalar(t: Tensor) -> float:
    return float(t.detach())


def evaluate_monitor(
    model: LadderNet, loader: torch.utils.data.DataLoader[Batch], epoch: int
) -> float:
    """Sum of ``model.objective(..., monitor=True).total`` over ``loader`` under ``no_grad``."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            total += _scalar(model.objective(batch, out, epoch, monitor=True).total)
    return total


def xi_ratio(model: LadderNet, batch: Batch, epoch: int) -> float:
    """``Xi_e = ||grad parts.penalty|| / ||grad parts.data||`` (A7.3; ``penalty = lambda_M(e) P``).

    Gradients over all trainable parameters on ``batch``; a zero data gradient raises.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    model.train()
    out = model(batch)
    parts = model.objective(batch, out, epoch).parts
    if "penalty" not in parts:
        raise KeyError("Xi_e needs an objective with a 'penalty' part (M4)")
    g_data = torch.autograd.grad(parts["data"], params, retain_graph=True, allow_unused=True)
    g_pen = torch.autograd.grad(parts["penalty"], params, allow_unused=True)

    def norm(grads: tuple[Tensor | None, ...]) -> float:
        sq = sum(float((g.detach() ** 2).sum()) for g in grads if g is not None)
        return math.sqrt(sq)

    n_data = norm(g_data)
    if not n_data > 0:
        raise FloatingPointError("zero data gradient: Xi_e undefined")
    return norm(g_pen) / n_data


def _clone_state(model: LadderNet) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def fit(
    model: LadderNet,
    train_set: SiteDataset,
    val_set: SiteDataset,
    cfg: TrainConfig,
    *,
    xi: bool = False,
    logger: logging.Logger | None = None,
) -> FitResult:
    """Train ``model`` on ``train_set`` and stop on ``val_set`` (protocol 4.2; P6-D5 (a)).

    The caller seeds torch before building the model (``src.utils.seeding.seed_torch``); the
    training loader is shuffled with ``cfg.seed``. ``xi=True`` logs ``Xi_e`` (M4). Returns the
    record; ``model`` ends in its best state (strict improvement, after the restart epoch).
    """
    start = time.perf_counter()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs, eta_min=0.0)
    train_loader = make_loader(train_set, cfg.batch_size, train=True, seed=cfg.seed)
    val_loader = make_loader(val_set, cfg.batch_size, train=False, seed=cfg.seed)
    full_train = train_set[list(range(len(train_set)))] if xi else None
    stopper = EarlyStopping(cfg.patience, cfg.restart_epoch)
    best_state: dict[str, Tensor] | None = None
    history: list[EpochRecord] = []
    stopped = False
    for epoch in range(cfg.max_epochs):
        t0 = time.perf_counter()
        lr = float(opt.param_groups[0]["lr"])
        model.train()
        total = 0.0
        parts: dict[str, float] = {}
        it_sum, it_rows, it_max = 0, 0, -1
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            out = model(batch)
            obj = model.objective(batch, out, epoch)
            if not bool(torch.isfinite(obj.total)):
                raise FloatingPointError(f"non-finite training objective at epoch {epoch}")
            torch.autograd.backward(obj.total)
            opt.step()
            total += _scalar(obj.total)
            for name, value in obj.parts.items():
                parts[name] = parts.get(name, 0.0) + _scalar(value)
            iters = out.extras.get("iterations")
            if iters is not None:
                it_sum += int(iters.sum())
                it_rows += int(iters.numel())
                it_max = max(it_max, int(iters.max()))
        sched.step()
        monitor = evaluate_monitor(model, val_loader, epoch)
        xi_e = xi_ratio(model, full_train, epoch) if full_train is not None else None
        improved, stop = stopper.update(epoch, monitor)
        if improved:
            best_state = _clone_state(model)
        record = EpochRecord(
            epoch=epoch,
            train=total,
            monitor=monitor,
            lr=lr,
            parts=parts,
            xi=xi_e,
            mean_iterations=it_sum / it_rows if it_rows else None,
            max_iterations=it_max if it_rows else None,
            seconds=time.perf_counter() - t0,
        )
        history.append(record)
        if logger is not None:
            logger.debug("epoch %d train %.6g monitor %.6g lr %.3g", epoch, total, monitor, lr)
        if stop:
            stopped = True
            break
    if best_state is None:  # pragma: no cover - unreachable: the first epoch always improves
        raise RuntimeError("no best state recorded")
    model.load_state_dict(best_state)
    model.eval()
    return FitResult(
        best_epoch=stopper.best_epoch,
        best_monitor=stopper.best,
        epochs_run=len(history),
        stopped_early=stopped,
        wall_s=time.perf_counter() - start,
        history=tuple(history),
    )


def predict(model: LadderNet, dataset: SiteDataset, batch_size: int) -> ModelOutput:
    """Concatenate ``model(batch)`` over ``dataset`` in row order under ``no_grad``.

    ``extras`` keeps the per-row tensors (first dimension = batch size) only.
    """
    model.eval()
    loader = make_loader(dataset, batch_size, train=False, seed=0)
    mus: list[Tensor] = []
    sigmas: list[Tensor] = []
    extras: dict[str, list[Tensor]] = {}
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            b = int(out.mu.shape[0])
            mus.append(out.mu)
            if out.sigma is not None:
                sigmas.append(out.sigma)
            for k, v in out.extras.items():
                if v.dim() >= 1 and int(v.shape[0]) == b:
                    extras.setdefault(k, []).append(v)
    sigma = torch.cat(sigmas) if sigmas else None
    return ModelOutput(
        mu=torch.cat(mus), sigma=sigma, extras={k: torch.cat(v) for k, v in extras.items()}
    )

"""Training of the treatment policy and of the PULSE phases.

Every phase trains on teacher-forced histories: the model reads the recorded
hours up to each position and is scored on the next one. The policy is
trained first and then frozen; each PULSE phase reads it for the
vasopressor descriptor of its preview pass (`pulse.feedback`).

The PULSE loss is the sum, in this order, of

* the negative log density of the standardised next values under the
  model's head (`pulse.heads`), averaged over recorded non-ordinal entries,
* the squared error of the location for the ordinal variables (SOFA
  score and oxygen requirement level), averaged over their recorded entries;
  a point head has instead the squared error over all recorded entries,
* the cross-entropy of the next-hour binary outcomes, where an absorbing
  outcome (death, release, pulmonary edema) counts only up to and including
  its first occurrence and is held at one afterwards,
* the mean cross-entropy of the observation logits against whether each
  variable is recorded in the next row,
* with the timing head, 0.05 times a smooth L1 loss between
  log(1 + predicted hours) and log(1 + hours until the variable is next
  recorded), those hours set to 48 when the variable is not recorded again.

The policy loss is the squared error of the next bolus volume in training
standard deviations, averaged over recorded entries, plus the cross-entropy
of the next vasopressor indicator.

The G-Transformer baseline (`pulse.gtransformer`) trains as the phase
`gtransformer`, outside the published phases, with the loss of a point
head; it does not read the policy.

Optimisation: Adam (learning rate 1e-3 for the policy, 5e-4 otherwise), the
rate multiplied by 0.3 after half of the epochs, gradients clipped at norm
one, batches of 32 patients in microbatches of 8, at most 50 epochs, early
stopping after five epochs without a better validation loss once ten epochs
have run, and the weights of the best validation epoch kept.
"""
from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pulse import gtransformer as G
from pulse import heads as H
from pulse import model as M
from pulse import variables as V
from pulse.data import Part, Prepared, Protocol
from pulse.feedback import ARMS, Arm, FeedbackPULSE, build_pulse, generator_states, restore_generators

PHASES = ("policy", "sample", "distribution", "quantile_only", "probability_only")
BASELINE = "gtransformer"
WAIT_WEIGHT = 0.05
WAIT_CENSOR_HOURS = 48.0
# A fit in progress is saved for continuation after every epoch and at least this often.
CONTINUATION_SECONDS = 120.0
# Seed of the epoch order. The reference used 724 here, for model
# construction (`model.SEED`) and for the choice of training patients
# (`data.SELECTION_SEED`); they are separate seeds that share a value.
ORDER_SEED = 724


@dataclass(frozen=True)
class Settings:
    epochs: int = 50
    batch: int = 32
    micro: int = 8
    patience: int = 5
    min_epochs: int = 10


# Losses ---------------------------------------------------------------------

def policy_loss(policy: M.Policy, batch: Part) -> torch.Tensor:
    out = policy.treatments(batch["continuous"], batch["binary"], batch["static"], batch["time_hours"])
    bolus, vaso = policy.treatment_cont, policy.treatment_bin
    scale = policy.cont_stds[bolus].reshape(1, 1, -1)
    error = ((out["continuous_treatments"][:, :-1] - batch["continuous_target"][:, 1:, bolus]) / scale).square()
    recorded = batch["continuous_mask"][:, 1:, bolus]
    loss = (error * recorded).sum() / recorded.sum().clamp_min(1.0)
    return loss + F.binary_cross_entropy_with_logits(out["binary_treatment_logits"][:, :-1],
                                                     batch["binary"][:, 1:, vaso])


def _binary_loss(logits: torch.Tensor, target: torch.Tensor, absorbing: torch.Tensor) -> torch.Tensor:
    """Two-class cross-entropy over a zero logit and `logits`, with absorbing outcomes held after onset."""
    pair = torch.stack([torch.zeros_like(logits), logits], -1)
    ever = target.cummax(dim=1).values
    before = torch.cat([torch.zeros_like(ever[:, :1, :]), ever[:, :-1, :]], 1)
    counted = torch.where(absorbing.view(1, 1, -1), before == 0, torch.ones_like(before))
    target = torch.where(absorbing.view(1, 1, -1), ever, target)
    loss = F.cross_entropy(pair.reshape(-1, 2), target.reshape(-1).long(), reduction="none").view_as(target)
    return (loss * counted).sum() / (counted.sum() + 1e-8)


def _hours_to_next_record(recorded: torch.Tensor, hours: torch.Tensor) -> torch.Tensor:
    """[B, T-1, V]: hours from each row to the next row recording the variable, else the censoring value."""
    recorded = recorded > 0
    b, t, v = recorded.shape
    waits = torch.full((b, t - 1, v), WAIT_CENSOR_HOURS, device=recorded.device)
    never = torch.full((), float("inf"), device=recorded.device)
    hours = hours.to(recorded.device, torch.float32)
    for step in range(t - 1):
        ahead = hours[:, step + 1:].unsqueeze(-1) - hours[:, step:step + 1].unsqueeze(-1)
        nearest = torch.where(recorded[:, step + 1:], ahead, never.expand_as(ahead)).min(dim=1).values
        waits[:, step] = torch.where(torch.isfinite(nearest), nearest, waits[:, step])
    return waits


def pulse_loss(model: FeedbackPULSE | G.GTransformer, batch: Part, arm: Arm) -> torch.Tensor:
    history = {k: batch[k] for k in ("continuous", "binary", "static", "time_hours", "continuous_mask")}
    if isinstance(model, G.GTransformer):
        core, out = model, model(history)
    else:
        core, out = model.pulse, model(history, quantiles=arm.quantiles, probabilities=arm.probabilities)
    idx = core.output_indices
    target = core.normalize(batch["continuous_target"])[:, 1:, idx]
    recorded = batch["continuous_mask"][:, 1:, idx]
    ordinal = torch.tensor([core.continuous_names[i] in V.ORDINAL for i in idx], device=target.device).view(1, 1, -1)

    kind = core.head
    if kind == "point":
        loss = ((target - out["mu"][:, :-1]) ** 2 * recorded).sum() / (recorded.sum() + 1e-8)
    else:
        density = recorded * (~ordinal).to(recorded.dtype)
        if kind == "gaussian":
            # The reference scored the Gaussian through its output log variance, which the head already clamps.
            params = {"mu": out["mu"][:, :-1], "log_var": out["log_var"][:, :-1]}
        else:
            params = H.stack(kind, out["dist_params_list"], slice(None, -1))
        nll = H.nll(kind, params, target)
        loss = (nll.reshape_as(density) * density).sum() / (density.sum() + 1e-8)
        squared = recorded * ordinal.to(recorded.dtype)
        loss = loss + ((target - out["mu"][:, :-1]) ** 2 * squared).sum() / (squared.sum() + 1e-8)

    absorbing = torch.tensor([core.binary_names[i] in V.ABSORBING for i in core.binary_output_indices],
                             device=target.device)
    loss = loss + _binary_loss(out["bin_logits"][:, :-1], batch["binary"][:, 1:, core.binary_output_indices],
                               absorbing)

    # Elementwise then mean: the fused mean reduction gives the same value but other gradient bits.
    observed = out["obs_logits"][:, :-1]
    loss = loss + F.binary_cross_entropy_with_logits(observed, (recorded > 0).to(observed.dtype),
                                                     reduction="none").mean()

    if out["delta_t_hours"] is None:
        return loss
    predicted = out["delta_t_hours"][:, :-1]
    waits = _hours_to_next_record(batch["continuous_mask"][:, :, idx], batch["time_hours"])
    valid = (torch.isfinite(predicted) & torch.isfinite(waits)).to(predicted.dtype)
    wait = F.smooth_l1_loss(torch.log1p(predicted.clamp_min(0.0)), torch.log1p(waits.clamp_min(0.0)),
                            reduction="none")
    return loss + WAIT_WEIGHT * ((wait * valid).sum() / valid.sum().clamp(min=1.0))


def batch_loss(model, part: Part, ids: np.ndarray, phase: str, device) -> torch.Tensor:
    batch = {k: v[ids].to(device) for k, v in part.items()}
    if phase == "policy":
        return policy_loss(model, batch)
    return pulse_loss(model, batch, ARMS["sample" if phase == BASELINE else phase])


# MPS attention dropout ------------------------------------------------------

def _math_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
                    enable_gqa=False):
    """Scaled dot-product attention with dropout from ordinary tensor operations."""
    if enable_gqa:
        key = key.repeat_interleave(query.shape[-3] // key.shape[-3], -3)
        value = value.repeat_interleave(query.shape[-3] // value.shape[-3], -3)
    weights = (query @ key.transpose(-2, -1)) * (scale if scale is not None else 1 / math.sqrt(query.shape[-1]))
    if is_causal:
        if attn_mask is not None:
            raise ValueError("causal attention and an explicit mask cannot be combined")
        keep = torch.ones(query.shape[-2], key.shape[-2], device=query.device, dtype=torch.bool).tril()
        weights = weights.masked_fill(~keep, float("-inf"))
    if attn_mask is not None:
        weights = weights.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else weights + attn_mask
    return F.dropout(torch.softmax(weights, -1).nan_to_num(), p=dropout_p, training=True) @ value


@contextlib.contextmanager
def mps_attention_dropout():
    """Route attention with dropout on MPS through `_math_attention`, as the reference did when training on MPS.

    The MPS attention kernel rejects nonzero dropout on some paths. Every
    other call, including all CPU and CUDA calls, keeps the original kernel.
    """
    original = F.scaled_dot_product_attention

    def attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None,
                  enable_gqa=False):
        if query.device.type == "mps" and dropout_p > 0:
            return _math_attention(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
        return original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                        scale=scale, enable_gqa=enable_gqa)

    F.scaled_dot_product_attention = attention
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original


# Loop -----------------------------------------------------------------------

def build(phase: str, prepared: Prepared, policy: M.Policy | None,
          architecture: M.Architecture = M.Architecture()):
    """A freshly seeded model for `phase`; PULSE phases read the frozen `policy`."""
    if phase == "policy":
        return M.build_policy(prepared.protocol)
    if phase == BASELINE:
        return G.build_gtransformer(prepared.protocol, prepared.train["binary"])
    if policy is None:
        raise ValueError(f"phase {phase} needs the trained policy")
    model = build_pulse(prepared.protocol, prepared.train["binary"], architecture)
    model.attach_policy(policy)
    return model


def optimiser(model, phase: str, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3 if phase == "policy" else 5e-4)
    return optimizer, torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 2), gamma=0.3)


def epoch_order(epoch: int, patients: int) -> np.ndarray:
    return np.random.default_rng(ORDER_SEED + epoch).permutation(patients)


def update(model, optimizer, part: Part, ids: np.ndarray, phase: str, micro: int, device) -> list[float]:
    """One optimiser step over `ids` in microbatches; returns the microbatch losses.

    Each microbatch loss is weighted by its share of the batch, so the
    accumulated gradient is the gradient of the batch mean.
    """
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for offset in range(0, len(ids), micro):
        part_ids = ids[offset:offset + micro]
        loss = batch_loss(model, part, part_ids, phase, device)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        (loss * len(part_ids) / len(ids)).backward()
        losses.append(float(loss.detach()))
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError("non-finite gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return losses


def validation(model, part: Part, phase: str, micro: int, device) -> float:
    """Mean loss per patient in evaluation mode; the generators are left as they were."""
    device = torch.device(device)
    states, numpy_state = generator_states(device), np.random.get_state()
    model.eval()
    total, n = 0.0, len(part["continuous"])
    with torch.no_grad():
        for start in range(0, n, micro):
            ids = np.arange(start, min(n, start + micro))
            total += float(batch_loss(model, part, ids, phase, device)) * len(ids)
    restore_generators(states, device)
    np.random.set_state(numpy_state)
    model.train()
    return total / n


def fit(phase: str, prepared: Prepared, policy: M.Policy | None, out: Path, device: str = "cpu",
        settings: Settings = Settings(), on_epoch: Callable[[dict], None] | None = None,
        architecture: M.Architecture = M.Architecture(), name: str | None = None) -> dict:
    """Train one phase; write the best validation weights to `<out>/<name>.pt`.

    `<out>/protocol.json` receives the protocol, whose cache digest the
    checkpoint also records; a directory holding another protocol is
    refused before training. Returns the model, the validation history,
    the best validation loss and whether training stopped early. The
    returned policy is frozen and in evaluation mode, ready for the PULSE
    phases. Only the best epoch's weights are written; there is no
    last-epoch checkpoint. `on_epoch`, when given, receives each
    epoch's entry of the validation history as soon as it is known; the
    command line prints it. `architecture` chooses the PULSE variant of
    the component table (`pulse.model.Architecture`), which a PULSE
    checkpoint records. `name` defaults to the phase; a row of the
    component table trains under its own name (`pulse.rows`).

    While a fit runs, `<out>/<name>.continue.pt` holds what it needs to
    go on: weights, optimiser, schedule, random generators, the position
    in the epoch, the validation history and the best weights so far. It
    is written after every epoch and at least every `CONTINUATION_SECONDS`
    within one, and removed once `<name>.pt` is written. A fit with the
    same phase, settings, architecture, protocol and device type continues
    from it, and any other fit is refused. On the CPU a fit that is
    stopped and continued gives the weights of one that is not.

    The published weights were trained on a CUDA device. Training on the
    CPU or on Apple MPS follows the same procedure, data order and
    stopping rule, but floating-point results differ between devices, so
    it yields different weights. The tests check the procedure bitwise
    against the reference implementation on the CPU.
    """
    if phase not in PHASES + (BASELINE,):
        raise ValueError(f"unknown phase {phase}")
    protocol = Path(out) / "protocol.json"
    if protocol.exists() and Protocol.load(protocol) != prepared.protocol:
        raise ValueError(f"{protocol} holds another protocol")
    with mps_attention_dropout() if torch.device(device).type == "mps" else contextlib.nullcontext():
        return _fit(phase, prepared, policy, Path(out), device, settings, on_epoch, architecture, name or phase)


@dataclass
class _Progress:
    """The next epoch and update of a fit, its validation history and its best weights so far."""

    epoch: int = 0
    step: int = 0
    updates: int = 0
    history: list[dict] = field(default_factory=list)
    best: float = float("inf")
    best_state: dict[str, torch.Tensor] | None = None


def _exhausted(history: list[dict], settings: Settings) -> bool:
    """Whether `patience` epochs have passed without a better validation loss, once `min_epochs` have run."""
    if not history or len(history) < settings.min_epochs:
        return False
    since_best = len(history) - 1 - int(np.argmin([h["validation_loss"] for h in history]))
    return since_best >= settings.patience


def _save_continuation(path: Path, identity: dict, progress: _Progress, model, optimizer, scheduler,
                       device: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    torch.save({"identity": identity, "progress": asdict(progress),
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "generators": generator_states(torch.device(device))}, pending)
    pending.replace(path)


def _continue(path: Path, identity: dict, model, optimizer, scheduler, device: str) -> _Progress:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["identity"] != identity:
        raise ValueError(f"{path} continues another fit; remove it to start this one")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    restore_generators(saved["generators"], torch.device(device))
    return _Progress(**saved["progress"])


def _fit(phase: str, prepared: Prepared, policy: M.Policy | None, out: Path, device: str,
         settings: Settings, on_epoch: Callable[[dict], None] | None, architecture: M.Architecture,
         name: str) -> dict:
    if policy is not None:
        policy.to(device)
    model = build(phase, prepared, policy, architecture).to(device)
    optimizer, scheduler = optimiser(model, phase, settings.epochs)
    patients = len(prepared.train["continuous"])
    updates_per_epoch = math.ceil(patients / settings.batch)
    continuation = out / f"{name}.continue.pt"
    identity = {"phase": phase, "settings": asdict(settings), "architecture": asdict(architecture),
                "cache_sha256": prepared.protocol.cache_sha256, "device": torch.device(device).type}
    progress = (_continue(continuation, identity, model, optimizer, scheduler, device) if continuation.exists()
                else _Progress())

    def save() -> float:
        _save_continuation(continuation, identity, progress, model, optimizer, scheduler, device)
        return time.monotonic()

    model.train()
    saved = time.monotonic()
    for epoch in range(progress.epoch, settings.epochs):
        if _exhausted(progress.history, settings):
            break
        order = epoch_order(epoch, patients)
        for step in range(progress.step, updates_per_epoch):
            ids = order[step * settings.batch:min((step + 1) * settings.batch, patients)]
            update(model, optimizer, prepared.train, ids, phase, settings.micro, device)
            progress.step, progress.updates = step + 1, progress.updates + 1
            if time.monotonic() - saved >= CONTINUATION_SECONDS:
                saved = save()
        loss = validation(model, prepared.val, phase, settings.micro, device)
        if not math.isfinite(loss):
            raise RuntimeError("non-finite validation loss")
        progress.history.append({"epoch": epoch + 1, "validation_loss": loss, "updates": progress.updates})
        if on_epoch is not None:
            on_epoch(dict(progress.history[-1]))
        if loss < progress.best:
            progress.best = loss
            progress.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step()
        progress.epoch, progress.step = epoch + 1, 0
        saved = save()

    model.load_state_dict(progress.best_state)
    model.eval()
    if phase == "policy":
        for p in model.parameters():
            p.requires_grad_(False)
    out.mkdir(parents=True, exist_ok=True)
    prepared.protocol.save(out / "protocol.json")
    M.save_checkpoint(out / f"{name}.pt", model, progress.history, prepared.protocol.cache_sha256,
                      architecture if isinstance(model, FeedbackPULSE) else None)
    continuation.unlink(missing_ok=True)
    return {"model": model, "history": progress.history, "validation_loss": progress.best,
            "stopped_early": len(progress.history) < settings.epochs}

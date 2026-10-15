"""Training of the treatment policy and of the PULSE phases.

Every phase trains on teacher-forced histories: the model reads the recorded
hours up to each position and is scored on the next one. The policy is
trained first and then frozen; each PULSE phase reads it for the
vasopressor descriptor of its preview pass (`pulse.feedback`).

The PULSE loss is the sum, in this order, of

* the CNF negative log density of the standardised next values, averaged
  over recorded non-ordinal entries,
* the squared error of the CNF location for the ordinal variables (SOFA
  score and oxygen requirement level), averaged over their recorded entries,
* the cross-entropy of the next-hour binary outcomes, where an absorbing
  outcome (death, release, pulmonary edema) counts only up to and including
  its first occurrence and is held at one afterwards,
* the mean cross-entropy of the observation logits against whether each
  variable is recorded in the next row,
* 0.05 times a smooth L1 loss between log(1 + predicted hours) and
  log(1 + hours until the variable is next recorded), those hours set to
  48 when the variable is not recorded again.

The policy loss is the squared error of the next bolus volume in training
standard deviations, averaged over recorded entries, plus the cross-entropy
of the next vasopressor indicator.

Optimisation: Adam (learning rate 1e-3 for the policy, 5e-4 otherwise), the
rate multiplied by 0.3 after half of the epochs, gradients clipped at norm
one, batches of 32 patients in microbatches of 8, at most 50 epochs, early
stopping after five epochs without a better validation loss once ten epochs
have run, and the weights of the best validation epoch kept.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pulse import cnf
from pulse import model as M
from pulse import simulate as S
from pulse import variables as V
from pulse.data import Part, Prepared, Protocol
from pulse.feedback import FeedbackPULSE, build_pulse, generator_states, restore_generators

PHASES = ("policy", "sample", "distribution", "quantile_only", "probability_only")
WAIT_WEIGHT = 0.05
WAIT_CENSOR_HOURS = 48.0
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


def pulse_loss(pulse: FeedbackPULSE, batch: Part, arm: S.Arm) -> torch.Tensor:
    core = pulse.pulse
    history = {k: batch[k] for k in ("continuous", "binary", "static", "time_hours", "continuous_mask")}
    out = pulse(history, quantiles=arm.quantiles, probabilities=arm.probabilities)
    idx = core.output_indices
    target = core.normalize(batch["continuous_target"])[:, 1:, idx]
    recorded = batch["continuous_mask"][:, 1:, idx]
    ordinal = torch.tensor([core.continuous_names[i] in V.ORDINAL for i in idx], device=target.device).view(1, 1, -1)

    density = recorded * (~ordinal).to(recorded.dtype)
    params = {k: torch.stack([p[k][:, :-1] for p in out["dist_params_list"]], 2)
              for k in cnf.FLOW_KEYS}
    nll = cnf.nll(params, target)
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
    return pulse_loss(model, batch, S.ARMS[phase])


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

def build(phase: str, prepared: Prepared, policy: M.Policy | None):
    """A freshly seeded model for `phase`; PULSE phases read the frozen `policy`."""
    if phase == "policy":
        return M.build_policy(prepared.protocol)
    if policy is None:
        raise ValueError(f"phase {phase} needs the trained policy")
    model = build_pulse(prepared.protocol, prepared.train["binary"])
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
        settings: Settings = Settings()) -> dict:
    """Train one phase; write the best validation weights to `<out>/<phase>.pt`.

    `<out>/protocol.json` receives the protocol, whose cache digest the
    checkpoint also records; a directory holding another protocol is
    refused before training. Returns the model, the validation history,
    the best validation loss and whether training stopped early. The
    returned policy is frozen and in evaluation mode, ready for the PULSE
    phases.
    """
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase}")
    protocol = Path(out) / "protocol.json"
    if protocol.exists() and Protocol.load(protocol) != prepared.protocol:
        raise ValueError(f"{protocol} holds another protocol")
    with mps_attention_dropout() if torch.device(device).type == "mps" else contextlib.nullcontext():
        return _fit(phase, prepared, policy, Path(out), device, settings)


def _fit(phase: str, prepared: Prepared, policy: M.Policy | None, out: Path, device: str,
         settings: Settings) -> dict:
    if policy is not None:
        policy.to(device)
    model = build(phase, prepared, policy).to(device)
    optimizer, scheduler = optimiser(model, phase, settings.epochs)
    patients = len(prepared.train["continuous"])
    updates_per_epoch = math.ceil(patients / settings.batch)
    best, best_state, history, updates = float("inf"), None, [], 0

    def exhausted() -> bool:
        if not history or len(history) < settings.min_epochs:
            return False
        since_best = len(history) - 1 - int(np.argmin([h["validation_loss"] for h in history]))
        return since_best >= settings.patience

    model.train()
    for epoch in range(settings.epochs):
        if exhausted():
            break
        order = epoch_order(epoch, patients)
        for step in range(updates_per_epoch):
            ids = order[step * settings.batch:min((step + 1) * settings.batch, patients)]
            update(model, optimizer, prepared.train, ids, phase, settings.micro, device)
            updates += 1
        loss = validation(model, prepared.val, phase, settings.micro, device)
        if not math.isfinite(loss):
            raise RuntimeError("non-finite validation loss")
        history.append({"epoch": epoch + 1, "validation_loss": loss, "updates": updates})
        print(f"{phase} epoch {epoch + 1}: validation loss {loss:.6f}", flush=True)
        if loss < best:
            best, best_state = loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step()

    model.load_state_dict(best_state)
    model.eval()
    if phase == "policy":
        for p in model.parameters():
            p.requires_grad_(False)
    Path(out).mkdir(parents=True, exist_ok=True)
    prepared.protocol.save(Path(out) / "protocol.json")
    M.save_checkpoint(Path(out) / f"{phase}.pt", model, history, prepared.protocol.cache_sha256)
    return {"model": model, "history": history, "validation_loss": best,
            "stopped_early": len(history) < settings.epochs}

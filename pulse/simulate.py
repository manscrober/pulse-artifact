"""Recursive Monte Carlo rollout of PULSE with the treatment policy.

A rollout starts from the recorded hours of a test patient up to the
forecast origin and generates the remaining hours one at a time, `paths`
times per patient. Each hour:

1. PULSE reads the patient's event history so far (padded to 24 rows) and
   predicts the next-hour binary outcomes, which variables will be
   recorded, the hours until they are, and a CNF law for every continuous
   variable.
2. The binary outcomes are drawn; the continuous variables are drawn from
   their CNF laws and clipped to their ranges; unrecorded variables keep
   their last value.
3. The policy proposes the bolus volume and draws the vasopressor
   indicator from the same history.
4. Derived variables are recomputed, and the new hour joins the event
   history if anything was recorded, an event occurred or a treatment
   changed.

The forecast origin `k` is the paper's: the first generated hour is hour
`k` (one-based), so the last recorded row is `k - 2`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from pulse import cnf
from pulse import variables as V
from pulse.data import Part, case_path, save_case
from pulse.model import GroupedModel, Policy

STEP_HOURS = 1.0  # length of one generated step
SEED = 500_000
# Patients per rollout call. Each call is seeded from the origin and its
# first patient, so the paths depend on this size; the published rollouts
# used calls of 8.
PER_CALL = 8


@dataclass(frozen=True)
class Arm:
    """How a rollout feeds its own output back. `weights` names the trained checkpoint."""

    weights: str


ARMS = {"sample": Arm("sample")}


def origin_step(k: int) -> int:
    """Last recorded row for forecast origin `k`."""
    if k not in (2, 6):
        raise ValueError("forecast origins are 2 and 6")
    return k - 2


def _gap(observed: torch.Tensor, hours_until: torch.Tensor) -> torch.Tensor:
    """Hours to the earliest recorded variable per row; zero where nothing is recorded."""
    gap = torch.where(observed, hours_until, torch.full_like(hours_until, float("inf"))).min(dim=1).values
    any_recorded = torch.isfinite(gap)
    if not any_recorded.any():
        return torch.zeros_like(gap)
    return torch.where(any_recorded, gap.clamp_min(1e-4), torch.zeros_like(gap))


def _absorb(previous: torch.Tensor, nxt: torch.Tensor, probs: torch.Tensor, outcome: list[int]) -> torch.Tensor:
    """Keep absorbing states on, and allow only one of death and release."""
    out = nxt.clone()
    absorbing = [V.BINARY.index(n) for n in V.ABSORBING]
    was = previous[:, absorbing] > 0.5
    if was.any():
        out[:, absorbing] = torch.where(was, torch.ones_like(was, dtype=out.dtype), out[:, absorbing])
    dead, released = previous[:, V.DEATH] > 0.5, previous[:, V.RELEASE] > 0.5
    if dead.any():
        out[dead, V.DEATH], out[dead, V.RELEASE] = 1.0, 0.0
    if released.any():
        out[released, V.RELEASE], out[released, V.DEATH] = 1.0, 0.0
    both = (out[:, V.DEATH] > 0.5) & (out[:, V.RELEASE] > 0.5)
    if both.any():
        # The more probable event wins; death wins a tie.
        death_wins = probs[:, outcome.index(V.DEATH)] >= probs[:, outcome.index(V.RELEASE)]
        out[both, V.DEATH] = death_wins[both].to(out.dtype)
        out[both, V.RELEASE] = (~death_wins[both]).to(out.dtype)
    return out


@torch.no_grad()
def rollout(pulse: GroupedModel, policy: Policy, arm: Arm, batch: Part, start: int,
            paths: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate hours `start + 1` onwards; returns continuous [N, P, T, V] and binary [N, P, T, K]."""
    pulse.eval(), policy.eval()
    n, steps, nv = batch["continuous"].shape
    rows_total = n * paths
    low, high = (b.to(batch["continuous"].device) for b in V.bounds())
    primary, outcome = pulse.output_indices, pulse.binary_output_indices
    bolus, vaso = V.BOLUS, V.VASOPRESSOR

    # Future hours are a fixed one-hour grid after the origin.
    times = batch["time_hours"].clone()
    grid = torch.arange(1, steps - start, device=times.device, dtype=times.dtype)
    times[:, start + 1:] = times[:, start:start + 1] + STEP_HOURS * grid
    keep = V.zero_invalid_mask(batch["continuous"], batch["continuous_mask"]) > 0

    def repeat(v):
        return v.repeat_interleave(paths, dim=0).clone()

    binary, times, static = repeat(batch["binary"]), repeat(times), repeat(batch["static"])
    keep = keep.repeat_interleave(paths, dim=0)
    means = pulse.cont_means.to(torch.float32)
    # Only events already recorded by the origin may alter the recorded rows.
    past = binary.clone()
    past[:, start + 1:] = 0
    death, release, _ = V.event_times(past)
    cont = V.fill_after_events(V.forward_fill(repeat(batch["continuous"]), keep, means), death, release, means)
    baseline = cont[:, 0].clone()

    # Event history: recorded rows first, generated rows appended, padding after.
    hist_cont = cont.new_zeros(rows_total, steps, nv)
    hist_bin = binary.new_zeros(rows_total, steps, binary.shape[-1])
    hist_mask = cont.new_zeros(rows_total, steps, nv)
    hist_time = times.new_zeros(rows_total, steps)
    hist_cont[:, :start + 1] = cont[:, :start + 1]
    hist_bin[:, :start + 1] = binary[:, :start + 1]
    hist_mask[:, :start + 1] = keep[:, :start + 1].to(cont.dtype)
    hist_time[:, :start + 1] = times[:, :start + 1]
    lengths = torch.full((rows_total,), start + 1, device=cont.device, dtype=torch.long)

    for t in range(start, steps - 1):
        padding = torch.arange(steps, device=cont.device).unsqueeze(0) >= lengths.unsqueeze(1)
        binary_hist = hist_bin.clone()
        history = {"continuous": hist_cont, "binary": binary_hist, "continuous_mask": hist_mask,
                   "time_hours": hist_time, "static": static, "padding_mask": padding}
        # The continuous encoders see next-hour binaries drawn afresh for
        # every history row, the last row's draw becoming the new hour
        # (reproduction constraint).
        x = pulse.inputs(history)
        logits = pulse.binary_logits(x)
        probs = torch.sigmoid(logits)
        drawn = (torch.rand_like(probs) < probs).float()
        out = pulse.next_outputs(x, logits, drawn, lengths)
        last = (lengths - 1).view(-1, 1, 1).expand(-1, 1, drawn.shape[-1])
        next_outcomes = drawn.gather(1, last)[:, 0, :]
        bin_probs = out["bin_probs"][:, 0, :]

        obs_probs = torch.sigmoid(out["obs_logits"][:, 0, :]).clamp(1e-6, 1.0 - 1e-6)
        observed = torch.rand_like(obs_probs) < obs_probs
        gap = _gap(observed, out["delta_t_hours"][:, 0, :])

        params = {k: torch.stack([p[k][:, 0] for p in out["dist_params_list"]], 1)
                  for k in cnf.FLOW_KEYS}
        draw = pulse.denormalize(cnf.sample(params).unsqueeze(1), primary).squeeze(1)
        draw = torch.min(torch.max(draw, low[primary]), high[primary])
        next_cont = cont[:, t, :].clone()
        next_cont[:, primary] = draw

        # The policy reads the whole padded history, and its output at the last
        # of the 24 rows, which is padding while the history is shorter, sets
        # the bolus volume and the vasopressor draw. Reading the last valid row
        # would be the cleaner form, but it changes every rollout; the published
        # runs read the last row.
        treatment = policy.treatments(hist_cont, binary_hist, static, hist_time)
        vaso_probs = torch.sigmoid(treatment["binary_treatment_logits"][:, -1, :])
        vaso_draw = (torch.rand_like(vaso_probs) < vaso_probs).float()
        next_cont[:, [bolus]] = treatment["continuous_treatments"][:, -1, :]

        next_bin = binary[:, t, :].clone()
        next_bin[:, outcome] = next_outcomes
        next_bin = _absorb(binary[:, t, :], next_bin, bin_probs, outcome)
        next_bin[:, [vaso]] = vaso_draw
        binary[:, t + 1, :] = next_bin
        event = (binary[:, t + 1, V.DEATH] > 0.5) | (binary[:, t + 1, V.RELEASE] > 0.5)

        recorded = observed & ~event.unsqueeze(1)
        missing = ~observed & ~event.unsqueeze(1)
        next_cont[:, primary] = torch.where(missing, cont[:, t, primary], next_cont[:, primary])
        V.forward_fill_zero_invalid(next_cont, cont[:, t, :], event)
        next_mask = torch.zeros_like(next_cont)
        next_mask[:, primary] = recorded.to(next_mask.dtype)
        next_mask[:, bolus] = 1.0
        cont[:, t + 1, :] = next_cont
        V.update_derived(cont, t + 1, baseline)

        any_recorded = observed.any(dim=1)
        append = (any_recorded | event | (next_cont[:, bolus] - cont[:, t, bolus]).abs().gt(1e-6)
                  | (next_bin[:, vaso] != binary[:, t, vaso]))
        next_time = torch.where(any_recorded, times[:, t] + gap, times[:, t + 1])
        any_recorded = any_recorded | event
        times[:, t + 1] = torch.where(any_recorded, next_time, times[:, t])
        if append.any():
            rows = append.nonzero(as_tuple=True)[0]
            cols = lengths[rows]
            hist_cont[rows, cols] = next_cont[rows]
            hist_bin[rows, cols] = next_bin[rows]
            hist_mask[rows, cols] = next_mask[rows]
            append_time = times[rows, t + 1]
            timed = rows[any_recorded[rows]]
            if timed.numel() > 0:
                # Recorded rows sit one gap after the previous history row.
                append_time[torch.searchsorted(rows, timed)] = hist_time[timed, lengths[timed] - 1] + gap[timed]
            hist_time[rows, cols] = append_time
            lengths[rows] += 1

    return cont.view(n, paths, steps, nv), binary.view(n, paths, steps, -1)


def evaluate(pulse: GroupedModel, policy: Policy, arm: Arm, test: Part, k: int, paths: int, out: Path,
             per_call: int = PER_CALL) -> None:
    """Roll out every test patient and write one case file each.

    Patients go in groups of `per_call`; each group is seeded from the
    origin and its first patient, so a group's paths do not depend on
    which groups ran before it, but they do depend on the group size.
    `out` also receives `rollout.json` with these settings, and a
    directory holding rollouts made with other settings is refused.
    """
    n = test["continuous"].shape[0]
    record = {"arm": asdict(arm), "origin": k, "paths": paths, "per_call": per_call, "seed": SEED + 1000 * k}
    settings = Path(out) / "rollout.json"
    if settings.exists() and json.loads(settings.read_text()) != record:
        raise ValueError(f"{out} holds rollouts made with other settings")
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps(record, indent=1) + "\n")
    for first in range(0, n, per_call):
        ids = list(range(first, min(first + per_call, n)))
        if all(case_path(out, i).exists() for i in ids):
            continue
        torch.manual_seed(SEED + 1000 * k + first)
        batch = {key: value[first:ids[-1] + 1] for key, value in test.items()}
        cont, binary = rollout(pulse, policy, arm, batch, origin_step(k), paths)
        for offset, i in enumerate(ids):
            save_case(out, i, cont[offset], binary[offset])

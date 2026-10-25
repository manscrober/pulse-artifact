"""Recursive Monte Carlo rollout of PULSE with the treatment policy.

A rollout starts from the recorded hours of a test patient up to the
forecast origin and generates the remaining hours one at a time, `paths`
times per patient. Each hour:

1. PULSE reads the patient's event history so far (padded to 24 rows) and
   predicts the next-hour binary outcomes, which variables will be
   recorded, the hours until they are (with the timing head), and a law
   for every continuous variable (`pulse.heads`).
2. The binary outcomes are drawn; the continuous variables are drawn from
   their laws and clipped to their ranges; unrecorded variables keep
   their last value. A point head has no law of its own: its value gets
   static Gaussian noise fitted to hour-to-hour changes of validation
   patients (`static_noise`).
3. The policy proposes the bolus volume and draws the vasopressor
   indicator from the same history.
4. Derived variables are recomputed, and the new hour joins the event
   history if anything was recorded, an event occurred or a treatment
   changed.

The forecast origin `k` is the paper's: the first generated hour is hour
`k` (one-based), so the last recorded row is `data.last_recorded_row(k)`.

The G-Transformer baseline (`pulse.gtransformer`) differs in step 1: it
reads the hourly rows generated so far, the form it is trained on, and the
binary outcomes it draws for its own later encoders become the next hour's
outcomes.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from pulse import heads as H
from pulse import variables as V
from pulse.data import Part, case_path, first_forecast_row, last_recorded_row, save_case
# ARMS is defined with the feedback it describes; the command line and the
# README also reach it as pulse.simulate.ARMS.
from pulse.feedback import ARMS, Arm, FeedbackPULSE, RolloutFeedback
from pulse.gtransformer import GTransformer
from pulse.model import Policy

STEP_HOURS = 1.0  # length of one generated step
SEED = 500_000
# Patients per rollout call. Each call is seeded from the origin and its
# first patient, so the paths depend on this size; the published rollouts
# used calls of 8.
PER_CALL = 8


def static_noise(part) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and SD [V] of the change between consecutive recorded hours, in physical units.

    `part` is a prepared part, the validation patients for the published
    rows. Both hours must be recorded (zero-invalid zeros count as not
    recorded); a variable without such a pair gets mean 0 and SD 1. Sums
    run in float64, patient by patient, as in the reference.
    """
    nv = part["continuous"].shape[-1]
    sums, squares, counts = np.zeros(nv), np.zeros(nv), np.zeros(nv, dtype=np.int64)
    for i in range(part["continuous"].shape[0]):
        cont = part["continuous"][i]
        mask = V.zero_invalid_mask(cont.unsqueeze(0), part["continuous_mask"][i].unsqueeze(0)).squeeze(0)
        valid = ((mask[:-1] > 0) & (mask[1:] > 0)).numpy()
        deltas = np.where(valid, (cont[1:] - cont[:-1]).numpy().astype(np.float64), 0.0)
        counts += valid.sum(axis=0)
        sums += deltas.sum(axis=0)
        squares += (deltas * deltas).sum(axis=0)
    means, sds = np.zeros(nv), np.ones(nv)
    present = counts > 0
    means[present] = sums[present] / counts[present]
    variance = np.zeros(nv)
    variance[present] = squares[present] / counts[present] - means[present] * means[present]
    sds[present] = np.sqrt(np.maximum(variance[present], 0.0))
    return torch.tensor(means, dtype=torch.float32), torch.tensor(sds, dtype=torch.float32)


def _draw(core, out: dict, noise: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
    """Next values [R, V_out] in physical units, drawn as the reference drew each head.

    The mixture and the flows are drawn in standardised units and scaled
    back. The Gaussian and the point head are drawn in physical units from
    a location and a standard deviation, which rounds differently.
    """
    kind, primary = core.head, core.output_indices
    if kind in ("mog2",) + H.FLOWS:
        params = H.stack(kind, out["dist_params_list"], index=0)
        return core.denormalize(H.sample(kind, params).unsqueeze(1), primary).squeeze(1)
    mu = core.denormalize(out["mu"][:, 0, :].unsqueeze(1), primary).squeeze(1)
    if kind == "gaussian":
        sigma = torch.exp(0.5 * torch.clamp(out["log_var"][:, 0, :], min=H.LOG_VAR_MIN))
        sigma = sigma * core.cont_stds.clone().to(mu.device)[primary]
        return mu + sigma * torch.randn_like(mu)
    if noise is None:
        raise ValueError("a point head needs the static noise of the validation patients")
    mean, sd = (v.to(mu.device)[primary] for v in noise)
    sigma = torch.sqrt(torch.zeros_like(mu) ** 2 + sd ** 2)
    return (mu + mean) + sigma * torch.randn_like(mu)


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


def _recorded_rows(batch: Part, k: int, paths: int, means: torch.Tensor):
    """Each patient repeated `paths` times, with the recorded rows filled as the rollout reads them.

    Returns continuous [R, T, V] forward-filled and filled after events
    recorded by the origin, binary [R, T, K], hours [R, T] with the
    generated hours on a one-hour grid after the origin, static [R, S] and
    the recorded-value mask [R, T, V].
    """
    steps, last, first = batch["continuous"].shape[1], last_recorded_row(k), first_forecast_row(k)
    times = batch["time_hours"].clone()
    grid = torch.arange(1, steps - last, device=times.device, dtype=times.dtype)
    times[:, first:] = times[:, last:last + 1] + STEP_HOURS * grid
    keep = V.zero_invalid_mask(batch["continuous"], batch["continuous_mask"]) > 0

    def repeat(v):
        return v.repeat_interleave(paths, dim=0).clone()

    binary, times, static = repeat(batch["binary"]), repeat(times), repeat(batch["static"])
    keep = keep.repeat_interleave(paths, dim=0)
    # Only events already recorded by the origin may alter the recorded rows.
    past = binary.clone()
    past[:, first:] = 0
    death, release, _ = V.event_times(past)
    cont = V.fill_after_events(V.forward_fill(repeat(batch["continuous"]), keep, means), death, release, means)
    return cont, binary, times, static, keep


class _EventHistory:
    """The rows PULSE reads: recorded rows first, generated rows appended, padding after.

    A generated hour joins only when something was recorded, an event
    occurred or a treatment changed, so `lengths` [R] can fall behind the
    hour index.
    """

    def __init__(self, cont: torch.Tensor, binary: torch.Tensor, keep: torch.Tensor, times: torch.Tensor,
                 k: int):
        rows, steps, nv = cont.shape
        self.continuous = cont.new_zeros(rows, steps, nv)
        self.binary = binary.new_zeros(rows, steps, binary.shape[-1])
        self.mask = cont.new_zeros(rows, steps, nv)
        self.time = times.new_zeros(rows, steps)
        first = first_forecast_row(k)
        self.continuous[:, :first] = cont[:, :first]
        self.binary[:, :first] = binary[:, :first]
        self.mask[:, :first] = keep[:, :first].to(cont.dtype)
        self.time[:, :first] = times[:, :first]
        self.lengths = torch.full((rows,), first, device=cont.device, dtype=torch.long)

    def model_input(self, static: torch.Tensor) -> dict[str, torch.Tensor]:
        steps = self.continuous.shape[1]
        padding = torch.arange(steps, device=self.continuous.device).unsqueeze(0) >= self.lengths.unsqueeze(1)
        return {"continuous": self.continuous, "binary": self.binary, "continuous_mask": self.mask,
                "time_hours": self.time, "static": static, "padding_mask": padding}

    def append(self, joins: torch.Tensor, timed: torch.Tensor, cont: torch.Tensor, binary: torch.Tensor,
               mask: torch.Tensor, hours: torch.Tensor, gap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append the new hour [R, ...] to the rows where `joins`; returns those rows and their columns.

        A row that is `timed` (a variable was recorded or an event
        occurred) sits one `gap` after its previous history row; the
        other rows take `hours` [R], the new hour on the simulator's clock.
        """
        if not joins.any():
            empty = torch.empty(0, device=cont.device, dtype=torch.long)
            return empty, empty
        rows = joins.nonzero(as_tuple=True)[0]
        cols = self.lengths[rows]
        self.continuous[rows, cols] = cont[rows]
        self.binary[rows, cols] = binary[rows]
        self.mask[rows, cols] = mask[rows]
        hour = hours[rows]
        timed_rows = rows[timed[rows]]
        if timed_rows.numel() > 0:
            hour[torch.searchsorted(rows, timed_rows)] = (self.time[timed_rows, self.lengths[timed_rows] - 1]
                                                          + gap[timed_rows])
        self.time[rows, cols] = hour
        self.lengths[rows] += 1
        return rows, cols


def _next_binaries(core, arm: Arm, x: dict, logits: torch.Tensor, history: _EventHistory,
                   additions) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """The next-hour outputs, the drawn outcomes [R, K_out] and their probabilities [R, K_out]."""
    outcome, lengths = core.binary_output_indices, history.lengths
    last = (lengths - 1).view(-1, 1, 1).expand(-1, 1, logits.shape[-1])
    if arm.two_pass:
        probs = torch.sigmoid(logits).gather(1, last)[:, 0, :]
        outcomes = (torch.rand_like(probs) < probs).float()
        # Row j conditions on the outcomes realised at row j + 1 and the
        # last row on the new draw.
        realised = history.binary[..., outcome]
        override = realised.clone()
        override[:, :-1] = realised[:, 1:]
        override[torch.arange(len(lengths), device=lengths.device), lengths - 1] = outcomes
        out = core.next_outputs(x, logits, override, lengths, additions)
        return out, outcomes, probs
    # The continuous encoders see next-hour binaries drawn afresh for every
    # history row, the last row's draw becoming the new hour. Drawing only
    # for the last row would be the cleaner form, but it consumes fewer
    # random numbers and so changes every later draw; the published sample
    # arm drew for every row. No descriptor adapters are passed, because
    # `Arm` allows feedback only with the two-pass step.
    probs = torch.sigmoid(logits)
    drawn = (torch.rand_like(probs) < probs).float()
    out = core.next_outputs(x, logits, drawn, lengths)
    return out, drawn.gather(1, last)[:, 0, :], out["bin_probs"][:, 0, :]


@torch.no_grad()
def rollout(model: FeedbackPULSE | GTransformer, policy: Policy, arm: Arm, batch: Part, k: int, paths: int, *,
            noise: tuple[torch.Tensor, torch.Tensor] | None = None,
            policy_at_last_event: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate the hours after forecast origin `k`, from row `first_forecast_row(k)` on.

    `noise` is the static noise of a point head (`static_noise`). With
    `policy_at_last_event` the policy's output is read at the last row of
    each event history; otherwise at the last of the 24 padded rows, as in
    the runs published before 19 October. A G-Transformer rolls out with
    the `sample` arm only. Returns continuous [N, P, T, V] and binary
    [N, P, T, K].
    """
    gtransformer = isinstance(model, GTransformer)
    if gtransformer and arm != ARMS["sample"]:
        raise ValueError("the G-Transformer rolls out with the sample arm only")
    core = model if gtransformer else model.pulse
    model.eval()
    policy.eval()
    n, steps, nv = batch["continuous"].shape
    rows_total = n * paths
    low, high = (b.to(batch["continuous"].device) for b in V.bounds())
    primary, outcome = core.output_indices, core.binary_output_indices
    bolus, vaso = V.BOLUS, V.VASOPRESSOR

    means = core.cont_means.to(torch.float32)
    cont, binary, times, static, keep = _recorded_rows(batch, k, paths, means)
    baseline = cont[:, 0].clone()
    history = _EventHistory(cont, binary, keep, times, k)
    feedback = (RolloutFeedback(model, rows_total, steps, quantiles=arm.quantiles, probabilities=arm.probabilities,
                                mean_state=arm.mean_state) if arm.feedback else None)

    last = last_recorded_row(k)
    for t in range(last, steps - 1):
        model_input = history.model_input(static)
        if gtransformer:
            hourly = {"continuous": cont[:, :t + 1], "binary": binary[:, :t + 1], "static": static}
            out = core.next_outputs(hourly)
            next_outcomes, bin_probs = out["bin_samples"][:, 0, :], out["bin_probs"][:, 0, :]
        else:
            if feedback is not None:
                if t == last:
                    feedback.prefix(model_input, policy, first_forecast_row(k))
                model_input["continuous"] = feedback.history_values(history.continuous)
            x = core.inputs(model_input)
            additions = None if feedback is None else feedback.additions()
            logits = core.binary_logits(x, additions)
            out, next_outcomes, bin_probs = _next_binaries(core, arm, x, logits, history, additions)

        obs_probs = torch.sigmoid(out["obs_logits"][:, 0, :]).clamp(1e-6, 1.0 - 1e-6)
        observed = torch.rand_like(obs_probs) < obs_probs
        # Without the timing head a generated hour sits on the one-hour grid.
        gap = None if out["delta_t_hours"] is None else _gap(observed, out["delta_t_hours"][:, 0, :])

        draw = _draw(core, out, noise)
        draw = torch.min(torch.max(draw, low[primary]), high[primary])
        next_cont = cont[:, t, :].clone()
        next_cont[:, primary] = draw

        # The policy reads the whole padded history. The runs published before
        # 19 October read its output at the last of the 24 rows, which is
        # padding while the history is shorter; the policy is causal, so the
        # last row of the event history gives the decision from the history.
        # The two readings also differ in the last bits: the sigmoid of a
        # strided view and of a gathered copy take different CPU kernels.
        treatment = policy.treatments(model_input["continuous"], history.binary, static, history.time)
        if policy_at_last_event:
            rows_all = torch.arange(rows_total, device=cont.device)
            read = (history.lengths - 1).clamp_min(0)
            vaso_logits = treatment["binary_treatment_logits"][rows_all, read, :]
            bolus_volume = treatment["continuous_treatments"][rows_all, read, :]
        else:
            vaso_logits = treatment["binary_treatment_logits"][:, -1, :]
            bolus_volume = treatment["continuous_treatments"][:, -1, :]
        vaso_probs = torch.sigmoid(vaso_logits)
        vaso_draw = (torch.rand_like(vaso_probs) < vaso_probs).float()
        next_cont[:, [bolus]] = bolus_volume

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

        # Drawn as recorded, even when an event then blanks the values.
        drawn_recorded = observed.any(dim=1)
        joins = (drawn_recorded | event | (next_cont[:, bolus] - cont[:, t, bolus]).abs().gt(1e-6)
                 | (next_bin[:, vaso] != binary[:, t, vaso]))
        timed = drawn_recorded | event
        if gap is None:
            times[:, t + 1] = torch.where(timed, times[:, t + 1], times[:, t])
            rows, cols = history.append(joins, torch.zeros_like(timed), next_cont, next_bin, next_mask,
                                        times[:, t + 1], gap)
        else:
            next_time = torch.where(drawn_recorded, times[:, t] + gap, times[:, t + 1])
            times[:, t + 1] = torch.where(timed, next_time, times[:, t])
            rows, cols = history.append(joins, timed, next_cont, next_bin, next_mask, times[:, t + 1], gap)
        if feedback is not None:
            feedback.record(out, vaso_probs, rows, cols, step=t, generated=cont[:, t + 1, :], recorded=next_mask,
                            baseline=baseline)

    return cont.view(n, paths, steps, nv), binary.view(n, paths, steps, -1)


def evaluate(model: FeedbackPULSE | GTransformer, policy: Policy, arm: Arm, test: Part, k: int, paths: int,
             out: Path, per_call: int = PER_CALL, provenance: dict[str, str] | None = None, *,
             noise: tuple[torch.Tensor, torch.Tensor] | None = None, policy_at_last_event: bool = False) -> None:
    """Roll out every test patient and write one case file each.

    Patients go in groups of `per_call`; each group is seeded from the
    origin and its first patient, so a group's paths do not depend on
    which groups ran before it, but they do depend on the group size.
    `out` also receives `rollout.json` with these settings, and a
    directory holding rollouts made with other settings is refused.
    `provenance`, for example the SHA-256 digests of the checkpoints, is
    stored and checked with the settings, so that existing case files are
    never kept for rollouts of other weights; an empty mapping counts as
    none. `noise` and `policy_at_last_event` are passed to `rollout` and
    recorded when set.
    """
    n = test["continuous"].shape[0]
    record = {"arm": asdict(arm), "origin": k, "paths": paths, "per_call": per_call, "seed": SEED + 1000 * k}
    if provenance:
        record["provenance"] = dict(provenance)
    if noise is not None:
        record["static_noise"] = {"mean": noise[0].tolist(), "sd": noise[1].tolist()}
    if policy_at_last_event:
        record["policy_at_last_event"] = True
    settings = Path(out) / "rollout.json"
    if settings.exists() and json.loads(settings.read_text()) != record:
        raise ValueError(f"{out} holds rollouts made with other settings")
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps(record, indent=1) + "\n")
    for first in range(0, n, per_call):
        ids = list(range(first, min(first + per_call, n)))
        # A group is rerun whole unless all its case files exist. Each group
        # is seeded on its own, so rerunning a partly written group rewrites
        # its existing files with identical contents.
        if all(case_path(out, i).exists() for i in ids):
            continue
        torch.manual_seed(SEED + 1000 * k + first)
        batch = {key: value[first:ids[-1] + 1] for key, value in test.items()}
        cont, binary = rollout(model, policy, arm, batch, k, paths, noise=noise,
                               policy_at_last_event=policy_at_last_event)
        for offset, i in enumerate(ids):
            save_case(out, i, cont[offset], binary[offset])

"""Distribution feedback: PULSE reads descriptors of its own earlier predictions.

At every history row the model may receive two descriptors besides the
values themselves:

- for each continuous variable, the 19 quantiles of the CNF law that was
  predicted for that row, summarised by their median, the log of the
  interquartile width and the quantiles relative to those two;
- for each binary variable, the probability that was predicted for it.

Each encoder adds the output of two small adapters, one per descriptor, to
its projected inputs. The last layer of every adapter starts at zero, so
an untrained adapter leaves the model unchanged.

In training the descriptors come from a preview pass of the same model on
the recorded history, shifted by one row so that no row sees a prediction
of itself.
"""
from __future__ import annotations

import torch
from torch import nn

from pulse import cnf
from pulse import variables as V
from pulse.data import Protocol
from pulse.model import SEED, Extra, GroupedModel, Policy, pulse_model

PROBABILITIES = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95,
                 0.975, 0.99, 0.995, 0.999)

Descriptors = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def quantiles_of(params: cnf.Params) -> torch.Tensor:
    """The 19 quantiles [..., 19] of CNF laws with parameters of leading shape [...]."""
    p = torch.as_tensor(PROBABILITIES, dtype=params["loc"].dtype, device=params["loc"].device)
    leading = params["loc"].shape
    stretched = {k: v.unsqueeze(len(leading)).expand(leading + (len(p),) + v.shape[len(leading):])
                 for k, v in params.items()}
    return cnf.quantile(stretched, p)


def quantile_features(quantiles: torch.Tensor) -> torch.Tensor:
    """Median, log interquartile width, then the quantiles centred and scaled by those two.

    The width has a floor of 1e-6 so that point masses are representable;
    whether a descriptor exists is supplied separately.
    """
    p = torch.as_tensor(PROBABILITIES, device=quantiles.device, dtype=quantiles.dtype)
    if not bool(torch.isfinite(quantiles).all()) or not bool((quantiles[..., 1:] >= quantiles[..., :-1]).all()):
        raise ValueError("quantiles must be finite and nondecreasing")

    def at(q: float) -> torch.Tensor:
        i = torch.searchsorted(p, p.new_tensor(q)).clamp(1, p.numel() - 1)
        weight = (q - p[i - 1]) / (p[i] - p[i - 1])
        return quantiles[..., i - 1] * (1 - weight) + quantiles[..., i] * weight

    median = at(0.5)
    width = (at(0.75) - at(0.25)).clamp_min(1e-6)
    return torch.cat((median.unsqueeze(-1), width.log().unsqueeze(-1),
                      (quantiles - median.unsqueeze(-1)) / width.unsqueeze(-1)), -1)


class DensityAdapter(nn.Module):
    """Maps the quantile descriptors of all continuous variables at a row [V, 19] to the encoder width."""

    def __init__(self, variables: int, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(variables * (len(PROBABILITIES) + 3), width), nn.SiLU(),
                                 nn.Linear(width, width))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, quantiles: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        q = torch.where(present.bool().unsqueeze(-1), quantiles, torch.zeros_like(quantiles))
        features = quantile_features(q) * present.unsqueeze(-1).to(q.dtype)
        features = torch.cat((features, present.unsqueeze(-1).to(features.dtype)), -1)
        out = self.net(features.flatten(-2))
        return out * present.bool().any(-1).unsqueeze(-1).to(out.dtype)


def _probability_adapter(binary: int, width: int) -> nn.Sequential:
    adapter = nn.Sequential(nn.Linear(2 * binary, width), nn.SiLU(), nn.Linear(width, width))
    nn.init.zeros_(adapter[-1].weight)
    nn.init.zeros_(adapter[-1].bias)
    return adapter


def generator_states(device: torch.device):
    if device.type == "mps":
        return torch.get_rng_state(), torch.mps.get_rng_state()
    if device.type == "cuda":
        return torch.get_rng_state(), torch.cuda.get_rng_state_all()
    return torch.get_rng_state(), None


def restore_generators(states, device: torch.device) -> None:
    torch.set_rng_state(states[0])
    if device.type == "mps":
        torch.mps.set_rng_state(states[1])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(states[1])


class FeedbackPULSE(nn.Module):
    """PULSE encoders (`pulse`) with one density and one probability adapter per encoder."""

    def __init__(self, core: GroupedModel):
        super().__init__()
        self.pulse = core
        width = core.encoders[0].input_proj.out_features
        # Construction order fixes the initial weights: density adapters first.
        self.adapters = nn.ModuleList([DensityAdapter(len(V.CONTINUOUS), width) for _ in core.encoders])
        self.probability_adapters = nn.ModuleList([_probability_adapter(len(V.BINARY), width)
                                                   for _ in core.encoders])
        self.policy: Policy | None = None

    def attach_policy(self, policy: Policy) -> None:
        """Use a trained policy, frozen and in evaluation mode, for vasopressor descriptors.

        Frozen parameters matter beyond training: with trainable parameters
        the transformer takes another kernel and the logits change in the
        last bits. The policy is stored outside the module tree, so its
        parameters stay out of `parameters()` and `state_dict()`: the
        optimiser never sees them and PULSE checkpoints do not hold them.
        A plain attribute assignment would register it as a submodule.
        """
        policy.eval()
        for p in policy.parameters():
            p.requires_grad_(False)
        object.__setattr__(self, "policy", policy)

    def extra(self, descriptors: Descriptors, quantiles: bool, probabilities: bool) -> Extra | None:
        """Additions to each encoder's projected inputs, probability adapter first."""
        q, present, p, p_present = descriptors
        if not (quantiles or probabilities):
            return None

        def add(index: int, h: torch.Tensor) -> torch.Tensor:
            if probabilities:
                values = torch.cat((p * p_present, p_present), -1)
                h = h + self.probability_adapters[index](values) * p_present.any(-1, keepdim=True)
            if quantiles:
                h = h + self.adapters[index](q, present).to(h.dtype)
            return h

        return add

    def distribution_tensors(self, outputs: dict, shape: torch.Size) -> Descriptors:
        """Descriptors [B, T, V, 19], [B, T, V], [B, T, K], [B, T, K] of model outputs.

        Bolus volume has no law and stays absent; the vasopressor probability
        comes from the policy and is filled in by the caller.
        """
        q = outputs["mu"].new_zeros(*shape, len(PROBABILITIES))
        present = q.new_zeros(shape)
        p = outputs["mu"].new_zeros(*shape[:2], len(V.BINARY))
        p_present = torch.zeros_like(p)
        params = outputs["dist_params_list"]
        stacked = {k: torch.stack([par[k] for par in params], 2) for k in cnf.FLOW_KEYS}
        indices = self.pulse.output_indices
        q[:, :, indices] = quantiles_of(stacked)
        present[:, :, indices] = 1.0
        p[:, :, self.pulse.binary_output_indices] = outputs["bin_probs"]
        p_present[:, :, self.pulse.binary_output_indices] = 1.0
        return q, present, p, p_present

    def forward(self, history: dict[str, torch.Tensor], *, quantiles: bool, probabilities: bool) -> dict:
        """Teacher-forced outputs with lagged descriptors from a preview pass.

        The preview runs without gradients and without descriptors, and the
        generator states are restored afterwards, so training draws the same
        dropout masks with and without it.
        """
        if not (quantiles or probabilities):
            return self.pulse(history)
        device = history["continuous"].device
        with torch.no_grad():
            states = generator_states(device)
            preview = self.pulse(history)
            q, present, p, p_present = self.distribution_tensors(preview, history["continuous"].shape)
            if self.policy is not None:
                po = self.policy.treatments(history["continuous"], history["binary"], history["static"],
                                            history.get("time_hours"))
                p[:, :, [V.VASOPRESSOR]] = po["binary_treatment_probs"]
                p_present[:, :, [V.VASOPRESSOR]] = 1.0
            restore_generators(states, device)

            def lag(v):
                return torch.cat((torch.zeros_like(v[:, :1]), v[:, :-1]), 1)

            descriptors = tuple(lag(v) for v in (q, present, p, p_present))
        return self.pulse(history, extra=self.extra(descriptors, quantiles, probabilities))


def build_pulse(protocol: Protocol, train_binary: torch.Tensor) -> FeedbackPULSE:
    """PULSE as trained: seeded construction, binary heads at training log-odds, then adapters."""
    torch.manual_seed(SEED)
    core = pulse_model(protocol)
    core.set_binary_biases(core.binary_initial_logits(train_binary))
    return FeedbackPULSE(core)


class RolloutFeedback:
    """Descriptor history of one rollout call, written as rows join the event history.

    Rows before the forecast origin get the descriptors that teacher forcing
    would give them: row j holds the laws predicted from rows before j with
    the recorded next-hour binaries. Each generated row holds the laws from
    which it was drawn: CNF quantiles for every continuous variable, recorded
    or not, the binary probabilities and the policy's vasopressor
    probability.

    With `mean_state` the model and the policy read a separate numerical
    history in place of the sampled values: each generated row holds the
    predictive means of the recorded variables, clipped to their ranges,
    the previous mean state for unrecorded ones, the bolus volume given,
    and derived variables recomputed on that state.
    """

    def __init__(self, pulse: FeedbackPULSE, rows: int, steps: int, *, quantiles: bool, probabilities: bool,
                 mean_state: bool = False):
        device = pulse.pulse.cont_means.device
        self.pulse, self.quantiles, self.probabilities = pulse, quantiles, probabilities
        self.mean_state = mean_state
        self.values = self.state = None
        nv, nk = len(V.CONTINUOUS), len(V.BINARY)
        self.descriptors = (torch.zeros(rows, steps, nv, len(PROBABILITIES), device=device),
                            torch.zeros(rows, steps, nv, device=device),
                            torch.zeros(rows, steps, nk, device=device), torch.zeros(rows, steps, nk, device=device))

    def extra(self) -> Extra | None:
        return self.pulse.extra(self.descriptors, self.quantiles, self.probabilities)

    def history_values(self, continuous: torch.Tensor) -> torch.Tensor:
        """The numerical history [R, T, V] the model and the policy read."""
        return self.values[:, :continuous.shape[1]] if self.mean_state else continuous

    @torch.no_grad()
    def prefix(self, history: dict[str, torch.Tensor], policy: Policy, length: int) -> None:
        """Teacher-forced descriptors for rows 1 to `length - 1` of the recorded prefix.

        Also starts the mean state from the recorded rows.
        """
        if self.mean_state:
            self.values = history["continuous"].clone()
            self.state = torch.zeros_like(self.values)
            self.state[:, :length] = history["continuous"][:, :length]
        core = self.pulse.pulse
        outcomes = history["binary"][:, :, core.binary_output_indices]
        hours = history.get("time_hours")
        po = policy.treatments(history["continuous"][:, :length], history["binary"][:, :length], history["static"],
                               None if hours is None else hours[:, :length])
        q, present, p, p_present = self.descriptors
        timed = ("continuous", "binary", "time_hours", "continuous_mask", "padding_mask")
        for j in range(1, length):
            part = {k: (v[:, :j] if k in timed and v is not None else v) for k, v in history.items()}
            x = core.inputs(part)
            out = core.next_outputs(x, core.binary_logits(x), outcomes[:, 1:j + 1])
            dq, dm, dp, dpm = self.pulse.distribution_tensors(out, (len(outcomes), 1, len(V.CONTINUOUS)))
            # The policy's probability at row j - 1, from its softmax over the prefix.
            dp[:, 0, V.VASOPRESSOR] = po["binary_treatment_probs"][:, j - 1, 0]
            dpm[:, 0, V.VASOPRESSOR] = 1.0
            q[:, j], present[:, j], p[:, j], p_present[:, j] = dq[:, 0], dm[:, 0], dp[:, 0], dpm[:, 0]

    def record(self, outputs: dict, vasopressor: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, *,
               step: int, generated: torch.Tensor, recorded: torch.Tensor, baseline: torch.Tensor) -> None:
        """Store the laws of the step at history column `cols` of the appended `rows`.

        `vasopressor` [R, 1] is the sigmoid of the policy logit used for the
        draw. `generated` [R, V] is the simulator's new row `step + 1` and
        `recorded` [R, V] marks its recorded variables; the mean state
        advances every step, whether or not a row joins the history.
        """
        if self.mean_state:
            self._advance_mean_state(outputs, step, generated, recorded, baseline)
        q, present, p, p_present = self.pulse.distribution_tensors(outputs, (len(vasopressor), 1, len(V.CONTINUOUS)))
        p[:, 0, V.VASOPRESSOR] = vasopressor[:, 0]
        p_present[:, 0, V.VASOPRESSOR] = 1.0
        for stored, value in zip(self.descriptors, (q, present, p, p_present)):
            stored[rows, cols] = value[rows, 0]
        if self.mean_state:
            self.values[rows, cols] = self.state[rows, step + 1]

    def _advance_mean_state(self, outputs: dict, step: int, generated: torch.Tensor, recorded: torch.Tensor,
                            baseline: torch.Tensor) -> None:
        core = self.pulse.pulse
        low, high = V.bounds()
        point = self.state[:, step].clone()
        for idx, params in zip(core.output_indices, outputs["dist_params_list"]):
            # One quadrature call per variable, on [R, 1], as in the reference
            # implementation. A single call on all variables gives the same
            # values on CPU in the tests here; the per-variable form is kept so
            # that other devices follow the published computation step by step.
            value = core.denormalize(cnf.mean(params).unsqueeze(-1), [idx])[:, 0, 0]
            value = value.clamp(low[idx].item(), high[idx].item())
            point[:, idx] = torch.where(recorded[:, idx].bool(), value, point[:, idx])
        bolus = V.BOLUS
        point[:, bolus] = generated[:, bolus]
        self.state[:, step + 1] = V.forward_fill_zero_invalid(point, self.state[:, step])
        V.update_derived(self.state, step + 1, baseline)

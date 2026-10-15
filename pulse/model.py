"""Grouped transformer encoders shared by PULSE and the treatment policy.

Variables are split into groups and each group has its own causal
transformer. Every encoder reads the whole history of the patient: the
standardised continuous values (zero where nothing was recorded), the
binary variables, an observation mask and the log of the hours since each
variable was last recorded, the treatments, and the static variables. The
binary encoder predicts the next hour's binary outcomes; the continuous
encoders additionally receive those next-hour binaries and predict the
next value of their variables. Time enters through a learned encoding of
the elapsed hours and of the gap since the previous row.

PULSE predicts each continuous variable with the residual CNF of
`pulse.cnf`, anchored at the variable's last recorded standardised value.
The policy predicts bolus volume as a point and the vasopressor indicator
as a logit.

Constructing a model draws its initial weights from the global generator,
so the order of construction below must not change. Two kinds of layers
exist only for that reason: heads for variables that a group holds but does
not predict, and the gate layers drawn and discarded in `Encoder`.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from pulse import cnf
from pulse import variables as V
from pulse.data import Protocol

BINARY_GROUP = "binary_outcomes"
SEED = 724

# Policy groups: a fixed clinical grouping, then the binary outcome, then the
# remaining variable (bolus volume) in its own group.
POLICY_GROUPS = {
    "vitals": ["Heart Rate", "Respiratory Rate", "spO2", "Temperature"],
    "bp": ["Systolic Blood Pressure", "Diastolic Blood Pressure", "Mean Blood Pressure",
           "Minimum Mean Blood Pressure", "Minimum Change in Mean Blood Pressure from Baseline",
           "Minimum Mean Blood Pressure from Baseline",
           "Minimum Change in Mean Blood Pressure from Previous"],
    "blood_meta": ["Platelet", "Hemoglobin", "Lactate", "BUN", "Creatinine", "Calcium", "Bicarbonate"],
    "abg": ["pH", "pO2", "pCO2", "sO2", "Base excess", "Total CO2"],
    "fluids_o2": ["Fluid Volume", "Urine Output", "Cumulative Fluids", "Fluid Balance in ICU", "Weight",
                  "Change in Weight", "O2 Requirement Level", "Change in O2 from Baseline",
                  "Change in O2 from Previous"],
    "scores": ["SOFA Score", "Change in SOFA Score from Baseline", "Change in SOFA Score from Previous"],
    BINARY_GROUP: [V.BINARY[V.VASOPRESSOR]],
    "others": [V.CONTINUOUS[V.BOLUS]],
}

# An optional function per encoder that adds to the projected inputs [B, T, H].
Extra = Callable[[int, torch.Tensor], torch.Tensor]


class TimeEncoding(nn.Module):
    """Hours since admission and since the previous row, as raw values, signed log1p and sinusoids."""

    def __init__(self, width: int, frequencies: int = 8, longest_hours: float = 24.0):
        super().__init__()
        periods = torch.logspace(0.0, math.log10(longest_hours), steps=frequencies)
        self.register_buffer("angular_frequencies", (2.0 * math.pi) / periods)
        self.proj = nn.Sequential(nn.Linear(4 + 4 * frequencies, width), nn.GELU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, hours: torch.Tensor) -> torch.Tensor:
        hours = hours.float()
        gaps = torch.zeros_like(hours)
        if hours.shape[1] > 1:
            gaps[:, 1:] = hours[:, 1:] - hours[:, :-1]
        w = self.angular_frequencies.view(1, 1, -1)
        phase_t, phase_dt = hours.unsqueeze(-1) * w, gaps.unsqueeze(-1) * w
        features = torch.cat([hours.unsqueeze(-1), gaps.unsqueeze(-1),
                              (torch.sign(hours) * torch.log1p(hours.abs())).unsqueeze(-1),
                              (torch.sign(gaps) * torch.log1p(gaps.abs())).unsqueeze(-1),
                              torch.sin(phase_t), torch.cos(phase_t), torch.sin(phase_dt), torch.cos(phase_dt)], -1)
        return self.norm(self.proj(features))


class Encoder(nn.Module):
    """One group: input projection, time encoding, causal transformer and one head per variable.

    `kind` is "binary" (one logit), "cnf" (26 flow parameters) or "point" (one
    value). Continuous encoders also have heads for the probability that the
    variable is recorded next hour and for the hours until it is.
    """

    def __init__(self, input_dim: int, outputs: list[str], kind: str, width: int, layers: int, heads: int,
                 dropout: float, gate_draws: bool):
        super().__init__()
        self.outputs, self.kind = list(outputs), kind
        self.input_proj = nn.Linear(input_dim, width)
        self.time_encoder = TimeEncoding(width)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads, dim_feedforward=4 * width,
                                           dropout=dropout, batch_first=True)
        # Padding masks must take the ordinary path on every device.
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.heads = nn.ModuleList()
        for _ in self.outputs:
            head = nn.Linear(width, cnf.NUM_PARAMS if kind == "cnf" else 1)
            if kind == "cnf":
                cnf.initialize(head)
            self.heads.append(head)
        if kind != "binary":
            self.obs_heads = nn.ModuleList([nn.Linear(width, 1) for _ in self.outputs])
            self.delta_t_heads = nn.ModuleList([nn.Linear(width, 1) for _ in self.outputs])
        if gate_draws:
            # The reference model builds a gate layer per variable that never
            # affects its output; drawing the same weights keeps the generator
            # in step for everything constructed afterwards.
            for _ in self.outputs:
                nn.Linear(width, 1)

    def encode(self, x: torch.Tensor, hours: torch.Tensor, mask: torch.Tensor, padding: torch.Tensor | None,
               extra: Callable[[torch.Tensor], torch.Tensor] | None = None) -> torch.Tensor:
        h = self.input_proj(x)
        if extra is not None:
            h = extra(h)
        h = h + self.time_encoder(hours).to(h.dtype)
        return self.transformer(h, mask=mask, src_key_padding_mask=padding, is_causal=True)

    def predict(self, h: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        out = {}
        for i, name in enumerate(self.outputs):
            out[name] = {"raw": self.heads[i](h)}
            if self.kind != "binary":
                out[name]["obs_logit"] = self.obs_heads[i](h).squeeze(-1)
                out[name]["delta_t"] = (F.softplus(self.delta_t_heads[i](h)) + 1e-4).squeeze(-1)
        return out


class GroupedModel(nn.Module):
    """Grouped encoders over the full history.

    `groups` maps group names to variable names in construction order and
    contains `binary_outcomes`. `outputs` are the continuous variables the
    model predicts; a group without any of them predicts all its variables
    (unused heads kept for weight compatibility).
    """

    def __init__(self, *, groups: dict[str, list[str]], outputs: list[str], binary_outputs: list[str],
                 cnf_heads: bool, gate_draws: bool, static_dim: int = len(V.STATIC), width: int = 64,
                 layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.continuous_names, self.binary_names = list(V.CONTINUOUS), list(V.BINARY)
        self.outputs, self.binary_outputs = list(outputs), list(binary_outputs)
        self.output_indices = [self.continuous_names.index(n) for n in self.outputs]
        self.binary_output_indices = [self.binary_names.index(n) for n in self.binary_outputs]
        self.treatment_cont = [self.continuous_names.index(n) for n in V.TREATMENTS if n in self.continuous_names]
        self.treatment_bin = [self.binary_names.index(n) for n in V.TREATMENTS if n in self.binary_names]
        self.group_names = list(groups)
        self.binary_group = self.group_names.index(BINARY_GROUP)
        nv, nk = len(self.continuous_names), len(self.binary_names)
        history_dim = nv + nk + len(V.TREATMENTS) + 2 * nv
        self.encoders = nn.ModuleList()
        for name in self.group_names:
            if name == BINARY_GROUP:
                encoder = Encoder(history_dim + static_dim, self.binary_outputs, "binary",
                                  width, layers, heads, dropout, gate_draws=False)
            else:
                predicted = [v for v in groups[name] if v in self.outputs] or list(groups[name])
                encoder = Encoder(history_dim + len(self.binary_outputs) + static_dim, predicted,
                                  "cnf" if cnf_heads else "point", width, layers, heads, dropout, gate_draws)
            self.encoders.append(encoder)
        self.register_buffer("cont_means", torch.zeros(nv))
        self.register_buffer("cont_stds", torch.ones(nv))

    def set_normalization(self, mean, sd) -> None:
        self.cont_means.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.cont_stds.copy_(torch.as_tensor(sd, dtype=torch.float32))

    def binary_initial_logits(self, binary: torch.Tensor) -> dict[str, float]:
        """Training-set log-odds of each binary output turning on in the next hour.

        `binary` is [N, T, K]. For an absorbing variable only hours before it
        first turned on count. Rates are kept inside [1e-4, 1 - 1e-4].
        """
        logits = {}
        for name, i in zip(self.binary_outputs, self.binary_output_indices):
            target = binary[:, 1:, i].float()
            if name in V.ABSORBING:
                before = binary[:, :-1, i].float().cummax(dim=1).values < 0.5
                rate = target[before].mean().item() if before.any() else 0.0
            else:
                rate = target.mean().item() if target.numel() else 0.0
            rate = min(max(float(rate), 1e-4), 1.0 - 1e-4)
            logits[name] = math.log(rate / (1.0 - rate))
        return logits

    def set_binary_biases(self, logits: dict[str, float]) -> None:
        """Start each binary head at a given logit with zero weights."""
        encoder = self.encoders[self.binary_group]
        with torch.no_grad():
            for head, name in zip(encoder.heads, encoder.outputs):
                if name in logits:
                    head.bias.fill_(float(logits[name]))
                    head.weight.zero_()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.cont_means.clone().to(x.device)) / self.cont_stds.clone().to(x.device)

    def denormalize(self, x: torch.Tensor, indices: list[int]) -> torch.Tensor:
        return x * self.cont_stds.clone().to(x.device)[indices] + self.cont_means.clone().to(x.device)[indices]

    # Inputs ---------------------------------------------------------------

    def inputs(self, history: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encoder inputs shared by all groups.

        `history` holds `continuous` [B, T, V] in physical units, `binary`
        [B, T, K], and optionally `continuous_mask` [B, T, V], `time_hours`
        [B, T], `static` [B, S] and `padding_mask` [B, T] (True at padding).
        """
        cont, binary = history["continuous"], history["binary"]
        if history.get("static") is None:
            raise ValueError("history needs the static variables [B, S]")
        b, t, nv = cont.shape
        norm = self.normalize(cont)
        hours = history.get("time_hours")
        if hours is None:
            hours = torch.arange(t, device=cont.device, dtype=cont.dtype).view(1, t).expand(b, t)
        mask = history.get("continuous_mask")
        mask = torch.ones_like(norm) if mask is None else mask.to(norm.device, norm.dtype)[:, :t].clamp(0.0, 1.0)

        # Hours since each variable was last recorded, and its last recorded value.
        times = hours[:, :t].to(cont.device, cont.dtype)
        last_time = times[:, :1].expand(b, nv).clone()
        last_value = norm.new_zeros((b, nv))
        ages, anchors = [], []
        for step in range(t):
            seen = mask[:, step, :] > 0.5
            now = times[:, step:step + 1].expand(b, nv)
            last_time = torch.where(seen, now, last_time)
            last_value = torch.where(seen, norm[:, step, :], last_value)
            ages.append((now - last_time).clamp_min(0.0))
            anchors.append(last_value)
        age = torch.log1p(torch.stack(ages, 1))

        value = norm * mask
        treatments = torch.cat([value[..., i:i + 1] for i in self.treatment_cont]
                               + [binary[..., i:i + 1] for i in self.treatment_bin], -1)
        cont_hist, bin_hist = value.clone(), binary.clone()
        cont_hist[..., self.treatment_cont] = 0.0
        bin_hist[..., self.treatment_bin] = 0.0
        return {"history": torch.cat([cont_hist, bin_hist, mask, age], -1), "treatments": treatments,
                "anchor": torch.stack(anchors, 1), "static": history["static"],
                "hours": hours.to(cont.device), "padding": history.get("padding_mask"),
                "causal": torch.triu(torch.ones(t, t, device=cont.device), diagonal=1).bool(),
                "factual_binary": binary}

    def _run(self, index: int, history: torch.Tensor, x: dict, extra: Extra | None) -> dict:
        t = history.shape[1]
        features = torch.cat([history, x["treatments"], x["static"].unsqueeze(1).expand(-1, t, -1)], -1)
        add = None if extra is None else (lambda h: extra(index, h))
        h = self.encoders[index].encode(features, x["hours"], x["causal"], x["padding"], add)
        return self.encoders[index].predict(h)

    # Outputs --------------------------------------------------------------

    def binary_logits(self, x: dict, extra: Extra | None = None) -> torch.Tensor:
        """Next-hour logits of the binary outcomes, [B, T, K_out]."""
        out = self._run(self.binary_group, x["history"], x, extra)
        return torch.stack([out[n]["raw"][..., 0] for n in self.binary_outputs], -1)

    def continuous_outputs(self, x: dict, next_binary: torch.Tensor, extra: Extra | None = None) -> dict:
        """Run every continuous group given next-hour binary outcomes [B, T, K_out]."""
        history = torch.cat([x["history"], next_binary], -1)
        out = {}
        for index, name in enumerate(self.group_names):
            if name == BINARY_GROUP:
                continue
            for var, pred in self._run(index, history, x, extra).items():
                if self.encoders[index].kind == "cnf":
                    pred["params"] = cnf.anchor(cnf.parse(pred["raw"]), x["anchor"][..., self.continuous_names.index(var)])
                    pred["mu"] = pred["params"]["mu"]
                else:
                    pred["mu"] = pred["raw"].squeeze(-1)
                out[var] = pred
        return out

    def _collect(self, per_var: dict, logits: torch.Tensor) -> dict:
        b, t, _ = logits.shape
        n = len(self.outputs)
        mu = torch.zeros(b, t, n, device=logits.device)
        log_var = torch.zeros(b, t, n, device=logits.device) if "params" in per_var[self.outputs[0]] else None
        obs, delta = torch.zeros(b, t, n, device=logits.device), torch.zeros(b, t, n, device=logits.device)
        params = []
        for i, name in enumerate(self.outputs):
            pred = per_var[name]
            mu[:, :, i] = pred["mu"]
            if log_var is not None:
                log_var[:, :, i] = pred["params"]["log_var"]
            obs[:, :, i] = pred["obs_logit"]
            delta[:, :, i] = pred["delta_t"]
            params.append(pred.get("params"))
        return {"mu": mu, "log_var": log_var, "obs_logits": obs, "delta_t_hours": delta,
                "bin_logits": logits, "dist_params_list": params if params[0] is not None else None}

    def next_binary_factual(self, x: dict) -> torch.Tensor:
        outcomes = x["factual_binary"][..., self.binary_output_indices]
        return torch.cat([outcomes[:, 1:, :], outcomes[:, -1:, :]], 1)

    def forward(self, history: dict[str, torch.Tensor], *, teacher_forcing: bool = True,
                extra: Extra | None = None) -> dict:
        """All positions. With teacher forcing the continuous encoders see the
        recorded next-hour binaries, otherwise the predicted probabilities.
        `bin_probs` is the softmax over a zero logit and the logit."""
        x = self.inputs(history)
        logits = self.binary_logits(x, extra)
        next_binary = self.next_binary_factual(x) if teacher_forcing else torch.sigmoid(logits)
        out = self._collect(self.continuous_outputs(x, next_binary, extra), logits)
        out["bin_probs"] = torch.softmax(torch.stack([torch.zeros_like(logits), logits], -1), -1)[..., 1]
        return out

    def predict_next(self, history: dict[str, torch.Tensor], next_binary: torch.Tensor | None = None,
                     lengths: torch.Tensor | None = None, extra: Extra | None = None) -> dict:
        """Outputs at the last valid row of each history, each with a time axis of one.

        `next_binary` [B, T, K or K_out] replaces the predicted next-hour
        binaries; `lengths` [B] gives the valid rows of padded histories.
        `bin_probs` is the sigmoid of the logit.
        """
        x = self.inputs(history)
        logits = self.binary_logits(x, extra)
        if next_binary is None:
            next_binary = torch.sigmoid(logits)
        elif next_binary.shape[-1] == len(self.binary_names):
            next_binary = next_binary[..., self.binary_output_indices]
        out = self._collect(self.continuous_outputs(x, next_binary, extra), logits)
        out["bin_probs"] = torch.sigmoid(logits)
        b, t, _ = logits.shape
        last = (torch.full((b,), t - 1, device=logits.device, dtype=torch.long) if lengths is None
                else lengths.to(logits.device, torch.long).clamp_min(1).clamp_max(t) - 1)

        def gather(v):
            if not isinstance(v, torch.Tensor) or v.dim() < 2:
                return v
            index = last.view([b, 1] + [1] * (v.dim() - 2)).expand(-1, 1, *v.shape[2:])
            return v.gather(1, index)

        params = out.pop("dist_params_list")
        result = {k: gather(v) for k, v in out.items()}
        result["dist_params_list"] = None if params is None else [{k: gather(v) for k, v in p.items()} for p in params]
        return result


def pulse_model(protocol: Protocol) -> GroupedModel:
    """The PULSE encoders: covariance groups from the protocol plus the binary outcomes.

    Encoders are built in the key order of `protocol.groups`: the covariance
    groups by label, then `continuous_treatments`, as in the reference.
    """
    if list(protocol.groups)[-1] != "continuous_treatments":
        raise ValueError("protocol groups must end with continuous_treatments")
    outputs = [n for n in V.CONTINUOUS if n != V.CONTINUOUS[V.BOLUS]]
    model = GroupedModel(groups={**protocol.groups, BINARY_GROUP: [n for n in V.BINARY if n != V.BINARY[V.VASOPRESSOR]]},
                         outputs=outputs, binary_outputs=[n for n in V.BINARY if n != V.BINARY[V.VASOPRESSOR]],
                         cnf_heads=True, gate_draws=True)
    model.set_normalization(protocol.mean, protocol.sd)
    return model


class Policy(GroupedModel):
    """Treatment policy: next-hour bolus volume and vasopressor logit."""

    def __init__(self, protocol: Protocol):
        super().__init__(groups=POLICY_GROUPS, outputs=[V.CONTINUOUS[V.BOLUS]], binary_outputs=[V.BINARY[V.VASOPRESSOR]],
                         cnf_heads=False, gate_draws=False)
        self.set_normalization(protocol.mean, protocol.sd)

    def treatments(self, continuous, binary, static, time_hours) -> dict:
        """All positions; the bolus volume is returned in physical units."""
        out = self({"continuous": continuous, "binary": binary, "static": static, "time_hours": time_hours},
                   teacher_forcing=False)
        return {"continuous_treatments": self.denormalize(out["mu"], self.output_indices),
                "binary_treatment_logits": out["bin_logits"], "binary_treatment_probs": out["bin_probs"],
                "observation_logits": out["obs_logits"], "delta_t_hours": out["delta_t_hours"]}


def build_policy(protocol: Protocol) -> Policy:
    torch.manual_seed(SEED)
    return Policy(protocol)


# Checkpoints -------------------------------------------------------------
#
# The artifact saves {"state": state_dict, "history": per-epoch validation
# losses, "cache_sha256": content digest of the training cache} and reads it
# without unpickling. Checkpoints of the reference implementation hold the
# state under "model" with longer names and pickle their configuration; only
# `read_reference_state` reads them, mapping the names as follows:
#
#   policy  model.model.<rest>                       -> <rest>
#   PULSE   pulse.model.<rest>                       -> pulse.<rest>
#           pulse.model.encoders.<i>.residual_gate_heads.*   dropped (never used)
#           adapters.*, probability_adapters.*       unchanged


def research_policy_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not all(k.startswith("model.model.") for k in state):
        raise ValueError("not a reference policy state")
    return {k.removeprefix("model.model."): v for k, v in state.items()}


def research_pulse_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(k.startswith("pulse.model.") for k in state):
        raise ValueError("not a reference PULSE state")
    return {("pulse." + k.removeprefix("pulse.model.") if k.startswith("pulse.model.") else k): v
            for k, v in state.items() if ".residual_gate_heads." not in k}


def save_checkpoint(path, model: nn.Module, history: list[dict], cache_sha256: str) -> None:
    """Weights, validation history and the cache digest of the protocol the model was trained with."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    torch.save({"state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "history": history, "cache_sha256": cache_sha256}, pending)
    pending.replace(path)


def read_state(path) -> dict[str, torch.Tensor]:
    """State dict of an artifact checkpoint, loaded without unpickling."""
    return torch.load(path, map_location="cpu", weights_only=True)["state"]


def read_reference_state(path) -> dict[str, torch.Tensor]:
    """State dict of a reference checkpoint, in artifact names, for the reference comparisons only.

    Reference checkpoints pickle their configuration and audit, so reading
    one unpickles arbitrary objects: use it only on files of known origin.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    return research_policy_state(state) if next(iter(state)).startswith("model.model.") else research_pulse_state(state)

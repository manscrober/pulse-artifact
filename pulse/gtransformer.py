"""G-Transformer baseline: one causal transformer per predicted variable.

G-Transformer (Xiong et al., 2024) simulates a patient with a separate
encoder for each variable it predicts. Variables are predicted in a fixed
order, binary outcomes first, and each encoder also reads the values of
the variables before it at the hour it predicts, so that within an hour the
joint law factorises as

    p(L_t | history) = p(L^0_t | history) p(L^1_t | L^0_t, history) ...

Every encoder reads, at each row, the two treatments (bolus volume and the
vasopressor indicator), the 36 continuous outcomes in standard deviations
and the 7 binary outcomes, the earlier variables of the same hour and the
static covariates, adds a sinusoidal encoding of the row index and applies
a causal transformer. A binary encoder ends in two class logits; a
continuous encoder in one predicted value and the logit that the variable
is recorded.

This is the reference implementation's `PerCovariateGTransformer` with the
settings of the published baseline: point predictions trained by squared
error, no learned variance. Its encoders take no padding mask, so the
prediction for the next hour is read from a row the caller names
(`next_outputs`).

`pulse.train` fits it as the phase `gtransformer` with the loss of the
point rows of the component table, and `pulse.simulate.rollout` rolls it
out with the plain `sample` arm. In a rollout it reads the hourly rows
generated so far, the form it is trained on, while the policy reads the
event history as for PULSE.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from pulse import variables as V
from pulse.model import SEED, GroupedModel

MAX_ROWS = 512


def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")


class Positions(nn.Module):
    """Sinusoidal encoding of the row index, added to the projected inputs."""

    def __init__(self, width: int, rows: int = MAX_ROWS):
        super().__init__()
        pe = torch.zeros(rows, width)
        position = torch.arange(0, rows, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class CovariateEncoder(nn.Module):
    """The encoder of the variable at position `index` of the prediction order."""

    def __init__(self, index: int, categorical: bool, inputs: int, width: int, layers: int, heads: int,
                 dropout: float, observation: bool):
        super().__init__()
        self.index, self.categorical = index, categorical
        self.input_proj = nn.Linear(inputs + index, width)
        self.pos_encoding = Positions(width)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads, dim_feedforward=4 * width, dropout=dropout,
                                           activation="relu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.output_head = nn.Linear(width, 2 if categorical else 1)
        if observation and not categorical:
            self.obs_head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.transformer(self.pos_encoding(self.input_proj(x)), mask=mask)
        out = self.output_head(hidden)
        if self.categorical:
            return {"logits": out, "probs": F.softmax(out, dim=-1)}
        return {"value": out.squeeze(-1),
                "obs_logit": self.obs_head(hidden).squeeze(-1) if hasattr(self, "obs_head") else None}


class GTransformer(nn.Module):
    """The published G-Transformer baseline: 7 binary and 36 continuous encoders.

    Outputs follow `pulse.model.GroupedModel`: `mu` [B, T, 36] in standard
    deviations, `bin_logits` and `bin_probs` [B, T, 7], `obs_logits`
    [B, T, 36]; row t predicts hour t + 1.
    """

    # The prediction is a point; rollouts add the static noise of `pulse.simulate.static_noise`.
    head = "point"

    def __init__(self, *, width: int = 64, layers: int = 2, heads: int = 4, dropout: float = 0.1,
                 observation: bool = True):
        super().__init__()
        self.continuous_names, self.binary_names = list(V.CONTINUOUS), list(V.BINARY)
        self.outputs = [n for n in V.CONTINUOUS if n != V.CONTINUOUS[V.BOLUS]]
        self.binary_outputs = [n for n in V.BINARY if n != V.BINARY[V.VASOPRESSOR]]
        self.output_indices = [self.continuous_names.index(n) for n in self.outputs]
        self.binary_output_indices = [self.binary_names.index(n) for n in self.binary_outputs]
        self.order = self.binary_outputs + self.outputs
        inputs = len(V.TREATMENTS) + len(self.order) + len(V.STATIC)
        self.encoders = nn.ModuleDict()
        for i, name in enumerate(self.order):
            self.encoders[safe_name(name)] = CovariateEncoder(i, name in self.binary_outputs, inputs, width, layers,
                                                              heads, dropout, observation)
        self.register_buffer("cont_means", torch.zeros(len(self.continuous_names)))
        self.register_buffer("cont_stds", torch.ones(len(self.continuous_names)))

    # Normalisation and initialisation, as in the reference ------------------

    set_normalization = GroupedModel.set_normalization
    binary_initial_logits = GroupedModel.binary_initial_logits

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.cont_means) / (self.cont_stds + 1e-8)

    def denormalize(self, x: torch.Tensor, indices: list[int]) -> torch.Tensor:
        return x * (self.cont_stds[indices] + 1e-8) + self.cont_means[indices]

    def set_binary_biases(self, logits: dict[str, float]) -> None:
        """Start each binary encoder at a given log-odds: class logits (0, logit), zero weights."""
        with torch.no_grad():
            for name in self.binary_outputs:
                if name in logits:
                    head = self.encoders[safe_name(name)].output_head
                    head.bias[0] = 0.0
                    head.bias[1] = float(logits[name])
                    head.weight.zero_()

    # Forward ----------------------------------------------------------------

    def run(self, continuous: torch.Tensor, binary: torch.Tensor, static: torch.Tensor, *, teacher_forcing: bool,
            sample_binary: bool = False) -> dict:
        """Every encoder in order over the whole history.

        With `teacher_forcing` each encoder reads the recorded values of the
        earlier variables at the hour it predicts. Otherwise it reads the
        predictions of the earlier encoders: the predicted value of a
        continuous variable and, for a binary one, a draw from its
        probability with `sample_binary` (kept in `draws`) or the
        probability itself.
        """
        b, t, _ = continuous.shape
        norm = self.normalize(continuous)
        treatments = torch.cat([norm[..., [V.BOLUS]], binary[..., [V.VASOPRESSOR]]], dim=-1)
        history = torch.cat([norm[..., self.output_indices], binary[..., self.binary_output_indices]], dim=-1)
        mask = torch.triu(torch.ones(t, t, device=continuous.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        static = static.unsqueeze(1).expand(-1, t, -1)
        recorded = None
        if teacher_forcing:
            recorded = torch.stack([norm[..., self.continuous_names.index(n)] if n in self.outputs
                                    else binary[..., self.binary_names.index(n)] for n in self.order], dim=-1)
            recorded = torch.cat([recorded[:, 1:, :], recorded[:, -1:, :]], dim=1)
        out = {"values": {}, "obs_logits": {}, "logits": {}, "probs": {}, "draws": {}}
        predicted = []
        for i, name in enumerate(self.order):
            parts = [treatments, history]
            if i > 0:
                parts.append(recorded[..., :i] if teacher_forcing else torch.stack(predicted, dim=-1))
            parts.append(static)
            o = self.encoders[safe_name(name)](torch.cat(parts, dim=-1), mask)
            if name in self.binary_outputs:
                probs = F.softmax(o["logits"], dim=-1)
                out["logits"][name], out["probs"][name] = o["logits"], probs
                positive = probs[..., 1]
                if not teacher_forcing and sample_binary:
                    out["draws"][name] = (torch.rand_like(positive) < positive).float()
                    predicted.append(out["draws"][name])
                else:
                    predicted.append(positive)
            else:
                out["values"][name] = o["value"]
                if o["obs_logit"] is not None:
                    out["obs_logits"][name] = o["obs_logit"]
                predicted.append(o["value"])
        return out

    def forward(self, history: dict[str, torch.Tensor]) -> dict:
        """Teacher-forced outputs for training, in the layout of `pulse.model.GroupedModel`."""
        out = self.run(history["continuous"], history["binary"], history["static"], teacher_forcing=True)
        mu = torch.stack([out["values"][n] for n in self.outputs], dim=-1)
        logits = torch.stack([out["logits"][n][..., 1] - out["logits"][n][..., 0] for n in self.binary_outputs], -1)
        probs = torch.softmax(torch.stack([torch.zeros_like(logits), logits], dim=-1), dim=-1)[..., 1]
        obs = torch.stack([out["obs_logits"][n] for n in self.outputs], dim=-1) if out["obs_logits"] else None
        return {"mu": mu, "log_var": None, "bin_logits": logits, "bin_probs": probs, "obs_logits": obs,
                "delta_t_hours": None, "dist_params_list": None}

    def next_outputs(self, history: dict[str, torch.Tensor], rows: torch.Tensor | None = None) -> dict:
        """Predictions for the next hour of each history, without teacher forcing.

        Binary predictions are drawn for the later encoders as in a rollout;
        `bin_samples` holds those draws, which a rollout takes as the next
        hour's outcomes. `rows` [B] names the row of each history to read;
        None reads the last row of the tensor, which is what the reference
        did for padded event histories too (issue G1 of the port tracker).
        """
        cont = history["continuous"]
        out = self.run(cont, history["binary"], history["static"], teacher_forcing=False, sample_binary=True)
        b, t, _ = cont.shape
        mu = torch.zeros(b, t, len(self.outputs), device=cont.device)
        obs = torch.zeros(b, t, len(self.outputs), device=cont.device) if out["obs_logits"] else None
        for i, name in enumerate(self.outputs):
            mu[:, :, i] = out["values"][name]
            if obs is not None:
                obs[:, :, i] = out["obs_logits"][name]
        logits = torch.zeros(b, t, len(self.binary_outputs), device=cont.device)
        probs = torch.zeros(b, t, len(self.binary_outputs), device=cont.device)
        for i, name in enumerate(self.binary_outputs):
            logits[:, :, i] = out["logits"][name][..., 1] - out["logits"][name][..., 0]
            probs[:, :, i] = out["probs"][name][..., 1]
        draws = torch.stack([out["draws"][n] for n in self.binary_outputs], dim=-1)

        def read(x):
            if x is None:
                return None
            return x[:, -1:, :] if rows is None else x[torch.arange(b, device=x.device), rows].unsqueeze(1)
        return {"mu": read(mu), "log_var": None, "bin_logits": read(logits), "bin_probs": read(probs),
                "bin_samples": read(draws), "obs_logits": read(obs), "delta_t_hours": None,
                "dist_params_list": None}


def gtransformer_model(protocol) -> GTransformer:
    """The baseline built and normalised as the reference built it, from the global generator."""
    model = GTransformer()
    model.set_normalization(protocol.mean, protocol.sd)
    return model


def build_gtransformer(protocol, train_binary: torch.Tensor) -> GTransformer:
    """Seeded construction and binary encoders at training log-odds, as for PULSE."""
    torch.manual_seed(SEED)
    model = gtransformer_model(protocol)
    model.set_binary_biases(model.binary_initial_logits(train_binary))
    return model

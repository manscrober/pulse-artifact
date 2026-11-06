r"""The predictive law of each continuous variable, one kind per component of Table 2.

Every head turns the raw outputs of one linear layer into parameters, a
location `mu` and, except for the point head, the log of a variance:

- `point`: one value, the location; rollouts add the static noise of
  `pulse.simulate.static_noise`.
- `gaussian`: a location and a log variance clamped below at -5.
- `mog2`: a mixture of two Gaussians, with one free mixing logit, two
  locations and two log variances; `mu` and `log_var` are the mixture's
  mean and the log of its total variance.
- `bspline`: a direct-location monotone cubic B-spline transform.
- `cnf`: the scalar flow of `pulse.cnf` on the level of the variable.
- `rescnf`: the same flow with its location added to the variable's last
  recorded standardised value (the anchor).

All values are in training standard deviations.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from pulse import cnf
from pulse.bspline import CubicUniformBSplineHead

SPLINE = CubicUniformBSplineHead(n_control_points=24)

KINDS = ("point", "gaussian", "mog2", "cnf", "rescnf", "bspline")
FLOWS = ("cnf", "rescnf", "bspline")
LOG_VAR_MIN = -5.0
MOG_SCALE_MIN = 1e-4

Params = dict[str, torch.Tensor]


def num_params(kind: str) -> int:
    return {"point": 1, "gaussian": 2, "mog2": 5, "cnf": cnf.NUM_PARAMS, "rescnf": cnf.NUM_PARAMS, "bspline": SPLINE.num_params}[kind]


@torch.no_grad()
def initialize(kind: str, head: torch.nn.Linear) -> None:
    """Initial weights; the point head keeps the default initialisation of `nn.Linear`."""
    if kind == "bspline":
        head.weight.zero_()
        head.bias.zero_()
    elif kind in FLOWS:
        cnf.initialize(head)
    elif kind == "gaussian":
        head.weight.zero_()
        head.bias.zero_()
        head.bias[1] = math.log(0.1 * 0.1)
    elif kind == "mog2":
        head.weight.zero_()
        head.bias.zero_()
        head.bias[3:5] = math.log(0.1 * 0.1)
        head.bias[1:3] = torch.tensor([-0.05, 0.05], device=head.bias.device, dtype=head.bias.dtype)


def parse(kind: str, raw: torch.Tensor, anchor: torch.Tensor) -> Params:
    """Parameters of raw outputs [..., num_params]; `anchor` [...] is used only by `rescnf`."""
    if kind == "point":
        return {"mu": raw[..., 0]}
    if kind == "gaussian":
        log_var = torch.clamp(raw[..., 1], min=LOG_VAR_MIN)
        return {"mu": raw[..., 0], "log_var": log_var, "sigma": torch.exp(0.5 * log_var)}
    if kind == "mog2":
        logits = torch.cat([torch.zeros_like(raw[..., :1]), raw[..., :1]], -1)
        pis = F.softmax(logits, dim=-1)
        mus = raw[..., 1:3]
        sigmas = torch.exp(0.5 * torch.clamp(raw[..., 3:5], min=LOG_VAR_MIN)) + MOG_SCALE_MIN
        mu = (pis * mus).sum(dim=-1)
        total = (pis * sigmas ** 2).sum(dim=-1) + (pis * (mus - mu.unsqueeze(-1)) ** 2).sum(dim=-1)
        return {"pis": pis, "mus": mus, "sigmas": sigmas, "mu": mu, "log_var": torch.log(total + 1e-8)}
    if kind == "bspline":
        return SPLINE.parse(raw)
    params = cnf.parse(raw)
    if kind == "rescnf":
        params = cnf.anchor(params, anchor)
    params["mu"] = params["loc"]
    return params


def stack(kind: str, per_variable: list[Params], index) -> Params:
    """Parameters of every variable stacked on a variable axis after batch and time.

    `index` selects along the time axis first: a slice such as
    `slice(None, -1)` keeps it ([B, T', V, ...]), an integer drops it ([B, V, ...]).
    """
    keys = {"gaussian": ("mu", "log_var"), "mog2": ("pis", "mus", "sigmas", "mu")}.get(kind, cnf.FLOW_KEYS)
    if kind == "bspline":
        keys = ("loc", "scale", "control_points", "mu")
    dim = 1 if isinstance(index, int) else 2
    return {k: torch.stack([p[k][:, index] for p in per_variable], dim) for k in keys}


def nll(kind: str, params: Params, target: torch.Tensor) -> torch.Tensor:
    """Negative log density of `target`, elementwise; the Gaussian form omits log(2 pi), as the reference did."""
    if kind == "gaussian":
        log_var = torch.clamp(params["log_var"], min=LOG_VAR_MIN)
        return 0.5 * (log_var + (target - params["mu"]) ** 2 * torch.exp(-log_var))
    if kind == "mog2":
        # Grouped as in the reference, which fixes the rounding.
        log_norms = -0.5 * math.log(2 * math.pi) - torch.log(params["sigmas"])
        squared = ((target.unsqueeze(-1) - params["mus"]) / params["sigmas"]) ** 2
        log_components = torch.log(params["pis"] + 1e-8) + log_norms - 0.5 * squared
        top = log_components.max(dim=-1, keepdim=True).values
        return -(top.squeeze(-1) + torch.log(torch.exp(log_components - top).sum(dim=-1)))
    if kind == "bspline":
        return SPLINE.nll(params, target)
    if kind in FLOWS:
        return cnf.nll(params, target)
    raise ValueError(f"the {kind} head has no density")


def sample(kind: str, params: Params) -> torch.Tensor:
    """Draws in standardised units for the mixture and the flows, from the global generator."""
    if kind == "mog2":
        pis = params["pis"]
        gumbel = -torch.log(-torch.log(torch.rand_like(pis) + 1e-8) + 1e-8)
        component = (torch.log(pis + 1e-8) + gumbel).argmax(dim=-1, keepdim=True)
        mu = params["mus"].gather(-1, component).squeeze(-1)
        sigma = params["sigmas"].gather(-1, component).squeeze(-1)
        return mu + sigma * torch.randn_like(mu)
    if kind == "bspline":
        return SPLINE.sample(params)
    if kind in FLOWS:
        return cnf.sample(params)
    raise ValueError(f"the {kind} head is drawn in physical units by the simulator")

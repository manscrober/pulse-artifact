r"""Residual scalar continuous normalising flow, the predictive law of each variable.

For one variable at one hour the network emits 26 raw numbers: a location
$\ell$, a log scale $\log s$, and amplitudes $a_j$, centres $c_j$ and widths
$b_j$ of eight basis functions. A standard normal draw $z$ is pushed through
the ordinary differential equation

$$\frac{dy}{d\tau} = \sum_{j=1}^{8} a_j \tanh\big(b_j (y - c_j)\big),
\qquad y(0) = z,$$

integrated over $\tau \in [0, 1]$ with 16 Heun steps, and the value is
$x = \ell + s\,y(1)$. Running the same steps backwards from $y(1)$ gives the
likelihood, with the log-Jacobian accumulated from $\partial v / \partial y$.
"Residual" means that the location is added to an anchor, the last value of
the variable, so the head predicts the change from the previous hour.

All values are in standardised units. Parameters travel as a dictionary of
tensors with matching leading shape; `amps`, `centers` and `widths` have a
trailing basis dimension.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

BASIS = 8
STEPS = 16
MAX_AMPLITUDE = 0.35
NUM_PARAMS = 2 + 3 * BASIS
LOG_SCALE_MIN, LOG_SCALE_MAX = -5.0, 3.0
QUANTILE_EPS = 1e-6

# Nine-node Gauss-Hermite rule for expectations under a standard normal.
_nodes, _weights = np.polynomial.hermite.hermgauss(9)
QUAD_NODES = torch.tensor(np.sqrt(2.0) * _nodes, dtype=torch.float32)
QUAD_WEIGHTS = torch.tensor(_weights / np.sqrt(np.pi), dtype=torch.float32)

Params = dict[str, torch.Tensor]

# The parameters the flow functions read; callers stack these over variables.
FLOW_KEYS = ("loc", "scale", "log_scale", "amps", "centers", "widths")


def parse(raw: torch.Tensor) -> Params:
    """Turn raw outputs [..., 26] into flow parameters."""
    loc = raw[..., 0]
    log_scale = torch.clamp(raw[..., 1], LOG_SCALE_MIN, LOG_SCALE_MAX)
    amps = MAX_AMPLITUDE * torch.tanh(raw[..., 2:2 + BASIS])
    centers = 4.0 * torch.tanh(raw[..., 2 + BASIS:2 + 2 * BASIS] / 4.0)
    widths = torch.clamp(F.softplus(raw[..., 2 + 2 * BASIS:2 + 3 * BASIS]) + 0.25, max=5.0)
    return {"loc": loc, "log_scale": log_scale, "scale": torch.exp(log_scale), "amps": amps,
            "centers": centers, "widths": widths, "log_var": 2.0 * log_scale}


def anchor(params: Params, value: torch.Tensor) -> Params:
    """Shift the location by `value`, the variable's last standardised value."""
    shifted = dict(params)
    shifted["loc"] = params["loc"] + value
    return shifted


def _velocity(y: torch.Tensor, p: Params) -> tuple[torch.Tensor, torch.Tensor]:
    tanh = torch.tanh(p["widths"] * (y.unsqueeze(-1) - p["centers"]))
    velocity = torch.sum(p["amps"] * tanh, dim=-1)
    slope = torch.sum(p["amps"] * p["widths"] * (1.0 - tanh.pow(2)), dim=-1)
    return velocity, slope


def _integrate(y: torch.Tensor, p: Params, direction: float) -> tuple[torch.Tensor, torch.Tensor]:
    dt = direction / STEPS
    logdet = torch.zeros_like(y)
    for _ in range(STEPS):
        velocity, slope = _velocity(y, p)
        proposal = y + dt * velocity
        velocity_next, slope_next = _velocity(proposal, p)
        y = y + 0.5 * dt * (velocity + velocity_next)
        logdet = logdet + 0.5 * dt * (slope + slope_next)
    return y, logdet


def forward(z: torch.Tensor, p: Params) -> torch.Tensor:
    """Map base draws `z` to values."""
    y, _ = _integrate(z, p, 1.0)
    return p["loc"] + p["scale"] * y


def nll(p: Params, target: torch.Tensor) -> torch.Tensor:
    """Negative log density of `target`, elementwise."""
    y = (target - p["loc"]) / p["scale"]
    z, logdet = _integrate(y, p, -1.0)
    log_base = -0.5 * (math.log(2.0 * math.pi) + z.pow(2))
    return -(log_base + (-p["log_scale"] + logdet))


def sample(p: Params, z: torch.Tensor | None = None) -> torch.Tensor:
    """Draw values; without `z` the base draws come from the global generator."""
    z = torch.randn_like(p["loc"]) if z is None else z.to(p["loc"]).reshape_as(p["loc"])
    return forward(z, p)


def quantile(p: Params, q: torch.Tensor | float) -> torch.Tensor:
    """Values at probabilities `q`, broadcast to the parameter shape."""
    q = torch.as_tensor(q, device=p["loc"].device, dtype=p["loc"].dtype).expand_as(p["loc"])
    q = q.clamp(QUANTILE_EPS, 1.0 - QUANTILE_EPS)
    return forward(math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0), p)


def mean(p: Params) -> torch.Tensor:
    """Predictive mean by Gauss-Hermite quadrature over the base draw."""
    nodes = QUAD_NODES.to(device=p["loc"].device, dtype=p["loc"].dtype)
    weights = QUAD_WEIGHTS.to(device=p["loc"].device, dtype=p["loc"].dtype)
    leading, count = p["loc"].shape, nodes.numel()
    z = nodes.view((1,) * len(leading) + (count,)).expand(leading + (count,))
    expanded = {"loc": p["loc"].unsqueeze(-1).expand_as(z), "scale": p["scale"].unsqueeze(-1).expand_as(z)}
    for key in ("amps", "centers", "widths"):
        expanded[key] = p[key].reshape(leading + (1, BASIS)).expand(leading + (count, BASIS))
    x = forward(z, expanded)
    return torch.sum(weights.view((1,) * len(leading) + (count,)) * x, dim=-1)


@torch.no_grad()
def initialize(head: torch.nn.Linear) -> None:
    """Start as a narrow normal around the anchor with evenly spread basis centres."""
    head.weight.zero_()
    head.bias.zero_()
    head.bias[1] = math.log(0.1)
    head.bias[2 + BASIS:2 + 2 * BASIS] = torch.linspace(-2.0, 2.0, BASIS, device=head.bias.device,
                                                       dtype=head.bias.dtype)
    head.bias[2 + 2 * BASIS:2 + 3 * BASIS] = math.log(math.expm1(0.75))

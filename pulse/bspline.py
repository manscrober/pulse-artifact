"""Monotone cubic B-spline output used by the PULSE-Spline model.

The source transform maps a standard Gaussian through sigmoid, a monotone
B-spline and logit, followed by the fitted location and scale.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

BSPLINE_EPS = 1e-6
BSPLINE_MIN_DELTA = 1e-4
BSPLINE_LOG_SCALE_MIN = -5.0
BSPLINE_LOG_SCALE_MAX = 3.0


def _open_uniform_knot_vector(
    n_control_points: int,
    degree: int = 3,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    n_spans = n_control_points - degree
    if n_spans <= 0:
        raise ValueError("n_control_points must be greater than degree")
    if n_spans == 1:
        interior = torch.empty(0, device=device, dtype=dtype)
    else:
        interior = torch.arange(1, n_spans, device=device, dtype=dtype) / float(n_spans)
    return torch.cat(
        [
            torch.zeros(degree + 1, device=device, dtype=dtype),
            interior,
            torch.ones(degree + 1, device=device, dtype=dtype),
        ]
    )


def _greville_abscissae(
    knots: torch.Tensor, n_control_points: int, degree: int = 3
) -> torch.Tensor:
    return torch.stack(
        [knots[i + 1 : i + degree + 1].mean() for i in range(n_control_points)], dim=0
    )


def _de_boor_curve(
    inputs: torch.Tensor,
    control_points: torch.Tensor,
    knots: torch.Tensor,
    *,
    degree: int,
) -> torch.Tensor:
    flat_inputs = inputs.reshape(-1)
    flat_ctrl = control_points.reshape(-1, control_points.shape[-1])
    flat_inputs = flat_inputs.clamp(0.0, 1.0 - BSPLINE_EPS)

    inner = knots[degree + 1 : -degree - 1]
    spans = torch.searchsorted(inner, flat_inputs, right=False) + degree

    offsets = torch.arange(degree + 1, device=flat_inputs.device)
    local_idx = spans.unsqueeze(-1) - degree + offsets.unsqueeze(0)
    local_ctrl = flat_ctrl.gather(1, local_idx)

    cols = [local_ctrl[:, j] for j in range(degree + 1)]
    for r in range(1, degree + 1):
        new_cols = list(cols)
        for j in range(degree, r - 1, -1):
            left = knots[spans - degree + j]
            right = knots[spans + 1 + j - r]
            denom = right - left
            alpha = torch.where(
                denom.abs() > 1e-12,
                (flat_inputs - left) / denom,
                torch.zeros_like(flat_inputs),
            )
            new_cols[j] = (1.0 - alpha) * cols[j - 1] + alpha * cols[j]
        cols = new_cols
    return cols[degree].reshape(inputs.shape)


def _cubic_bspline_unit_transform(
    inputs: torch.Tensor,
    control_points: torch.Tensor,
    knots: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    outputs = _de_boor_curve(inputs, control_points, knots, degree=3)
    deriv_ctrl = (
        3.0
        * (control_points[..., 1:] - control_points[..., :-1])
        / (
            knots[4 : control_points.shape[-1] + 3]
            - knots[1 : control_points.shape[-1]]
        )
    )
    derivatives = _de_boor_curve(inputs, deriv_ctrl, knots[1:-1], degree=2)
    return outputs, torch.clamp(derivatives, min=1e-8)


class CubicUniformBSplineHead:
    """Conditional monotone cubic B-spline on a standard Gaussian base."""

    def __init__(
        self,
        n_control_points: int = 24,
        newton_iters: int = 10,
        *,
        residual: bool = False,
        mode_grid_size: int = 257,
    ):
        if n_control_points < 6:
            raise ValueError("n_control_points must be at least 6 for cubic B-splines")
        self.n_control_points = int(n_control_points)
        self.newton_iters = int(newton_iters)
        self.residual = bool(residual)
        self.mode_grid_size = int(mode_grid_size)
        self.num_params = 2 + (self.n_control_points - 1)
        prefix = "resbspline" if self.residual else "bspline"
        self.name = f"{prefix}{self.n_control_points}"
        knots = _open_uniform_knot_vector(
            self.n_control_points, degree=3, dtype=torch.float32
        )
        base_ctrl = _greville_abscissae(knots, self.n_control_points, degree=3)
        self._base_log_deltas = torch.log(torch.diff(base_ctrl))
        self._knot_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def _split_raw(
        self, raw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loc = raw[..., 0]
        log_scale = raw[..., 1]
        delta_logits = raw[..., 2:]
        return loc, log_scale, delta_logits

    def _knots(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        ctrl = params["control_points"]
        key = (ctrl.device, ctrl.dtype)
        knots = self._knot_cache.get(key)
        if knots is None:
            knots = _open_uniform_knot_vector(
                self.n_control_points,
                degree=3,
                device=ctrl.device,
                dtype=ctrl.dtype,
            )
            self._knot_cache[key] = knots
        return knots

    def _unit_forward(
        self,
        u: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _cubic_bspline_unit_transform(
            u, params["control_points"], self._knots(params)
        )

    def _unit_value(
        self,
        u: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return _de_boor_curve(
            u, params["control_points"], self._knots(params), degree=3
        )

    def _inverse_u(
        self, target_v: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        target_v = target_v.clamp(BSPLINE_EPS, 1.0 - BSPLINE_EPS)
        with torch.no_grad():
            lo = torch.full_like(target_v, BSPLINE_EPS)
            hi = torch.full_like(target_v, 1.0 - BSPLINE_EPS)
            for _ in range(self.newton_iters):
                mid = 0.5 * (lo + hi)
                value, _ = self._unit_forward(mid, params)
                hi = torch.where(value > target_v, mid, hi)
                lo = torch.where(value <= target_v, mid, lo)
            u0 = 0.5 * (lo + hi)
        value, deriv = self._unit_forward(u0, params)
        u = u0 - (value - target_v) / torch.clamp(deriv, min=1e-6)
        return u.clamp(BSPLINE_EPS, 1.0 - BSPLINE_EPS)

    def _forward_value_from_base(
        self,
        z: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        u = torch.sigmoid(z).clamp(BSPLINE_EPS, 1.0 - BSPLINE_EPS)
        v = self._unit_value(u, params).clamp(BSPLINE_EPS, 1.0 - BSPLINE_EPS)
        return params["loc"] + params["scale"] * torch.logit(v)

    def _inverse_to_base(
        self,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = params["scale"]
        v_target = torch.sigmoid((x - params["loc"]) / scale).clamp(
            BSPLINE_EPS, 1.0 - BSPLINE_EPS
        )
        u = self._inverse_u(v_target, params)
        v, dv_du = self._unit_forward(u, params)
        v = v.clamp(BSPLINE_EPS, 1.0 - BSPLINE_EPS)
        z = torch.logit(u)
        logabsdet_forward = (
            torch.log(scale)
            + torch.log(torch.clamp(dv_du, min=1e-8))
            + torch.log(u)
            + torch.log1p(-u)
            - torch.log(v)
            - torch.log1p(-v)
        )
        return z, -logabsdet_forward

    def parse(
        self, raw: torch.Tensor, variance_param: str = "softplus"
    ) -> Dict[str, torch.Tensor]:
        del variance_param
        loc, log_scale, delta_logits = self._split_raw(raw)
        base_log_deltas = self._base_log_deltas.to(device=raw.device, dtype=raw.dtype)
        soft_deltas = F.softmax(delta_logits + base_log_deltas, dim=-1)
        min_delta = min(BSPLINE_MIN_DELTA, 0.25 / float(self.n_control_points - 1))
        deltas = (
            min_delta + (1.0 - min_delta * (self.n_control_points - 1)) * soft_deltas
        )
        control_points = torch.cat(
            [torch.zeros_like(loc.unsqueeze(-1)), torch.cumsum(deltas, dim=-1)], dim=-1
        )
        log_scale = torch.clamp(log_scale, BSPLINE_LOG_SCALE_MIN, BSPLINE_LOG_SCALE_MAX)
        scale = torch.exp(log_scale)
        # The location and scale are proxies for moments. Asymmetric learned
        # transforms can move the true mean and median away from the location.
        mean = loc
        log_var = 2.0 * log_scale
        median = loc
        return {
            "loc": loc,
            "log_scale": log_scale,
            "scale": scale,
            "control_points": control_points,
            "mu": mean,
            "log_var": log_var,
            "median": median,
        }

    def nll(
        self, params: Dict[str, torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        z, logabsdet = self._inverse_to_base(target, params)
        log_p_z = -0.5 * (math.log(2.0 * math.pi) + z.pow(2))
        return -(log_p_z + logabsdet)

    def sample(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = params.get("base_noise")
        if z is None:
            z = torch.randn_like(params["mu"])
        else:
            z = z.to(device=params["mu"].device, dtype=params["mu"].dtype)
            z = z.reshape_as(params["mu"])
        return self._forward_value_from_base(z, params)

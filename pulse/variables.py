"""Variables of the sepsis cohort and the rules that tie them together.

Continuous variables are stored in physical units in the order below. Some
are derived from others (running minima, changes from the first hour or the
previous hour, cumulative fluids); `update_derived` recomputes them after a
new hour is written. Death and hospital release are absorbing: once either
indicator turns on it stays on and the trajectory ends.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

CONTINUOUS = (
    "Heart Rate",
    "Diastolic Blood Pressure",
    "Systolic Blood Pressure",
    "Mean Blood Pressure",
    "Minimum Mean Blood Pressure",
    "Minimum Change in Mean Blood Pressure from Baseline",
    "Minimum Mean Blood Pressure from Baseline",
    "Minimum Change in Mean Blood Pressure from Previous",
    "Temperature",
    "SOFA Score",
    "Change in SOFA Score from Baseline",
    "Change in SOFA Score from Previous",
    "Platelet",
    "Hemoglobin",
    "Calcium",
    "BUN",
    "Creatinine",
    "Bicarbonate",
    "Lactate",
    "O2 Requirement Level",
    "Change in O2 from Baseline",
    "Change in O2 from Previous",
    "pO2",
    "sO2",
    "spO2",
    "pCO2",
    "Total CO2",
    "pH",
    "Base excess",
    "Weight",
    "Change in Weight",
    "Respiratory Rate",
    "Fluid Volume",
    "Urine Output",
    "Bolus Volume",
    "Cumulative Fluids",
    "Fluid Balance in ICU",
)

BINARY = (
    "Cumulative Edema",
    "Pulmonary Edema Indicator",
    "Diuretics Indicator",
    "Dialysis Indicator",
    "Mechanical Ventilation Indicator",
    "Vasopressor Indicator",
    "In-Hospital Mortality Indicator",
    "Release Indicator",
)

STATIC_CONTINUOUS = ("Age", "Pre-ICU Fluid Amount", "Elixhauser Score")
STATIC_BINARY = ("Gender", "End Stage Renal Failure", "Congestive Heart Failure")
STATIC = STATIC_CONTINUOUS + STATIC_BINARY

# Plausible physical range of each continuous variable. Simulated values are
# clipped to it, and a variable whose lower bound is positive treats an exact
# zero as a missing value.
RANGES = {
    "Heart Rate": (20, 250),
    "Diastolic Blood Pressure": (20, 140),
    "Systolic Blood Pressure": (50, 250),
    "Mean Blood Pressure": (30, 180),
    "Minimum Mean Blood Pressure": (40, 150),
    "Minimum Change in Mean Blood Pressure from Baseline": (-80, 0),
    "Minimum Mean Blood Pressure from Baseline": (-80, 0),
    "Minimum Change in Mean Blood Pressure from Previous": (-80, 0),
    "Temperature": (32, 42),
    "SOFA Score": (0, 24),
    "Change in SOFA Score from Baseline": (-24, 24),
    "Change in SOFA Score from Previous": (-24, 24),
    "Platelet": (10, 1000),
    "Hemoglobin": (2, 20),
    "Calcium": (3, 15),
    "BUN": (1, 200),
    "Creatinine": (0.1, 20),
    "Bicarbonate": (5, 50),
    "Lactate": (0.1, 20),
    "O2 Requirement Level": (0, 3),
    "Change in O2 from Baseline": (-3, 3),
    "Change in O2 from Previous": (-3, 3),
    "pO2": (15, 600),
    "sO2": (20, 100),
    "spO2": (50, 100),
    "pCO2": (10, 150),
    "Total CO2": (4, 60),
    "pH": (6.8, 7.8),
    "Base excess": (-30, 30),
    "Weight": (30, 200),
    "Change in Weight": (-20, 20),
    "Respiratory Rate": (4, 60),
    "Fluid Volume": (0, 5000),
    "Urine Output": (0, 2000),
    "Bolus Volume": (0, 2000),
    "Cumulative Fluids": (-10000, 50000),
    "Fluid Balance in ICU": (-2000, 5000),
}

# Ordinal variables with their number of levels; they are modelled on the
# continuous scale.
ORDINAL = {"SOFA Score": 25, "O2 Requirement Level": 4}

ABSORBING = ("Pulmonary Edema Indicator", "In-Hospital Mortality Indicator", "Release Indicator")

ZERO_INVALID = tuple(i for i, name in enumerate(CONTINUOUS) if RANGES[name][0] > 0)

DEATH = BINARY.index("In-Hospital Mortality Indicator")
RELEASE = BINARY.index("Release Indicator")

_C = {name: i for i, name in enumerate(CONTINUOUS)}


def bounds() -> tuple[torch.Tensor, torch.Tensor]:
    """Lower and upper range bounds as float32 tensors of shape [V]."""
    low = torch.tensor([float(RANGES[name][0]) for name in CONTINUOUS], dtype=torch.float32)
    high = torch.tensor([float(RANGES[name][1]) for name in CONTINUOUS], dtype=torch.float32)
    return low, high


def update_derived(cont: torch.Tensor, step: int, baseline: torch.Tensor | None = None) -> torch.Tensor:
    """Recompute derived variables of hour `step` in place.

    `cont` is [N, T, V] in physical units and `baseline` is [N, V], the first
    hour. The two minimum-change variables have no recorded source and keep
    whatever value the model wrote; SOFA and the oxygen requirement level are
    modelled directly.
    """
    cf, fb = _C["Cumulative Fluids"], _C["Fluid Balance in ICU"]
    if step == 0:
        cont[:, 0, cf] = cont[:, 0, fb]
    else:
        cont[:, step, cf] = cont[:, step - 1, cf] + cont[:, step, fb]

    mbp, min_mbp, min_mbp_base = _C["Mean Blood Pressure"], _C["Minimum Mean Blood Pressure"], _C["Minimum Mean Blood Pressure from Baseline"]
    if step == 0:
        cont[:, 0, min_mbp] = cont[:, 0, mbp]
    else:
        cont[:, step, min_mbp] = torch.min(cont[:, step - 1, min_mbp], cont[:, step, mbp])
    if baseline is not None:
        if step == 0:
            cont[:, 0, min_mbp_base] = cont[:, 0, mbp] - baseline[:, mbp]
        else:
            cont[:, step, min_mbp_base] = torch.min(cont[:, step - 1, min_mbp_base], cont[:, step, mbp] - baseline[:, mbp])

    for source, from_baseline, from_previous in (
        ("SOFA Score", "Change in SOFA Score from Baseline", "Change in SOFA Score from Previous"),
        ("O2 Requirement Level", "Change in O2 from Baseline", "Change in O2 from Previous"),
        ("Weight", None, "Change in Weight"),
    ):
        src = _C[source]
        if from_baseline is not None and baseline is not None:
            cont[:, step, _C[from_baseline]] = cont[:, step, src] - baseline[:, src]
        if step > 0:
            cont[:, step, _C[from_previous]] = cont[:, step, src] - cont[:, step - 1, src]
        else:
            cont[:, 0, _C[from_previous]] = 0.0
    return cont


def forward_fill(cont: torch.Tensor, keep: torch.Tensor, fill: torch.Tensor) -> torch.Tensor:
    """Carry the last kept value of [N, T, V] forward; before the first one use `fill` [V]."""
    filled = cont.clone()
    last = fill.view(1, -1).expand(filled.shape[0], -1).clone()
    for t in range(filled.shape[1]):
        kept = keep[:, t, :]
        filled[:, t, :] = torch.where(kept, filled[:, t, :], last)
        last = torch.where(kept, filled[:, t, :], last)
    return filled


def zero_invalid_mask(cont: torch.Tensor, mask: torch.Tensor, event: torch.Tensor | None = None) -> torch.Tensor:
    """Clear the mask [N, T, V] where a zero-invalid variable is zero before an event."""
    if event is None:
        event = torch.zeros(mask.shape[:2], dtype=torch.bool, device=mask.device)
    updated = mask.clone()
    for i in ZERO_INVALID:
        valid = (cont[:, :, i] != 0) | event
        updated[:, :, i] = updated[:, :, i] * valid.to(updated.dtype)
    return updated


def forward_fill_zero_invalid(current: torch.Tensor, previous: torch.Tensor, event: torch.Tensor | None = None) -> torch.Tensor:
    """Replace zeros of zero-invalid variables in hour [N, V] by the previous hour, in place.

    Rows where death or release happens now (`event` [N]) keep their zeros.
    """
    if event is None:
        event = torch.zeros(current.shape[0], dtype=torch.bool, device=current.device)
    event = event.to(current.device)
    for i in ZERO_INVALID:
        invalid = (current[:, i] == 0) & ~event
        if invalid.any():
            current[invalid, i] = previous[invalid, i]
    return current


def event_times(binary: torch.Tensor) -> tuple[list[int], list[int], torch.Tensor]:
    """First hour of death and of release per row of [N, T, K], and the mask [N, T] from then on.

    A row without an event gets T. When both indicators occur the earlier
    one wins, and death wins a tie.
    """
    n, steps, _ = binary.shape
    event = torch.zeros(n, steps, dtype=torch.bool, device=binary.device)
    death, release = [steps] * n, [steps] * n
    for b in range(n):
        for index, times in ((DEATH, death), (RELEASE, release)):
            hits = (binary[b, :, index] > 0.5).nonzero(as_tuple=True)[0]
            if hits.numel() > 0:
                times[b] = int(hits[0].item())
                event[b, times[b]:] = True
        if release[b] < death[b]:
            death[b] = steps
        elif death[b] < steps:
            release[b] = steps
    return death, release, event


def fill_after_events(cont: torch.Tensor, death: Sequence[int] | None = None, release: Sequence[int] | None = None,
                      means: torch.Tensor | None = None) -> torch.Tensor:
    """Fill [N, T, V] from the hour of death or release onwards.

    After death a variable takes its lower range bound when that is negative
    and zero otherwise; after release it takes the training mean `means` [V].
    `death` and `release` hold one hour per row, or a single hour that
    applies to every row; an hour of T or later means no event.
    """
    filled = cont.clone()
    n, steps, _ = filled.shape
    device = filled.device
    hours = torch.arange(steps, device=device).view(1, steps)

    def as_tensor(times):
        t = torch.as_tensor(times, device=device).reshape(-1)
        if t.numel() == 1:
            return t.expand(n)
        if t.numel() != n:
            raise ValueError(f"expected one event hour or one per row ({n}), got {t.numel()}")
        return t

    if death is not None:
        values = torch.tensor([RANGES[name][0] if RANGES[name][0] < 0 else 0.0 for name in CONTINUOUS],
                              device=device, dtype=filled.dtype)
        after = (hours >= as_tensor(death).view(-1, 1)).unsqueeze(-1)
        filled = torch.where(after, values.view(1, 1, -1), filled)
    if release is not None and means is not None:
        means = means.to(device) if torch.is_tensor(means) else torch.tensor(means, device=device, dtype=filled.dtype)
        after = (hours >= as_tensor(release).view(-1, 1)).unsqueeze(-1)
        filled = torch.where(after, means.view(1, 1, -1), filled)
    return filled

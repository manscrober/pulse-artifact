"""Seeded synthetic patients for comparisons with the reference implementation.

The values are random draws inside each variable's range with some exact
zeros, missing entries and death or release events; they carry no clinical
meaning and contain no patient data.
"""
from __future__ import annotations

import torch

from pulse import variables as V


def trajectories(seed: int = 0, n: int = 8, steps: int = 24) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    low, high = V.bounds()
    cont = low + (high - low) * torch.rand(n, steps, len(V.CONTINUOUS), generator=g)
    zeros = torch.rand(n, steps, len(V.CONTINUOUS), generator=g) < 0.1
    cont = torch.where(zeros, torch.zeros_like(cont), cont)
    keep = torch.rand(n, steps, len(V.CONTINUOUS), generator=g) < 0.6
    binary = (torch.rand(n, steps, len(V.BINARY), generator=g) < 0.3).float()
    binary[:, :, [V.DEATH, V.RELEASE]] = 0.0
    # Rows cycle through: no event, death, release, release before death,
    # death before release, and a tie.
    for b in range(n):
        death, release = [(None, None), (9, None), (None, 5), (15, 7), (6, 12), (10, 10)][b % 6]
        if death is not None:
            binary[b, death:, V.DEATH] = 1.0
        if release is not None:
            binary[b, release:, V.RELEASE] = 1.0
    means = low + (high - low) * torch.rand(len(V.CONTINUOUS), generator=g)
    event_now = torch.rand(n, generator=g) < 0.3
    return {"continuous": cont, "keep": keep, "binary": binary, "means": means, "event_now": event_now}

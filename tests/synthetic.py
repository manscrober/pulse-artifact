"""Seeded synthetic patients for comparisons with the reference implementation.

The values are random draws inside each variable's range with some exact
zeros, missing entries and death or release events; they carry no clinical
meaning and contain no patient data.
"""
from __future__ import annotations

import torch

from pulse import variables as V


def cache(seed: int = 0, subjects: int = 9000, stays: int = 9600, steps: int = 24) -> dict:
    """A cache with the published split sizes: more subjects than the split uses, some with several stays.

    Continuous values follow four shared random walks so that the covariance
    groups are not ties. A few recorded entries are NaN.
    """
    g = torch.Generator().manual_seed(seed)
    v = len(V.CONTINUOUS)
    low, high = V.bounds()
    walks = torch.randn(stays, steps, 4, generator=g).cumsum(1) / 3
    loadings = torch.randn(4, v, generator=g) * (torch.rand(4, v, generator=g) < 0.5)
    level = walks @ loadings + 0.5 * torch.randn(stays, steps, v, generator=g)
    continuous = low + (high - low) * torch.sigmoid(level)
    mask = (torch.rand(stays, steps, v, generator=g) < 0.5).float()
    continuous[torch.rand(stays, steps, v, generator=g) < 0.001] = torch.nan
    owners = torch.cat([torch.arange(subjects), torch.randint(subjects, (stays - subjects,), generator=g)])
    owners = owners[torch.randperm(stays, generator=g)]
    subject_ids = 1000 + 7 * torch.randperm(subjects, generator=g)
    static = torch.cat([60 + 15 * torch.randn(stays, 3, generator=g),
                        (torch.rand(stays, 3, generator=g) < 0.3).float()], 1)
    return {"continuous": continuous, "continuous_mask": mask,
            "binary": (torch.rand(stays, steps, len(V.BINARY), generator=g) < 0.2).float(),
            "time_hours": torch.arange(steps).float() + torch.rand(stays, steps, generator=g),
            "static": static, "subject_ids": subject_ids[owners],
            "stay_ids": torch.arange(stays) + 50000, "hadm_ids": torch.arange(stays) + 90000,
            "continuous_names": list(V.CONTINUOUS), "binary_names": list(V.BINARY), "static_names": list(V.STATIC)}


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

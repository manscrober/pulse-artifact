"""Seeded synthetic patients for comparisons with the reference implementation.

The values are random draws inside each variable's range with some exact
zeros, missing entries and death or release events; they carry no clinical
meaning and contain no patient data.
"""
from __future__ import annotations

import torch

from pulse import data
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


# Death and release hours of the first scored test patients: none, death
# before both origins, death between them, release, release before death,
# a tie, none, late death.
SCORED_EVENTS = ((None, None), (0, None), (3, None), (None, 10), (15, 12), (9, 9), (None, None), (20, None))


def scored_cache(seed: int = 5) -> dict:
    """`cache(seed)` with the events above and recorded zeros in the first test patients."""
    c = cache(seed)
    rows = data.split(c["subject_ids"])["test"][:len(SCORED_EVENTS)]
    for row, (death, release) in zip(rows, SCORED_EVENTS):
        c["binary"][row, :, [V.DEATH, V.RELEASE]] = 0.0
        if death is not None:
            c["binary"][row, death:, V.DEATH] = 1.0
        if release is not None:
            c["binary"][row, release:, V.RELEASE] = 1.0
    zero_invalid = list(V.ZERO_INVALID[:4])
    c["continuous"][rows[0], 6:9, zero_invalid] = 0.0
    c["continuous_mask"][rows[0], 6:9, zero_invalid] = 1.0
    return c


def paths(seed: int, count: int, steps: int = 24) -> dict[str, torch.Tensor]:
    """Simulated paths [P, T, V] and [P, T, K]; some paths die or are released."""
    g = torch.Generator().manual_seed(seed)
    low, high = V.bounds()
    continuous = low + (high - low) * torch.rand(count, steps, len(V.CONTINUOUS), generator=g)
    binary = (torch.rand(count, steps, len(V.BINARY), generator=g) < 0.2).float()
    binary[:, :, [V.DEATH, V.RELEASE]] = 0.0
    hours = torch.randint(1, 2 * steps, (count, 2), generator=g)
    for p in range(count):
        death, release = hours[p].tolist()
        if death < steps:
            binary[p, death:, V.DEATH] = 1.0
        if release < steps and p % 3 == 0:
            binary[p, release:, V.RELEASE] = 1.0
    return {"continuous": continuous, "binary": binary}


# Scored arms with the number of leading paths each contributes.
SCORED_ARMS = {"first": None, "second": None, "third": 3}


def scored_rollouts(count: int = 5) -> dict[tuple[str, int, int], dict[str, torch.Tensor]]:
    """Paths by (arm, origin, case). Arm "second" has a NaN in case 3 and no case 6."""
    result = {}
    for a, arm in enumerate(SCORED_ARMS):
        for origin in (2, 6):
            for case in range(len(SCORED_EVENTS)):
                if (arm, case) == ("second", 6):
                    continue
                result[arm, origin, case] = paths(100 * a + 10 * origin + case, count)
                if (arm, case) == ("second", 3):
                    result[arm, origin, case]["continuous"][1, 10, 3] = torch.nan
    return result


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

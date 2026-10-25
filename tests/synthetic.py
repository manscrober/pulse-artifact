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
    rows = data.select(c["subject_ids"])["test"][:len(SCORED_EVENTS)]
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


def source_stays(seed: int = 0, count: int = 12) -> list[tuple[dict, dict]]:
    """Source records of ICU stays in the shape `pulse.hourly.bin_stay` reads, with diagnoses.

    Records fall before, during and after the first 24 hours; values include
    out-of-range ones, Fahrenheit temperatures, litres, zero rates, missing
    end times and boluses. Stays die, are released, both or neither.
    """
    import numpy as np
    import pandas as pd

    from pulse import hourly

    rng = np.random.default_rng(seed)
    low, high = (b.numpy().astype(float) for b in V.bounds())
    stays = []
    for s in range(count):
        intime = pd.Timestamp("2150-01-01") + pd.Timedelta(minutes=int(rng.integers(0, 10**6)))

        def at(hours):
            return intime + pd.to_timedelta(np.round(np.asarray(hours) * 3600), unit="s")

        rows = []
        for name in hourly.MEASURED:
            i = V.CONTINUOUS.index(name)
            n = int(rng.integers(0, 30))
            span = high[i] - low[i]
            values = rng.uniform(low[i] - 0.1 * span, high[i] + 0.1 * span, n)
            items = rng.choice([hourly.FAHRENHEIT_ITEM, 223762] if name == "Temperature" else [1000 + i, 2000 + i], n)
            if name == "Temperature":
                values = np.where(items == hourly.FAHRENHEIT_ITEM, values * 9 / 5 + 32, values)
            rows.append(pd.DataFrame({"variable": name, "itemid": items, "charttime": at(rng.uniform(-6, 30, n)),
                                      "valuenum": values}))
        measurements = pd.concat(rows, ignore_index=True)

        n = 40
        start = rng.uniform(-8, 28, n)
        minutes = np.where(rng.random(n) < 0.3, rng.uniform(0, 4.9, n), rng.uniform(5, 600, n))
        labels = rng.choice(["Norepinephrine", "Phenylephrine (50/250)", "Furosemide (Lasix)", "NaCl 0.9%",
                             "Dextrose 5%", "Vasopressin"], n)
        rateuom = rng.choice(["mL/hour", "mL/min", "mcg/kg/min", None], n)
        inputs = pd.DataFrame({
            "stay_id": s, "starttime": at(start), "endtime": at(start + minutes / 60),
            "amount": np.where(rng.random(n) < 0.1, np.nan, rng.uniform(1, 1500, n)),
            "amountuom": rng.choice(["mL", "ml", "L", "mg"], n, p=[0.6, 0.2, 0.1, 0.1]),
            "rate": np.where(rng.random(n) < 0.15, 0.0, rng.uniform(0.5, 200, n)),
            "rateuom": rateuom, "label": labels})
        inputs.loc[rng.random(n) < 0.1, "endtime"] = pd.NaT
        inputs.loc[rng.random(n) < 0.15, "rate"] = np.nan

        n = 30
        outputs = pd.DataFrame({"charttime": at(rng.uniform(-3, 27, n)), "value": rng.uniform(0, 900, n),
                                "valueuom": rng.choice(["mL", "L", "cm"], n, p=[0.8, 0.1, 0.1])})
        n = 6
        start = rng.uniform(-30, 26, n)
        procedures = pd.DataFrame({"starttime": at(start), "endtime": at(start + rng.uniform(-1, 40, n)),
                                   "itemid": rng.choice(list(hourly.VENTILATION_ITEMS + hourly.DIALYSIS_ITEMS) + [225999], n)})
        diagnoses = pd.DataFrame({"hadm_id": 7000 + s, "icd_code": ["I509", "J810", "N186"], "icd_version": 10,
                                  "long_title": ["heart failure", "pulmonary edema", "renal failure"]})
        death, release = [(None, 30), (10.5, None), (None, 7.25), (15.0, 5.0), (40.0, 18.0), (3.0, 3.5)][s % 6]
        stay = {"stay_id": s, "subject_id": 100 + s, "hadm_id": 7000 + s, "intime": intime,
                "outtime": intime + pd.Timedelta(hours=release if release is not None else 50 + s),
                "dod": None if death is None else intime + pd.Timedelta(hours=death),
                "anchor_age": int(rng.integers(18, 90)), "gender": "M" if s % 2 else "F"}
        frames = {"measurements": measurements, "inputevents": inputs, "outputevents": outputs,
                  "procedureevents": procedures}
        stays.append((frames, dict(stay, diagnoses=diagnoses)))
    return stays


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


def cnf_inputs(seed: int = 0, rows: int = 5, variables: int = 36) -> dict[str, torch.Tensor]:
    """Raw head outputs [R, V, 26], anchors, targets and base draws [R, V]."""
    g = torch.Generator().manual_seed(seed)
    return {"raw": 1.5 * torch.randn(rows, variables, 26, generator=g),
            "anchor": torch.randn(rows, variables, generator=g),
            "target": 2.0 * torch.randn(rows, variables, generator=g),
            "z": torch.randn(rows, variables, generator=g)}


def protocol():
    """A protocol with six interleaved covariance groups and unit-free statistics."""
    from pulse.data import Protocol

    names = [n for n in V.CONTINUOUS if n not in V.TREATMENTS]
    groups = {f"cov_group_{g + 1:02d}": names[g::6] for g in range(6)}
    groups["continuous_treatments"] = [V.CONTINUOUS[V.BOLUS]]
    low, high = V.bounds()
    return Protocol(continuous_names=list(V.CONTINUOUS), binary_names=list(V.BINARY),
                    static_names=list(V.STATIC), mean=((low + high) / 2).tolist(),
                    sd=((high - low) / 6).tolist(), static_mean=[0.0] * 6, static_sd=[1.0] * 6,
                    groups=groups, cache_sha256="0" * 64, split={"train": 0, "val": 0, "test": 0})


def part(seed: int = 0, n: int = 4, steps: int = 24) -> dict[str, torch.Tensor]:
    """One split in the prepared format: irregular hours, forward-filled inputs and masks."""
    g = torch.Generator().manual_seed(seed)
    data = trajectories(seed, n, steps)
    mask = data["keep"].float()
    mask[:, 0] = 1.0
    target = torch.where(mask > 0, data["continuous"], data["means"])
    filled = target.clone()
    for t in range(1, steps):
        filled[:, t] = torch.where(mask[:, t] > 0, filled[:, t], filled[:, t - 1])
    gaps = 0.25 + 1.5 * torch.rand(n, steps, generator=g)
    hours = torch.cumsum(gaps, 1) - gaps[:, :1]
    static = torch.randn(n, len(V.STATIC), generator=g)
    static[:, 3:] = (static[:, 3:] > 0).float()
    return {"continuous": filled, "continuous_target": target, "continuous_mask": mask,
            "binary": data["binary"], "time_hours": hours, "static": static}


def perturb(module: torch.nn.Module, seed: int = 1, scale: float = 0.1, rename=lambda name: name) -> None:
    """Add seeded noise to every parameter so zero-initialised layers act.

    Noise is drawn in the order of the artifact's parameter names; `rename`
    maps another module's names to those, or to None for parameters it lacks.
    """
    g = torch.Generator().manual_seed(seed)
    named = {rename(name): p for name, p in module.named_parameters()}
    with torch.no_grad():
        for name in sorted(k for k in named if k is not None):
            named[name].add_(scale * torch.randn(named[name].shape, generator=g))

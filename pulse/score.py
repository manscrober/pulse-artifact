"""Aligned probabilistic scores of saved rollouts with a paired whole-patient bootstrap.

Every arm is scored on the same target points: recorded test values from the
forecast origin onwards, forward-filled in physical units and filled after
death or release by the evaluation rule, divided by the training SD of each
variable. Paths are filled the same way. Per variable the scores are the
empirical CRPS, the weighted interval score over 499 central intervals and
the central-90% coverage, both from inverse-CDF quantiles of the paths; the
vector RMSE compares the path mean with the target over a panel of
variables at each recorded hour. A panel score is the ratio of summed scores
to recorded points within each variable, averaged over variables with
points. Confidence intervals resample whole test patients, with the same
resamples for every arm, so differences between arms are paired.

Two rules decide the patients. A test patient whose first recorded death is
at least one hour before the origin's first forecast hour is skipped, and so
is a patient for whom any arm has no case file; skipped positions leave gaps
in the case numbering. A target point counts only if every arm's paths are
finite there, so an arm's score depends on the arms scored with it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pulse import data
from pulse import variables as V
from pulse.data import Protocol

# Per-variable scores, named as in the scored rows: the WIS over 499 central
# intervals and the central-90% coverage from inverse-CDF quantiles.
INTERVALS = 499
FIELDS = ("crps", "wis499", "coverage90_inverse")
METRICS = ("vector_rmse",) + FIELDS
PANELS = {35: [i for i, name in enumerate(V.CONTINUOUS) if name not in V.ORDINAL],
          37: list(range(len(V.CONTINUOUS)))}


def _quantiles(sorted_samples: np.ndarray, probabilities) -> np.ndarray:
    """Inverse of the empirical CDF along the first axis."""
    q = np.asarray(probabilities, float)
    return sorted_samples[np.ceil(q * len(sorted_samples)).astype(int) - 1]


def empirical_scores(samples: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    """CRPS, WIS and central-90% coverage of the empirical law of `samples` [M, ...] at `truth` [...]."""
    x = np.sort(np.asarray(samples, float), axis=0)
    y = np.asarray(truth, float)
    m = len(x)
    coefficient = (2 * np.arange(1, m + 1) - m - 1).reshape((m,) + (1,) * y.ndim)
    crps = np.abs(x - y).mean(axis=0) - (coefficient * x).sum(axis=0) / m ** 2
    alphas = np.arange(1, INTERVALS + 1) / (INTERVALS + 1)
    lo, hi = _quantiles(x, alphas / 2), _quantiles(x, 1 - alphas / 2)
    median = _quantiles(x, [.5])[0]
    weight = alphas.reshape((-1,) + (1,) * y.ndim) / 2
    denominator = len(alphas) + .5
    # Terms are summed in the order of the published scorer.
    wis = (.5 * np.abs(median - y) / denominator + np.sum(weight * (hi - lo), axis=0) / denominator
           + np.maximum(lo - y, 0).sum(axis=0) / denominator + np.maximum(y - hi, 0).sum(axis=0) / denominator)
    level = .9
    low, high = _quantiles(x, [(1 - level) / 2, (1 + level) / 2])
    return {"crps": crps, "wis499": wis, "coverage90_inverse": ((low <= y) & (y <= high)).astype(float)}


def _first(column: torch.Tensor) -> int | None:
    hours = torch.where(column > .5)[0]
    return int(hours[0]) if len(hours) else None


def fill_truth(continuous: torch.Tensor, mask: torch.Tensor, binary: torch.Tensor,
               mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Target values [T, V] and the mask of recorded points of one patient.

    Zeros of variables that cannot be zero are unrecorded. Values carry
    forward from the training mean, and the earlier of death and release
    (death on a tie) fills the hours from then on.
    """
    mask = mask.bool() & torch.isfinite(continuous)
    for i in V.ZERO_INVALID:
        mask[:, i] &= continuous[:, i] != 0
    filled = V.forward_fill(continuous[None], mask[None], mean)
    death, release, _ = V.event_times(binary[None])
    return V.fill_after_events(filled, death, release, mean)[0], mask


def fill_paths(continuous: torch.Tensor, binary: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """Fill paths [P, T, V] after death, or after release on paths without death."""
    steps = continuous.shape[1]
    death = [_first(b[:, V.DEATH]) for b in binary]
    release = [_first(b[:, V.RELEASE]) if d is None else None for b, d in zip(binary, death)]
    return V.fill_after_events(continuous, [steps if d is None else d for d in death],
                               [steps if r is None else r for r in release], mean)


def collect(arms: dict[str, tuple[Path, int | None]], cache: dict, test_rows: np.ndarray, mean: torch.Tensor,
            origin: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[int]]:
    """Targets [n, H, V], masks and filled paths [n, P, H, V] from the origin's first forecast hour on."""
    first = data.first_forecast_row(origin)
    truth, masks, cases = [], [], []
    samples = {label: [] for label in arms}
    for case, row in enumerate(test_rows):
        death = _first(cache["binary"][row][:, V.DEATH])
        if death is not None and death <= data.last_recorded_row(origin):
            continue
        directories = {label: Path(directory) / f"k{origin}" for label, (directory, _) in arms.items()}
        if not all(data.case_path(d, case).exists() for d in directories.values()):
            continue
        y, m = fill_truth(cache["continuous"][row].float(), cache["continuous_mask"][row], cache["binary"][row], mean)
        for label, (_, max_paths) in arms.items():
            paths = data.load_case(directories[label], case)
            filled = fill_paths(paths["continuous"][:max_paths], paths["binary"][:max_paths], mean)
            samples[label].append(filled[:, first:].numpy())
        truth.append(y[first:].numpy())
        masks.append(m[first:].numpy())
        cases.append(case)
    return np.stack(truth), np.stack(masks), {label: np.stack(x) for label, x in samples.items()}, cases


def patient_statistics(y: np.ndarray, mask: np.ndarray, predictions: dict[str, np.ndarray],
                       sd: np.ndarray) -> dict[str, np.ndarray]:
    """Per patient: recorded points and score sums per variable, and vector RMSE per panel."""
    n, _, variables = y.shape
    common = mask & np.isfinite(y)
    for x in predictions.values():
        common &= np.isfinite(x).all(axis=1)
    stats = {"counts": common.sum(axis=1).astype(np.int64)}
    owners = np.broadcast_to(np.arange(n)[:, None], y.shape[:2])
    for arm, x in predictions.items():
        sums = np.zeros((n, variables, len(FIELDS)))
        for v in range(variables):
            valid = common[:, :, v]
            if not valid.any():
                continue
            draws = x[:, :, :, v].transpose(1, 0, 2)[:, valid] / sd[v]
            target = y[:, :, v][valid] / sd[v]
            scores = empirical_scores(draws, target)
            for f, source in enumerate(FIELDS):
                sums[:, v, f] = np.bincount(owners[valid], weights=scores[source], minlength=n)
        stats[f"{arm}_sums"] = sums
        for size, panel in PANELS.items():
            valid = common[:, :, panel]
            errors = (x[:, :, :, panel].mean(1) - y[:, :, panel]) / sd[panel]
            squared = np.where(valid, errors ** 2, 0).sum(-1)
            stats[f"{arm}_vector_{size}"] = np.sqrt(squared.sum(1) / valid.any(-1).sum(1).clip(1))
    return stats


def estimates(stats: dict[str, np.ndarray], weights: np.ndarray, size: int, arm: str) -> np.ndarray:
    """Panel scores [resamples + 1, 4] for patient weights [resamples + 1, n]."""
    panel = PANELS[size]
    denominators = weights @ stats["counts"][:, panel]
    sums = stats[f"{arm}_sums"][:, panel]
    totals = (weights @ sums.reshape(len(sums), -1)).reshape(len(weights), len(panel), len(FIELDS))
    ratios = np.divide(totals, denominators[:, :, None], out=np.full_like(totals, np.nan),
                       where=denominators[:, :, None] > 0)
    supported = (denominators > 0).sum(1)
    scores = np.nansum(ratios, axis=1) / supported[:, None]
    vector = weights @ stats[f"{arm}_vector_{size}"] / weights.sum(1)
    return np.column_stack((vector, scores))


def score(arms: dict[str, tuple[Path, int | None]], cache: dict, protocol: Protocol, origins=data.ORIGINS,
          resamples: int = 2000, seed: int = 20261006) -> dict:
    """Score arms {label: (directory, first M paths or None)}; the first arm is the reference.

    Each directory holds `k{origin}/case_XXXX.pt` for every origin. Rows give
    the point estimate and the 2.5% and 97.5% bootstrap quantiles of each
    metric, and for every other arm the same for its difference from the
    reference. Next to the rows, the result records under `arms` each arm's
    directory and path limit, and the number of resamples and the seed, so
    that a saved table names the rollouts it was computed from.
    """
    if protocol.cache_sha256 != cache["sha256"]:
        raise ValueError("the cache differs from the one the protocol was prepared from")
    reference = next(iter(arms))
    mean = torch.tensor(protocol.mean)
    sd = np.array(protocol.sd)
    test_rows = data.select(cache["subject_ids"])["test"]
    rows = []
    for k in origins:
        y, mask, predictions, cases = collect(arms, cache, test_rows, mean, k)
        stats = patient_statistics(y, mask, predictions, sd)
        n = len(cases)
        rng = np.random.default_rng(seed + k)
        draws = np.stack([np.bincount(r, minlength=n) for r in rng.integers(n, size=(resamples, n))])
        weights = np.concatenate((np.ones((1, n), dtype=np.int64), draws))
        for size in PANELS:
            values = {arm: estimates(stats, weights, size, arm) for arm in arms}
            for m, metric in enumerate(METRICS):
                for arm in arms:
                    point, boot = values[arm][0, m], values[arm][1:, m]
                    row = dict(origin=k, panel=size, metric=metric, arm=arm, patients=n,
                               paths=int(predictions[arm].shape[1]), estimate=float(point),
                               ci=[float(q) for q in np.quantile(boot, [.025, .975])])
                    if arm != reference:
                        d = values[arm][:, m] - values[reference][:, m]
                        row["minus_reference"] = float(d[0])
                        row["minus_reference_ci"] = [float(q) for q in np.quantile(d[1:], [.025, .975])]
                    rows.append(row)
    return {"reference": reference, "arms": {label: {"directory": str(d), "max_paths": m} for label, (d, m) in arms.items()},
            "resamples": resamples, "seed": seed, "rows": rows}

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

Two further scores follow the component table of the paper. The
normalised RMSE is the RMSE of the path mean per variable, pooled over
recorded points, averaged over a panel; its primary form leaves out the
derived and accumulated variables of `RMSE_EXCLUDED` (25 of the 37
remain). Optionally, the randomized PIT distance is the Kolmogorov-Smirnov
distance of one randomized probability integral transform per target
point to the uniform law, per variable, averaged over the panel.
`binary_panel` scores the patient-level forecast of each binary outcome.

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
from scipy.sparse import csr_matrix
from sklearn.metrics import average_precision_score, roc_auc_score

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
# Left out of the primary normalised RMSE: accumulated fluids, the blood-pressure minima and changes
# from baseline or the previous hour.
RMSE_EXCLUDED = ("Cumulative Fluids", "Fluid Balance in ICU", "Fluid Volume", "Minimum Mean Blood Pressure",
                 "Minimum Mean Blood Pressure from Baseline", "Minimum Change in Mean Blood Pressure from Baseline",
                 "Minimum Change in Mean Blood Pressure from Previous", "Change in SOFA Score from Baseline",
                 "Change in SOFA Score from Previous", "Change in O2 from Baseline", "Change in O2 from Previous",
                 "Change in Weight")
PIT_SEED = 20261005
BINARY_SEED = 20261019


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
    return {"crps": crps, "wis499": wis, "coverage90_inverse": ((low <= y) & (y <= high)).astype(float),
            "cdf_left": (x < y).mean(axis=0), "cdf_right": (x <= y).mean(axis=0)}


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
                       sd: np.ndarray, origin: int | None = None) -> dict[str, np.ndarray]:
    """Per patient: recorded points, score sums and squared errors of the path mean per variable,
    and vector RMSE per panel.

    With `origin`, also each target point's randomized PIT and its patient,
    per variable; ties are broken with a generator seeded from the origin
    and the variable, the same for every arm.
    """
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
            if origin is not None:
                rng = np.random.default_rng(PIT_SEED + 100 * origin + v)
                stats[f"{arm}_pit_{v}"] = (scores["cdf_left"]
                                           + rng.random(len(target)) * (scores["cdf_right"] - scores["cdf_left"]))
                stats[f"{arm}_owner_{v}"] = owners[valid]
        stats[f"{arm}_sums"] = sums
        error = (x.mean(1) - y) / sd
        stats[f"{arm}_squared"] = np.where(common, error ** 2, 0).sum(1)
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


def normalized_rmse(stats: dict[str, np.ndarray], weights: np.ndarray, size: int, arm: str) -> np.ndarray:
    """Normalised RMSE [resamples + 1, 2] over the panel and over its variables outside `RMSE_EXCLUDED`."""
    panel = PANELS[size]
    counts = weights @ stats["counts"][:, panel]
    rmse = np.sqrt(np.divide(weights @ stats[f"{arm}_squared"][:, panel], counts, out=np.full(counts.shape, np.nan),
                             where=counts > 0))
    primary = np.array([V.CONTINUOUS[v] not in RMSE_EXCLUDED for v in panel])
    return np.column_stack((np.nanmean(rmse, 1), np.nanmean(rmse[:, primary], 1)))


def weighted_pit_ks(values: np.ndarray, owners: np.ndarray, weights: np.ndarray, batch: int = 32) -> np.ndarray:
    """KS distance to the uniform law of PIT values [N] whose patients carry multiplicities [resamples + 1, n].

    Equal values are pooled, and the empirical CDF is compared with the
    diagonal on both sides of each of its jumps.
    """
    result = np.full(len(weights), np.nan)
    if not len(values):
        return result
    unique, group = np.unique(values, return_inverse=True)
    counts = csr_matrix((np.ones(len(values)), (group, owners)), shape=(len(unique), weights.shape[1]))
    for start in range(0, len(weights), batch):
        jumps = np.ascontiguousarray((counts @ weights[start:start + batch].T).T)
        totals = jumps.sum(1)
        after = np.cumsum(jumps, axis=1)
        before = after - jumps
        np.divide(after, totals[:, None], out=after, where=totals[:, None] > 0)
        np.divide(before, totals[:, None], out=before, where=totals[:, None] > 0)
        ks = np.maximum((after - unique).max(1), (unique - before).max(1))
        result[start:start + len(jumps)] = np.where(totals > 0, ks, np.nan)
    return result


def _paired(row: dict, loss: dict[str, np.ndarray], arm: str, previous: str | None, best: str) -> None:
    """Loss differences of `arm` against the previous arm in the given order and against the best arm."""
    for key, other in (("previous", previous), ("best", best)):
        if other is None or other == arm:
            continue
        d = loss[arm] - loss[other]
        row[f"loss_minus_{key}"] = float(d[0])
        row[f"loss_minus_{key}_ci"] = [float(q) for q in np.quantile(d[1:], [.025, .975])]
    row["best"] = best


def score(arms: dict[str, tuple[Path, int | None]], cache: dict, protocol: Protocol, origins=data.ORIGINS,
          resamples: int = 2000, seed: int = 20261006, pit: bool = False) -> dict:
    """Score arms {label: (directory, first M paths or None)}; the first arm is the reference.

    Each directory holds `k{origin}/case_XXXX.pt` for every origin. Rows give
    the point estimate and the 2.5% and 97.5% bootstrap quantiles of each
    metric, and for every other arm the same for its difference from the
    reference. Each row also compares the arm's loss with the arm before it
    in the given order, which reads a component table row by row, and with
    the best arm; coverage counts as its distance from 0.9 and every other
    metric as itself. Next to the rows, the result records under `arms`
    each arm's directory and path limit, and the number of resamples and
    the seed, so that a saved table names the rollouts it was computed from.
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
        stats = patient_statistics(y, mask, predictions, sd, k if pit else None)
        n = len(cases)
        rng = np.random.default_rng(seed + k)
        draws = np.stack([np.bincount(r, minlength=n) for r in rng.integers(n, size=(resamples, n))])
        weights = np.concatenate((np.ones((1, n), dtype=np.int64), draws))
        metrics = METRICS + (("pit_ks",) if pit else ()) + ("nrmse", "nrmse_primary")
        if pit:
            ks = {arm: np.stack([weighted_pit_ks(stats[f"{arm}_pit_{v}"], stats[f"{arm}_owner_{v}"], weights)
                                 if f"{arm}_pit_{v}" in stats else np.full(len(weights), np.nan)
                                 for v in range(len(V.CONTINUOUS))], 1) for arm in arms}
        for size, panel in PANELS.items():
            values = {arm: estimates(stats, weights, size, arm) for arm in arms}
            if pit:
                values = {arm: np.column_stack((v, np.nanmean(ks[arm][:, panel], axis=1))) for arm, v in values.items()}
            values = {arm: np.column_stack((v, normalized_rmse(stats, weights, size, arm))) for arm, v in values.items()}
            for m, metric in enumerate(metrics):
                loss = {arm: np.abs(v[:, m] - .9) if metric.startswith("coverage90") else v[:, m]
                        for arm, v in values.items()}
                best = min(arms, key=lambda arm: loss[arm][0])
                for i, arm in enumerate(arms):
                    point, boot = values[arm][0, m], values[arm][1:, m]
                    row = dict(origin=k, panel=size, metric=metric, arm=arm, patients=n,
                               paths=int(predictions[arm].shape[1]), estimate=float(point),
                               ci=[float(q) for q in np.quantile(boot, [.025, .975])])
                    if arm != reference:
                        d = values[arm][:, m] - values[reference][:, m]
                        row["minus_reference"] = float(d[0])
                        row["minus_reference_ci"] = [float(q) for q in np.quantile(d[1:], [.025, .975])]
                    _paired(row, loss, arm, list(arms)[i - 1] if i else None, best)
                    rows.append(row)
    return {"reference": reference, "arms": {label: {"directory": str(d), "max_paths": m} for label, (d, m) in arms.items()},
            "resamples": resamples, "seed": seed, "rows": rows}


def _binary_scores(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    out = {"event_rate": float(target.mean()), "predicted_rate": float(probability.mean()),
           "brier": float(((probability - target) ** 2).mean())}
    predicted = probability >= .5
    tp = float((predicted & (target == 1)).sum())
    out["f1"] = 2 * tp / max(float(predicted.sum() + target.sum()), 1.)
    if 0 < target.sum() < len(target):
        out["auroc"] = float(roc_auc_score(target, probability))
        out["auprc"] = float(average_precision_score(target, probability))
    return out


def binary_panel(arms: dict[str, Path], cache: dict, protocol: Protocol, origins=data.ORIGINS,
                 resamples: int = 1000, seed: int = BINARY_SEED) -> dict:
    """Patient-level forecasts of each binary channel, with paired whole-patient bootstrap intervals.

    For origin k the target is whether the event is recorded at any hour
    from the first forecast hour on, and the forecast is the share of paths
    in which it occurs. Per channel the result gives AUROC, AUPRC, Brier
    score, F1 at a share of 0.5 and the recorded and forecast event rates.
    The macro averages run over outcome channels with at least one event and
    one non-event among the scored patients; the vasopressor indicator, a
    treatment, is reported but not averaged. Patients are chosen as in
    `score`.
    """
    if protocol.cache_sha256 != cache["sha256"]:
        raise ValueError("the cache differs from the one the protocol was prepared from")
    names = list(V.BINARY)
    outcomes = [j for j in range(len(names)) if j != V.VASOPRESSOR]
    test_rows = data.select(cache["subject_ids"])["test"]
    result = {"arms": {label: str(d) for label, d in arms.items()}, "channels": names, "rows": [],
              "channels_by_arm": {}, "resamples": resamples, "seed": seed}
    for k in origins:
        first = data.first_forecast_row(k)
        targets, probs = [], {a: [] for a in arms}
        for case, row in enumerate(test_rows):
            death = _first(cache["binary"][row][:, V.DEATH])
            if death is not None and death <= data.last_recorded_row(k):
                continue
            directories = {a: Path(d) / f"k{k}" for a, d in arms.items()}
            if not all(data.case_path(d, case).exists() for d in directories.values()):
                continue
            targets.append(cache["binary"][row][first:].float().amax(0).numpy())
            for a, d in directories.items():
                b = data.load_case(d, case)["binary"]
                probs[a].append((b[:, first:] > .5).any(1).float().mean(0).numpy())
        y, p = np.stack(targets), {a: np.stack(v) for a, v in probs.items()}
        n = len(y)
        draws = np.random.default_rng(seed + k).integers(n, size=(resamples, n))
        usable = [j for j in outcomes if 0 < y[:, j].sum() < n]

        def macro(arm, idx):
            return {"auroc": np.mean([roc_auc_score(y[idx, j], p[arm][idx, j]) for j in usable
                                      if 0 < y[idx, j].sum() < len(idx)]),
                    "auprc": np.mean([average_precision_score(y[idx, j], p[arm][idx, j]) for j in usable
                                      if y[idx, j].sum() > 0]),
                    "brier": float(((p[arm][idx][:, usable] - y[idx][:, usable]) ** 2).mean())}
        boot = {a: [macro(a, idx) for idx in draws] for a in arms}
        full = {a: macro(a, np.arange(n)) for a in arms}
        labels = list(arms)
        for i, a in enumerate(labels):
            result["channels_by_arm"].setdefault(str(k), {})[a] = {
                names[j]: _binary_scores(y[:, j], p[a][:, j]) for j in range(len(names))}
            for m in ("auroc", "auprc", "brier"):
                # Losses are lower-is-better: AUROC and AUPRC enter with a negative sign.
                sign = 1. if m == "brier" else -1.
                loss = {b: sign * np.array([full[b][m]] + [x[m] for x in boot[b]]) for b in arms}
                best = min(arms, key=lambda b: loss[b][0])
                values = np.array([x[m] for x in boot[a]])
                row = {"origin": k, "panel": 37, "metric": f"binary_{m}", "arm": a, "patients": n,
                       "estimate": float(full[a][m]), "ci": np.quantile(values, [.025, .975]).tolist(), "best": best,
                       "macro_channels": [names[j] for j in usable]}
                for key, other in (("previous", labels[i - 1] if i else None), ("best", best)):
                    if other is None or other == a:
                        continue
                    diff = loss[a] - loss[other]
                    row[f"loss_minus_{key}"] = float(diff[0])
                    row[f"loss_minus_{key}_ci"] = np.quantile(diff[1:], [.025, .975]).tolist()
                result["rows"].append(row)
    return result

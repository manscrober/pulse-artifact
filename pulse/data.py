"""Cohort cache, data preparation, the training protocol and rollout case files.

The cache holds every ICU stay of the cohort on a 24-hour grid in physical
units. Preparation splits subjects, scales with training statistics only,
fills missing values and groups the continuous variables for the encoders.

A `Part` is one split as a dictionary of tensors with N patients, T hours,
V continuous variables, K binary variables and S static columns:

    continuous         [N, T, V]  inputs, forward-filled from training means
    continuous_target  [N, T, V]  targets, missing entries set to training means
    continuous_mask    [N, T, V]  1 where a value was recorded
    binary             [N, T, K]
    time_hours         [N, T]
    static             [N, S]     standardised with training statistics
"""
from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from pulse import variables as V

Part = dict[str, torch.Tensor]


@dataclass(frozen=True)
class Protocol:
    continuous_names: list[str]
    binary_names: list[str]
    static_names: list[str]
    mean: list[float]
    sd: list[float]
    static_mean: list[float]
    static_sd: list[float]
    groups: dict[str, list[str]]
    cache_sha256: str
    split: dict[str, int]

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> Protocol:
        return cls(**json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class Prepared:
    train: Part
    val: Part
    test: Part
    protocol: Protocol


def case_path(directory: Path, case: int) -> Path:
    return Path(directory) / f"case_{case:04d}.pt"


CASE_FORMAT = "compact-1"


def _pack_paths(paths: torch.Tensor) -> dict:
    """Paths [P, T, ...] without loss: the leading rows that every path shares, which are the recorded
    hours, stored once; a tensor of exact zeros and ones stored as bits; the bytes compressed."""
    values = paths.detach().cpu().contiguous().numpy()
    count, steps = values.shape[:2]
    words = values.view(f"u{values.itemsize}").reshape(count, steps, -1)
    shared = (words == words[:1]).all(axis=(0, 2))
    rows = steps if shared.all() else int(np.argmin(shared))
    flat = np.concatenate((values[0, :rows].ravel(), values[:, rows:].ravel()))
    zero, one = (np.array(x, dtype=values.dtype).view(words.dtype) for x in (0, 1))
    as_words = flat.view(words.dtype)
    bits = bool(((as_words == zero) | (as_words == one)).all())
    raw = np.packbits(as_words == one).tobytes() if bits else flat.tobytes()
    return {"dtype": str(values.dtype), "shape": list(values.shape), "shared_rows": rows, "bits": bits,
            "data": torch.from_numpy(np.frombuffer(zlib.compress(raw, 6), dtype=np.uint8).copy())}


def _unpack_paths(packed: dict) -> torch.Tensor:
    dtype, shape, rows = np.dtype(packed["dtype"]), packed["shape"], packed["shared_rows"]
    count, steps, rest = shape[0], shape[1], shape[2:]
    width = int(np.prod(rest, dtype=np.int64))
    size = rows * width + count * (steps - rows) * width
    raw = zlib.decompress(packed["data"].numpy().tobytes())
    if packed["bits"]:
        flat = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), count=size).astype(dtype)
    else:
        flat = np.frombuffer(raw, dtype=dtype)
    prefix = np.broadcast_to(flat[:rows * width].reshape(1, rows, *rest), (count, rows, *rest))
    return torch.from_numpy(np.concatenate((prefix, flat[rows * width:].reshape(count, steps - rows, *rest)), axis=1))


def save_case(directory: Path, case: int, continuous: torch.Tensor, binary: torch.Tensor) -> None:
    """Write the paths of one test patient: continuous [P, T, V], binary [P, T, K].

    The file holds the same values, bit for bit, in 34-44% of the space of
    a file of the tensors themselves (`CASE_FORMAT`; 420 rollout case files
    of seven arms with 100 paths)."""
    path = case_path(directory, case)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    torch.save({"case": case, "format": CASE_FORMAT, "continuous": _pack_paths(continuous),
                "binary": _pack_paths(binary)}, pending)
    pending.replace(path)


def load_case(directory: Path, case: int) -> dict:
    """The paths of one test patient, from a compact case file or from one that holds the tensors."""
    payload = torch.load(case_path(directory, case), map_location="cpu", weights_only=True)
    if payload["case"] != case:
        raise ValueError(f"case file {case} holds case {payload['case']}")
    if "format" not in payload:
        return payload
    if payload["format"] != CASE_FORMAT:
        raise ValueError(f"case file {case} has the unknown format {payload['format']}")
    return {"case": case, "continuous": _unpack_paths(payload["continuous"]),
            "binary": _unpack_paths(payload["binary"])}


# Forecast origins of the paper. At origin k the first forecast hour is hour
# k, counted from one, so rows 0 to k - 2 are recorded and row k - 1 is the
# first forecast row.
ORIGINS = (2, 6)


def last_recorded_row(k: int) -> int:
    """The last recorded row for forecast origin `k`; the rollout starts after it."""
    if k not in ORIGINS:
        raise ValueError(f"forecast origins are {ORIGINS}")
    return k - 2


def first_forecast_row(k: int) -> int:
    """The first row that is forecast, and scored, for forecast origin `k`."""
    return last_recorded_row(k) + 1


# Subjects per split in the published cohort (strict first stay per subject).
SPLIT = {"train": 7147, "val": 893, "test": 894}
SPLIT_SEED = 42
SELECTION_SEED = 724
CACHE_TENSORS = ("continuous", "continuous_mask", "binary", "time_hours", "static",
                 "subject_ids", "stay_ids", "hadm_ids")


def save_cache(cache: dict, path: Path, keys: tuple[str, ...] = CACHE_TENSORS) -> Path:
    """Write the cohort tensors [N, T, ...] named by `keys` and their variable names."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: cache[key] for key in keys}
    payload.update(continuous_names=list(V.CONTINUOUS), binary_names=list(V.BINARY), static_names=list(V.STATIC))
    pending = path.with_suffix(".pending")
    torch.save(payload, pending)
    pending.replace(path)
    return path


NAME_LISTS = ("continuous_names", "binary_names", "static_names")


def cache_digest(cache: dict, keys: tuple[str, ...] = CACHE_TENSORS) -> str:
    """SHA-256 over the cache content: each tensor's dtype, shape and bytes, then the name lists.

    The digest is independent of the file name and of the serialisation
    format, which both change the bytes of a `torch.save` file.
    """
    digest = hashlib.sha256()
    for key in keys:
        tensor = cache[key].contiguous()
        digest.update(f"{key}:{tensor.dtype}:{list(tensor.shape)}\n".encode())
        digest.update(tensor.numpy().tobytes())
    for key in NAME_LISTS:
        digest.update(json.dumps([key, list(cache[key])]).encode())
    return digest.hexdigest()


def load_cache(path: Path, keys: tuple[str, ...] = CACHE_TENSORS) -> dict:
    """Read a cache written by `save_cache` and record its content digest as `sha256`."""
    cache = torch.load(path, map_location="cpu", weights_only=True)
    for key, names in zip(NAME_LISTS, (V.CONTINUOUS, V.BINARY, V.STATIC)):
        if list(cache[key]) != list(names):
            raise ValueError(f"{path}: {key} differ from pulse.variables")
    cache["sha256"] = cache_digest(cache, keys)
    return cache


def split(subject_ids: torch.Tensor) -> dict[str, np.ndarray]:
    """Cache rows of the train, validation and test stays.

    Subjects are shuffled with a fixed seed and cut into consecutive blocks
    of the sizes in `SPLIT`; each subject contributes only its first stay in
    cache order. Rows keep cache order within each split.
    """
    ids = np.asarray(subject_ids)
    subjects = np.unique(ids)
    np.random.default_rng(SPLIT_SEED).shuffle(subjects)
    if len(subjects) < sum(SPLIT.values()):
        raise ValueError(f"the cache holds {len(subjects)} subjects; the split needs {sum(SPLIT.values())}")
    eligible = np.sort(np.unique(ids, return_index=True)[1])
    ends = np.cumsum(list(SPLIT.values()))
    blocks = np.split(subjects, ends)[:len(SPLIT)]
    return {name: eligible[np.isin(ids[eligible], block)] for name, block in zip(SPLIT, blocks)}


def select(subject_ids: torch.Tensor) -> dict[str, np.ndarray]:
    """Cache rows of each prepared part, in the order of its patients.

    The split of `split`, with the training rows permuted by
    `SELECTION_SEED`; validation and test rows keep cache order.
    """
    rows = split(subject_ids)
    return {"train": np.random.default_rng(SELECTION_SEED).permutation(rows["train"]),
            "val": rows["val"], "test": rows["test"]}


def _mean_filled(values: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    values, mask = values.numpy(), mask.numpy()
    means = (values * mask).sum(axis=(0, 1)) / np.maximum(mask.sum(axis=(0, 1)), 1.0)
    return np.where(mask > 0.5, values, means.reshape(1, 1, -1)).reshape(-1, values.shape[-1])


def _abs_correlation(matrix: np.ndarray) -> np.ndarray:
    corr = np.abs(np.nan_to_num(np.corrcoef(matrix, rowvar=False), nan=0.0, posinf=0.0, neginf=0.0))
    np.fill_diagonal(corr, 1.0)
    return corr


def covariance_groups(continuous: torch.Tensor, mask: torch.Tensor, groups: int = 6) -> dict[str, list[str]]:
    """Group the non-treatment variables by average-linkage clustering, then add the treatments.

    `continuous` [N, T, V] are the training inputs and `mask` [N, T, V] marks
    recorded values. Similarity combines absolute correlations of levels and
    of hour-to-hour changes, each filled with its own mean where unrecorded.
    """
    outcomes = [i for i, name in enumerate(V.CONTINUOUS) if name not in V.TREATMENTS]
    x, m = continuous[..., outcomes], mask[..., outcomes]
    levels = _abs_correlation(_mean_filled(x, m))
    changes = _abs_correlation(_mean_filled(x[:, 1:, :] - x[:, :-1, :], m[:, 1:, :] * m[:, :-1, :]))
    # The reference implementation weighted a residual correlation by 0.5 and
    # used the change correlation for it; the sum keeps its rounding.
    similarity = 0.2 * levels + 0.3 * changes + 0.5 * changes
    np.fill_diagonal(similarity, 1.0)
    distance = 1.0 - np.clip(similarity, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    labels = fcluster(linkage(squareform(distance, checks=False), method="average"), t=groups, criterion="maxclust")
    result: dict[str, list[str]] = {}
    for label, name in sorted(zip(labels.tolist(), [V.CONTINUOUS[i] for i in outcomes])):
        result.setdefault(f"cov_group_{label:02d}", []).append(name)
    result["continuous_treatments"] = [name for name in V.CONTINUOUS if name in V.TREATMENTS]
    return result


def _part(cache: dict, ids: np.ndarray, x: torch.Tensor, mask: torch.Tensor, mean: torch.Tensor,
          static: torch.Tensor) -> Part:
    filled = torch.where(mask[ids], x[ids], mean)
    for t in range(1, filled.shape[1]):
        filled[:, t] = torch.where(mask[ids, t], filled[:, t], filled[:, t - 1])
    return {"continuous": filled, "continuous_target": torch.where(mask[ids], x[ids], mean),
            "continuous_mask": mask[ids].float(), "binary": cache["binary"][ids].float(),
            "time_hours": cache["time_hours"][ids].float(), "static": static[ids]}


def prepare(cache: dict, protocol: Protocol | None = None) -> Prepared:
    """Split, select, scale and fill the cohort; fit the covariance groups on training patients.

    Means and SDs of continuous variables use recorded training entries only
    (SD floor 0.01). Inputs carry the last recorded value forward and start
    from the training mean; targets put the training mean at unrecorded
    entries. Static binary columns are left unscaled.

    Given the protocol a checkpoint was trained with, preparation takes the
    means and SDs from it and checks that the cache, the split and the
    covariance groups agree. Recomputed statistics can differ from the
    shipped ones in the last bit on another platform.
    """
    chosen = select(cache["subject_ids"])
    x = cache["continuous"].float()
    mask = cache["continuous_mask"].bool() & torch.isfinite(x)
    train = chosen["train"]
    observed = torch.where(mask[train], x[train], torch.nan)
    mean = torch.nanmean(observed.flatten(0, 1), 0).nan_to_num()
    sd = torch.nanmean((observed - mean).square().flatten(0, 1), 0).sqrt().nan_to_num(1.0).clamp_min(0.01)
    static = cache["static"].float()
    static_mean = static[train].mean(0)
    static_sd = static[train].std(0, unbiased=False).clamp_min(0.01)
    static_mean[len(V.STATIC_CONTINUOUS):] = 0.0
    static_sd[len(V.STATIC_CONTINUOUS):] = 1.0
    if protocol is not None:
        if protocol.cache_sha256 != cache["sha256"]:
            raise ValueError("the protocol was prepared from a different cache")
        if protocol.split != {name: len(ids) for name, ids in chosen.items()}:
            raise ValueError("the protocol's split sizes differ from this cache")
        mean, sd = torch.tensor(protocol.mean), torch.tensor(protocol.sd)
        static_mean, static_sd = torch.tensor(protocol.static_mean), torch.tensor(protocol.static_sd)
    scaled = (static - static_mean) / static_sd
    parts = {name: _part(cache, ids, x, mask, mean, scaled) for name, ids in chosen.items()}
    groups = covariance_groups(parts["train"]["continuous"], parts["train"]["continuous_mask"])
    if protocol is not None:
        if protocol.groups != groups:
            raise ValueError("the protocol's covariance groups differ from those of this cache")
        return Prepared(train=parts["train"], val=parts["val"], test=parts["test"], protocol=protocol)
    protocol = Protocol(
        continuous_names=list(V.CONTINUOUS), binary_names=list(V.BINARY), static_names=list(V.STATIC),
        mean=mean.tolist(), sd=sd.tolist(), static_mean=static_mean.tolist(), static_sd=static_sd.tolist(),
        groups=groups,
        cache_sha256=cache["sha256"], split={name: len(ids) for name, ids in chosen.items()})
    return Prepared(train=parts["train"], val=parts["val"], test=parts["test"], protocol=protocol)

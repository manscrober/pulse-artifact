"""Prepared data, the training protocol and rollout case files.

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

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

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


def save_case(directory: Path, case: int, continuous: torch.Tensor, binary: torch.Tensor) -> None:
    """Write the paths of one test patient: continuous [P, T, V], binary [P, T, K]."""
    path = case_path(directory, case)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    torch.save({"case": case, "continuous": continuous.cpu(), "binary": binary.cpu()}, pending)
    pending.replace(path)


def load_case(directory: Path, case: int) -> dict:
    payload = torch.load(case_path(directory, case), map_location="cpu", weights_only=True)
    if payload["case"] != case:
        raise ValueError(f"case file {case} holds case {payload['case']}")
    return payload

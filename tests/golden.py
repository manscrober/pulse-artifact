"""Digests of tensors produced by the research reference implementation.

Fixtures live in tests/fixtures/<platform>/<module>.json and map a name to
the shape, dtype and SHA-256 of a tensor's bytes. Equal digests mean
bitwise-equal tensors. Results of matrix products and transcendental
functions differ between CPU instruction sets, so those modules have
fixtures per platform. A module whose operations are all exactly rounded
(comparisons, copies, single additions and subtractions) gives the same
bits everywhere and keeps one set in tests/fixtures/any, used on every
platform.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

FIXTURES = Path(__file__).parent / "fixtures"
PORTABLE = "any"


def platform_key() -> str:
    return f"{sys.platform}-{platform.machine()}"


def digest(tensor: torch.Tensor) -> dict:
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch."),
            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}


def fixture_path(module: str, key: str | None = None) -> Path:
    return FIXTURES / (key or platform_key()) / f"{module}.json"


def write(module: str, digests: dict[str, dict], meta: dict, portable: bool = False) -> Path:
    """Store digests for this platform, or for every platform when `portable`."""
    path = fixture_path(module, PORTABLE if portable else None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "digests": digests}, indent=1, sort_keys=True) + "\n")
    return path


def load(module: str) -> dict[str, dict] | None:
    """Digests for this platform, else the portable ones, else None."""
    for path in (fixture_path(module), fixture_path(module, PORTABLE)):
        if path.exists():
            return json.loads(path.read_text())["digests"]
    return None

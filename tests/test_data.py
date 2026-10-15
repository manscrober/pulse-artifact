import dataclasses
import json

import numpy as np
import pytest
import torch

from pulse import data
from synthetic import cache as synthetic_cache


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    path = data.save_cache(synthetic_cache(seed=5), tmp_path_factory.mktemp("cache") / "cache.pt")
    cache = data.load_cache(path)
    return cache, data.prepare(cache)


def test_split_keeps_first_stays_of_disjoint_subjects(prepared):
    cache, _ = prepared
    subjects = cache["subject_ids"].numpy()
    rows = data.split(cache["subject_ids"])
    assert {name: len(ids) for name, ids in rows.items()} == data.SPLIT
    sets = [set(subjects[ids].tolist()) for ids in rows.values()]
    assert all(len(s) == len(ids) for s, ids in zip(sets, rows.values()))
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    first = {}
    for row, subject in enumerate(subjects.tolist()):
        first.setdefault(subject, row)
    assert all(first[subjects[row]] == row for ids in rows.values() for row in ids)


def test_protocol(prepared):
    cache, result = prepared
    protocol = result.protocol
    assert protocol.split == data.SPLIT and protocol.cache_sha256 == cache["sha256"]
    assert list(protocol.groups)[-1] == "continuous_treatments"
    assert sorted(n for names in protocol.groups.values() for n in names) == sorted(protocol.continuous_names)
    assert protocol.static_mean[3:] == [0.0] * 3 and protocol.static_sd[3:] == [1.0] * 3


def test_split_rejects_a_short_cache():
    with pytest.raises(ValueError):
        data.split(torch.arange(sum(data.SPLIT.values()) - 1))


def test_digest_depends_on_content_only(tmp_path):
    cache = synthetic_cache(seed=1, subjects=20, stays=25)
    first = data.load_cache(data.save_cache(cache, tmp_path / "a" / "cache.pt"))["sha256"]
    assert data.load_cache(data.save_cache(cache, tmp_path / "cohort.pt"))["sha256"] == first
    cache["continuous"][0, 0, 0] += 1.0
    assert data.load_cache(data.save_cache(cache, tmp_path / "changed.pt"))["sha256"] != first


def test_prepare_with_a_given_protocol(prepared):
    cache, result = prepared
    again = data.prepare(cache, result.protocol)
    assert again.protocol == result.protocol
    for name in ("train", "val", "test"):
        for key, value in getattr(result, name).items():
            assert torch.equal(getattr(again, name)[key], value)
    shifted = dataclasses.replace(result.protocol, static_mean=[v + 1.0 for v in result.protocol.static_mean])
    assert not torch.equal(data.prepare(cache, shifted).train["static"], result.train["static"])
    for change in ({"cache_sha256": "0" * 64}, {"split": {"train": 1, "val": 1, "test": 1}},
                   {"groups": {"continuous_treatments": []}}):
        with pytest.raises(ValueError):
            data.prepare(cache, dataclasses.replace(result.protocol, **change))


def test_load_cache_rejects_other_variables(tmp_path):
    path = data.save_cache(synthetic_cache(seed=1, subjects=20, stays=25), tmp_path / "cache.pt")
    payload = torch.load(path, weights_only=True)
    payload["continuous_names"] = payload["continuous_names"][::-1]
    torch.save(payload, path)
    with pytest.raises(ValueError):
        data.load_cache(path)


def test_against_reference(prepared, expected):
    cache, result = prepared
    rows = data.split(cache["subject_ids"])
    chosen = {"train": np.random.default_rng(data.SELECTION_SEED).permutation(rows["train"]),
              "val": rows["val"], "test": rows["test"]}
    for name in ("train", "val", "test"):
        expected(f"{name}_subjects", cache["subject_ids"][chosen[name]])
        for key, value in getattr(result, name).items():
            expected(f"{name}_{key}", value)
    protocol = result.protocol
    for key in ("mean", "sd", "static_mean", "static_sd"):
        expected(key, torch.tensor(getattr(protocol, key)))
    expected("groups", torch.tensor(list(json.dumps(protocol.groups).encode()), dtype=torch.uint8))

import json

import numpy as np
import pytest
import torch

import synthetic
from pulse import data, score


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("score")
    cache = data.load_cache(data.save_cache(synthetic.scored_cache(), root / "cache.pt"))
    protocol = data.prepare(cache).protocol
    for (arm, origin, case), p in synthetic.scored_rollouts().items():
        data.save_case(root / arm / f"k{origin}", case, p["continuous"], p["binary"])
    arms = {arm: (root / arm, m) for arm, m in synthetic.SCORED_ARMS.items()}
    return cache, protocol, arms


def as_bytes(value) -> torch.Tensor:
    return torch.tensor(list(json.dumps(value).encode()), dtype=torch.uint8)


def test_empirical_scores_of_two_draws():
    s = score.empirical_scores(np.array([[0.0], [2.0]]), np.array([0.5]))
    assert s["crps"][0] == pytest.approx(0.5 * (0.5 + 1.5) - 0.25 * 2.0)
    assert s["coverage90_inverse"][0] == 1.0
    assert score.empirical_scores(np.array([[0.0], [2.0]]), np.array([3.0]))["coverage90_inverse"][0] == 0.0


def test_exclusions(inputs):
    cache, protocol, arms = inputs
    rows = data.select(cache["subject_ids"])["test"]
    mean = torch.tensor(protocol.mean)
    # Case 1 dies at hour 0 and case 2 at hour 3, before origin 6's first
    # forecast hour; case 6 lacks a file in one arm.
    assert score.collect(arms, cache, rows, mean, 2)[3] == [0, 2, 3, 4, 5, 7]
    assert score.collect(arms, cache, rows, mean, 6)[3] == [0, 3, 4, 5, 7]
    y, mask, predictions, _ = score.collect(arms, cache, rows, mean, 2)
    assert predictions["third"].shape[1] == 3 and predictions["first"].shape[1] == 5
    # The NaN of arm "second" (case 3, hour 10, variable 3) removes that
    # point from every arm.
    counts = score.patient_statistics(y, mask, predictions, np.array(protocol.sd))["counts"]
    expected_counts = mask.sum(1)
    expected_counts[2, 3] -= mask[2, 9, 3]
    assert np.array_equal(counts, expected_counts)


def test_rejects_other_cache(inputs):
    cache, protocol, arms = inputs
    with pytest.raises(ValueError):
        score.score(arms, dict(cache, sha256="0" * 64), protocol)


def test_against_reference(inputs, expected):
    cache, protocol, arms = inputs
    rows = data.select(cache["subject_ids"])["test"]
    mean, sd = torch.tensor(protocol.mean), np.array(protocol.sd)
    for k in (2, 6):
        y, mask, predictions, cases = score.collect(arms, cache, rows, mean, k)
        expected(f"k{k}_truth", torch.from_numpy(y))
        expected(f"k{k}_mask", torch.from_numpy(mask))
        for arm, x in predictions.items():
            expected(f"k{k}_{arm}_paths", torch.from_numpy(x))
        for key, value in score.patient_statistics(y, mask, predictions, sd, k).items():
            expected(f"k{k}_{key}", torch.from_numpy(np.ascontiguousarray(value)))
    result = score.score(arms, cache, protocol, resamples=200, pit=True)
    expected("rows", as_bytes(result["rows"]))
    panel = score.binary_panel({arm: d for arm, (d, _) in arms.items()}, cache, protocol, resamples=200)
    expected("binary_rows", as_bytes(panel["rows"]))
    expected("binary_channels_by_arm", as_bytes(panel["channels_by_arm"]))


def test_primary_rmse_panel_has_25_variables():
    assert len([v for v in score.PANELS[37] if score.V.CONTINUOUS[v] not in score.RMSE_EXCLUDED]) == 25

import json

import pytest
import torch

from pulse import model as M
from pulse import simulate as S
from pulse.data import load_case
from synthetic import part, perturb, protocol

PATHS, PER_CALL = 4, 2


def models():
    torch.manual_seed(M.SEED)
    pulse = M.pulse_model(protocol())
    policy = M.build_policy(protocol())
    perturb(pulse, seed=1)
    perturb(policy, seed=2)
    return pulse, policy


@pytest.mark.parametrize("arm", ["sample", "sample_twopass"])
def test_against_reference(expected, arm, tmp_path):
    pulse, policy = models()
    test = part(seed=5, n=3)
    for k in (6, 2):
        S.evaluate(pulse, policy, S.ARMS[arm], test, k, PATHS, tmp_path / f"k{k}", per_call=PER_CALL)
        for i in range(3):
            case = load_case(tmp_path / f"k{k}", i)
            expected(f"{arm}.k{k}.case{i}.continuous", case["continuous"])
            expected(f"{arm}.k{k}.case{i}.binary", case["binary"])


@pytest.mark.parametrize("arm", ["sample", "sample_twopass"])
def test_generator_after_each_call(expected, arm):
    pulse, policy = models()
    test = part(seed=5, n=3)
    for k in (6, 2):
        for first in range(0, 3, PER_CALL):
            torch.manual_seed(S.SEED + 1000 * k + first)
            S.rollout(pulse, policy, S.ARMS[arm], {key: v[first:first + PER_CALL] for key, v in test.items()},
                      S.origin_step(k), PATHS)
            expected(f"{arm}.k{k}.first{first}.after", torch.rand(2))


def test_rollout_settings_are_recorded_and_checked(tmp_path):
    pulse, policy = models()
    test = part(seed=5, n=1)
    S.evaluate(pulse, policy, S.ARMS["sample"], test, 6, 1, tmp_path)
    record = json.loads((tmp_path / "rollout.json").read_text())
    assert (record["origin"], record["paths"], record["per_call"]) == (6, 1, S.PER_CALL)
    with pytest.raises(ValueError):
        S.evaluate(pulse, policy, S.ARMS["sample"], test, 6, 2, tmp_path)

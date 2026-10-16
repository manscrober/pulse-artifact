import json

import pytest
import torch

from pulse import data as D
from pulse import feedback as F
from pulse import model as M
from pulse import simulate as S
from pulse import variables as V
from pulse.data import load_case
from synthetic import part, perturb, protocol

PATHS, PER_CALL = 4, 2
ARMS = list(S.ARMS)


def models():
    torch.manual_seed(M.SEED)
    pulse = M.pulse_model(protocol())
    policy = M.build_policy(protocol())
    perturb(pulse, seed=1)
    perturb(policy, seed=2)
    torch.manual_seed(77)
    model = F.FeedbackPULSE(pulse)
    perturb(model, seed=3, rename=lambda n: None if n.startswith("pulse.") else n)
    return model, policy


@pytest.mark.parametrize("arm", ARMS)
def test_against_reference(expected, arm, tmp_path):
    pulse, policy = models()
    test = part(seed=5, n=3)
    for k in (6, 2):
        S.evaluate(pulse, policy, S.ARMS[arm], test, k, PATHS, tmp_path / f"k{k}", per_call=PER_CALL)
        for i in range(3):
            case = load_case(tmp_path / f"k{k}", i)
            expected(f"{arm}.k{k}.case{i}.continuous", case["continuous"])
            expected(f"{arm}.k{k}.case{i}.binary", case["binary"])


@pytest.mark.parametrize("arm", ARMS)
def test_generator_after_each_call(expected, arm):
    pulse, policy = models()
    test = part(seed=5, n=3)
    for k in (6, 2):
        for first in range(0, 3, PER_CALL):
            torch.manual_seed(S.SEED + 1000 * k + first)
            S.rollout(pulse, policy, S.ARMS[arm], {key: v[first:first + PER_CALL] for key, v in test.items()},
                      k, PATHS)
            expected(f"{arm}.k{k}.first{first}.after", torch.rand(2))


def test_sobol_table_leaves_generator_alone():
    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    table = S.sobol_table(8)
    assert table.shape == (8, 24, 8) and torch.equal(torch.rand(3), expected)
    assert torch.equal(table, S.sobol_table(8))


def test_arm_rules():
    with pytest.raises(ValueError):
        S.Arm("sample", sobol=True)
    with pytest.raises(ValueError):
        S.Arm("sample", two_pass=True, quantiles=True)


def test_rollout_settings_are_recorded_and_checked(tmp_path):
    pulse, policy = models()
    test = part(seed=5, n=1)
    S.evaluate(pulse, policy, S.ARMS["sample"], test, 6, 1, tmp_path)
    record = json.loads((tmp_path / "rollout.json").read_text())
    assert (record["origin"], record["paths"], record["per_call"]) == (6, 1, S.PER_CALL)
    with pytest.raises(ValueError):
        S.evaluate(pulse, policy, S.ARMS["sample"], test, 6, 2, tmp_path)


def test_rollout_provenance_is_recorded_and_checked(tmp_path):
    pulse, policy = models()
    test = part(seed=5, n=1)
    first = {"policy": "a" * 64, "weights": "b" * 64}
    S.evaluate(pulse, policy, S.ARMS["sample"], test, 6, 1, tmp_path, provenance=first)
    assert json.loads((tmp_path / "rollout.json").read_text())["provenance"] == first
    written = S.case_path(tmp_path, 0).read_bytes()
    S.evaluate(pulse, policy, S.ARMS["sample"], test, 6, 1, tmp_path, provenance=dict(first))
    assert S.case_path(tmp_path, 0).read_bytes() == written
    for other in ({**first, "weights": "c" * 64}, None):
        with pytest.raises(ValueError):
            S.evaluate(pulse, policy, S.ARMS["sample"], test, 6, 1, tmp_path, provenance=other)


SKIP_ARMS = ["sample", "sample_sobol", "distribution", "mean_only"]


@pytest.fixture
def nothing_changes(monkeypatch):
    """Nothing is recorded, the bolus is held at zero and the vasopressor is never started.

    With death and release also made improbable (see `quiet_outcomes`),
    every generated hour after the first then skips the history append, so
    the history grows by at most one row after the origin. The reference
    digests come from the same changes applied to the reference
    implementation.
    """
    next_outputs, treatments = M.GroupedModel.next_outputs, M.Policy.treatments

    def quiet_next(self, *args, **kwargs):
        out = next_outputs(self, *args, **kwargs)
        out["obs_logits"] = torch.full_like(out["obs_logits"], -30.0)
        return out

    def quiet_treatments(self, *args, **kwargs):
        out = dict(treatments(self, *args, **kwargs))
        out["continuous_treatments"] = torch.zeros_like(out["continuous_treatments"])
        out["binary_treatment_logits"] = torch.full_like(out["binary_treatment_logits"], -30.0)
        return out

    monkeypatch.setattr(M.GroupedModel, "next_outputs", quiet_next)
    monkeypatch.setattr(M.Policy, "treatments", quiet_treatments)


def quiet_outcomes(pulse: F.FeedbackPULSE) -> None:
    """Set the death and release output biases to -30."""
    heads = pulse.pulse.encoders[pulse.pulse.binary_group].heads
    names = pulse.pulse.binary_outputs
    with torch.no_grad():
        for index in (V.DEATH, V.RELEASE):
            heads[names.index(V.BINARY[index])].bias.fill_(-30.0)


@pytest.mark.parametrize("arm", SKIP_ARMS)
def test_skipped_appends_against_reference(expected, arm, nothing_changes, monkeypatch):
    pulse, policy = models()
    quiet_outcomes(pulse)
    # The rows of the event history that the patient model last read, per
    # rollout row: the last call of `inputs` on the patient model is the
    # final generated hour's.
    inputs, last_lengths = M.GroupedModel.inputs, []

    def spy(self, history):
        if self is pulse.pulse:
            last_lengths[:] = [(~history["padding_mask"]).sum(1)]
        return inputs(self, history)

    monkeypatch.setattr(M.GroupedModel, "inputs", spy)
    test = part(seed=5, n=3)
    for k in (6, 2):
        start = D.last_recorded_row(k)
        torch.manual_seed(S.SEED + 1000 * k)
        cont, binary = S.rollout(pulse, policy, S.ARMS[arm], test, k, PATHS)
        for i in range(3):
            expected(f"skip.{arm}.k{k}.case{i}.continuous", cont[i])
            expected(f"skip.{arm}.k{k}.case{i}.binary", binary[i])
        expected(f"skip.{arm}.k{k}.after", torch.rand(2))
        # The first generated hour joins the history (its bolus changes from
        # the recorded one) and no later hour does, so the final hour reads
        # the recorded rows and one generated row.
        assert last_lengths and bool((last_lengths[0] == start + 2).all()), \
            "a generated hour after the first joined the history"
        # Nothing was recorded either: unrecorded variables carry forward,
        # apart from those that `update_derived` recomputes every hour.
        probe = torch.randn(1, 2, cont.shape[-1], generator=torch.Generator().manual_seed(0))
        written = (V.update_derived(probe.clone(), 1, probe[:, 0]) != probe)[0, 1]
        carried = [i for i in pulse.pulse.output_indices if not written[i]]
        generated = cont[..., start + 1:, carried]
        assert torch.equal(generated[..., 1:, :], generated[..., :-1, :]), "a generated hour recorded a value"

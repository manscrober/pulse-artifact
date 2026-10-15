import numpy as np
import pytest
import torch

from pulse import data as D
from pulse import train as T
from synthetic import part, perturb, protocol

BATCH, MICRO = 4, 3


def prepared() -> D.Prepared:
    return D.Prepared(train=part(seed=7, n=7), val=part(seed=8, n=5), test=part(seed=9, n=1), protocol=protocol())


def frozen_policy():
    from pulse import model as M
    policy = M.build_policy(protocol())
    perturb(policy, seed=2)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


def flat(named) -> torch.Tensor:
    items = {n: v for n, v in named if v is not None}
    return torch.cat([items[n].detach().reshape(-1) for n in sorted(items)])


@pytest.mark.parametrize("phase", T.PHASES)
def test_first_updates(expected, phase):
    """One epoch of two updates, validation, then the first update of the second epoch."""
    data = prepared()
    model = T.build(phase, data, None if phase == "policy" else frozen_policy())
    optimizer, scheduler = T.optimiser(model, phase, epochs=2)
    model.train()
    order = T.epoch_order(0, 7)
    for cursor in range(2):
        losses = T.update(model, optimizer, data.train, order[cursor * BATCH:(cursor + 1) * BATCH], phase, MICRO, "cpu")
        expected(f"{phase}.epoch1.update{cursor}.losses", torch.tensor(losses, dtype=torch.float64))
    loss = T.validation(model, data.val, phase, MICRO, "cpu")
    expected(f"{phase}.validation", torch.tensor([loss], dtype=torch.float64))
    scheduler.step()
    losses = T.update(model, optimizer, data.train, T.epoch_order(1, 7)[:BATCH], phase, MICRO, "cpu")
    expected(f"{phase}.epoch2.update0.losses", torch.tensor(losses, dtype=torch.float64))
    expected(f"{phase}.epoch2.update0.grads", flat((n, p.grad) for n, p in model.named_parameters()))
    expected(f"{phase}.epoch2.update0.parameters", flat(model.named_parameters()))
    expected(f"{phase}.after", torch.rand(3))


@pytest.mark.parametrize("phase", ["policy", "distribution"])
def test_fit(expected, phase, tmp_path):
    settings = T.Settings(epochs=3, batch=BATCH, micro=MICRO, patience=1, min_epochs=2)
    result = T.fit(phase, prepared(), None if phase == "policy" else frozen_policy(), tmp_path, "cpu", settings)
    losses = [h["validation_loss"] for h in result["history"]]
    expected(f"fit.{phase}.validation", torch.tensor(losses, dtype=torch.float64))
    expected(f"fit.{phase}.parameters", flat(result["model"].named_parameters()))
    expected(f"fit.{phase}.after", torch.rand(3))
    assert result["validation_loss"] == min(losses)
    saved = torch.load(tmp_path / f"{phase}.pt", weights_only=True)
    assert saved["history"] == result["history"]
    assert saved["cache_sha256"] == protocol().cache_sha256
    assert D.Protocol.load(tmp_path / "protocol.json") == protocol()
    assert all(torch.equal(saved["state"][k], v) for k, v in result["model"].state_dict().items())
    if phase == "policy":
        assert not result["model"].training
        assert not any(p.requires_grad for p in result["model"].parameters())


def test_waiting_time_targets():
    """Hours to the next record, censored at 48 when the variable is not recorded again."""
    recorded = torch.tensor([[[1.0], [0.0], [1.0], [0.0]]])
    hours = torch.tensor([[0.0, 0.5, 2.0, 3.5]])
    assert T._hours_to_next_record(recorded, hours).flatten().tolist() == [2.0, 1.5, 48.0]


def test_absorbing_outcome_counts_until_onset():
    logits = torch.zeros(1, 4, 2)
    target = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 1.0]]])
    absorbing = torch.tensor([True, False])
    # The absorbing column counts rows 0 and 1 only; the other column counts all four.
    assert T._binary_loss(logits, target, absorbing).item() == pytest.approx(np.log(2.0))


def test_fit_refuses_another_protocol(tmp_path):
    import dataclasses
    dataclasses.replace(protocol(), cache_sha256="1" * 64).save(tmp_path / "protocol.json")
    with pytest.raises(ValueError):
        T.fit("policy", prepared(), None, tmp_path)


def test_fit_copies_the_given_protocol(tmp_path):
    """A shipped protocol is written next to the weights as it is, never recomputed."""
    import dataclasses
    shipped = tmp_path / "shipped.json"
    protocol().save(shipped)
    data = dataclasses.replace(prepared(), protocol=D.Protocol.load(shipped))
    settings = T.Settings(epochs=1, batch=BATCH, micro=MICRO, patience=1, min_epochs=1)
    T.fit("policy", data, None, tmp_path / "out", "cpu", settings)
    assert (tmp_path / "out" / "protocol.json").read_bytes() == shipped.read_bytes()

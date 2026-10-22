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
def test_fit(expected, phase, tmp_path, capsys):
    settings = T.Settings(epochs=3, batch=BATCH, micro=MICRO, patience=1, min_epochs=2)
    reported = []
    result = T.fit(phase, prepared(), None if phase == "policy" else frozen_policy(), tmp_path, "cpu", settings,
                   on_epoch=reported.append)
    assert reported == result["history"] and capsys.readouterr().out == ""
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


def _attention_inputs(device="cpu"):
    g = torch.Generator().manual_seed(3)
    return [torch.randn(2, 4, 6, 16, generator=g).to(device) for _ in range(3)]


def test_mps_attention_dropout_is_scoped_and_leaves_cpu_calls_unchanged():
    import torch.nn.functional as F
    original = F.scaled_dot_product_attention
    q, k, v = _attention_inputs()
    expected = original(q, k, v, is_causal=True)
    with T.mps_attention_dropout():
        assert F.scaled_dot_product_attention is not original
        assert torch.equal(F.scaled_dot_product_attention(q, k, v, is_causal=True), expected)
    assert F.scaled_dot_product_attention is original


def test_math_attention_agrees_with_the_kernel():
    import torch.nn.functional as F
    q, k, v = _attention_inputs()
    mask = torch.rand(6, 6, generator=torch.Generator().manual_seed(1)) > 0.3
    mask.fill_diagonal_(True)
    for kwargs in ({"is_causal": True}, {"attn_mask": mask}):
        assert torch.allclose(T._math_attention(q, k, v, **kwargs), F.scaled_dot_product_attention(q, k, v, **kwargs),
                              atol=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an MPS device")
def test_mps_attention_with_dropout_runs():
    import torch.nn.functional as F
    q, k, v = _attention_inputs("mps")
    with T.mps_attention_dropout():
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.5, is_causal=True)
    assert out.shape == q.shape and torch.isfinite(out).all()


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


class Interrupted(Exception):
    pass


def interrupt_at(monkeypatch, call: int) -> None:
    """Stop the fit before its `call`-th update, after a continuation is saved at every update."""
    update, calls = T.update, []

    def counted(*args, **kwargs):
        calls.append(1)
        if len(calls) == call:
            raise Interrupted
        return update(*args, **kwargs)

    monkeypatch.setattr(T, "CONTINUATION_SECONDS", 0.0)
    monkeypatch.setattr(T, "update", counted)


@pytest.mark.parametrize("phase", ["policy", "sample", T.BASELINE])
@pytest.mark.parametrize("call", [2, 3, 6])
def test_fit_continues_where_it_stopped(phase, call, tmp_path, monkeypatch):
    """Stopped within the first epoch (call 2), at the start of the second (3) or within the third (6)."""
    settings = T.Settings(epochs=4, batch=BATCH, micro=MICRO, patience=1, min_epochs=2)
    policy = None if phase == "policy" else frozen_policy()
    whole = T.fit(phase, prepared(), policy, tmp_path / "whole", "cpu", settings)
    with monkeypatch.context() as patch:
        interrupt_at(patch, call)
        with pytest.raises(Interrupted):
            T.fit(phase, prepared(), policy, tmp_path / "parts", "cpu", settings)
    assert (tmp_path / "parts" / f"{phase}.continue.pt").exists()
    torch.manual_seed(1)  # the continuation restores the generators
    parts = T.fit(phase, prepared(), policy, tmp_path / "parts", "cpu", settings)
    assert parts["history"] == whole["history"]
    assert all(torch.equal(v, whole["model"].state_dict()[k]) for k, v in parts["model"].state_dict().items())
    assert not (tmp_path / "parts" / f"{phase}.continue.pt").exists()


def test_fit_refuses_the_continuation_of_another_fit(tmp_path, monkeypatch):
    settings = T.Settings(epochs=3, batch=BATCH, micro=MICRO, patience=1, min_epochs=2)
    with monkeypatch.context() as patch:
        interrupt_at(patch, 2)
        with pytest.raises(Interrupted):
            T.fit("policy", prepared(), None, tmp_path, "cpu", settings)
    with pytest.raises(ValueError, match="continues another fit"):
        T.fit("policy", prepared(), None, tmp_path, "cpu", T.Settings(epochs=4, batch=BATCH, micro=MICRO))

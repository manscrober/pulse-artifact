import dataclasses

import numpy as np
import pytest
import torch

from pulse import data as D
from pulse import model as M
from pulse import train as T
from pulse import variables as V
from pulse.feedback import ARMS
from pulse.rows import ROWS
from synthetic import part, perturb, protocol

BATCH, MICRO = 4, 3


def prepared() -> D.Prepared:
    return D.Prepared(train=part(seed=7, n=7), val=part(seed=8, n=5), test=part(seed=9, n=1), protocol=protocol())


def frozen_policy():
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
    model = T.build(phase, data, None if phase == "policy" else frozen_policy(), M.Architecture(head="rescnf"))
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
                   on_epoch=reported.append, architecture=M.Architecture(head="rescnf"))
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
    # PULSE checkpoints record their variant; the policy records none.
    assert saved.get("architecture") == (None if phase == "policy" else dataclasses.asdict(M.Architecture(head="rescnf")))
    if phase == "policy":
        assert not result["model"].training
        assert not any(p.requires_grad for p in result["model"].parameters())


def test_fit_of_a_row_is_saved_under_its_name_with_its_variant(tmp_path):
    settings = T.Settings(epochs=1, batch=BATCH, micro=MICRO, patience=1, min_epochs=1)
    T.fit("sample", prepared(), frozen_policy(), tmp_path, "cpu", settings, architecture=ROWS["flow"].architecture,
          name="flow")
    saved = torch.load(tmp_path / "flow.pt", weights_only=True)
    assert saved["architecture"] == dataclasses.asdict(ROWS["flow"].architecture)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["flow.pt", "protocol.json"]


def _sample_model_and_batch():
    data = prepared()
    model = T.build("sample", data, frozen_policy(), M.Architecture(head="rescnf"))
    perturb(model, seed=4)
    model.eval()
    return model, {k: v[:4] for k, v in data.train.items()}


def test_masks_of_a_source_that_records_everything_leave_the_loss_unchanged():
    model, batch = _sample_model_and_batch()
    flagged = {**batch, "observable": torch.ones(4, len(V.CONTINUOUS)), "outcomes_known": torch.ones(4, len(V.BINARY))}
    with torch.no_grad():
        assert torch.allclose(T.pulse_loss(model, flagged, ARMS["sample"]), T.pulse_loss(model, batch, ARMS["sample"]),
                              rtol=1e-6, atol=0)


def test_a_source_leaves_the_terms_of_what_it_never_records_out_of_the_loss():
    """Changing the observation, timing and outcome heads of what a source never records leaves its loss alone."""
    model, batch = _sample_model_and_batch()
    variable, outcome = "Lactate", "Cumulative Edema"
    observable = torch.ones(4, len(V.CONTINUOUS))
    observable[:, V.CONTINUOUS.index(variable)] = 0.0
    known = torch.ones(4, len(V.BINARY))
    known[:, V.BINARY.index(outcome)] = 0.0
    flagged = {**batch, "observable": observable, "outcomes_known": known}

    def losses():
        with torch.no_grad():
            return [T.pulse_loss(model, b, ARMS["sample"]).item() for b in (flagged, batch)]

    before = losses()
    core = model.pulse
    encoder = next(e for e in core.encoders if variable in e.outputs)
    i = encoder.outputs.index(variable)
    binary = core.encoders[core.binary_group]
    with torch.no_grad():
        for head in (encoder.obs_heads[i], encoder.delta_t_heads[i], binary.heads[binary.outputs.index(outcome)]):
            head.weight.add_(0.5)
            head.bias.add_(0.5)
    after = losses()
    assert after[0] == before[0]
    assert after[1] != before[1]


def test_initial_outcome_rates_count_only_the_stays_that_record_the_outcome():
    data = prepared()
    core = M.pulse_model(protocol(), M.Architecture(head="rescnf"))
    j = V.BINARY.index("Cumulative Edema")
    binary = data.train["binary"].clone()
    binary[:3, :, j] = 0.0
    known = torch.ones(len(binary), len(V.BINARY))
    known[:3, j] = 0.0
    logits = core.binary_initial_logits(binary, known)
    assert logits["Cumulative Edema"] == core.binary_initial_logits(data.train["binary"][3:])["Cumulative Edema"]
    assert logits["Dialysis Indicator"] == core.binary_initial_logits(binary)["Dialysis Indicator"]


def test_a_cotrained_fit_records_its_sources(tmp_path):
    data = prepared()
    joined = D.Prepared(train=D.concatenate([data.train, part(seed=10, n=3)]), val=data.val, test=data.test,
                        protocol=data.protocol, sources=("e" * 64,))
    settings = T.Settings(epochs=1, batch=BATCH, micro=MICRO, patience=1, min_epochs=1)
    T.fit("sample", joined, frozen_policy(), tmp_path, "cpu", settings, architecture=ROWS["anchor"].architecture,
          name="anchor")
    saved = torch.load(tmp_path / "anchor.pt", weights_only=True)
    assert saved["sources"] == ["e" * 64] and saved["cache_sha256"] == protocol().cache_sha256


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
    dataclasses.replace(protocol(), cache_sha256="1" * 64).save(tmp_path / "protocol.json")
    with pytest.raises(ValueError):
        T.fit("policy", prepared(), None, tmp_path)


def test_fit_copies_the_given_protocol(tmp_path):
    """A shipped protocol is written next to the weights as it is, never recomputed."""
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
    whole = T.fit(phase, prepared(), policy, tmp_path / "whole", "cpu", settings, architecture=M.Architecture(head="rescnf"))
    with monkeypatch.context() as patch:
        interrupt_at(patch, call)
        with pytest.raises(Interrupted):
            T.fit(phase, prepared(), policy, tmp_path / "parts", "cpu", settings, architecture=M.Architecture(head="rescnf"))
    assert (tmp_path / "parts" / f"{phase}.continue.pt").exists()
    torch.manual_seed(1)  # the continuation restores the generators
    parts = T.fit(phase, prepared(), policy, tmp_path / "parts", "cpu", settings, architecture=M.Architecture(head="rescnf"))
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


def test_spline_warm_start_from_direct_flow(tmp_path):
    source = tmp_path / "flow.pt"
    data = prepared()
    flow = T.build("sample", data, frozen_policy(), M.Architecture(head="cnf"))
    M.save_checkpoint(source, flow, [], data.protocol.cache_sha256, M.Architecture(head="cnf"))
    settings = T.Settings(epochs=1, batch=7, micro=7, patience=1, min_epochs=1)
    result = T.fit("sample", data, frozen_policy(), tmp_path / "fit", "cpu", settings,
                   warm_start_cnf=source)
    checkpoint = torch.load(tmp_path / "fit/sample.pt", weights_only=True)
    assert result["model"].pulse.head == "bspline"
    assert checkpoint["architecture"]["head"] == "bspline"
    assert checkpoint["initialization"]["kind"] == "direct_cnf"
    assert checkpoint["initialization"]["copied_heads"] > 0
    assert len(checkpoint["initialization"]["sha256"]) == 64

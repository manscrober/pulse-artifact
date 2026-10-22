import torch

from pulse import gtransformer as G
from synthetic import part, perturb, protocol


def build():
    model = G.build_gtransformer(protocol(), part(seed=6, n=16)["binary"])
    return model, torch.rand(3)


def record(expected, prefix, o):
    for key in ("mu", "bin_logits", "bin_probs", "obs_logits"):
        expected(f"{prefix}.{key}", o[key])


def test_initial_weights(expected):
    model, after = build()
    expected("after_init", after)
    for name, value in model.state_dict().items():
        expected("init." + name, value)


def test_outputs(expected):
    model, _ = build()
    perturb(model, seed=1)
    data = part(seed=4, n=4)
    history = {k: data[k] for k in ("continuous", "binary", "static")}
    model.eval()
    with torch.no_grad():
        record(expected, "teacher", model(history))
        torch.manual_seed(31)
        o = model.next_outputs(history)
        for key in ("mu", "bin_logits", "bin_probs", "obs_logits"):
            expected(f"next.{key}", o[key])
        expected("next.after", torch.rand(3))
        torch.manual_seed(31)
        o = model.next_outputs(history, rows=torch.tensor([23, 16, 8, 0]))
        for key in ("mu", "bin_probs", "obs_logits"):
            expected(f"rows.{key}", o[key])
        expected("rows.after", torch.rand(3))
    model.train()
    torch.manual_seed(9)
    o = model(history)
    record(expected, "train", o)
    (o["mu"].square().mean() + o["bin_logits"].mean() + o["obs_logits"].mean()).backward()
    for name, p in model.named_parameters():
        if p.grad is not None and ("Heart_Rate" in name or "Cumulative_Edema" in name):
            expected("grad." + name, p.grad)
    expected("after_train", torch.rand(3))


def test_last_row_read_differs_from_named_rows_only_where_padded():
    model, _ = build()
    perturb(model, seed=1)
    model.eval()
    data = part(seed=4, n=4)
    history = {k: data[k] for k in ("continuous", "binary", "static")}
    with torch.no_grad():
        torch.manual_seed(0)
        last = model.next_outputs(history)
        torch.manual_seed(0)
        named = model.next_outputs(history, rows=torch.full((4,), 23))
    assert all(torch.equal(last[k], named[k]) for k in ("mu", "bin_probs", "obs_logits"))


# Training and rollout in the package ------------------------------------------

def prepared():
    from pulse.data import Prepared
    return Prepared(train=part(seed=7, n=7), val=part(seed=8, n=5), test=part(seed=9, n=1), protocol=protocol())


def test_predicts_the_variables_pulse_predicts():
    from pulse import model as M
    pulse = M.pulse_model(protocol())
    model = G.gtransformer_model(protocol())
    assert model.output_indices == pulse.output_indices
    assert model.binary_output_indices == pulse.binary_output_indices


def test_phase_loss_is_the_point_row_loss():
    """Squared error over recorded values, binary cross-entropy with absorbing outcomes held, observation BCE."""
    from pulse import train as T
    from pulse import variables as V

    data = prepared()
    model = T.build(T.BASELINE, data, None)
    perturb(model, seed=1)
    model.eval()
    ids = torch.arange(4).numpy()
    batch = {k: v[ids] for k, v in data.train.items()}
    with torch.no_grad():
        loss = T.batch_loss(model, data.train, ids, T.BASELINE, "cpu")
        out = model(batch)
    idx, outcome = model.output_indices, model.binary_output_indices
    target = model.normalize(batch["continuous_target"])[:, 1:, idx]
    recorded = batch["continuous_mask"][:, 1:, idx]
    absorbing = torch.tensor([model.binary_names[i] in V.ABSORBING for i in outcome])
    manual = (((target - out["mu"][:, :-1]) ** 2 * recorded).sum() / (recorded.sum() + 1e-8)
              + T._binary_loss(out["bin_logits"][:, :-1], batch["binary"][:, 1:, outcome], absorbing)
              + torch.nn.functional.binary_cross_entropy_with_logits(out["obs_logits"][:, :-1], recorded))
    assert torch.allclose(loss, manual, rtol=1e-6, atol=0)


def test_fit_trains_the_observation_heads(tmp_path):
    from pulse import train as T

    data = prepared()
    start = T.build(T.BASELINE, data, None).state_dict()
    settings = T.Settings(epochs=2, batch=4, micro=2, patience=1, min_epochs=1)
    result = T.fit(T.BASELINE, data, None, tmp_path, "cpu", settings)
    trained = torch.load(tmp_path / f"{T.BASELINE}.pt", weights_only=True)["state"]
    assert len(result["history"]) == 2
    observation = [k for k in trained if k.endswith("obs_head.weight")]
    assert len(observation) == 36
    assert all(not torch.equal(trained[k], start[k]) for k in observation)


def rollout_setup():
    from pulse import model as M
    from pulse import simulate as S

    model = G.build_gtransformer(protocol(), part(seed=6, n=16)["binary"])
    perturb(model, seed=1)
    policy = M.build_policy(protocol())
    perturb(policy, seed=2)
    return model, policy, S.static_noise(part(seed=8, n=5)), part(seed=5, n=2)


def test_rollout_reads_the_hourly_rows_and_realises_the_draws(monkeypatch):
    from pulse import data as D
    from pulse import simulate as S
    from pulse import variables as V

    model, policy, noise, test = rollout_setup()
    calls, next_outputs = [], G.GTransformer.next_outputs

    def spy(self, history, rows=None):
        out = next_outputs(self, history, rows)
        calls.append((history["continuous"].shape[1], out["bin_samples"][:, 0, :].clone()))
        return out

    monkeypatch.setattr(G.GTransformer, "next_outputs", spy)
    k = 6
    torch.manual_seed(0)
    cont, binary = S.rollout(model, policy, S.ARMS["sample"], test, k, 3, noise=noise, policy_at_last_event=True)
    last, steps = D.last_recorded_row(k), test["continuous"].shape[1]
    assert [rows for rows, _ in calls] == list(range(last + 1, steps))
    # Outcomes other than death, release and the absorbing ones are taken as drawn.
    free = [j for j, i in enumerate(model.binary_output_indices) if V.BINARY[i] not in V.ABSORBING]
    flat = binary.reshape(-1, steps, binary.shape[-1])
    for (rows, drawn), t in zip(calls, range(last, steps - 1)):
        for j in free:
            assert torch.equal(flat[:, t + 1, model.binary_output_indices[j]], drawn[:, j])


def test_rollout_takes_only_the_sample_arm():
    import pytest

    from pulse import simulate as S

    model, policy, noise, test = rollout_setup()
    with pytest.raises(ValueError):
        S.rollout(model, policy, S.ARMS["sample_sobol"], test, 6, 2, noise=noise)

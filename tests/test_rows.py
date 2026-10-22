"""Every PULSE variant of the component table against the reference driver.

The reference digests come from the reference driver, run with
each row's architecture: initial weights, outputs after a perturbation, two
training updates of the sample phase, and rollouts of three synthetic
patients at both origins with the policy read at the last event and, for
point heads, the static noise of synthetic validation patients.
"""
import numpy as np
import pytest
import torch

from pulse import feedback as F
from pulse import model as M
from pulse import simulate as S
from pulse import train as T
from pulse.rows import ROWS
from synthetic import part, perturb, protocol

PATHS, PER_CALL, MICRO = 4, 2, 3
ARCHITECTURES = {row.reference: row.architecture for row in ROWS.values()}
ROLLOUT_ARMS = {}
for row in ROWS.values():
    ROLLOUT_ARMS.setdefault(row.reference, [])
    if row.arm not in ROLLOUT_ARMS[row.reference]:
        ROLLOUT_ARMS[row.reference].append(row.arm)


def build(architecture):
    return F.build_pulse(protocol(), part(seed=6, n=16)["binary"], architecture)


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


def record(expected, prefix, o):
    for key in ("mu", "log_var", "bin_logits", "bin_probs", "obs_logits", "delta_t_hours"):
        if o.get(key) is not None:
            expected(f"{prefix}.{key}", o[key])


def test_rows_name_known_settings():
    assert len(ROWS) == 10 and len(ARCHITECTURES) == 8
    assert all(row.weights in (None, *ROWS) for row in ROWS.values())


@pytest.mark.parametrize("reference", list(ARCHITECTURES))
def test_weights_and_outputs(expected, reference):
    model = build(ARCHITECTURES[reference])
    expected(f"{reference}.after_init", torch.rand(3))
    for name, value in model.state_dict().items():
        expected(f"{reference}.init.{name}", value)
    perturb(model, seed=1)
    model.eval()
    core = model.pulse
    data = part(seed=4, n=4)
    history = {k: data[k] for k in ("continuous", "binary", "static", "time_hours", "continuous_mask")}
    with torch.no_grad():
        record(expected, f"{reference}.teacher", model(history, quantiles=False, probabilities=False))
        lengths = torch.tensor([24, 17, 9, 1])
        padding = torch.arange(24).view(1, -1) >= lengths.view(-1, 1)
        override = (torch.rand(4, 24, 8, generator=torch.Generator().manual_seed(3)) < 0.4).float()
        x = core.inputs(dict(history, padding_mask=padding))
        record(expected, f"{reference}.next", core.next_outputs(
            x, core.binary_logits(x), override[..., core.binary_output_indices], lengths))


@pytest.mark.parametrize("reference", list(ARCHITECTURES))
def test_first_updates(expected, reference):
    train, val = part(seed=7, n=7), part(seed=8, n=5)
    policy = frozen_policy()
    model = build(ARCHITECTURES[reference])
    model.attach_policy(policy)
    optimizer, _ = T.optimiser(model, "sample", epochs=2)
    model.train()
    order = T.epoch_order(0, 7)
    for cursor in range(2):
        losses = T.update(model, optimizer, train, order[cursor * 4:(cursor + 1) * 4], "sample", MICRO, "cpu")
        expected(f"{reference}.update{cursor}.losses", torch.tensor(losses, dtype=torch.float64))
    expected(f"{reference}.update1.grads", flat((n, p.grad) for n, p in model.named_parameters()))
    expected(f"{reference}.update1.parameters", flat(model.named_parameters()))
    loss = T.validation(model, val, "sample", MICRO, "cpu")
    expected(f"{reference}.validation", torch.tensor([loss], dtype=torch.float64))


@pytest.mark.parametrize("reference", list(ARCHITECTURES))
def test_rollouts(expected, reference):
    architecture = ARCHITECTURES[reference]
    noise = S.static_noise(part(seed=8, n=5)) if architecture.head == "point" else None
    policy = frozen_policy()
    test = part(seed=5, n=3)
    for arm in ROLLOUT_ARMS[reference]:
        model = build(architecture)
        perturb(model, seed=1)
        model.attach_policy(policy)
        for k in (6, 2):
            for first in range(0, 3, PER_CALL):
                torch.manual_seed(S.SEED + 1000 * k + first)
                batch = {key: v[first:first + PER_CALL] for key, v in test.items()}
                cont, binary = S.rollout(model, policy, S.ARMS[arm], batch, k, PATHS, noise=noise,
                                         policy_at_last_event=True)
                for offset in range(cont.shape[0]):
                    expected(f"{reference}.{arm}.k{k}.case{first + offset}.continuous", cont[offset])
                    expected(f"{reference}.{arm}.k{k}.case{first + offset}.binary", binary[offset])
                expected(f"{reference}.{arm}.k{k}.first{first}.after", torch.rand(2))


def test_static_noise_without_pairs():
    data = part(seed=8, n=2)
    data = dict(data, continuous_mask=torch.zeros_like(data["continuous_mask"]))
    mean, sd = S.static_noise(data)
    assert torch.equal(mean, torch.zeros_like(mean)) and torch.equal(sd, torch.ones_like(sd))
    assert np.isfinite(mean.numpy()).all()

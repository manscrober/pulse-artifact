import torch

from pulse import model as M
from synthetic import part, perturb, protocol


def build():
    torch.manual_seed(M.SEED)
    pulse = M.pulse_model(protocol())
    after_pulse = torch.rand(3)
    pulse.set_binary_biases(pulse.binary_initial_logits(part(seed=6, n=16)["binary"]))
    policy = M.build_policy(protocol())
    after_policy = torch.rand(3)
    return pulse, policy, after_pulse, after_policy


def record(expected, prefix, o):
    for key in ("mu", "log_var", "bin_logits", "bin_probs", "obs_logits", "delta_t_hours"):
        expected(f"{prefix}.{key}", o[key])
    for j in (0, 7, 35):
        for key in ("loc", "scale", "amps", "centers", "widths"):
            expected(f"{prefix}.params{j}.{key}", o["dist_params_list"][j][key])


def next_step(model, history, next_binary, lengths):
    x = model.inputs(history)
    return model.next_outputs(x, model.binary_logits(x), next_binary, lengths)


def test_initial_weights(expected):
    pulse, policy, after_pulse, after_policy = build()
    expected("pulse_after_init", after_pulse)
    expected("policy_after_init", after_policy)
    logits = pulse.binary_initial_logits(part(seed=6, n=16)["binary"])
    expected("binary_initial_logits", torch.tensor([logits[n] for n in pulse.binary_outputs], dtype=torch.float64))
    for name, value in pulse.state_dict().items():
        expected("pulse_init." + name, value)
    for name, value in policy.state_dict().items():
        expected("policy_init." + name, value)


def test_outputs(expected):
    pulse, policy, _, _ = build()
    perturb(pulse, seed=1)
    perturb(policy, seed=2)
    data = part(seed=4, n=4)
    history = {k: data[k] for k in ("continuous", "binary", "static", "time_hours", "continuous_mask")}
    pulse.eval()
    policy.eval()
    with torch.no_grad():
        # Rollout size: 800 rows, padded histories of random length, 7-column override.
        big = part(seed=7, n=800)
        g = torch.Generator().manual_seed(8)
        lengths = torch.randint(1, 25, (800,), generator=g)
        padding = torch.arange(24).view(1, -1) >= lengths.view(-1, 1)
        override = (torch.rand(800, 24, 7, generator=g) < 0.3).float()
        big_history = {k: big[k] for k in ("continuous", "binary", "static", "time_hours", "continuous_mask")}
        record(expected, "big", next_step(pulse, dict(big_history, padding_mask=padding), override, lengths))
        p = policy.treatments(big["continuous"], big["binary"], big["static"], big["time_hours"])
        expected("big_policy.continuous_treatments", p["continuous_treatments"][:, -1])
        expected("big_policy.binary_treatment_logits", p["binary_treatment_logits"][:, -1])
        record(expected, "teacher", pulse(history))
        lengths = torch.tensor([24, 17, 9, 1])
        padding = torch.arange(24).view(1, -1) >= lengths.view(-1, 1)
        override = (torch.rand(4, 24, 8, generator=torch.Generator().manual_seed(3)) < 0.4).float()
        override = override[..., pulse.binary_output_indices]
        record(expected, "next", next_step(pulse, dict(history, padding_mask=padding), override, lengths))
        p = policy.treatments(data["continuous"], data["binary"], data["static"], data["time_hours"])
        for key, value in p.items():
            expected("policy." + key, value)

    pulse.train()
    policy.train()
    torch.manual_seed(9)
    o = pulse(history)
    record(expected, "train", o)
    loss = o["mu"].square().mean() + o["bin_logits"].mean() + o["dist_params_list"][3]["amps"].sum()
    loss.backward()
    for name, p_ in pulse.named_parameters():
        if p_.grad is not None and ("input_proj" in name or "heads.0" in name):
            expected("grad." + name, p_.grad)
    p = policy.treatments(data["continuous"], data["binary"], data["static"], data["time_hours"])
    expected("policy_train.continuous_treatments", p["continuous_treatments"])
    expected("policy_train.binary_treatment_logits", p["binary_treatment_logits"])
    expected("after_train", torch.rand(3))

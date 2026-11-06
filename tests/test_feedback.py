import pytest
import torch

from pulse import feedback as F
from pulse import model as M
from synthetic import part, perturb, protocol

SWITCHES = {"distribution": (True, True), "quantile_only": (True, False),
            "probability_only": (False, True), "sample": (False, False)}


def frozen_policy():
    policy = M.build_policy(protocol())
    perturb(policy, seed=2)
    return policy


@pytest.mark.parametrize("phase", list(SWITCHES))
def test_training_forward(expected, phase):
    data = part(seed=4, n=4)
    history = {k: data[k] for k in ("continuous", "binary", "static", "time_hours", "continuous_mask")}
    policy = frozen_policy()
    model = F.build_pulse(protocol(), part(seed=6, n=16)["binary"], M.Architecture(head="rescnf"))
    model.attach_policy(policy)
    expected(f"{phase}.after_init", torch.rand(3))
    if phase == "distribution":
        for name, value in model.state_dict().items():
            if name.startswith(("adapters.", "probability_adapters.")):
                expected("init." + name, value)
    perturb(model, seed=3)
    model.train()
    torch.manual_seed(11)
    quantiles, probabilities = SWITCHES[phase]
    o = model(history, quantiles=quantiles, probabilities=probabilities)
    for key in ("mu", "log_var", "bin_logits", "bin_probs", "obs_logits", "delta_t_hours"):
        expected(f"{phase}.{key}", o[key])
    for j in (0, 20):
        for key in ("loc", "scale", "amps"):
            expected(f"{phase}.params{j}.{key}", o["dist_params_list"][j][key])
    (o["mu"].square().mean() + o["bin_logits"].mean()).backward()
    for name, p in model.named_parameters():
        if p.grad is not None and ("adapters.0." in name or "adapters.7." in name):
            expected(f"{phase}.grad.{name}", p.grad)
    expected(f"{phase}.after", torch.rand(3))


def test_quantile_features(expected):
    q = torch.sort(torch.randn(2, 3, 5, 19, generator=torch.Generator().manual_seed(4)), -1).values
    q[0, 0, 0] = 0.25
    expected("quantile_features", F.quantile_features(q))


def test_frozen_policy_is_independent_of_grad_mode():
    """The reference calls the trained policy frozen; then grad mode cannot change its outputs."""
    policy = frozen_policy()
    model = F.FeedbackPULSE(M.pulse_model(protocol(), M.Architecture(head="rescnf")))
    model.attach_policy(policy)
    data = part(seed=4, n=4)
    args = (data["continuous"], data["binary"], data["static"], data["time_hours"])
    with torch.no_grad():
        a = policy.treatments(*args)["binary_treatment_logits"]
    b = policy.treatments(*args)["binary_treatment_logits"]
    assert torch.equal(a, b) and not any(p.requires_grad for p in policy.parameters())

import torch

from pulse import cnf
from synthetic import cnf_inputs

# The probability grid of the distribution descriptors.
PROBABILITIES = torch.tensor([0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95,
                              0.975, 0.99, 0.995, 0.999])


def test_against_reference(expected):
    data = cnf_inputs(seed=11)
    raw = data["raw"].clone().requires_grad_(True)
    params = cnf.anchor(cnf.parse(raw), data["anchor"])
    for key in ("loc", "scale", "amps", "centers", "widths"):
        expected(f"param_{key}", params[key])
    loss = cnf.nll(params, data["target"])
    expected("nll", loss)
    loss.mean().backward()
    expected("nll_grad", raw.grad)
    with torch.no_grad():
        expected("sample_given", cnf.sample(params, data["z"]))
        torch.manual_seed(5)
        expected("sample_global", cnf.sample(params))
        expected("after_sample", torch.rand(3))
        expected("mean", cnf.mean(params))
        leading = params["mu"].shape
        stretched = {k: v.unsqueeze(2).expand(leading + (len(PROBABILITIES),) + v.shape[2:])
                     for k, v in params.items()}
        expected("quantiles", cnf.quantile(stretched, PROBABILITIES))
    head = torch.nn.Linear(4, cnf.NUM_PARAMS)
    cnf.initialize(head)
    expected("initial_bias", head.bias.detach())


def test_quantile_inverts_cdf():
    data = cnf_inputs(seed=2, rows=3, variables=4)
    with torch.no_grad():
        params = cnf.parse(data["raw"])
        x = cnf.quantile(params, 0.3)
        z = (x - params["loc"]) / params["scale"]
        back, _ = cnf._integrate(z, params, -1.0)
        assert torch.allclose(0.5 * (1 + torch.erf(back / 2 ** 0.5)), torch.full_like(back, 0.3), atol=1e-3)

"""Behavior checks for the default B-spline distribution."""
import math

import torch

from pulse import heads, model


def test_identity_start_and_density():
    layer = torch.nn.Linear(4, heads.num_params("bspline"))
    heads.initialize("bspline", layer)
    raw = layer(torch.zeros(5, 4))
    params = heads.parse("bspline", raw, torch.zeros(5))
    z = torch.tensor([-1.5, -0.7, 0.0, 0.7, 1.5])
    transformed = heads.SPLINE._forward_value_from_base(z, params)
    assert torch.allclose(transformed, z, atol=2e-3)
    nll = heads.nll("bspline", params, z)
    assert torch.allclose(nll, 0.5 * (z.square() + math.log(2 * math.pi)), atol=3e-2)
    assert model.Architecture().head == "bspline"


def test_trainable_shape_and_finite_sample():
    torch.manual_seed(21)
    raw = (torch.randn(3, 25) * 0.2).requires_grad_()
    params = heads.parse("bspline", raw, torch.zeros(3))
    controls = params["control_points"]
    assert torch.all(controls[..., 1:] > controls[..., :-1])
    loss = heads.nll("bspline", params, torch.tensor([-1.0, 0.2, 1.2])).mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(raw.grad).all()
    assert torch.isfinite(heads.sample("bspline", params)).all()


def test_feedback_training_explains_supported_heads(tmp_path):
    from pulse import train
    import pytest
    with pytest.raises(ValueError, match="continuous-flow head"):
        train.fit("distribution", None, None, tmp_path)

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

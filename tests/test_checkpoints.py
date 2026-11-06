import json
import pickle
from pathlib import Path, PurePosixPath

import pytest
import torch

from pulse import model as M
from synthetic import perturb, protocol


def test_policy_key_map(tmp_path):
    policy = M.build_policy(protocol())
    perturb(policy)
    reference = {"model.model." + k: v for k, v in policy.state_dict().items()}
    torch.save({"model": reference, "config": {}}, tmp_path / "policy_best.pt")
    fresh = M.build_policy(protocol())
    fresh.load_state_dict(M.read_reference_state(tmp_path / "policy_best.pt"), strict=True)
    assert all(torch.equal(a, b) for a, b in zip(fresh.state_dict().values(), policy.state_dict().values()))


def test_pulse_key_map(tmp_path):
    pulse = M.pulse_model(protocol(), M.Architecture(head="rescnf"))
    perturb(pulse)
    reference = {"pulse.model." + k: v for k, v in pulse.state_dict().items()}
    reference["pulse.model.encoders.0.residual_gate_heads.0.weight"] = torch.zeros(1, 64)
    reference["adapters.0.net.0.bias"] = torch.ones(64)
    mapped = M.reference_pulse_state(reference)
    assert mapped["adapters.0.net.0.bias"] is reference["adapters.0.net.0.bias"]
    assert not any("residual_gate" in k for k in mapped)
    fresh = M.pulse_model(protocol(), M.Architecture(head="rescnf"))
    fresh.load_state_dict({k.removeprefix("pulse."): v for k, v in mapped.items() if k.startswith("pulse.")}, strict=True)
    assert all(torch.equal(a, b) for a, b in zip(fresh.state_dict().values(), pulse.state_dict().values()))


def test_artifact_checkpoint(tmp_path):
    policy = M.build_policy(protocol())
    M.save_checkpoint(tmp_path / "policy.pt", policy, [{"epoch": 1, "validation_loss": 0.5}], "0" * 64)
    payload = torch.load(tmp_path / "policy.pt", weights_only=True)
    assert payload["history"][0]["epoch"] == 1 and payload["cache_sha256"] == "0" * 64
    assert M.read_state(tmp_path / "policy.pt").keys() == policy.state_dict().keys()


def test_artifact_reader_does_not_unpickle(tmp_path):
    """A checkpoint that pickles other objects, as reference checkpoints do, is refused."""
    torch.save({"model": {}, "config": PurePosixPath("config")}, tmp_path / "reference.pt")
    with pytest.raises(pickle.UnpicklingError):
        M.read_state(tmp_path / "reference.pt")


def test_reference_names_load_strictly():
    """Names and shapes of reference checkpoints built from the synthetic protocol."""
    keys = json.loads((Path(__file__).parent / "fixtures" / "reference_keys.json").read_text())
    pulse, policy = M.pulse_model(protocol(), M.Architecture(head="rescnf")), M.build_policy(protocol())
    reference = {k: torch.zeros(shape) for k, shape in keys["policy"].items()}
    policy.load_state_dict(M.reference_policy_state(reference), strict=True)
    reference = {k: torch.zeros(shape) for k, shape in keys["pulse"].items()}
    mapped = M.reference_pulse_state(reference)
    pulse.load_state_dict({k.removeprefix("pulse."): v for k, v in mapped.items()}, strict=True)

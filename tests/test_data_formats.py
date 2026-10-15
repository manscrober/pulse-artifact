import torch

from pulse.data import Protocol, load_case, save_case


def test_protocol_round_trip(tmp_path):
    protocol = Protocol(continuous_names=["a", "b"], binary_names=["c"], static_names=["d"],
                        mean=[0.5, -1.0], sd=[1.0, 2.0], static_mean=[3.0], static_sd=[1.0],
                        groups={"g": ["a"], "continuous_treatments": ["b"]},
                        cache_sha256="0" * 64, split={"train": 3, "val": 1, "test": 1})
    protocol.save(tmp_path / "protocol.json")
    assert Protocol.load(tmp_path / "protocol.json") == protocol


def test_case_round_trip(tmp_path):
    continuous, binary = torch.randn(4, 24, 37), torch.randint(0, 2, (4, 24, 8)).float()
    save_case(tmp_path, 12, continuous, binary)
    case = load_case(tmp_path, 12)
    assert case["case"] == 12
    assert torch.equal(case["continuous"], continuous) and torch.equal(case["binary"], binary)

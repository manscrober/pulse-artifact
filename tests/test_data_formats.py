import pytest
import torch

from pulse.data import Protocol, case_path, load_case, save_case


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


def _same_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.int32), b.view(torch.int32))


def test_case_file_keeps_every_bit(tmp_path):
    # Rollout-like paths: recorded hours shared by every path, then paths that differ, with NaN and -0.0.
    continuous = torch.randn(100, 24, 37)
    continuous[:, :5] = torch.randn(5, 37)
    continuous[3, 9, 2], continuous[7, 10, 4] = float("nan"), -0.0
    binary = torch.randint(0, 2, (100, 24, 8)).float()
    binary[:, :5] = binary[:1, :5]
    save_case(tmp_path, 0, continuous, binary)
    case = load_case(tmp_path, 0)
    assert _same_bits(case["continuous"], continuous) and _same_bits(case["binary"], binary)
    # Event values other than exact zeros and ones are stored as numbers.
    binary[1, 20, 3], binary[2, 21, 1] = 0.5, -0.0
    save_case(tmp_path, 1, continuous, binary)
    assert _same_bits(load_case(tmp_path, 1)["binary"], binary)
    # Paths that agree in every hour, one path, and an empty event panel.
    for i, (c, b) in enumerate(((continuous[:1].expand(3, -1, -1), binary[:3]), (continuous[:1], binary[:1]),
                                (continuous[:2], torch.zeros(2, 24, 0)))):
        save_case(tmp_path, 2 + i, c, b)
        case = load_case(tmp_path, 2 + i)
        assert _same_bits(case["continuous"], c.contiguous()) and _same_bits(case["binary"], b.contiguous())


def test_case_file_is_compact_and_reads_the_tensor_format(tmp_path):
    continuous = torch.randn(100, 24, 37)
    continuous[:, :5] = torch.randn(5, 37)
    binary = (torch.rand(100, 24, 8) < 0.05).float()
    save_case(tmp_path, 0, continuous, binary)
    tensors = tmp_path / "tensors"
    tensors.mkdir()
    torch.save({"case": 0, "continuous": continuous, "binary": binary}, case_path(tensors, 0))
    assert case_path(tmp_path, 0).stat().st_size < 0.85 * case_path(tensors, 0).stat().st_size
    case = load_case(tensors, 0)
    assert torch.equal(case["continuous"], continuous) and torch.equal(case["binary"], binary)
    torch.save({"case": 0, "format": "other"}, case_path(tensors, 0))
    with pytest.raises(ValueError, match="unknown format"):
        load_case(tensors, 0)

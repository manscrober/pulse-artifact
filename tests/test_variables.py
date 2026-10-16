import pytest
import torch

from pulse import variables as V
from synthetic import trajectories


def test_lists():
    assert len(V.CONTINUOUS) == 37 and len(V.BINARY) == 8 and len(V.STATIC) == 6
    assert set(V.RANGES) == set(V.CONTINUOUS)
    assert all(name in V.CONTINUOUS for name in V.ORDINAL)
    assert all(name in V.BINARY for name in V.ABSORBING)


@pytest.fixture(scope="module")
def data():
    """Synthetic trajectories shared by the reference tests, which only read them."""
    return trajectories(seed=3)


def test_bounds(expected):
    low, high = V.bounds()
    expected("bounds_low", low)
    expected("bounds_high", high)


def test_update_derived(expected, data):
    derived = data["continuous"].clone()
    baseline = derived[:, 0].clone()
    for t in range(derived.shape[1]):
        V.update_derived(derived, t, baseline)
    expected("derived", derived)


def test_forward_fill_zero_invalid(expected, data):
    cont = data["continuous"]
    expected("fill_zero_invalid", V.forward_fill_zero_invalid(cont[:, 5].clone(), cont[:, 4], data["event_now"]))
    expected("fill_zero_invalid_no_event", V.forward_fill_zero_invalid(cont[:, 5].clone(), cont[:, 4]))


def test_forward_fill_zero_invalid_over_all_hours(expected, data):
    sequence = data["continuous"].clone()
    for t in range(1, sequence.shape[1]):
        V.forward_fill_zero_invalid(sequence[:, t], sequence[:, t - 1])
    expected("fill_zero_invalid_all_hours", sequence)


def test_event_times(expected, data):
    death, release, event = V.event_times(data["binary"])
    expected("death", torch.tensor(death))
    expected("release", torch.tensor(release))
    expected("event", event)


def test_zero_invalid_mask(expected, data):
    expected("zero_invalid_mask", V.zero_invalid_mask(data["continuous"], data["keep"].float()))


def test_forward_fill(expected, data):
    expected("forward_fill", V.forward_fill(data["continuous"], data["keep"], data["means"]))


def test_fill_after_events(expected, data):
    death, release, _ = V.event_times(data["binary"])
    filled = V.forward_fill(data["continuous"], data["keep"], data["means"])
    expected("fill_after_events", V.fill_after_events(filled, death, release, data["means"]))


def test_event_hours_need_one_per_row():
    cont = torch.zeros(3, 6, len(V.CONTINUOUS))
    with pytest.raises(ValueError):
        V.fill_after_events(cont, [2, 4], [6, 6, 6], torch.zeros(len(V.CONTINUOUS)))


def test_event_rule():
    binary = torch.zeros(3, 6, len(V.BINARY))
    binary[0, 2:, V.RELEASE] = 1
    binary[0, 4:, V.DEATH] = 1
    binary[1, 3:, V.RELEASE] = 1
    binary[1, 3:, V.DEATH] = 1
    death, release, event = V.event_times(binary)
    assert (death, release) == ([6, 3, 6], [2, 6, 6])
    assert event.sum(1).tolist() == [4, 3, 0]
